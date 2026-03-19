# src/agent/agent.py
import json
import logging
from datetime import datetime

import litellm

logger = logging.getLogger(__name__)
from src.config import AgentConfig
from src.llm import call_llm
from src.skills import build_skill_manifest, parse_frontmatter
from src.tools.dispatcher import execute_tool_call
from src.tools.schemas import get_tool_schemas


_MAX_HISTORY_CHARS = 250_000  # ~80k tokens, leaves room for system prompt + tool schemas + max_tokens

# Map tool names to their key argument(s) for log display
_TOOL_KEY_ARGS: dict[str, list[str]] = {
    "web_fetch": ["url"],
    "bash": ["command"],
    "write": ["file_path"],
    "read": ["file_path"],
    "edit": ["file_path"],
    "glob": ["pattern"],
    "grep": ["pattern"],
    "load_skill": ["name"],
    "delegate": ["task"],
    "attach": ["path"],
}

_MAX_ARG_LEN = 120


def _display_name(tool_name: str) -> str:
    """Convert snake_case tool name to PascalCase display name."""
    return "".join(part.capitalize() for part in tool_name.split("_"))


def _tool_detail_lines(name: str, args_str: str) -> list[str]:
    """Build tree-formatted detail lines for a tool call."""
    try:
        args = json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        return [f"├─ {_display_name(name)} (unparseable args)"]

    key_names = _TOOL_KEY_ARGS.get(name, list(args.keys())[:1])
    lines = []
    for key in key_names:
        val = args.get(key, "")
        val_str = str(val)
        if len(val_str) > _MAX_ARG_LEN:
            val_str = val_str[:_MAX_ARG_LEN] + "..."
        lines.append(f"{_display_name(name)} {val_str}")

    if not lines:
        return [f"╰─ {_display_name(name)}"]

    result = []
    for i, line in enumerate(lines):
        connector = "╰─" if i == len(lines) - 1 else "├─"
        result.append(f"{connector} {line}")
    return result


def _estimate_chars(messages: list[dict]) -> int:
    """Rough character count across all message contents."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                total += len(str(block))
    return total


def _trim_history(history: list[dict], max_chars: int = _MAX_HISTORY_CHARS) -> None:
    """Remove oldest messages in coherent groups until under the char limit.

    Groups: user+assistant pairs, or assistant(tool_calls)+tool+...+tool sequences.
    Always removes from the front so the most recent context is preserved.
    Keeps at least one user message to avoid empty-messages API errors.
    When only one user message exists (e.g. system-task sessions), trims
    assistant+tool groups after it.
    """
    user_count = sum(1 for m in history if m["role"] == "user")

    if user_count > 1:
        # Multi-user-message session: trim by user message groups, keep at least one
        while user_count > 1 and _estimate_chars(history) > max_chars:
            if history[0]["role"] == "user":
                user_count -= 1
            history.pop(0)
            while history and history[0]["role"] != "user":
                history.pop(0)
    elif user_count == 1 and len(history) > 1:
        # Single user message (e.g. system task): keep first message,
        # trim assistant+tool groups after it
        while len(history) > 1 and _estimate_chars(history) > max_chars:
            del history[1]
            while len(history) > 1 and history[1]["role"] == "tool":
                del history[1]


def _is_context_overflow(exc: Exception) -> bool:
    """Check if an exception is a context window / input length overflow."""
    if isinstance(exc, litellm.ContextWindowExceededError):
        return True
    msg = str(exc).lower()
    return "context limit" in msg or "prompt is too long" in msg or "exceed" in msg and "token" in msg


def _parse_skill_tools(skill_content: str) -> list[str]:
    """Extract required tool names from a skill's frontmatter."""
    fm = parse_frontmatter(skill_content)
    tools_str = fm.get("tools", "")
    if not tools_str:
        return []
    return [t.strip() for t in tools_str.split(",") if t.strip()]


class Agent:
    def __init__(self, config: AgentConfig, tools: list[str] | None = None):
        self.config = config
        self.sessions: dict[str, list[dict]] = {}
        if not config.identity_file.exists():
            raise FileNotFoundError(
                f"Identity file not found: {config.identity_file}. "
                "Curunir requires an identity file to start."
            )
        self._identity: str = config.identity_file.read_text()
        self._skill_manifest: str = build_skill_manifest(config.skills_dir)
        self.tools = tools  # None = all tools
        self._session_tools: dict[str, set[str]] = {}  # extra tools loaded by skills

    def _refresh_manifest(self) -> None:
        self._skill_manifest = build_skill_manifest(self.config.skills_dir)

    def _build_system_prompt(self) -> str:
        parts = [self._identity]
        if self._skill_manifest:
            parts.append(self._skill_manifest)
        parts.append(f"Current time: {datetime.now().isoformat()}")
        return "\n\n".join(parts)

    def _get_tool_schemas(self, session_id: str | None = None) -> list[dict]:
        base = get_tool_schemas(self.tools)
        if session_id and session_id in self._session_tools:
            extra = get_tool_schemas(list(self._session_tools[session_id]))
            base = base + extra
        return base

    async def handle(
        self, message: str | list, session_id: str,
        on_tool_call=None, attachments: list[dict] | None = None,
        system_task_prompt: str | None = None,
    ) -> str:
        """Process a message and return the agent's response.

        Args:
            message: User input text, or a list of content blocks for multimodal input.
            session_id: Session identifier for conversation history.
            on_tool_call: Optional async callback called with (name, args_str)
                          for each tool call, enabling real-time UI updates.
            attachments: Optional list that will be populated with any files
                         the agent attaches during this request via the attach tool.
        """
        if session_id not in self.sessions:
            self._refresh_manifest()
        history = self.sessions.setdefault(session_id, [])

        if system_task_prompt:
            # System-initiated task: inject task as a user message so all LLM
            # providers accept the request (some reject system-only conversations).
            history.append({"role": "user", "content": f"## Scheduled Task\n{system_task_prompt}"})
        else:
            history.append({"role": "user", "content": message})
        system_prompt = self._build_system_prompt()
        _trim_history(history)
        messages = [{"role": "system", "content": system_prompt}] + history

        sid = session_id[:8]
        msg_chars = _estimate_chars(history)
        logger.info("[%s] agent loop start — %d messages, ~%dk chars", sid, len(history), msg_chars // 1000)

        tool_schemas = self._get_tool_schemas(session_id)

        for iteration in range(self.config.max_iterations):
            logger.debug("[%s] iteration %d — calling LLM (%d messages)", sid, iteration + 1, len(messages))
            try:
                response = await call_llm(self.config.model, messages, tool_schemas, api_base=self.config.api_base, openrouter_provider=self.config.openrouter_provider)
            except (litellm.ContextWindowExceededError, litellm.BadRequestError) as e:
                if not _is_context_overflow(e):
                    raise
                logger.warning("[%s] context window exceeded, trimming history to %dk chars", sid, _MAX_HISTORY_CHARS // 2000)
                _trim_history(history, max_chars=_MAX_HISTORY_CHARS // 2)
                if not history:
                    return "Sorry, the message was too long for me to process."
                messages = [{"role": "system", "content": system_prompt}] + history
                try:
                    response = await call_llm(self.config.model, messages, tool_schemas, api_base=self.config.api_base, openrouter_provider=self.config.openrouter_provider)
                except (litellm.ContextWindowExceededError, litellm.BadRequestError) as e2:
                    if not _is_context_overflow(e2):
                        raise
                    logger.error("[%s] context window still exceeded after trim, aborting", sid)
                    return "Sorry, the conversation is too long. Please start a new thread."

            if response.tool_calls:
                assistant_msg: dict = {"role": "assistant", "tool_calls": response.tool_calls}
                if response.text:
                    assistant_msg["content"] = response.text
                history.append(assistant_msg)

                for tool_call in response.tool_calls:
                    name = tool_call["function"]["name"]
                    args_str = tool_call["function"]["arguments"]
                    detail_lines = _tool_detail_lines(name, args_str)
                    logger.info("[%s] tool call: %s", sid, name)
                    for line in detail_lines:
                        logger.info("  %s", line)

                    if on_tool_call:
                        await on_tool_call(name, args_str)

                    result = await execute_tool_call(
                        name,
                        json.loads(args_str),
                        self.config,
                        attachments=attachments,
                        on_tool_call=on_tool_call,
                    )

                    # After load_skill, check for required tools in frontmatter
                    if name == "load_skill":
                        required = _parse_skill_tools(result)
                        if required:
                            self._session_tools.setdefault(session_id, set()).update(required)
                            tool_schemas = self._get_tool_schemas(session_id)
                            logger.info("[%s] skill loaded tools: %s", sid, required)

                    result_preview = result[:200] if result else "(empty)"
                    logger.debug("[%s] tool result: %s", sid, result_preview)
                    history.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result,
                    })

                _trim_history(history)
                messages = [{"role": "system", "content": system_prompt}] + history
                continue

            if response.text:
                logger.info("[%s] agent done after %d iteration(s), response length: %d chars", sid, iteration + 1, len(response.text))
                history.append({"role": "assistant", "content": response.text})
                if system_task_prompt:
                    self.sessions.pop(session_id, None)
                return response.text

            logger.warning("[%s] LLM returned empty response", sid)
            history.append({"role": "assistant", "content": ""})
            if system_task_prompt:
                self.sessions.pop(session_id, None)
            return "Error: LLM returned empty response."

        logger.warning("[%s] iteration limit reached (%d)", sid, self.config.max_iterations)
        if system_task_prompt:
            self.sessions.pop(session_id, None)
        return "Iteration limit reached."
