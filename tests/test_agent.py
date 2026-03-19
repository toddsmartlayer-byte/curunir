# tests/test_agent.py
import json
from unittest.mock import AsyncMock, patch

import pytest

from src.agent.agent import Agent
from src.llm import LLMResponse
from src.skills import build_skill_manifest


@pytest.fixture
def agent(agent_config):
    return Agent(agent_config)


class TestAgentHandle:
    async def test_returns_text_response(self, agent):
        mock_response = LLMResponse(text="Hello!", tool_calls=None)
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response):
            result = await agent.handle("hi", "test-session")
        assert result == "Hello!"

    async def test_session_persistence(self, agent):
        response1 = LLMResponse(text="First", tool_calls=None)
        response2 = LLMResponse(text="Second", tool_calls=None)
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, side_effect=[response1, response2]):
            await agent.handle("msg1", "s1")
            await agent.handle("msg2", "s1")
        history = agent.sessions["s1"]
        assert len(history) == 4  # user, assistant, user, assistant
        assert history[0]["content"] == "msg1"
        assert history[1]["content"] == "First"

    async def test_separate_sessions(self, agent):
        mock_response = LLMResponse(text="Reply", tool_calls=None)
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response):
            await agent.handle("msg1", "session-a")
            await agent.handle("msg2", "session-b")
        assert len(agent.sessions["session-a"]) == 2
        assert len(agent.sessions["session-b"]) == 2

    async def test_executes_tool_calls(self, agent):
        tool_response = LLMResponse(
            text=None,
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": "echo tool_test"})},
            }],
        )
        text_response = LLMResponse(text="Done!", tool_calls=None)

        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, side_effect=[tool_response, text_response]):
            result = await agent.handle("run something", "s1")

        assert result == "Done!"
        history = agent.sessions["s1"]
        # user, assistant(tool_calls), tool, assistant(text)
        assert len(history) == 4
        assert history[1].get("tool_calls") is not None
        assert history[2]["role"] == "tool"
        assert "tool_test" in history[2]["content"]

    async def test_handles_combined_text_and_tool_calls(self, agent):
        combined = LLMResponse(
            text="Let me check",
            tool_calls=[{
                "id": "call_2",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": "echo combined"})},
            }],
        )
        final = LLMResponse(text="All done", tool_calls=None)

        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, side_effect=[combined, final]):
            result = await agent.handle("check", "s1")

        assert result == "All done"
        # The combined response should have content preserved
        assert agent.sessions["s1"][1].get("content") == "Let me check"
        assert agent.sessions["s1"][1].get("tool_calls") is not None

    async def test_empty_response_returns_error(self, agent):
        empty = LLMResponse(text=None, tool_calls=None)
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=empty):
            result = await agent.handle("hello", "s1")
        assert "error" in result.lower()

    async def test_max_iterations(self, agent_config):
        agent_config.max_iterations = 2
        agent = Agent(agent_config)

        tool_response = LLMResponse(
            text=None,
            tool_calls=[{
                "id": "call_loop",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": "echo loop"})},
            }],
        )
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=tool_response):
            result = await agent.handle("loop forever", "s1")
        assert "iteration limit" in result.lower()


class TestDelegateToolExecution:
    async def test_delegate_via_agent_handle(self, agent):
        """Delegate tool calls go through the unified execute_tool_call."""
        tool_response = LLMResponse(
            text=None,
            tool_calls=[{
                "id": "call_delegate",
                "type": "function",
                "function": {"name": "delegate", "arguments": json.dumps({"task": "say hello"})},
            }],
        )
        text_response = LLMResponse(text="Done", tool_calls=None)

        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, side_effect=[tool_response, text_response]), \
             patch("src.agent.agent.execute_tool_call", new_callable=AsyncMock, return_value="sub-agent result"):
            result = await agent.handle("delegate this", "s1")

        assert result == "Done"
        # Verify the tool result was recorded in history
        history = agent.sessions["s1"]
        tool_msg = [m for m in history if m["role"] == "tool"][0]
        assert tool_msg["content"] == "sub-agent result"


class TestToolAllowlist:
    async def test_only_allowed_tools_in_schemas(self, agent_config):
        agent = Agent(agent_config, tools=["read", "grep"])
        mock_response = LLMResponse(text="Hi", tool_calls=None)
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
            await agent.handle("hello", "s1")
        schemas = mock_llm.call_args[0][2]
        tool_names = {s["function"]["name"] for s in schemas}
        assert tool_names == {"read", "grep"}

    async def test_none_means_all_tools(self, agent_config):
        agent = Agent(agent_config)  # tools=None (default)
        mock_response = LLMResponse(text="Hi", tool_calls=None)
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
            await agent.handle("hello", "s1")
        schemas = mock_llm.call_args[0][2]
        assert len(schemas) == 10  # all tools including delegate, web_fetch, and schedule


class TestHistoryTruncation:
    async def test_trims_old_messages_when_over_limit(self, agent_config):
        agent = Agent(agent_config)
        session_id = "s-trunc"
        history = agent.sessions.setdefault(session_id, [])
        for i in range(100):
            history.append({"role": "user", "content": "x" * 10_000})
            history.append({"role": "assistant", "content": "y" * 10_000})

        mock_response = LLMResponse(text="ok", tool_calls=None)
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
            await agent.handle("new message", "s-trunc")

        messages = mock_llm.call_args[0][1]
        # Should be trimmed (system + trimmed history + new user msg)
        assert len(messages) < 202

    async def test_no_trim_when_single_user_message(self, agent_config):
        """Sub-agents have one user message — trimming must not remove it."""
        agent = Agent(agent_config)
        session_id = "s-single"
        history = agent.sessions.setdefault(session_id, [])
        # Simulate sub-agent: one user msg, then a huge tool result
        history.append({"role": "user", "content": "analyze this"})
        history.append({"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}]})
        history.append({"role": "tool", "tool_call_id": "c1", "content": "x" * 1_000_000})

        mock_response = LLMResponse(text="ok", tool_calls=None)
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
            await agent.handle("continue", session_id)

        messages = mock_llm.call_args[0][1]
        non_system = [m for m in messages if m["role"] != "system"]
        # Must still have at least one user message
        assert any(m["role"] == "user" for m in non_system)

    async def test_truncation_preserves_message_pairs(self, agent_config):
        """Truncation should not leave orphaned tool results or split pairs."""
        agent = Agent(agent_config)
        session_id = "s-pairs"
        history = agent.sessions.setdefault(session_id, [])
        for i in range(50):
            history.append({"role": "user", "content": "x" * 20_000})
            history.append({"role": "assistant", "content": "y" * 20_000})

        mock_response = LLMResponse(text="ok", tool_calls=None)
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
            await agent.handle("new message", "s-pairs")

        messages = mock_llm.call_args[0][1]
        # After system prompt, first message should be "user" (not orphaned assistant/tool)
        non_system = [m for m in messages if m["role"] != "system"]
        assert non_system[0]["role"] == "user"

    async def test_system_task_trims_tool_groups_after_user_prompt(self, agent_config):
        """System tasks (single user message) should trim old tool groups, not the task prompt."""
        agent = Agent(agent_config)
        session_id = "sched:trim:1"
        history = agent.sessions.setdefault(session_id, [])
        # Simulate: task prompt + many large tool call rounds
        history.append({"role": "user", "content": "## Scheduled Task\nDo something."})
        for i in range(20):
            history.append({"role": "assistant", "tool_calls": [{"id": f"c{i}", "function": {"name": "bash", "arguments": "{}"}}]})
            history.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 100_000})

        mock_response = LLMResponse(text="done", tool_calls=None)
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
            result = await agent.handle("", session_id, system_task_prompt="Do something.")

        messages = mock_llm.call_args[0][1]
        non_system = [m for m in messages if m["role"] != "system"]
        # Task prompt must survive trimming
        assert non_system[0]["role"] == "user"
        assert "Scheduled Task" in non_system[0]["content"]
        # History should have been trimmed (less than original 41 messages)
        assert len(non_system) < 41


class TestSystemTaskMode:
    async def test_system_task_sends_user_message(self, agent):
        mock_response = LLMResponse(text="Task done.", tool_calls=None)
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
            result = await agent.handle("", "sched:test:123", system_task_prompt="Do the thing.")
        assert result == "Task done."
        # Task prompt should be sent as a user message for provider compatibility
        messages = mock_llm.call_args[0][1]
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert "Do the thing." in user_msgs[0]["content"]

    async def test_system_task_prompt_in_user_message(self, agent):
        mock_response = LLMResponse(text="Done", tool_calls=None)
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
            await agent.handle("", "sched:test:123", system_task_prompt="Check PRs.")
        messages = mock_llm.call_args[0][1]
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert "## Scheduled Task" in user_msgs[0]["content"]
        assert "Check PRs." in user_msgs[0]["content"]

    async def test_system_task_cleans_up_session(self, agent):
        mock_response = LLMResponse(text="Done", tool_calls=None)
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response):
            await agent.handle("", "sched:test:123", system_task_prompt="Do it.")
        # Session should be cleaned up after completion
        assert "sched:test:123" not in agent.sessions

    async def test_system_task_with_tool_calls(self, agent):
        tool_response = LLMResponse(
            text=None,
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": "echo scheduled"})},
            }],
        )
        text_response = LLMResponse(text="Task complete.", tool_calls=None)
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, side_effect=[tool_response, text_response]):
            result = await agent.handle("", "sched:test:456", system_task_prompt="Run a command.")
        assert result == "Task complete."
        # Session cleaned up after completion
        assert "sched:test:456" not in agent.sessions

    async def test_normal_handle_unchanged(self, agent):
        """Ensure regular user messages still work as before."""
        mock_response = LLMResponse(text="Hello!", tool_calls=None)
        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response):
            result = await agent.handle("hi", "normal-session")
        assert result == "Hello!"
        history = agent.sessions["normal-session"]
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "hi"


class TestAgentInit:
    def test_loads_identity(self, agent):
        assert "test assistant" in agent._identity.lower()

    def test_missing_identity_raises(self, tmp_path, tmp_skills):
        from src.config import AgentConfig
        config = AgentConfig(
            identity_file=tmp_path / "nonexistent.md",
            skills_dir=tmp_skills,
        )
        with pytest.raises(FileNotFoundError):
            Agent(config)


class TestRuntimeSkillDetection:
    async def test_new_session_picks_up_new_skills(self, agent_config):
        agent = Agent(agent_config)
        mock_response = LLMResponse(text="ok", tool_calls=None)

        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
            await agent.handle("hello", "s1")

        # Add a new skill after first session
        skill_dir = agent_config.skills_dir / "new_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: new_skill\ndescription: A dynamically added skill\n---\n"
        )

        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
            await agent.handle("hello", "s2")

        system_prompt = mock_llm.call_args[0][1][0]["content"]
        assert "new_skill" in system_prompt

    async def test_same_session_does_not_rescan(self, agent_config):
        agent = Agent(agent_config)
        mock_response = LLMResponse(text="ok", tool_calls=None)

        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response), \
             patch("src.agent.agent.build_skill_manifest", wraps=build_skill_manifest) as mock_manifest:
            await agent.handle("msg1", "s1")
            mock_manifest.reset_mock()
            await agent.handle("msg2", "s1")

        mock_manifest.assert_not_called()

    async def test_empty_skills_dir_after_refresh(self, agent_config):
        agent = Agent(agent_config)
        mock_response = LLMResponse(text="ok", tool_calls=None)

        with patch("src.agent.agent.call_llm", new_callable=AsyncMock, return_value=mock_response):
            result = await agent.handle("hello", "s1")

        assert result == "ok"
