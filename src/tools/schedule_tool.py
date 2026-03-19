# src/tools/schedule_tool.py
"""CRUD operations for scheduled tasks stored in context/schedules.json."""

import json
import os
import tempfile
import time
from pathlib import Path

from croniter import croniter

from src.config import AgentConfig


def _schedule_path(config: AgentConfig) -> Path:
    return config.context_dir / "schedules.json"


def _load(config: AgentConfig) -> list[dict]:
    path = _schedule_path(config)
    if not path.exists():
        return []
    return json.loads(path.read_text())


def _save(config: AgentConfig, tasks: list[dict]) -> None:
    path = _schedule_path(config)
    # Atomic write: temp file + rename to prevent partial reads
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(tasks, f, indent=2)
        os.rename(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _validate_cron(expr: str) -> bool:
    try:
        croniter(expr)
        return True
    except (ValueError, KeyError):
        return False


def exec_schedule(args: dict, config: AgentConfig) -> str:
    action = args.get("action")
    if not action:
        return "Error: missing 'action' field. Use: list, add, update, remove."

    match action:
        case "list":
            return _list(config)
        case "add":
            return _add(args, config)
        case "update":
            return _update(args, config)
        case "remove":
            return _remove(args, config)
        case _:
            return f"Error: unknown action '{action}'. Use: list, add, update, remove."


def _list(config: AgentConfig) -> str:
    tasks = _load(config)
    if not tasks:
        return "No scheduled tasks."
    lines = []
    for t in tasks:
        status = "enabled" if t.get("enabled", True) else "disabled"
        skill = f" (skill: {t['skill']})" if t.get("skill") else ""
        lines.append(f"- **{t['id']}** `{t['cron']}` [{status}]{skill}\n  {t['prompt']}")
    return "\n".join(lines)


def _add(args: dict, config: AgentConfig) -> str:
    task_id = args.get("id")
    cron = args.get("cron")
    prompt = args.get("prompt")

    if not task_id or not cron or not prompt:
        return "Error: missing required fields — 'add' needs 'id', 'cron', and 'prompt'."

    if not _validate_cron(cron):
        return f"Error: invalid cron expression '{cron}'."

    tasks = _load(config)
    if any(t["id"] == task_id for t in tasks):
        return f"Error: task '{task_id}' already exists. Use 'update' to modify it."

    tasks.append({
        "id": task_id,
        "cron": cron,
        "prompt": prompt,
        "skill": args.get("skill"),
        "enabled": True,
        "last_run": 0,
    })
    _save(config, tasks)
    return f"Task '{task_id}' added — scheduled at `{cron}`."


def _update(args: dict, config: AgentConfig) -> str:
    task_id = args.get("id")
    if not task_id:
        return "Error: 'update' requires 'id' field."

    tasks = _load(config)
    task = next((t for t in tasks if t["id"] == task_id), None)
    if not task:
        return f"Error: task '{task_id}' not found."

    if "cron" in args:
        if not _validate_cron(args["cron"]):
            return f"Error: invalid cron expression '{args['cron']}'."
        task["cron"] = args["cron"]
    if "prompt" in args:
        task["prompt"] = args["prompt"]
    if "skill" in args:
        task["skill"] = args["skill"]
    if "enabled" in args:
        task["enabled"] = bool(args["enabled"])

    _save(config, tasks)
    return f"Task '{task_id}' updated."


def _remove(args: dict, config: AgentConfig) -> str:
    task_id = args.get("id")
    if not task_id:
        return "Error: 'remove' requires 'id' field."

    tasks = _load(config)
    original_len = len(tasks)
    tasks = [t for t in tasks if t["id"] != task_id]
    if len(tasks) == original_len:
        return f"Error: task '{task_id}' not found."

    _save(config, tasks)
    return f"Task '{task_id}' removed."
