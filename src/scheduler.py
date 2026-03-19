# src/scheduler.py
"""Async scheduler that fires scheduled tasks via agent.handle()."""

import asyncio
import json
import logging
import os
import tempfile
import time

from croniter import croniter

from src.skills import load_skill

logger = logging.getLogger(__name__)


def _schedule_path(config):
    return config.context_dir / "schedules.json"


def _load_tasks(config) -> list[dict]:
    path = _schedule_path(config)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read schedules.json: %s", e)
        return []


def _update_last_run(config, task_id: str, timestamp: int) -> None:
    """Atomically update a task's last_run timestamp in the schedule file."""
    path = _schedule_path(config)
    try:
        tasks = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, FileNotFoundError):
        return
    for t in tasks:
        if t["id"] == task_id:
            t["last_run"] = timestamp
            break
    else:
        return  # task not found
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


def _next_fire_time(task: dict, now: float) -> float | None:
    """Return the fire time if task is due, else None."""
    last_run = task.get("last_run", 0)
    if last_run >= now:
        return None
    try:
        cron = croniter(task["cron"], last_run)
        next_fire = cron.get_next(float)
        return next_fire if next_fire <= now else None
    except (ValueError, KeyError):
        return None


async def _run_task(agent, task_id: str, session_id: str, prompt: str) -> None:
    """Run a single scheduled task. Called via asyncio.create_task() for concurrency."""
    try:
        await agent.handle(
            message="",
            session_id=session_id,
            system_task_prompt=prompt,
        )
        logger.info("Scheduled task completed: %s", task_id)
    except Exception as e:
        logger.error("Scheduled task failed: %s — %s", task_id, e)


async def _check_and_fire(agent) -> list[str]:
    """Check all tasks and fire any that are due. Returns list of fired task IDs."""
    tasks = _load_tasks(agent.config)
    fired = []
    now = time.time()

    for task in tasks:
        if not task.get("enabled", True):
            continue
        fire_time = _next_fire_time(task, now)
        if fire_time is None:
            continue

        task_id = task["id"]
        prompt = task["prompt"]

        # Load skill content if specified
        if task.get("skill"):
            skill_content = load_skill(task["skill"], agent.config.skills_dir)
            if not skill_content.startswith("Skill not found"):
                prompt = skill_content + "\n\n" + prompt

        # Advance last_run past the matched fire time to prevent double-fires
        timestamp = max(int(now), int(fire_time))
        session_id = f"sched:{task_id}:{timestamp}"

        _update_last_run(agent.config, task_id, timestamp)

        logger.info("Firing scheduled task: %s (session %s)", task_id, session_id)
        asyncio.create_task(_run_task(agent, task_id, session_id, prompt))
        fired.append(task_id)

    return fired


async def run_scheduler(agent, interval_sec: int = 60):
    """Main scheduler loop. Runs forever, checking for due tasks every interval."""
    logger.info("Scheduler started (interval: %ds)", interval_sec)
    while True:
        await asyncio.sleep(interval_sec)
        try:
            fired = await _check_and_fire(agent)
            if fired:
                logger.info("Scheduler tick: fired %s", fired)
        except Exception as e:
            logger.error("Scheduler tick error: %s", e)
