# tests/test_scheduler.py
import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from croniter import croniter

from src.scheduler import _check_and_fire, _update_last_run, _next_fire_time


@pytest.fixture
def schedule_file(tmp_path, agent_config):
    agent_config.context_dir = tmp_path
    sf = tmp_path / "schedules.json"
    sf.write_text("[]")
    return sf


class TestNextFireTime:
    def test_never_run_task_is_due(self):
        task = {"cron": "* * * * *", "last_run": 0}
        assert _next_fire_time(task, time.time()) is not None

    def test_recently_run_task_not_due(self):
        task = {"cron": "0 9 * * *", "last_run": int(time.time())}
        assert _next_fire_time(task, time.time()) is None

    def test_last_run_in_future_not_due(self):
        task = {"cron": "* * * * *", "last_run": int(time.time()) + 3600}
        assert _next_fire_time(task, time.time()) is None

    def test_invalid_cron_not_due(self):
        task = {"cron": "not valid", "last_run": 0}
        assert _next_fire_time(task, time.time()) is None

    def test_hourly_task_due_after_one_hour(self):
        now = time.time()
        task = {"cron": "0 * * * *", "last_run": int(now) - 3601}
        assert _next_fire_time(task, now) is not None

    def test_returns_actual_fire_time(self):
        now = time.time()
        task = {"cron": "0 * * * *", "last_run": int(now) - 3601}
        fire_time = _next_fire_time(task, now)
        assert fire_time is not None
        assert fire_time <= now

    def test_no_double_fire_when_last_run_before_cron_match(self):
        """Regression: if last_run is set to a time before the cron match,
        firing once and setting last_run=max(now, fire_time) should prevent
        the same match from triggering again."""
        # Simulate: cron matches at :00 of some hour, last_run is 30s before that
        now = time.time()
        # Find the most recent top-of-hour
        cron_match = int(now) - (int(now) % 3600)
        # last_run is 30s before the cron match (simulates premature first fire)
        task = {"cron": "0 * * * *", "last_run": cron_match - 30}
        fire_time = _next_fire_time(task, now)
        assert fire_time is not None  # first fire is due

        # After firing, last_run is set to max(now, fire_time)
        task["last_run"] = max(int(now), int(fire_time))
        # The same cron match should NOT trigger again
        assert _next_fire_time(task, now) is None


class TestUpdateLastRun:
    def test_updates_last_run(self, schedule_file, agent_config):
        schedule_file.write_text(json.dumps([
            {"id": "t1", "cron": "* * * * *", "prompt": "p", "skill": None, "enabled": True, "last_run": 0},
        ]))
        now = int(time.time())
        _update_last_run(agent_config, "t1", now)
        tasks = json.loads(schedule_file.read_text())
        assert tasks[0]["last_run"] == now

    def test_update_nonexistent_task_is_noop(self, schedule_file, agent_config):
        schedule_file.write_text(json.dumps([
            {"id": "t1", "cron": "* * * * *", "prompt": "p", "skill": None, "enabled": True, "last_run": 0},
        ]))
        _update_last_run(agent_config, "nope", 123)
        tasks = json.loads(schedule_file.read_text())
        assert tasks[0]["last_run"] == 0  # unchanged


class TestCheckAndFire:
    async def test_fires_due_task(self, schedule_file, agent_config):
        # Task with last_run=0 (never run) and cron="* * * * *" (every minute) → should fire
        schedule_file.write_text(json.dumps([
            {"id": "t1", "cron": "* * * * *", "prompt": "do it", "skill": None, "enabled": True, "last_run": 0},
        ]))
        mock_agent = AsyncMock()
        mock_agent.config = agent_config

        fired = await _check_and_fire(mock_agent)
        # Let the create_task coroutine run
        await asyncio.sleep(0)

        assert "t1" in fired
        mock_agent.handle.assert_called_once()
        call_kwargs = mock_agent.handle.call_args
        assert call_kwargs.kwargs["system_task_prompt"] == "do it"
        assert call_kwargs.kwargs["session_id"].startswith("sched:t1:")

    async def test_skips_disabled_task(self, schedule_file, agent_config):
        schedule_file.write_text(json.dumps([
            {"id": "t1", "cron": "* * * * *", "prompt": "do it", "skill": None, "enabled": False, "last_run": 0},
        ]))
        mock_agent = AsyncMock()
        mock_agent.config = agent_config

        fired = await _check_and_fire(mock_agent)

        assert fired == []
        mock_agent.handle.assert_not_called()

    async def test_skips_recently_run_task(self, schedule_file, agent_config):
        now = int(time.time())
        schedule_file.write_text(json.dumps([
            {"id": "t1", "cron": "* * * * *", "prompt": "do it", "skill": None, "enabled": True, "last_run": now},
        ]))
        mock_agent = AsyncMock()
        mock_agent.config = agent_config

        fired = await _check_and_fire(mock_agent)

        assert fired == []
        mock_agent.handle.assert_not_called()

    async def test_handles_malformed_json(self, schedule_file, agent_config):
        schedule_file.write_text("not json{{{")
        mock_agent = AsyncMock()
        mock_agent.config = agent_config

        fired = await _check_and_fire(mock_agent)

        assert fired == []

    async def test_handles_missing_file(self, tmp_path, agent_config):
        agent_config.context_dir = tmp_path
        # No schedules.json exists
        mock_agent = AsyncMock()
        mock_agent.config = agent_config

        fired = await _check_and_fire(mock_agent)

        assert fired == []

    async def test_loads_skill_into_prompt(self, schedule_file, agent_config, tmp_skills):
        skill_dir = tmp_skills / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: test\n---\nDo special things.")

        schedule_file.write_text(json.dumps([
            {"id": "t1", "cron": "* * * * *", "prompt": "do it", "skill": "my-skill", "enabled": True, "last_run": 0},
        ]))
        mock_agent = AsyncMock()
        mock_agent.config = agent_config

        fired = await _check_and_fire(mock_agent)
        await asyncio.sleep(0)

        call_kwargs = mock_agent.handle.call_args
        prompt = call_kwargs.kwargs["system_task_prompt"]
        assert "Do special things." in prompt
        assert "do it" in prompt

    async def test_fires_multiple_tasks_concurrently(self, schedule_file, agent_config):
        schedule_file.write_text(json.dumps([
            {"id": "t1", "cron": "* * * * *", "prompt": "p1", "skill": None, "enabled": True, "last_run": 0},
            {"id": "t2", "cron": "* * * * *", "prompt": "p2", "skill": None, "enabled": True, "last_run": 0},
        ]))
        mock_agent = AsyncMock()
        mock_agent.config = agent_config

        fired = await _check_and_fire(mock_agent)
        await asyncio.sleep(0)

        assert "t1" in fired
        assert "t2" in fired
        assert mock_agent.handle.call_count == 2
