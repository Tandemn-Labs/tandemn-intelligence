"""Tests for monitor task liveness — the /health truthfulness invariant.

Phase 2e of contract-hardening. The bug being fixed: before this phase,
a background monitor task could exit cleanly (clean return) or raise,
and /health would still report "ok" — silent degradation. These tests
prove that both failure modes now surface as /health 503 with a clear
`fatal` message.

The intentional stop() path must NOT trigger `_fatal`.
"""

import asyncio

import pytest
import pytest_asyncio

from koi.monitor import MonitoringLoop


class _FakeOrca:
    """Minimal stub — monitor loops are patched out in these tests."""

    async def get_job_metrics(self, *a, **kw):
        return {}


@pytest.fixture
def fresh_monitor():
    return MonitoringLoop(orca=_FakeOrca(), telemetry_interval=0.01)


class TestFatalInitialization:
    def test_fatal_is_none_in_init(self, fresh_monitor):
        """_fatal must be initialized in __init__ so /health can read it
        even before start() is called."""
        assert fresh_monitor._fatal is None

    def test_fatal_resets_on_start(self, fresh_monitor):
        fresh_monitor._fatal = "stale from a previous run"

        async def run():
            await fresh_monitor.start()
            await fresh_monitor.stop()

        asyncio.run(run())
        assert fresh_monitor._fatal is None


class TestCleanExitIsFatal:
    """If a loop returns cleanly while _running is True, that's fatal —
    the service is alive but its monitoring internals have stopped."""

    @pytest.mark.asyncio
    async def test_clean_exit_of_telemetry_loop_sets_fatal(self, fresh_monitor):
        # Patch loops to exit immediately.
        async def _noop():
            return  # clean return — fatal

        fresh_monitor._telemetry_loop = _noop
        fresh_monitor._trigger_dispatcher = _noop
        await fresh_monitor.start()
        # Yield once so done_callbacks fire.
        await asyncio.sleep(0.01)
        assert fresh_monitor._fatal is not None
        assert "exited unexpectedly" in fresh_monitor._fatal
        await fresh_monitor.stop()

    @pytest.mark.asyncio
    async def test_fatal_names_the_task(self, fresh_monitor):
        async def _noop():
            return

        fresh_monitor._telemetry_loop = _noop
        fresh_monitor._trigger_dispatcher = _noop
        await fresh_monitor.start()
        await asyncio.sleep(0.01)
        # One of the two tasks produced the fatal; name must be in the message.
        assert any(
            name in (fresh_monitor._fatal or "")
            for name in ("telemetry", "triggers")
        )
        await fresh_monitor.stop()


class TestExceptionIsFatal:
    @pytest.mark.asyncio
    async def test_exception_in_task_sets_fatal(self, fresh_monitor):
        async def _boom():
            raise RuntimeError("synthetic fault")

        async def _forever():
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                raise

        fresh_monitor._telemetry_loop = _boom
        # Keep the other task alive so it doesn't overwrite _fatal with its
        # own "clean exit" message in this test.
        fresh_monitor._trigger_dispatcher = _forever
        await fresh_monitor.start()
        await asyncio.sleep(0.01)
        assert fresh_monitor._fatal is not None
        assert "synthetic fault" in fresh_monitor._fatal
        await fresh_monitor.stop()


class TestCleanStopIsNotFatal:
    @pytest.mark.asyncio
    async def test_intentional_stop_does_not_set_fatal(self, fresh_monitor):
        """stop() cancels the tasks; cancellation is expected, not fatal."""

        async def _forever():
            try:
                while True:
                    await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                raise  # proper shutdown

        fresh_monitor._telemetry_loop = _forever
        fresh_monitor._trigger_dispatcher = _forever
        await fresh_monitor.start()
        await asyncio.sleep(0.01)
        assert fresh_monitor._fatal is None
        await fresh_monitor.stop()
        assert fresh_monitor._fatal is None


class TestHealthReturns503OnFatal:
    """End-to-end: /health returns 503 when a monitor task dies."""

    @pytest.mark.asyncio
    async def test_health_503_when_fatal_set(self):
        # Reuse the test_server fixture pattern to drive /health via HTTP.
        from unittest.mock import AsyncMock, MagicMock
        from httpx import AsyncClient, ASGITransport

        from koi.resource_ledger import ResourceLedger
        from koi.runtime_state import RuntimeStateStore
        from koi.server import app
        from koi.tools.memory import AgenticMemory

        app.state.perfdb = MagicMock()
        app.state.perfdb.record_count = 0
        app.state.memory = AgenticMemory(db_path=":memory:")
        app.state.runtime_state = RuntimeStateStore(":memory:")
        app.state.ledger = ResourceLedger()
        app.state.decide_lock = asyncio.Lock()
        app.state.orca = None
        app.state.agent = MagicMock()
        app.state.agent.model = "test"
        app.state.agent.decide = AsyncMock()

        monitor = MagicMock()
        monitor.tracked_jobs = {}
        monitor._fatal = None
        app.state.monitor = monitor

        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r1 = await c.get("/health")
            assert r1.status_code == 200
            assert r1.json()["status"] == "ok"

            # Simulate a monitor task dying.
            monitor._fatal = "telemetry exited unexpectedly (clean return)"
            r2 = await c.get("/health")
            assert r2.status_code == 503
            body = r2.json()
            assert body["status"] == "fatal"
            assert "telemetry" in body["fatal"]

    @pytest.mark.asyncio
    async def test_health_exposes_inbox_counters(self):
        from unittest.mock import AsyncMock, MagicMock
        from httpx import AsyncClient, ASGITransport

        from koi.resource_ledger import ResourceLedger
        from koi.runtime_state import RuntimeStateStore
        from koi.server import app
        from koi.tools.memory import AgenticMemory

        app.state.perfdb = MagicMock()
        app.state.perfdb.record_count = 0
        app.state.memory = AgenticMemory(db_path=":memory:")
        runtime = RuntimeStateStore(":memory:")
        runtime.claim_event("ev-1", "t", "j")
        runtime.claim_event("ev-2", "t", "j")
        runtime.mark_processed("ev-2")
        app.state.runtime_state = runtime
        app.state.ledger = ResourceLedger()
        app.state.decide_lock = asyncio.Lock()
        app.state.orca = None
        app.state.agent = MagicMock()
        app.state.agent.model = "test"
        monitor = MagicMock()
        monitor.tracked_jobs = {}
        monitor._fatal = None
        app.state.monitor = monitor

        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["inbox_processed"] == 1
        assert body["inbox_processing"] == 1
        assert body["stale_inbox_claims"] == 0
