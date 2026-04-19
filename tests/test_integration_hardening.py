"""Cross-repo integration test: Orca outbox → Koi inbox on the real wire.

Phase 6 of contract-hardening. Unit tests cover each half in isolation —
this proves the two halves actually talk. Lives in the Koi repo because
Koi's venv has all the deps (anthropic, fastapi, httpx) while still
being able to import `orca_server.outbox` from the sibling checkout
(it's stdlib-only except for a lazy `requests` import inside one method).

Pipeline exercised end-to-end:

   Orca business code
        └─ enqueue(...) → OutboxDB (sqlite row, envelope injected)
               └─ OutboxPublisher.drain_once()
                       └─ TestClient.post(path, json=envelope+payload)
                               └─ Koi.server handler
                                       ├─ claim_event()   → inbox 'processing'
                                       ├─ run handler     → side effects
                                       └─ mark_processed  → inbox 'processed'

Proven end-to-end:

  1. Happy path: a single replica_failed event lands in Koi's inbox as
     'processed'; no duplicate outcomes.
  2. Watchdog + monitor_replica collision: two enqueues with the same
     dedup_key collapse to one row at Orca, deliver once to Koi.
  3. Duplicate delivery across retries: if the publisher retries a row
     that Koi already processed, Koi short-circuits and no double side
     effects land.
  4. Handler returns 500 (simulating crash-mid-handler): outbox row NOT
     marked delivered; next drain retries and succeeds after reclaim.
  5. Koi unreachable during outage: publisher buffers, then drains on
     recovery.

Not covered here (needs real infra or money):
  - Real GPU R1/R3/R4 from E2E_REAL_GPU_PLAYBOOK.md
  - Actual 25-hour outage replay (inbox retention covers it in unit tests)
  - iptables-level network partition (covered by test_outbox_publisher.py
    with injected exceptions)
"""

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

# Load Orca's outbox module from the sibling checkout.
_ORCA_PATH = Path(__file__).resolve().parents[2] / "Tandemn-orca"
if str(_ORCA_PATH) not in sys.path:
    sys.path.insert(0, str(_ORCA_PATH))

from orca_server.outbox import OutboxDB, OutboxPublisher  # noqa: E402

from koi.resource_ledger import ResourceLedger
from koi.runtime_state import RuntimeStateStore as KoiRuntimeStateStore
from koi.schemas import EngineConfig, JobTracker, PlacementConfig
from koi.server import app as koi_app
from koi.tools.memory import AgenticMemory


@pytest.fixture
def wired():
    """Orca OutboxDB + Publisher routed at a fully-configured Koi TestClient."""
    koi_memory = AgenticMemory(db_path=":memory:")
    koi_runtime = KoiRuntimeStateStore(":memory:")

    koi_app.state.perfdb = MagicMock()
    koi_app.state.perfdb.record_count = 0
    koi_app.state.memory = koi_memory
    koi_app.state.runtime_state = koi_runtime
    koi_app.state.ledger = ResourceLedger()
    koi_app.state.decide_lock = asyncio.Lock()
    koi_app.state.orca = None
    koi_app.state.agent = MagicMock()
    koi_app.state.agent.model = "test"
    koi_app.state.agent.decide = AsyncMock()
    koi_app.state.agent.handle_trigger = AsyncMock()

    # Pre-register a tracker (single-chain → /job/complete Mode 1).
    config = PlacementConfig(
        gpu_type="L40S",
        instance_type="g6e.12xlarge",
        num_gpus=8,
        num_instances=1,
        tp=4,
        pp=2,
        dp=1,
        region="us-east-1",
        engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=2),
        market="on_demand",
    )
    tracker = JobTracker(
        job_id="mo-abc",
        decision_id="d-1",
        group_id=None,
        config=config,
        slo_deadline_hours=8.0,
        total_tokens=1_000_000,
        predicted_tps=1200.0,
        tokens_remaining=1_000_000,
    )

    monitor = MagicMock()
    monitor.tracked_jobs = {"mo-abc": tracker}
    monitor._pending_launches = {}
    monitor._pending_replica_decisions = {}
    monitor._koi_initiated_kills = set()
    monitor._fatal = None
    monitor._trigger_queue = asyncio.Queue()
    monitor.persist_job = MagicMock()
    monitor.unregister_job = MagicMock(
        side_effect=lambda jid: monitor.tracked_jobs.pop(jid, None)
    )
    monitor.unregister_group = MagicMock(return_value=[])
    monitor.get_group_chains = MagicMock(return_value={})
    monitor.get_pending_launch = MagicMock(return_value={})
    monitor.clear_pending_launch = MagicMock()
    monitor.register_job = MagicMock()
    monitor.consume_pending_replica_decision = MagicMock(return_value=None)
    koi_app.state.monitor = monitor

    koi_client = TestClient(koi_app)

    # Orca-side wiring: in-memory outbox + publisher whose post_fn routes
    # to Koi's ASGI TestClient. No real sockets, but the envelope
    # contract is exercised exactly as it would be over HTTP.
    outbox = OutboxDB(":memory:")

    def _post_to_koi(url, json_payload, timeout):
        path = url if url.startswith("/") else "/" + url.split("/", 3)[-1]
        return koi_client.post(path, json=json_payload, timeout=timeout)

    publisher = OutboxPublisher(
        outbox,
        koi_base_url="",
        post_fn=_post_to_koi,
        poll_interval=0.01,
    )

    try:
        yield SimpleNamespace(
            outbox=outbox,
            publisher=publisher,
            koi_client=koi_client,
            koi_runtime=koi_runtime,
            koi_memory=koi_memory,
            koi_monitor=monitor,
            koi_tracker=tracker,
        )
    finally:
        outbox.close()
        koi_client.close()


# ----------------------------------------------------------------------


class TestHappyPath:
    def test_replica_failed_end_to_end(self, wired):
        wired.outbox.enqueue(
            "/job/replica-failed",
            "replica_failed",
            {
                "job_id": "mo-abc",
                "replica_id": "mo-abc",
                "group_id": "mo-abc",
                "decision_id": "d-1",
                "status": "failed",
                "reason": "test failure",
            },
            job_id="mo-abc",
            dedup_key="replica_failed:mo-abc",
        )
        delivered = wired.publisher.drain_once()
        assert delivered == 1
        assert wired.outbox.pending_count() == 0
        assert wired.koi_runtime.inbox_count(status="processed") == 1
        assert wired.koi_runtime.inbox_count(status="processing") == 0

    def test_job_complete_records_outcome(self, wired):
        wired.outbox.enqueue(
            "/job/complete",
            "job_complete",
            {
                "job_id": "mo-abc",
                "status": "succeeded",
                "metrics": {"avg_generation_throughput_toks_per_s": 1234.5},
            },
            job_id="mo-abc",
            dedup_key="job_complete:mo-abc",
        )
        wired.publisher.drain_once()
        assert wired.koi_memory.outcome_count() == 1


class TestWatchdogMonitorCollision:
    """Two enqueues with the same dedup_key must reach Koi exactly once."""

    def test_dedup_collapses_at_orca(self, wired):
        # monitor_replica path fires first
        wired.outbox.enqueue(
            "/job/replica-failed",
            "replica_failed",
            {"job_id": "mo-abc", "group_id": "mo-abc", "reason": "chunks pending"},
            job_id="mo-abc",
            dedup_key="replica_failed:mo-abc",
        )
        # watchdog fires 30s later, same key
        wired.outbox.enqueue(
            "/job/replica-failed",
            "replica_failed",
            {"job_id": "mo-abc", "group_id": "mo-abc", "reason": "heartbeat timeout"},
            job_id="mo-abc",
            dedup_key="replica_failed:mo-abc",
        )
        assert wired.outbox.pending_count() == 1  # collapsed at source

        wired.publisher.drain_once()
        assert wired.koi_runtime.inbox_count(status="processed") == 1


class TestDuplicateDeliveryRetry:
    def test_replay_after_processed_is_noop(self, wired):
        wired.outbox.enqueue(
            "/job/complete",
            "job_complete",
            {"job_id": "mo-abc", "status": "succeeded", "metrics": {}},
            job_id="mo-abc",
            dedup_key="job_complete:mo-abc",
        )
        wired.publisher.drain_once()
        assert wired.koi_memory.outcome_count() == 1

        # Simulate a retry: reset delivered_at so the publisher sees the row
        # as pending again, and drain. Koi's inbox recognizes the event_id
        # and returns duplicate_ignored.
        with wired.outbox._lock:
            wired.outbox._conn.execute(
                "UPDATE outbox SET delivered_at = NULL WHERE event_id = ?",
                ("job_complete:mo-abc",),
            )
            wired.outbox._conn.commit()

        wired.publisher.drain_once()
        assert wired.koi_memory.outcome_count() == 1  # no second outcome


class TestHandlerReturnsErrorRetryLater:
    def test_handler_raises_row_stays_pending(self, wired, monkeypatch):
        # record_outcome raises once, then behaves. Models a transient fault.
        original = AgenticMemory.record_outcome
        calls = {"n": 0}

        def flaky_record(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("synthetic fault")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(AgenticMemory, "record_outcome", flaky_record)

        wired.outbox.enqueue(
            "/job/complete",
            "job_complete",
            {"job_id": "mo-abc", "status": "succeeded", "metrics": {}},
            job_id="mo-abc",
            dedup_key="job_complete:mo-abc",
        )

        # First drain: handler raises → 500 → mark_failure. Row stays.
        wired.publisher.drain_once()
        assert wired.outbox.pending_count() == 1
        assert wired.koi_runtime.inbox_count(status="processed") == 0

        # Force the outbox row back into the claim window and Koi's inbox
        # claim past its reclaim window (default 120s).
        with wired.outbox._lock:
            wired.outbox._conn.execute(
                "UPDATE outbox SET next_attempt_at = ? "
                "WHERE event_id = 'job_complete:mo-abc'",
                (time.time(),),
            )
            wired.outbox._conn.commit()
        with wired.koi_runtime._lock:
            wired.koi_runtime._conn.execute(
                "UPDATE inbox SET claimed_at = ? "
                "WHERE event_id = 'job_complete:mo-abc'",
                (time.time() - 300,),
            )
            wired.koi_runtime._conn.commit()

        wired.publisher.drain_once()
        assert wired.outbox.pending_count() == 0
        assert wired.koi_memory.outcome_count() == 1
        assert calls["n"] == 2


class TestKoiUnreachableBuffersAndRecovers:
    def test_connection_error_buffers_then_drains(self, wired, monkeypatch):
        # Phase 1: Koi "down" — publisher gets ConnectionError.
        def down(url, json_payload, timeout):
            raise ConnectionError("Koi is down")

        monkeypatch.setattr(wired.publisher, "_post_fn", down)

        wired.outbox.enqueue(
            "/job/replica-failed",
            "replica_failed",
            {"job_id": "mo-abc", "group_id": "mo-abc", "reason": "test"},
            job_id="mo-abc",
            dedup_key="replica_failed:mo-abc",
        )
        wired.outbox.enqueue(
            "/job/complete",
            "job_complete",
            {"job_id": "mo-abc", "status": "succeeded", "metrics": {}},
            job_id="mo-abc",
            dedup_key="job_complete:mo-abc",
        )
        wired.publisher.drain_once()
        assert wired.outbox.pending_count() == 2
        assert wired.koi_runtime.inbox_count(status="processed") == 0

        # Phase 2: Koi recovers. Restore the working post_fn, force rows
        # back into the claim window, drain.
        koi_client = wired.koi_client

        def recovered(url, json_payload, timeout):
            path = url if url.startswith("/") else "/" + url.split("/", 3)[-1]
            return koi_client.post(path, json=json_payload, timeout=timeout)

        monkeypatch.setattr(wired.publisher, "_post_fn", recovered)
        with wired.outbox._lock:
            wired.outbox._conn.execute(
                "UPDATE outbox SET next_attempt_at = ?", (time.time(),)
            )
            wired.outbox._conn.commit()

        wired.publisher.drain_once()
        assert wired.outbox.pending_count() == 0
        assert wired.koi_runtime.inbox_count(status="processed") == 2
