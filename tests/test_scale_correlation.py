"""Tests for per-replica scale correlation (blocker #5 fix).

Phase 3 of contract-hardening. The bug being fixed: previously, Koi
tracked pending scale decisions as a FIFO queue keyed by group_id,
decrementing a "remaining" counter as new replicas arrived. Under
overlapping scale ops with out-of-order replica arrivals, replicas
from scale-op-B could be attributed to decision-A (or vice versa).

The fix: scale_chain_tool now registers each new replica_id → decision
mapping directly (using the new_replicas list Orca returns in its
/job/{id}/scale response). /job/started does an exact replica_id
lookup. No FIFO guesswork.
"""

import pytest

from koi.monitor import MonitoringLoop
from koi.runtime_state import RuntimeStateStore
from unittest.mock import MagicMock


@pytest.fixture
def monitor():
    return MonitoringLoop(orca=MagicMock(), runtime_state=None)


class TestRegisterAndConsume:
    def test_register_stores_decision(self, monitor):
        monitor.register_pending_replica_decision(
            replica_id="r0",
            decision_id="d-1",
            scale_request_id="sr-1",
            decision={"gpu_type": "L40S"},
        )
        pending = monitor.consume_pending_replica_decision("r0")
        assert pending is not None
        assert pending["decision_id"] == "d-1"
        assert pending["scale_request_id"] == "sr-1"
        assert pending["decision"] == {"gpu_type": "L40S"}

    def test_consume_is_one_shot(self, monitor):
        monitor.register_pending_replica_decision("r0", "d-1")
        assert monitor.consume_pending_replica_decision("r0") is not None
        assert monitor.consume_pending_replica_decision("r0") is None

    def test_consume_unknown_replica_returns_none(self, monitor):
        """Initial-launch path: replica has no pending decision, handler
        uses its own decision_id path. consume must not raise."""
        assert monitor.consume_pending_replica_decision("unknown-replica") is None


class TestOverlappingScaleOps:
    """The core bug scenario: two scale ops overlap, their replicas arrive
    out of launch order. Each replica must get its correct decision_id."""

    def test_out_of_order_arrivals_map_correctly(self, monitor):
        # Scale op A: decision-a produced r-A0 and r-A1
        for rid in ("r-A0", "r-A1"):
            monitor.register_pending_replica_decision(
                replica_id=rid, decision_id="d-A", decision={"gpu_type": "L40S"}
            )
        # Scale op B: decision-b produced r-B0 and r-B1
        for rid in ("r-B0", "r-B1"):
            monitor.register_pending_replica_decision(
                replica_id=rid, decision_id="d-B", decision={"gpu_type": "L4"}
            )

        # Replicas arrive out of launch order: B0, A1, B1, A0
        assert monitor.consume_pending_replica_decision("r-B0")["decision_id"] == "d-B"
        assert monitor.consume_pending_replica_decision("r-A1")["decision_id"] == "d-A"
        assert monitor.consume_pending_replica_decision("r-B1")["decision_id"] == "d-B"
        assert monitor.consume_pending_replica_decision("r-A0")["decision_id"] == "d-A"

        # All pending decisions consumed.
        assert monitor._pending_replica_decisions == {}

    def test_partial_scale_up_leaves_others_pending(self, monitor):
        """If a scale op launched 3 replicas but only 2 reach model_ready,
        the third pending decision stays until either consumed or evicted.
        (Eviction is a future hardening — for now it lingers harmlessly.)"""
        for rid in ("r-0", "r-1", "r-2"):
            monitor.register_pending_replica_decision(rid, "d-x")
        monitor.consume_pending_replica_decision("r-0")
        monitor.consume_pending_replica_decision("r-1")
        # r-2 never shows up (launch failed after Orca reported it)
        assert monitor._pending_replica_decisions.keys() == {"r-2"}


class TestPersistenceRoundTrip:
    def test_decisions_survive_monitor_restart(self, tmp_path):
        db_path = str(tmp_path / "runtime.sqlite")
        store = RuntimeStateStore(db_path)
        monitor = MonitoringLoop(orca=MagicMock(), runtime_state=store)

        monitor.register_pending_replica_decision(
            replica_id="r0",
            decision_id="d-persisted",
            scale_request_id="sr-1",
            decision={"gpu_type": "L40S", "tp": 4, "pp": 2},
        )

        # Simulate restart by instantiating a fresh monitor on the same DB.
        restored = MonitoringLoop(
            orca=MagicMock(), runtime_state=RuntimeStateStore(db_path)
        )
        restored.restore_runtime_state()

        pending = restored.consume_pending_replica_decision("r0")
        assert pending is not None
        assert pending["decision_id"] == "d-persisted"
        assert pending["scale_request_id"] == "sr-1"

    def test_consume_deletes_from_disk(self, tmp_path):
        db_path = str(tmp_path / "runtime.sqlite")
        store = RuntimeStateStore(db_path)
        monitor = MonitoringLoop(orca=MagicMock(), runtime_state=store)

        monitor.register_pending_replica_decision("r0", "d-1")
        monitor.consume_pending_replica_decision("r0")

        assert store.load_pending_replica_decisions() == {}
