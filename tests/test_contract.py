"""Tests for koi/contract.py — shared Orca ↔ Koi event contract."""

import pytest
from pydantic import ValidationError

from koi.contract import EventEnvelope, ReasonCode, TERMINAL_PHASES


class TestEventEnvelope:
    def test_all_fields_optional(self):
        env = EventEnvelope()
        assert env.event_id is None
        assert env.event_type is None
        assert env.emitted_at is None
        assert env.correlation_id is None

    def test_accepts_valid_payload(self):
        env = EventEnvelope(
            event_id="replica_failed:mo-abc-r0",
            event_type="replica_failed",
            emitted_at=1234567890.5,
            correlation_id="sr-xyz",
        )
        assert env.event_id == "replica_failed:mo-abc-r0"
        assert env.event_type == "replica_failed"
        assert env.emitted_at == 1234567890.5
        assert env.correlation_id == "sr-xyz"

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            EventEnvelope(unknown_field="oops")


class TestReasonCode:
    def test_stable_string_values(self):
        """Values must match Orca's koi_contract.py literal strings."""
        assert ReasonCode.HEARTBEAT_TIMEOUT.value == "heartbeat_timeout"
        assert ReasonCode.CLEAN_EXIT_PENDING_CHUNKS.value == "clean_exit_pending_chunks"
        assert ReasonCode.LOG_STREAM_ERROR.value == "log_stream_error"
        assert ReasonCode.SPOT_PREEMPTION.value == "spot_preemption"
        assert ReasonCode.LAUNCH_CAPACITY_EXHAUSTED.value == "launch_capacity_exhausted"
        assert ReasonCode.MONITOR_THREAD_EXITED.value == "monitor_thread_exited"
        assert ReasonCode.KOI_INITIATED_KILL.value == "koi_initiated_kill"
        assert ReasonCode.MODEL_LOAD_TIMEOUT.value == "model_load_timeout"
        assert ReasonCode.UNKNOWN.value == "unknown"

    def test_is_string_enum(self):
        assert ReasonCode.HEARTBEAT_TIMEOUT == "heartbeat_timeout"


class TestTerminalPhases:
    def test_contents(self):
        assert "completed" in TERMINAL_PHASES
        assert "failed" in TERMINAL_PHASES
        assert "dead" in TERMINAL_PHASES
        assert "killed" in TERMINAL_PHASES
        assert "swapped_out" in TERMINAL_PHASES

    def test_is_frozenset(self):
        assert isinstance(TERMINAL_PHASES, frozenset)

    def test_non_terminal_phases_excluded(self):
        assert "launching" not in TERMINAL_PHASES
        assert "running" not in TERMINAL_PHASES
        assert "model_ready" not in TERMINAL_PHASES
