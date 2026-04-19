"""Tests for the 8 inbound Pydantic webhook models in koi/server.py.

Phase 2a of the contract-hardening rollout — verify that:
  1. All models accept envelope fields (event_id, event_type, emitted_at, correlation_id).
  2. All models still accept legacy payloads without envelope fields.
  3. All models still reject unknown fields (extra="forbid" preserved).
  4. Explicit entity IDs (replica_id, scale_request_id, group_id) accepted where added.
  5. ReasonCode accepted where added.
"""

import pytest
from pydantic import ValidationError

from koi.contract import ReasonCode
from koi.server import (
    ConfigAttemptRequest,
    DecideRequest,
    JobCompleteRequest,
    JobLaunchHeartbeatRequest,
    JobLaunchingRequest,
    JobStartedRequest,
    LaunchFailedRequest,
    ReplicaFailedRequest,
)


_ENVELOPE = {
    "event_id": "job_complete:mo-abc",
    "event_type": "job_complete",
    "emitted_at": 1234567890.5,
    "correlation_id": "sr-xyz",
}


class TestDecideRequest:
    def test_legacy_payload(self):
        req = DecideRequest(job_request={"model_name": "X"}, resource_map={})
        assert req.event_id is None

    def test_with_envelope(self):
        req = DecideRequest(job_request={"m": "x"}, resource_map={}, **_ENVELOPE)
        assert req.event_id == "job_complete:mo-abc"
        assert req.event_type == "job_complete"
        assert req.emitted_at == 1234567890.5
        assert req.correlation_id == "sr-xyz"

    def test_rejects_unknown(self):
        with pytest.raises(ValidationError):
            DecideRequest(job_request={}, resource_map={}, bogus=1)


class TestJobCompleteRequest:
    def test_legacy_payload(self):
        req = JobCompleteRequest(job_id="mo-abc", status="succeeded")
        assert req.event_id is None
        assert req.group_id is None
        assert req.decision_id is None
        assert req.reason_code is None

    def test_with_envelope_and_ids(self):
        req = JobCompleteRequest(
            job_id="mo-abc",
            status="succeeded",
            group_id="mo-abc",
            decision_id="d-123",
            reason_code=ReasonCode.UNKNOWN,
            reason_detail="test",
            **_ENVELOPE,
        )
        assert req.group_id == "mo-abc"
        assert req.decision_id == "d-123"
        assert req.reason_code == ReasonCode.UNKNOWN
        assert req.reason_detail == "test"

    def test_reason_code_accepts_string(self):
        req = JobCompleteRequest(
            job_id="mo-abc", status="failed", reason_code="heartbeat_timeout"
        )
        assert req.reason_code == ReasonCode.HEARTBEAT_TIMEOUT

    def test_rejects_unknown(self):
        with pytest.raises(ValidationError):
            JobCompleteRequest(job_id="x", status="s", bogus=1)


class TestJobStartedRequest:
    _BASE = dict(
        job_id="mo-abc-r0",
        gpu_type="L40S",
        instance_type="g6e.12xlarge",
        tp=4,
        pp=2,
        slo_deadline_hours=2.0,
        total_tokens=1_000_000,
    )

    def test_legacy_payload(self):
        req = JobStartedRequest(**self._BASE)
        assert req.event_id is None
        assert req.replica_id is None
        assert req.scale_request_id is None

    def test_with_envelope_and_ids(self):
        req = JobStartedRequest(
            **self._BASE,
            replica_id="mo-abc-r0",
            scale_request_id="sr-xyz",
            **_ENVELOPE,
        )
        assert req.replica_id == "mo-abc-r0"
        assert req.scale_request_id == "sr-xyz"
        assert req.event_id == "job_complete:mo-abc"

    def test_rejects_unknown(self):
        with pytest.raises(ValidationError):
            JobStartedRequest(**self._BASE, bogus=1)


class TestJobLaunchingRequest:
    def test_legacy_payload(self):
        req = JobLaunchingRequest(job_id="mo-abc-r0")
        assert req.event_id is None
        assert req.replica_id is None

    def test_with_envelope_and_ids(self):
        req = JobLaunchingRequest(
            job_id="mo-abc-r0",
            replica_id="mo-abc-r0",
            scale_request_id="sr-1",
            **_ENVELOPE,
        )
        assert req.replica_id == "mo-abc-r0"
        assert req.scale_request_id == "sr-1"

    def test_rejects_unknown(self):
        with pytest.raises(ValidationError):
            JobLaunchingRequest(job_id="x", bogus=1)


class TestJobLaunchHeartbeatRequest:
    def test_legacy_payload(self):
        req = JobLaunchHeartbeatRequest(job_id="mo-abc-r0", phase="provisioned")
        assert req.event_id is None
        assert req.replica_id is None

    def test_with_envelope_and_ids(self):
        req = JobLaunchHeartbeatRequest(
            job_id="mo-abc-r0",
            phase="loading",
            replica_id="mo-abc-r0",
            scale_request_id="sr-1",
            **_ENVELOPE,
        )
        assert req.replica_id == "mo-abc-r0"

    def test_rejects_unknown(self):
        with pytest.raises(ValidationError):
            JobLaunchHeartbeatRequest(job_id="x", phase="p", bogus=1)


class TestReplicaFailedRequest:
    def test_legacy_payload(self):
        req = ReplicaFailedRequest(job_id="r0", group_id="mo-abc")
        assert req.event_id is None
        assert req.replica_id is None
        assert req.reason_code is None

    def test_with_envelope_and_ids(self):
        req = ReplicaFailedRequest(
            job_id="r0",
            group_id="mo-abc",
            replica_id="r0",
            reason_code=ReasonCode.HEARTBEAT_TIMEOUT,
            reason_detail="missed 3 heartbeats",
            **_ENVELOPE,
        )
        assert req.replica_id == "r0"
        assert req.reason_code == ReasonCode.HEARTBEAT_TIMEOUT
        assert req.reason_detail == "missed 3 heartbeats"

    def test_rejects_unknown(self):
        with pytest.raises(ValidationError):
            ReplicaFailedRequest(job_id="r0", group_id="g", bogus=1)


class TestConfigAttemptRequest:
    _BASE = dict(
        job_id="mo-abc",
        instance_type="g6e.12xlarge",
        gpu_type="L40S",
        region="us-east-1",
        launched=True,
    )

    def test_legacy_payload(self):
        req = ConfigAttemptRequest(**self._BASE)
        assert req.event_id is None
        assert req.group_id is None

    def test_with_envelope_and_group(self):
        req = ConfigAttemptRequest(**self._BASE, group_id="mo-abc", **_ENVELOPE)
        assert req.group_id == "mo-abc"
        assert req.event_id == "job_complete:mo-abc"

    def test_rejects_unknown(self):
        with pytest.raises(ValidationError):
            ConfigAttemptRequest(**self._BASE, bogus=1)


class TestLaunchFailedRequest:
    def test_legacy_payload(self):
        req = LaunchFailedRequest(job_id="mo-abc")
        assert req.event_id is None
        assert req.group_id is None
        assert req.reason_code is None

    def test_with_envelope_and_reason(self):
        req = LaunchFailedRequest(
            job_id="mo-abc",
            group_id="mo-abc",
            reason_code=ReasonCode.LAUNCH_CAPACITY_EXHAUSTED,
            reason_detail="no spot in us-east-1",
            **_ENVELOPE,
        )
        assert req.group_id == "mo-abc"
        assert req.reason_code == ReasonCode.LAUNCH_CAPACITY_EXHAUSTED

    def test_rejects_unknown(self):
        with pytest.raises(ValidationError):
            LaunchFailedRequest(job_id="x", bogus=1)
