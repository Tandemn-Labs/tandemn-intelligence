"""Phase 5: tests for P4 replica recovery harness."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Optional

import pytest
from pydantic_ai.models.test import TestModel

from koi.agent import KoiAgent
from koi.harness.p4 import (
    P4DiagnosisInput,
    build_p4_packet,
    run_replica_recovery,
)
from koi.resource_ledger import ResourceLedger
from koi.schemas import EngineConfig, JobTracker, MonitoringStatus, PlacementConfig
from koi.tools.memory import AgenticMemory


class StubPerfDB:
    _rows = [
        {
            "gpu_type": "L40S",
            "instance_type": "g6e.12xlarge",
            "tp": 4,
            "pp": 1,
            "dp": 1,
            "throughput_tps": 1200.0,
        },
        {
            "gpu_type": "A100-80GB",
            "instance_type": "p4de.24xlarge",
            "tp": 8,
            "pp": 1,
            "dp": 1,
            "throughput_tps": 2400.0,
        },
    ]

    def query(self, **kwargs):
        records = list(self._rows)
        gpu_type = kwargs.get("gpu_type")
        tp = kwargs.get("tp")
        pp = kwargs.get("pp")
        if gpu_type:
            records = [r for r in records if r["gpu_type"] == gpu_type]
        if tp is not None:
            records = [r for r in records if r["tp"] == tp]
        if pp is not None:
            records = [r for r in records if r["pp"] == pp]
        return records[: kwargs.get("limit", 20)]


class FakeOrca:
    def __init__(self):
        self.scale_calls: list[dict[str, Any]] = []

    async def get_resources(self):
        return {
            "instances": [
                {
                    "instance_type": "g6e.12xlarge",
                    "gpu_type": "L40S",
                    "gpus_per_instance": 4,
                    "vcpus": 48,
                    "quota_family": "G",
                    "gpu_memory_gb": 48.0,
                    "cost_per_instance_hour_usd": 10.49,
                    "interconnect": "PCIe",
                },
                {
                    "instance_type": "p4de.24xlarge",
                    "gpu_type": "A100-80GB",
                    "gpus_per_instance": 8,
                    "vcpus": 96,
                    "quota_family": "P",
                    "gpu_memory_gb": 80.0,
                    "cost_per_instance_hour_usd": 40.96,
                    "interconnect": "NVLink",
                },
            ],
            "quotas": [
                {"family": "G", "region": "us-east-1", "market": "spot", "baseline_vcpus": 384, "used_vcpus": 0},
                {"family": "G", "region": "us-east-1", "market": "on_demand", "baseline_vcpus": 384, "used_vcpus": 0},
                {"family": "P", "region": "us-east-1", "market": "on_demand", "baseline_vcpus": 384, "used_vcpus": 0},
            ],
        }

    async def scale_job(self, job_id, gpu_type, tp, pp, count, **kwargs):
        self.scale_calls.append(
            {"job_id": job_id, "gpu_type": gpu_type, "tp": tp, "pp": pp, "count": count, **kwargs}
        )
        return {
            "status": "scaling",
            "new_replicas": [f"{job_id}-r-new"],
            "scale_request_id": "scale-p4-1",
        }


class FakeMonitor:
    def __init__(self, chains: Optional[dict[str, Any]] = None):
        self._chains = chains or {}
        self.tracked_jobs = dict(self._chains)
        self._koi_initiated_kills: set[str] = set()

    def get_group_chains(self, group_id):
        return self._chains

    def persist_job(self, job_id):
        return None

    def register_pending_replica_decision(self, **kwargs):
        return None


@pytest.fixture
def memory() -> AgenticMemory:
    return AgenticMemory(db_path=":memory:")


def _config(
    gpu_type: str = "L40S",
    instance_type: str = "g6e.12xlarge",
    tp: int = 4,
    pp: int = 1,
    dp: int = 1,
    market: str = "spot",
    region: str = "us-east-1",
) -> PlacementConfig:
    return PlacementConfig(
        gpu_type=gpu_type,
        instance_type=instance_type,
        num_gpus=tp * pp * dp,
        num_instances=1,
        tp=tp,
        pp=pp,
        dp=dp,
        region=region,
        engine_config=EngineConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp),
        market=market,
    )


def _record_parent(memory: AgenticMemory) -> str:
    return memory.record_decision(
        job_id="job-p4",
        model_name="Qwen/Qwen3-32B",
        instance_type="g6e.12xlarge",
        gpu_type="L40S",
        tp=4,
        pp=1,
        dp=1,
        num_gpus=4,
        predicted_tps=1200.0,
        predicted_cost_per_hour=10.49,
        slo_deadline_hours=2.0,
        objective="cheapest",
        avg_input_tokens=512,
        avg_output_tokens=512,
        num_requests=5000,
        market="spot",
        cost_roofline_usd=200.0,
    )


def _tracker(decision_id: str, *, group_id: str = "job-p4") -> JobTracker:
    tracker = JobTracker(
        job_id="job-p4-r0",
        decision_id=decision_id,
        group_id=group_id,
        config=_config(),
        slo_deadline_hours=2.0,
        total_tokens=5_000_000,
        predicted_tps=1200.0,
        tokens_remaining=4_000_000,
        tokens_completed=1_000_000,
    )
    tracker.smoothed_tps = 1180.0
    tracker.elapsed_hours = 0.5
    tracker.status = MonitoringStatus.FAILED
    return tracker


def _agent(memory: AgenticMemory, *, with_orca: bool = True, with_model: bool = True) -> KoiAgent:
    agent = KoiAgent(
        perfdb=StubPerfDB(),
        memory=memory,
        orca=FakeOrca() if with_orca else None,
        api_key="test-key",
    )
    if with_model:
        agent._model = TestModel()
    else:
        agent._model = None
    agent.model = "test-model"
    agent.monitor = FakeMonitor()
    return agent


def _failed_req(group_id: str = "job-p4") -> SimpleNamespace:
    return SimpleNamespace(
        job_id="job-p4-r0",
        replica_id="job-p4-r0",
        group_id=group_id,
        decision_id=None,
        instance_type="g6e.12xlarge",
        region="us-east-1",
        market="spot",
        status="failed",
        reason="SpotInstanceInterruption",
        reason_code=None,
        reason_detail=None,
        event_id="evt-p4-1",
        correlation_id=None,
    )


@pytest.mark.asyncio
async def test_p4_packet_emits_diagnosis_aligned_menu(memory):
    decision_id = _record_parent(memory)
    agent = _agent(memory)
    tracker = _tracker(decision_id)

    diagnosis = P4DiagnosisInput(
        diagnosis_code="spot_preemption",
        bottleneck="market_capacity",
        next_fix="retry_same_topology_on_demand",
        failure_scope="L40S|g6e.12xlarge|us-east-1|spot",
        rationale="Replica was interrupted on spot capacity",
        cooloff_minutes=30,
    )

    packet = await build_p4_packet(
        agent=agent,
        req=_failed_req(),
        tracker=tracker,
        memory=memory,
        diagnosis=diagnosis,
        region="us-east-1",
        market="spot",
    )

    assert packet.transition_type.value == "replica_recovery"
    assert packet.state.value == "replica_recovery"
    assert packet.policy_context["force_on_demand"] is True
    assert packet.policy_context["retry_budget_remaining_before_choice"] >= 1
    types = [option.action_type for option in packet.action_options]
    # Must include hold_noop and abort_recovery as terminal options.
    assert types[-2:] == ["hold_noop", "abort_recovery"]
    # Must include at least one diagnosis-aligned replacement option.
    replacement_types = [
        t for t in types if t not in {"hold_noop", "abort_recovery"}
    ]
    assert replacement_types, types
    # Spot-preemption diagnosis prefers replace_market or migrate_gpu_family
    # over replace_same.
    assert "replace_same" not in replacement_types[:1], (
        f"replace_same should not be the top option for spot_preemption; got {replacement_types}"
    )


@pytest.mark.asyncio
async def test_p4_retry_budget_exhausted_only_emits_hold_or_abort(memory, monkeypatch):
    decision_id = _record_parent(memory)
    agent = _agent(memory)
    tracker = _tracker(decision_id)

    monkeypatch.setenv("KOI_HARNESS_P4_RETRY_BUDGET", "0")
    packet = await build_p4_packet(
        agent=agent,
        req=_failed_req(),
        tracker=tracker,
        memory=memory,
        diagnosis={"diagnosis_code": "spot_preemption", "next_fix": "retry_same_topology_on_demand"},
        region="us-east-1",
        market="spot",
    )

    types = [option.action_type for option in packet.action_options]
    assert "abort_recovery" in types
    assert all(
        t in {"hold_noop", "abort_recovery"} for t in types
    ), f"Expected only hold_noop / abort_recovery, got {types}"


@pytest.mark.asyncio
async def test_run_replica_recovery_records_child_decision_and_reserves(memory):
    decision_id = _record_parent(memory)
    agent = _agent(memory)
    tracker = _tracker(decision_id)
    ledger = ResourceLedger()

    plan = await run_replica_recovery(
        agent=agent,
        req=_failed_req(),
        tracker=tracker,
        memory=memory,
        diagnosis={
            "diagnosis_code": "spot_preemption",
            "next_fix": "retry_same_topology_on_demand",
            "failure_scope": "L40S|g6e.12xlarge|us-east-1|spot",
            "rationale": "Spot preemption on test scope",
        },
        region="us-east-1",
        market="spot",
        ledger=ledger,
    )

    assert plan["action"] in {"replace_market", "migrate_gpu_family", "replace_alt_topology"}
    assert plan["decision_id"]
    child = memory.get_decision(plan["decision_id"])
    assert child is not None
    assert child["parent_decision_id"] == decision_id
    assert child["triggered_by"] == "replica_recovery"
    assert child["market"] == "on_demand"
    assert ledger.pending_count == 1
    # The agent's scale primitive should have been called once.
    assert len(agent.orca.scale_calls) == 1
    call = agent.orca.scale_calls[0]
    assert call["job_id"] == "job-p4"
    assert call["count"] == 1
    assert call["on_demand"] is True


@pytest.mark.asyncio
async def test_run_replica_recovery_falls_back_when_no_llm(memory):
    decision_id = _record_parent(memory)
    agent = _agent(memory, with_model=False)
    tracker = _tracker(decision_id)
    ledger = ResourceLedger()

    plan = await run_replica_recovery(
        agent=agent,
        req=_failed_req(),
        tracker=tracker,
        memory=memory,
        diagnosis={
            "diagnosis_code": "spot_preemption",
            "next_fix": "retry_same_topology_on_demand",
        },
        region="us-east-1",
        market="spot",
        ledger=ledger,
    )

    assert plan["action"] in {"replace_market", "migrate_gpu_family", "replace_alt_topology"}
    assert plan["decision_id"]


@pytest.mark.asyncio
async def test_run_replica_recovery_recent_failure_downranks_same_scope(memory):
    decision_id = _record_parent(memory)
    agent = _agent(memory)
    tracker = _tracker(decision_id)
    now = time.time()
    # Same scope as the failed replica -> recently failed.
    memory.record_cooloff(
        key="L40S|g6e.12xlarge|us-east-1|on_demand",
        gpu_type="L40S",
        instance_type="g6e.12xlarge",
        region="us-east-1",
        market="on_demand",
        tp=4,
        pp=1,
        dp=1,
        reason="recent capacity issue",
        diagnosis_code="no_capacity",
        avoid_until=now + 30 * 60,
    )

    packet = await build_p4_packet(
        agent=agent,
        req=_failed_req(),
        tracker=tracker,
        memory=memory,
        diagnosis={
            "diagnosis_code": "spot_preemption",
            "next_fix": "retry_same_topology_on_demand",
        },
        region="us-east-1",
        market="spot",
    )

    replacement_options = [
        option
        for option in packet.action_options
        if option.action_type not in {"hold_noop", "abort_recovery"}
    ]
    assert replacement_options, packet.action_options
    top = replacement_options[0]
    # The L40S on_demand "replace_market" option has a fresh failure annotation.
    # It should not be the top recommendation; A100 migrate should rank above.
    if top.evidence.get("source") == "replace_market":
        assert top.evidence.get("recent_failure") is None, top.evidence
