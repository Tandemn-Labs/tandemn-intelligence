"""Phase 4: tests for P5c chain post-mortem harness."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest
from pydantic_ai.models.test import TestModel

from koi.agent import KoiAgent
from koi.harness.p5c import (
    P5cDiagnosis,
    build_p5c_packet,
    deterministic_diagnosis,
    render_p5c_prompt,
    run_chain_postmortem,
)
from koi.schemas import (
    EngineConfig,
    JobTracker,
    MonitoringStatus,
    PlacementConfig,
)
from koi.tools.memory import AgenticMemory


class _ReplicaFailedReq(SimpleNamespace):
    pass


@pytest.fixture
def memory() -> AgenticMemory:
    return AgenticMemory(db_path=":memory:")


def _config(
    gpu_type: str = "L40S",
    instance_type: str = "g6e.12xlarge",
    tp: int = 4,
    pp: int = 1,
    dp: int = 1,
    region: str = "us-east-1",
    market: str = "spot",
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


def _tracker(
    *,
    config: Optional[PlacementConfig] = None,
    decision_id: str = "dec-parent",
    smoothed_tps: float = 1180.0,
) -> JobTracker:
    cfg = config or _config()
    tracker = JobTracker(
        job_id="r0",
        decision_id=decision_id,
        group_id="job-group",
        config=cfg,
        slo_deadline_hours=8.0,
        total_tokens=6_000_000,
        predicted_tps=1200.0,
        tokens_remaining=4_500_000,
        tokens_completed=1_500_000,
    )
    tracker.smoothed_tps = smoothed_tps
    tracker.elapsed_hours = 0.5
    tracker.slo_headroom_pct = 12.0
    tracker.gpu_cache_usage = 0.7
    tracker.gpu_sm_util = 80.0
    tracker.gpu_mem_bw_util = 75.0
    tracker.status = MonitoringStatus.FAILED
    return tracker


def _req(reason: str = "SpotInstanceInterruption") -> _ReplicaFailedReq:
    return _ReplicaFailedReq(
        job_id="r0",
        replica_id="r0",
        group_id="job-group",
        decision_id="dec-parent",
        instance_type="g6e.12xlarge",
        region="us-east-1",
        market="spot",
        status="failed",
        reason=reason,
        reason_code=None,
        reason_detail=None,
        event_id="evt-1",
        correlation_id=None,
    )


class TestDeterministicDiagnosis:
    def test_spot_preemption_emits_cooloff(self):
        diag = deterministic_diagnosis(
            req=_req("SpotInstanceInterruption"),
            tracker=_tracker(),
            failure_category="spot_preemption",
            region="us-east-1",
            market="spot",
            actual_tps_before_death=1180.0,
            now=1000.0,
        )

        assert diag.diagnosis_code == "spot_preemption"
        assert diag.bottleneck == "market_capacity"
        assert diag.next_fix == "retry_same_topology_on_demand"
        assert diag.cooloff_minutes == 30
        assert diag.cooloff_key == diag.failure_scope
        assert diag.avoid_until == 1000.0 + 30 * 60
        assert diag.hard_until == 1000.0 + 10 * 60

    def test_no_capacity_emits_shorter_cooloff(self):
        diag = deterministic_diagnosis(
            req=_req("InsufficientCapacity"),
            tracker=_tracker(),
            failure_category="no_capacity",
            region="us-east-1",
            market="on_demand",
            actual_tps_before_death=None,
            now=2000.0,
        )

        assert diag.diagnosis_code == "no_capacity"
        assert diag.next_fix == "switch_market_or_gpu_family"
        assert diag.cooloff_minutes == 20
        assert diag.avoid_until == 2000.0 + 20 * 60

    def test_oom_uses_topology_in_scope(self):
        diag = deterministic_diagnosis(
            req=_req("CUDA out of memory"),
            tracker=_tracker(),
            failure_category="oom",
            region="us-east-1",
            market="spot",
            actual_tps_before_death=900.0,
            now=3000.0,
        )

        assert diag.diagnosis_code == "oom"
        assert diag.bottleneck == "memory_bound"
        assert diag.cooloff_minutes == 60
        assert "4|1|1" in diag.failure_scope

    def test_heartbeat_does_not_emit_cooloff(self):
        diag = deterministic_diagnosis(
            req=_req("Heartbeat timeout (45s)"),
            tracker=_tracker(),
            failure_category="unknown",
            region="us-east-1",
            market="on_demand",
            actual_tps_before_death=420.0,
            now=4000.0,
        )

        assert diag.diagnosis_code == "heartbeat_timeout"
        assert diag.bottleneck == "runtime_unhealthy"
        assert diag.avoid_until is None
        assert diag.cooloff_minutes == 0


class TestPacketBuilder:
    def test_packet_includes_failure_runtime_memory_sections(self, memory):
        memory.record_decision(
            job_id="job-group",
            model_name="Qwen/Qwen3-32B",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1200.0,
            predicted_cost_per_hour=10.49,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=1024,
            avg_output_tokens=1024,
            num_requests=1500,
            market="spot",
        )
        tracker = _tracker()
        req = _req()

        packet = build_p5c_packet(
            req=req,
            tracker=tracker,
            memory=memory,
            failure_category="spot_preemption",
            region="us-east-1",
            market="spot",
            actual_tps_before_death=1180.0,
        )

        assert packet.transition_type.value == "chain_postmortem"
        assert packet.state.value == "replica_recovery"
        assert "failure:chain" in packet.detail_sections
        assert "runtime_metrics:chain" in packet.detail_sections
        assert "config:chain" in packet.detail_sections
        memory_section = packet.detail_sections["memory:chain"]
        assert "failure_summary" in memory_section
        assert "active_cooloffs" in memory_section
        prompt = render_p5c_prompt(packet)
        assert "P5C CHAIN POST-MORTEM" in prompt
        assert "spot_preemption" in prompt


class TestRunChainPostmortem:
    @pytest.mark.asyncio
    async def test_returns_deterministic_diagnosis_on_timeout(self, memory):
        agent = KoiAgent(perfdb=None, memory=memory, api_key="test-key")

        class _BoomModel:
            async def request(self, *args, **kwargs):
                raise TimeoutError("forced timeout")

            async def request_stream(self, *args, **kwargs):
                raise TimeoutError("forced timeout")

        agent._model = TestModel()  # default model attribute

        from koi.harness import p5c as p5c_module

        async def _raise_timeout(*args, **kwargs):
            import asyncio

            raise asyncio.TimeoutError()

        original_run_typed = p5c_module.KoiToolRunner.run_typed
        p5c_module.KoiToolRunner.run_typed = _raise_timeout
        try:
            diag = await run_chain_postmortem(
                agent=agent,
                req=_req(),
                tracker=_tracker(),
                memory=memory,
                failure_category="spot_preemption",
                region="us-east-1",
                market="spot",
                actual_tps_before_death=1180.0,
            )
        finally:
            p5c_module.KoiToolRunner.run_typed = original_run_typed

        assert isinstance(diag, P5cDiagnosis)
        assert diag.diagnosis_code == "spot_preemption"
        assert diag.cooloff_minutes == 30

    @pytest.mark.asyncio
    async def test_uses_llm_diagnosis_when_available(self, memory):
        agent = KoiAgent(perfdb=None, memory=memory, api_key="test-key")
        agent._model = TestModel()

        from koi.harness import p5c as p5c_module

        async def _stub_run_typed(self, prompt, **kwargs):  # noqa: ARG002
            return 0, P5cDiagnosis(
                diagnosis_code="memory_bound",
                bottleneck="kv_cache_pressure",
                next_fix="reduce_max_seqs",
                failure_scope="custom",
                event_at=42.0,
                cooloff_minutes=15,
                rationale="LLM derived",
            )

        original = p5c_module.KoiToolRunner.run_typed
        p5c_module.KoiToolRunner.run_typed = _stub_run_typed
        try:
            diag = await run_chain_postmortem(
                agent=agent,
                req=_req(),
                tracker=_tracker(),
                memory=memory,
                failure_category="oom",
                region="us-east-1",
                market="spot",
                actual_tps_before_death=900.0,
            )
        finally:
            p5c_module.KoiToolRunner.run_typed = original

        assert diag.diagnosis_code == "memory_bound"
        assert diag.cooloff_minutes == 15
        # _normalize_diagnosis fills avoid_until from cooloff_minutes
        assert diag.avoid_until == 42.0 + 15 * 60
        assert diag.cooloff_key == "custom"
