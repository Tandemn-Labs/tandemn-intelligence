"""Tests for koi/server.py — FastAPI endpoints."""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from koi.schemas import (
    AgentDecision, PlacementConfig, EngineConfig, DataSource, MonitoringStatus,
)
from koi.server import app
from koi.resource_ledger import ResourceLedger
from koi.tools.memory import AgenticMemory
from koi.tools.perfdb import PerfDB


def _mock_decision(job_id="job-test123"):
    config = PlacementConfig(
        gpu_type="L40S", instance_type="g6e.12xlarge",
        num_gpus=8, num_instances=2, tp=4, pp=2, dp=1,
        region="us-east-1",
        engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=2),
    )
    return AgentDecision(
        job_id=job_id, model_name="Qwen/Qwen2.5-72B-Instruct",
        config=config,
        predicted_tps=833.0, predicted_cost_per_hour=13.35,
        predicted_total_cost=33.38, predicted_runtime_hours=2.5,
        reasoning="PerfDB shows L40S TP=4 PP=2 gets 833 TPS",
        confidence=0.85, data_source=DataSource.EXACT_MATCH,
    )


@pytest_asyncio.fixture
async def client():
    """Set up app state manually to avoid needing real API keys."""
    import asyncio

    memory = AgenticMemory(db_path=":memory:")

    app.state.perfdb = MagicMock()
    app.state.perfdb.record_count = 307
    app.state.memory = memory
    app.state.ledger = ResourceLedger()
    app.state.decide_lock = asyncio.Lock()
    app.state.orca = None
    app.state.agent = MagicMock()
    app.state.agent.model = "claude-sonnet-4-6"
    app.state.agent.decide = AsyncMock(return_value=_mock_decision())
    app.state.agent.handle_trigger = AsyncMock(return_value="ok")
    app.state.monitor = MagicMock()
    app.state.monitor.tracked_jobs = {}
    app.state.monitor._pending_launches = {}
    app.state.monitor.register_job = MagicMock()
    app.state.monitor.get_group_chains = MagicMock(return_value={})
    app.state.monitor.unregister_group = MagicMock(return_value=[])

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealth:
    @pytest.mark.asyncio
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == "2.0"
        assert "tracked_jobs" in body
        assert "memory_decisions" in body


class TestDecide:
    @pytest.mark.asyncio
    async def test_valid_decide(self, client):
        resp = await client.post("/decide", json={
            "job_request": {
                "model_name": "Qwen/Qwen2.5-72B-Instruct",
                "task_type": "batch",
                "avg_input_tokens": 953,
                "avg_output_tokens": 1024,
                "num_requests": 5000,
                "slo_deadline_hours": 8.0,
                "objective": "cheapest",
            },
            "resource_map": {
                "instances": [
                    {"instance_type": "g6e.12xlarge", "gpu_type": "L40S",
                     "gpus_per_instance": 4, "vcpus": 48, "quota_family": "G",
                     "gpu_memory_gb": 48.0, "cost_per_instance_hour_usd": 10.49}
                ],
                "quotas": [
                    {"family": "G", "region": "us-east-1", "market": "on_demand",
                     "baseline_vcpus": 192, "used_vcpus": 0}
                ],
            },
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == "job-test123"
        assert body["config"]["gpu_type"] == "L40S"

    @pytest.mark.asyncio
    async def test_empty_resources_422(self, client):
        resp = await client.post("/decide", json={
            "job_request": {
                "model_name": "test", "avg_input_tokens": 512, "avg_output_tokens": 256,
            },
            "resource_map": {"instances": [], "quotas": []},
        })
        assert resp.status_code == 422


class TestDecideConcurrency:
    @pytest.mark.asyncio
    async def test_parallel_decide_requests_do_not_double_book(self, client):
        """Concurrent /decide calls should not see the same 8 GPUs as free."""
        import asyncio

        seen_available = []

        async def decide_once(job_request, resource_map):
            resource = resource_map.get_resource("L40S")
            seen_available.append(resource.available_gpus if resource else None)
            await asyncio.sleep(0.05)
            if not resource or resource.available_gpus < 8:
                raise RuntimeError("insufficient adjusted resources")
            return _mock_decision(job_id=f"job-{len(seen_available)}")

        app.state.agent.decide = decide_once

        payload = {
            "job_request": {
                "model_name": "Qwen/Qwen2.5-72B-Instruct",
                "task_type": "batch",
                "avg_input_tokens": 953,
                "avg_output_tokens": 1024,
                "num_requests": 5000,
                "slo_deadline_hours": 8.0,
                "objective": "cheapest",
            },
            "resource_map": {
                "instances": [
                    {
                        "instance_type": "g6e.12xlarge",
                        "gpu_type": "L40S",
                        "gpus_per_instance": 4,
                        "vcpus": 48,
                        "quota_family": "G",
                        "gpu_memory_gb": 48.0,
                        "cost_per_instance_hour_usd": 10.49,
                    },
                ],
                "quotas": [
                    {
                        "family": "G",
                        "region": "us-east-1",
                        "market": "on_demand",
                        "baseline_vcpus": 96,
                        "used_vcpus": 0,
                    },
                ],
            },
        }

        resp1, resp2 = await asyncio.gather(
            client.post("/decide", json=payload),
            client.post("/decide", json=payload),
        )

        status_codes = sorted([resp1.status_code, resp2.status_code])
        assert status_codes == [200, 500]
        assert seen_available == [8, 0]
        assert app.state.ledger.pending_count == 1


class TestJobComplete:
    @pytest.mark.asyncio
    async def test_webhook(self, client):
        # Register a job first
        app.state.monitor.tracked_jobs = {
            "job-done": MagicMock(
                decision_id="dec-123",
                group_id=None,
                elapsed_hours=2.5,
                slo_headroom_pct=80.0,
            ),
        }

        resp = await client.post("/job/complete", json={
            "job_id": "job-done",
            "status": "succeeded",
            "metrics": {"avg_generation_throughput_toks_per_s": 1500.0},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"

    @pytest.mark.asyncio
    async def test_unknown_job(self, client):
        app.state.monitor.tracked_jobs = {}
        resp = await client.post("/job/complete", json={
            "job_id": "job-unknown",
            "status": "succeeded",
            "metrics": {},
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "unknown_job"


class TestLaunchFailed:
    @pytest.mark.asyncio
    async def test_records_failures(self, client):
        app.state.monitor.tracked_jobs = {}
        resp = await client.post("/job/launch-failed", json={
            "job_id": "job-fail1",
            "configs_tried": [
                {"gpu_type": "A100-80GB", "instance_type": "p4de.24xlarge", "region": "us-west-2"},
                {"gpu_type": "L40S", "instance_type": "g6e.12xlarge", "region": "us-east-1"},
            ],
            "failure_reasons": ["InsufficientCapacity", "QuotaExceeded"],
            "total_time_seconds": 180.0,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["attempts_recorded"] == 2

    @pytest.mark.asyncio
    async def test_releases_ledger_and_unregisters_job(self, client):
        """All-failed launches should release reservations and unregister the job."""
        app.state.ledger.reserve("dec-launchfail", "L40S", 8, region="us-east-1")
        app.state.monitor.tracked_jobs = {
            "job-fail2": MagicMock(decision_id="dec-launchfail"),
        }
        app.state.monitor.unregister_job = MagicMock()

        resp = await client.post("/job/launch-failed", json={
            "job_id": "job-fail2",
            "configs_tried": [
                {
                    "gpu_type": "L40S",
                    "instance_type": "g6e.12xlarge",
                    "region": "us-east-1",
                    "market": "spot",
                },
                {
                    "gpu_type": "A100-80GB",
                    "instance_type": "p4de.24xlarge",
                    "region": "us-west-2",
                    "market": "on_demand",
                },
            ],
            "failure_reasons": ["InsufficientCapacity", "QuotaExceeded"],
            "total_time_seconds": 240.0,
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "recorded"
        assert body["attempts_recorded"] == 2
        assert app.state.ledger.pending_count == 0
        app.state.monitor.unregister_job.assert_called_once_with("job-fail2")

        spot = app.state.memory.get_failure_summary(
            "L40S", region="us-east-1", market="spot",
        )
        on_demand = app.state.memory.get_failure_summary(
            "A100-80GB", region="us-west-2", market="on_demand",
        )
        assert spot["effective_observations"] == 1
        assert spot["availability_pct"] < 50.0
        assert on_demand["effective_observations"] == 1
        assert on_demand["availability_pct"] < 50.0


class TestReplicaFailed:
    @pytest.mark.asyncio
    async def test_captures_tps_before_zeroing(self, client):
        """actual_tps should be recorded from smoothed_tps BEFORE it's set to 0."""
        import asyncio
        from koi.schemas import JobTracker, MonitoringStatus

        config = PlacementConfig(
            gpu_type="L40S", instance_type="g6e.12xlarge",
            num_gpus=4, num_instances=1, tp=4, pp=1, dp=1,
            region="us-east-1",
            engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=1),
        )
        tracker = JobTracker(
            job_id="r0", config=config,
            slo_deadline_hours=8.0, total_tokens=6_000_000,
            predicted_tps=1200.0, tokens_remaining=5_000_000,
        )
        tracker.smoothed_tps = 1850.0  # was running at 1850 TPS before death
        # Record a decision so the outcome JOIN works
        dec_id = app.state.memory.record_decision(
            job_id="parent-job", model_name="Qwen/Qwen3-32B",
            instance_type="g6e.12xlarge", gpu_type="L40S",
            tp=4, pp=1, dp=1, num_gpus=4,
            predicted_tps=1200.0, predicted_cost_per_hour=6.85,
            slo_deadline_hours=8.0, objective="cheapest",
            avg_input_tokens=953, avg_output_tokens=1024,
        )
        tracker.decision_id = dec_id

        app.state.monitor.tracked_jobs = {"r0": tracker}
        app.state.monitor._koi_initiated_kills = set()
        app.state.monitor._trigger_queue = asyncio.Queue()
        app.state.monitor._pending_launches = {}

        resp = await client.post("/job/replica-failed", json={
            "job_id": "r0",
            "group_id": "parent-job",
            "status": "failed",
            "reason": "Heartbeat timeout (45s)",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "trigger_emitted"

        # Verify tracker is zeroed
        assert tracker.smoothed_tps == 0
        assert tracker.status == MonitoringStatus.FAILED

        # Verify outcome was recorded with actual_tps=1850
        outcomes = app.state.memory.query_outcomes(status="replica_failed")
        assert len(outcomes) >= 1
        last = outcomes[0]
        assert last["actual_tps"] == 1850.0

    @pytest.mark.asyncio
    async def test_dedup_already_failed(self, client):
        """Second /job/replica-failed for same ID returns already_failed."""
        import asyncio
        from koi.schemas import JobTracker, MonitoringStatus

        config = PlacementConfig(
            gpu_type="L40S", instance_type="g6e.12xlarge",
            num_gpus=4, num_instances=1, tp=4, pp=1, dp=1,
            region="us-east-1",
            engine_config=EngineConfig(tensor_parallel_size=4, pipeline_parallel_size=1),
        )
        tracker = JobTracker(
            job_id="r0", config=config,
            slo_deadline_hours=8.0, total_tokens=6_000_000,
            predicted_tps=1200.0,
        )
        tracker.status = MonitoringStatus.FAILED  # already dead

        app.state.monitor.tracked_jobs = {"r0": tracker}
        app.state.monitor._koi_initiated_kills = set()
        app.state.monitor._trigger_queue = asyncio.Queue()
        app.state.monitor._pending_launches = {}

        resp = await client.post("/job/replica-failed", json={
            "job_id": "r0",
            "group_id": "parent-job",
            "status": "failed",
            "reason": "Heartbeat timeout (45s)",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "already_failed"


class TestClassifyFailure:
    def test_spot_preemption(self):
        from koi.server import _classify_failure
        assert _classify_failure("SpotInstanceInterruption") == "spot_preemption"
        assert _classify_failure("spot preempted by EC2") == "spot_preemption"

    def test_no_capacity(self):
        from koi.server import _classify_failure
        assert _classify_failure("InsufficientCapacity in us-east-1") == "no_capacity"

    def test_oom(self):
        from koi.server import _classify_failure
        assert _classify_failure("CUDA OOM: tried to allocate 40GB") == "oom"
        assert _classify_failure("OutOfMemoryError") == "oom"

    def test_quota(self):
        from koi.server import _classify_failure
        assert _classify_failure("QuotaExceeded for p5.48xlarge") == "quota"

    def test_unknown(self):
        from koi.server import _classify_failure
        assert _classify_failure("some random error") == "unknown"
        assert _classify_failure("") == "unknown"


class TestJobLaunching:
    @pytest.mark.asyncio
    async def test_launching_tracked(self, client):
        """POST /job/launching stores in _pending_launches."""
        resp = await client.post("/job/launching", json={
            "job_id": "r0",
            "group_id": "parent-job",
            "gpu_type": "L40S",
            "instance_type": "g6e.12xlarge",
            "tp": 4, "pp": 1,
            "region": "us-east-1",
            "market": "on_demand",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "tracked"
        assert "r0" in app.state.monitor._pending_launches

    @pytest.mark.asyncio
    async def test_launching_visible_in_jobs(self, client):
        """Pending launches appear in /jobs with status=launching."""
        import time
        app.state.monitor._pending_launches = {
            "r0": {"group_id": "parent", "gpu_type": "L40S",
                    "instance_type": "g6e.12xlarge", "tp": 4, "pp": 1,
                    "region": "us-east-1", "market": "on_demand",
                    "launched_at": time.time()},
        }
        resp = await client.get("/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pending_launches"] == 1
        launching = [j for j in data["jobs"] if j["status"] == "launching"]
        assert len(launching) == 1
        assert launching[0]["gpu_type"] == "L40S"


class TestConfigAttempted:
    @pytest.mark.asyncio
    async def test_config_attempted_separates_spot_and_on_demand_priors(self, client):
        """Spot failures and on-demand successes should update different priors."""
        spot_resp = await client.post("/job/config-attempted", json={
            "job_id": "job-market",
            "decision_id": "dec-market",
            "instance_type": "g6e.12xlarge",
            "gpu_type": "L40S",
            "region": "us-east-1",
            "market": "spot",
            "launched": False,
            "failure_reason": "InsufficientCapacity",
            "attempt_index": 0,
        })
        assert spot_resp.status_code == 200
        assert spot_resp.json()["launched"] is False

        on_demand_resp = await client.post("/job/config-attempted", json={
            "job_id": "job-market",
            "decision_id": "dec-market",
            "instance_type": "g6e.12xlarge",
            "gpu_type": "L40S",
            "region": "us-east-1",
            "market": "on_demand",
            "launched": True,
            "time_to_launch": 45.0,
            "attempt_index": 1,
        })
        assert on_demand_resp.status_code == 200
        assert on_demand_resp.json()["launched"] is True

        spot = app.state.memory.get_failure_summary(
            "L40S", region="us-east-1", market="spot",
        )
        on_demand = app.state.memory.get_failure_summary(
            "L40S", region="us-east-1", market="on_demand",
        )
        assert spot["effective_observations"] == 1
        assert spot["availability_pct"] < 50.0
        assert on_demand["effective_observations"] == 1
        assert on_demand["availability_pct"] > 50.0


class TestJobStarted:
    @pytest.mark.asyncio
    async def test_started_clears_pending_and_registers_job(self, client):
        """POST /job/started should clear pending state and register the job."""
        app.state.ledger.reserve("dec-started", "L40S", 4, region="unknown")
        app.state.monitor._pending_launches = {
            "r0": {
                "group_id": "parent-job",
                "gpu_type": "L40S",
                "instance_type": "g6e.12xlarge",
                "tp": 4,
                "pp": 1,
                "region": "us-east-1",
                "market": "spot",
                "launched_at": 0,
            },
        }
        existing = MagicMock(
            group_id="parent-job",
            action_in_progress=True,
            action_freeze_until=123.0,
        )
        app.state.monitor.tracked_jobs = {"existing-r1": existing}

        resp = await client.post("/job/started", json={
            "job_id": "r0",
            "decision_id": "dec-started",
            "group_id": "parent-job",
            "gpu_type": "L40S",
            "instance_type": "g6e.12xlarge",
            "tp": 4,
            "pp": 1,
            "dp": 1,
            "slo_deadline_hours": 8.0,
            "total_tokens": 6_000_000,
            "predicted_tps": 1200.0,
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "registered"
        assert body["decision_id"] == "dec-started"
        assert "r0" not in app.state.monitor._pending_launches
        assert app.state.ledger.pending_count == 0

        kwargs = app.state.monitor.register_job.call_args.kwargs
        assert kwargs["job_id"] == "r0"
        assert kwargs["decision_id"] == "dec-started"
        assert kwargs["group_id"] == "parent-job"
        assert kwargs["predicted_tps"] == 1200.0
        assert kwargs["config"].gpu_type == "L40S"
        assert kwargs["config"].instance_type == "g6e.12xlarge"

        assert existing.action_in_progress is False
        assert existing.action_freeze_until is None

        summary = app.state.memory.get_failure_summary(
            "L40S", region="unknown", market="spot",
        )
        assert summary["effective_observations"] == 1
        assert summary["availability_pct"] > 50.0

    @pytest.mark.asyncio
    async def test_started_fallback_creates_child_decision(self, client):
        """Fallback launches should create a child decision for the actual config."""
        original_decision_id = app.state.memory.record_decision(
            job_id="parent-job",
            model_name="Qwen/Qwen2.5-72B-Instruct",
            instance_type="g6e.12xlarge",
            gpu_type="L40S",
            tp=4,
            pp=1,
            dp=1,
            num_gpus=4,
            predicted_tps=1200.0,
            predicted_cost_per_hour=6.85,
            slo_deadline_hours=8.0,
            objective="cheapest",
            avg_input_tokens=953,
            avg_output_tokens=1024,
            num_requests=5000,
            market="spot",
        )
        app.state.ledger.reserve(original_decision_id, "L40S", 4, region="unknown")

        resp = await client.post("/job/started", json={
            "job_id": "r1",
            "decision_id": original_decision_id,
            "group_id": "parent-job",
            "gpu_type": "A100-80GB",
            "instance_type": "p4de.24xlarge",
            "tp": 8,
            "pp": 1,
            "dp": 1,
            "slo_deadline_hours": 8.0,
            "total_tokens": 8_000_000,
            "predicted_tps": 0.0,
            "is_fallback": True,
        })

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "registered"
        assert body["decision_id"] != original_decision_id
        assert app.state.ledger.pending_count == 0

        kwargs = app.state.monitor.register_job.call_args.kwargs
        assert kwargs["job_id"] == "r1"
        assert kwargs["decision_id"] == body["decision_id"]
        assert kwargs["config"].gpu_type == "A100-80GB"
        assert kwargs["config"].instance_type == "p4de.24xlarge"
        assert kwargs["config"].tp == 8

        decisions = app.state.memory.query_decisions(model_name="Qwen/Qwen2.5-72B-Instruct")
        child = next(d for d in decisions if d["decision_id"] == body["decision_id"])
        assert child["parent_decision_id"] == original_decision_id
        assert child["triggered_by"] == "fallback"
        assert child["gpu_type"] == "A100-80GB"
        assert child["instance_type"] == "p4de.24xlarge"
        assert child["market"] == "on_demand"

        on_demand = app.state.memory.get_failure_summary(
            "A100-80GB", region="unknown", market="on_demand",
        )
        spot = app.state.memory.get_failure_summary(
            "A100-80GB", region="unknown", market="spot",
        )
        assert on_demand["effective_observations"] == 1
        assert on_demand["availability_pct"] > 50.0
        assert spot["effective_observations"] == 0
        assert spot["availability_pct"] == pytest.approx(50.0)


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_decide_started_complete_flow(self, client):
        """Decision, start, and complete should preserve one clean learning record."""
        def register_job_side_effect(**kwargs):
            app.state.monitor.tracked_jobs[kwargs["job_id"]] = MagicMock(
                decision_id=kwargs["decision_id"],
                group_id=None,
                elapsed_hours=2.5,
                slo_headroom_pct=70.0,
            )

        app.state.monitor.register_job.side_effect = register_job_side_effect

        decide_resp = await client.post("/decide", json={
            "job_request": {
                "model_name": "Qwen/Qwen2.5-72B-Instruct",
                "task_type": "batch",
                "avg_input_tokens": 953,
                "avg_output_tokens": 1024,
                "num_requests": 5000,
                "slo_deadline_hours": 8.0,
                "objective": "cheapest",
            },
            "resource_map": {
                "instances": [
                    {
                        "instance_type": "g6e.12xlarge",
                        "gpu_type": "L40S",
                        "gpus_per_instance": 4,
                        "vcpus": 48,
                        "quota_family": "G",
                        "gpu_memory_gb": 48.0,
                        "cost_per_instance_hour_usd": 10.49,
                    },
                ],
                "quotas": [
                    {
                        "family": "G",
                        "region": "us-east-1",
                        "market": "on_demand",
                        "baseline_vcpus": 192,
                        "used_vcpus": 0,
                    },
                ],
            },
        })
        assert decide_resp.status_code == 200
        decide_body = decide_resp.json()
        decision_id = decide_body["_decision_id"]
        assert app.state.ledger.pending_count == 1
        assert app.state.memory.decision_count() == 1

        started_resp = await client.post("/job/started", json={
            "job_id": decide_body["job_id"],
            "decision_id": decision_id,
            "gpu_type": decide_body["config"]["gpu_type"],
            "instance_type": decide_body["config"]["instance_type"],
            "tp": decide_body["config"]["tp"],
            "pp": decide_body["config"]["pp"],
            "dp": decide_body["config"]["dp"],
            "slo_deadline_hours": 8.0,
            "total_tokens": 6_000_000,
            "predicted_tps": decide_body["predicted_tps"],
        })
        assert started_resp.status_code == 200
        assert started_resp.json()["decision_id"] == decision_id
        assert app.state.ledger.pending_count == 0
        assert decide_body["job_id"] in app.state.monitor.tracked_jobs

        complete_resp = await client.post("/job/complete", json={
            "job_id": decide_body["job_id"],
            "status": "succeeded",
            "metrics": {
                "avg_generation_throughput_toks_per_s": 1500.0,
                "cost_per_hour": 10.49,
            },
        })
        assert complete_resp.status_code == 200
        assert complete_resp.json()["status"] == "recorded"
        assert app.state.memory.outcome_count() == 1

        outcomes = app.state.memory.query_outcomes(status="succeeded")
        assert len(outcomes) == 1
        assert outcomes[0]["decision_id"] == decision_id
        assert outcomes[0]["actual_tps"] == 1500.0
        assert outcomes[0]["actual_cost_per_hour"] == 10.49


class TestListJobs:
    @pytest.mark.asyncio
    async def test_empty(self, client):
        app.state.monitor.tracked_jobs = {}
        app.state.monitor._pending_launches = {}
        resp = await client.get("/jobs")
        assert resp.status_code == 200
        assert resp.json()["tracked_jobs"] == 0
