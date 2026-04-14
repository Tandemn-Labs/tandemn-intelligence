"""Compatibility tests for simulation/mock_orca.py.

These tests intentionally freeze the current legacy HTTP contract so we can
extract a shared simulator engine without breaking:
  - simulation/mock_orca.py
  - simulation/sim_ctl.py
  - simulation/run_sim_tests.py
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import simulation.mock_orca as mock_orca


class _DummyTask:
    def cancel(self):
        return None


async def _noop(*args, **kwargs):
    return None


def _make_job() -> mock_orca.SimJob:
    job = mock_orca.SimJob(
        job_id="mo-qwen32b-sim",
        model_name="Qwen/Qwen3-32B",
        total_chunks=500,
        completed_chunks=125,
        slo_deadline_hours=8.0,
        tokens_per_chunk=12000,
        decision_id="dec-sim-1",
    )
    now = time.time() - 120
    job.replicas["mo-qwen32b-sim-r0"] = mock_orca.SimReplica(
        replica_id="mo-qwen32b-sim-r0",
        phase="running",
        base_tps=1200.0,
        gpu_type="L40S",
        instance_type="g6e.12xlarge",
        tp=4,
        pp=2,
        region="us-east-1",
        market="on_demand",
        started_at=now,
        warmup_seconds=0,
    )
    job.replicas["mo-qwen32b-sim-r1"] = mock_orca.SimReplica(
        replica_id="mo-qwen32b-sim-r1",
        phase="running",
        base_tps=1100.0,
        gpu_type="L40S",
        instance_type="g6e.12xlarge",
        tp=4,
        pp=2,
        region="us-east-1",
        market="on_demand",
        started_at=now,
        warmup_seconds=0,
    )
    return job


@pytest_asyncio.fixture
async def client(monkeypatch):
    mock_orca.SIM.jobs.clear()
    mock_orca.SIM.koi_url = "http://localhost:8090"

    monkeypatch.setattr(mock_orca, "_notify_koi_replica_failed", _noop)
    monkeypatch.setattr(mock_orca, "_notify_koi_complete", _noop)
    monkeypatch.setattr(mock_orca, "_notify_koi_launching", _noop)
    monkeypatch.setattr(mock_orca, "_notify_koi_config_attempted", _noop)
    monkeypatch.setattr(mock_orca, "_notify_koi_replica_started", _noop)

    transport = ASGITransport(app=mock_orca.app, raise_app_exceptions=True)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    mock_orca.SIM.jobs.clear()


@pytest.fixture
def seeded_job():
    job = _make_job()
    mock_orca.SIM.jobs[job.job_id] = job
    return job


class TestLegacyResourceShape:
    @pytest.mark.asyncio
    async def test_resources_returns_shape_c_contract(self, client):
        resp = await client.get("/resources")
        assert resp.status_code == 200

        body = resp.json()
        assert set(body.keys()) == {"instances", "quotas"}
        assert body["instances"], "expected non-empty instances list"
        assert body["quotas"], "expected non-empty quotas list"

        inst = body["instances"][0]
        assert {"instance_type", "gpu_type", "gpus_per_instance", "cost_per_instance_hour_usd"} <= set(inst.keys())

        quota = body["quotas"][0]
        assert {"family", "region", "market", "baseline_vcpus", "used_vcpus"} <= set(quota.keys())

    def test_mock_orca_script_help_runs_from_repo_root(self):
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "simulation/mock_orca.py", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr
        assert "Mock Orca server for Koi testing" in result.stdout


class TestLegacyJobEndpoints:
    @pytest.mark.asyncio
    async def test_job_metrics_and_chunk_progress_contract(self, client, seeded_job):
        metrics = await client.get(f"/job/{seeded_job.job_id}/metrics")
        assert metrics.status_code == 200
        body = metrics.json()
        assert {
            "avg_generation_throughput_toks_per_s",
            "gpu_cache_usage_perc",
            "num_requests_running",
            "num_requests_waiting",
            "gpu_sm_util_pct",
            "gpu_mem_bw_util_pct",
        } <= set(body.keys())
        assert body["avg_generation_throughput_toks_per_s"] > 0

        progress = await client.get(f"/job/{seeded_job.job_id}/chunks/progress")
        assert progress.status_code == 200
        assert progress.json() == {
            "total": 500,
            "pending": 375,
            "inflight": 0,
            "completed": 125,
            "failed": 0,
            "all_done": False,
        }

    @pytest.mark.asyncio
    async def test_replicas_and_state_contract(self, client, seeded_job):
        replicas = await client.get(f"/job/{seeded_job.job_id}/replicas")
        assert replicas.status_code == 200
        rep_body = replicas.json()
        assert "replicas" in rep_body
        assert len(rep_body["replicas"]) == 2
        assert {
            "replica_id",
            "phase",
            "region",
            "market",
            "instance_type",
            "has_metrics",
        } <= set(rep_body["replicas"][0].keys())

        state = await client.get("/sim/state")
        assert state.status_code == 200
        sim_body = state.json()
        assert seeded_job.job_id in sim_body
        job_state = sim_body[seeded_job.job_id]
        assert {"status", "model", "chunks", "aggregate_tps", "replicas_alive", "replicas_total", "replicas"} <= set(job_state.keys())
        assert job_state["chunks"].startswith("125/500")
        assert job_state["replicas_alive"] == 2
        assert {"phase", "tps", "gpu", "tp", "pp"} <= set(job_state["replicas"]["mo-qwen32b-sim-r0"].keys())

    @pytest.mark.asyncio
    async def test_submit_batch_returns_existing_job_launching(self, client, seeded_job):
        resp = await client.post("/submit/batch", json={"model_name": seeded_job.model_name})
        assert resp.status_code == 200
        assert resp.json() == {
            "job_id": seeded_job.job_id,
            "status": "launching",
            "replicas": 2,
        }


class TestLegacyMutations:
    @pytest.mark.asyncio
    async def test_scale_returns_scaling_and_creates_launching_replicas(self, client, seeded_job, monkeypatch):
        def fake_create_task(coro):
            coro.close()
            return _DummyTask()

        monkeypatch.setattr(mock_orca.asyncio, "create_task", fake_create_task)

        resp = await client.post(
            f"/job/{seeded_job.job_id}/scale",
            json={"count": 2, "gpu_type": "A100", "tp_size": 8, "pp_size": 1, "on_demand": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "scaling"
        assert len(body["new_replicas"]) == 2

        new_ids = body["new_replicas"]
        for rid in new_ids:
            replica = seeded_job.replicas[rid]
            assert replica.phase == "launching"
            assert replica.gpu_type == "A100"
            assert replica.tp == 8
            assert replica.pp == 1

    @pytest.mark.asyncio
    async def test_kill_replica_marks_replica_dead_in_state(self, client, seeded_job):
        resp = await client.post(
            "/sim/kill-replica/mo-qwen32b-sim-r0",
            json={"reason": "Simulated EC2 termination"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "killed"

        state = await client.get("/sim/state")
        body = state.json()[seeded_job.job_id]
        assert body["replicas"]["mo-qwen32b-sim-r0"]["phase"] == "dead"
        assert body["replicas_alive"] == 1
