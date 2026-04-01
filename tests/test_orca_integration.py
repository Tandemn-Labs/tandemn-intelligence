"""
Tests for Orca integration — Shape C parsing + server endpoints.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from koi.intake import _parse_resource_map_response, _dict_to_job_request
from koi.schemas import (
    GPUResource, ResourceMap, PlacementDecision, PlacementConfig,
    PredictedMetrics, EngineConfig, DataSource,
)


# ---------------------------------------------------------------------------
# Fixtures: Orca-style payloads
# ---------------------------------------------------------------------------

ORCA_SINGLE_FAMILY = {
    "instances": [
        {
            "instance_type": "g6e.12xlarge",
            "gpu_type": "L40S",
            "gpus_per_instance": 4,
            "vcpus": 48,
            "quota_family": "G",
            "gpu_memory_gb": 48.0,
            "interconnect": "PCIe",
            "cost_per_instance_hour_usd": 10.49,
        }
    ],
    "quotas": [
        {
            "family": "G",
            "region": "us-east-1",
            "market": "on_demand",
            "baseline_vcpus": 192,
            "used_vcpus": 0,
        }
    ],
}

ORCA_MULTI_FAMILY = {
    "instances": [
        {
            "instance_type": "g6e.12xlarge",
            "gpu_type": "L40S",
            "gpus_per_instance": 4,
            "vcpus": 48,
            "quota_family": "G",
            "gpu_memory_gb": 48.0,
            "interconnect": "PCIe",
            "cost_per_instance_hour_usd": 10.49,
        },
        {
            "instance_type": "p5.48xlarge",
            "gpu_type": "H100",
            "gpus_per_instance": 8,
            "vcpus": 192,
            "quota_family": "P5",
            "gpu_memory_gb": 80.0,
            "interconnect": "NVLink",
            "cost_per_instance_hour_usd": 98.32,
        },
        {
            "instance_type": "p4d.24xlarge",
            "gpu_type": "A100",
            "gpus_per_instance": 8,
            "vcpus": 96,
            "quota_family": "P4d",
            "gpu_memory_gb": 80.0,
            "interconnect": "NVLink",
            "cost_per_instance_hour_usd": 32.77,
        },
    ],
    "quotas": [
        {"family": "G", "region": "us-east-1", "market": "on_demand", "baseline_vcpus": 192, "used_vcpus": 0},
        {"family": "P5", "region": "us-east-1", "market": "on_demand", "baseline_vcpus": 384, "used_vcpus": 0},
        {"family": "P4d", "region": "us-west-2", "market": "on_demand", "baseline_vcpus": 192, "used_vcpus": 0},
    ],
}

ORCA_MULTI_REGION_QUOTAS = {
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
        {"family": "G", "region": "us-east-1", "market": "on_demand", "baseline_vcpus": 48, "used_vcpus": 0},
        {"family": "G", "region": "us-west-2", "market": "on_demand", "baseline_vcpus": 192, "used_vcpus": 0},
    ],
}


# ---------------------------------------------------------------------------
# Shape C parsing tests
# ---------------------------------------------------------------------------

class TestShapeCParsing:
    def test_single_family(self):
        rm = _parse_resource_map_response(ORCA_SINGLE_FAMILY)
        assert len(rm.resources) == 1
        r = rm.resources[0]
        assert r.gpu_type == "L40S"
        assert r.instance_type == "g6e.12xlarge"
        assert r.gpus_per_instance == 4
        assert r.region == "us-east-1"
        assert r.allocated_gpus == 0
        # 192 / 48 = 4 instances, capped at 4 → 4 * 4 = 16 GPUs
        assert r.total_gpus == 16
        assert r.gpu_memory_gb == 48.0
        assert r.interconnect == "PCIe"
        assert r.cost_per_instance_hour_usd == 10.49

    def test_multi_family(self):
        rm = _parse_resource_map_response(ORCA_MULTI_FAMILY)
        assert len(rm.resources) == 3
        types = {r.gpu_type for r in rm.resources}
        assert types == {"L40S", "H100", "A100"}

    def test_best_region_picked(self):
        """us-west-2 has 192 vCPU vs us-east-1 with 48 → should pick us-west-2."""
        rm = _parse_resource_map_response(ORCA_MULTI_REGION_QUOTAS)
        assert len(rm.resources) == 1
        assert rm.resources[0].region == "us-west-2"
        # us-west-2: 192 / 48 = 4 instances (capped at 4) → 16 GPUs
        assert rm.resources[0].total_gpus == 16

    def test_large_quota_fully_reported(self):
        """All quota-derived instances are reported — no arbitrary cap."""
        data = {
            "instances": [
                {
                    "instance_type": "p5.48xlarge",
                    "gpu_type": "H100",
                    "gpus_per_instance": 8,
                    "vcpus": 192,
                    "quota_family": "P5",
                    "gpu_memory_gb": 80.0,
                    "cost_per_instance_hour_usd": 98.32,
                }
            ],
            "quotas": [
                {"family": "P5", "region": "us-east-1", "market": "on_demand",
                 "baseline_vcpus": 960, "used_vcpus": 0},
            ],
        }
        rm = _parse_resource_map_response(data)
        # 960 // 192 = 5 instances → 40 GPUs (no cap)
        assert rm.resources[0].total_gpus == 40

    def test_zero_quota_filtered(self):
        """Families with 0 baseline vCPUs are excluded."""
        data = {
            "instances": [
                {
                    "instance_type": "g6e.12xlarge",
                    "gpu_type": "L40S",
                    "gpus_per_instance": 4,
                    "vcpus": 48,
                    "quota_family": "G",
                    "gpu_memory_gb": 48.0,
                    "cost_per_instance_hour_usd": 10.49,
                }
            ],
            "quotas": [
                {"family": "G", "region": "us-east-1", "market": "on_demand",
                 "baseline_vcpus": 0, "used_vcpus": 0},
            ],
        }
        with pytest.raises(ValueError, match="no GPU resources"):
            _parse_resource_map_response(data)

    def test_used_vcpus_reduce_capacity(self):
        """used_vcpus reduces available instances."""
        data = {
            "instances": [
                {
                    "instance_type": "g6e.12xlarge",
                    "gpu_type": "L40S",
                    "gpus_per_instance": 4,
                    "vcpus": 48,
                    "quota_family": "G",
                    "gpu_memory_gb": 48.0,
                    "cost_per_instance_hour_usd": 10.49,
                }
            ],
            "quotas": [
                {"family": "G", "region": "us-east-1", "market": "on_demand",
                 "baseline_vcpus": 192, "used_vcpus": 144},
            ],
        }
        rm = _parse_resource_map_response(data)
        # (192 - 144) / 48 = 1 instance → 4 GPUs
        assert rm.resources[0].total_gpus == 4

    def test_fully_used_quota_filtered(self):
        """If all vCPUs used, family is filtered out."""
        data = {
            "instances": [
                {
                    "instance_type": "g6e.12xlarge",
                    "gpu_type": "L40S",
                    "gpus_per_instance": 4,
                    "vcpus": 48,
                    "quota_family": "G",
                    "gpu_memory_gb": 48.0,
                    "cost_per_instance_hour_usd": 10.49,
                }
            ],
            "quotas": [
                {"family": "G", "region": "us-east-1", "market": "on_demand",
                 "baseline_vcpus": 192, "used_vcpus": 192},
            ],
        }
        with pytest.raises(ValueError, match="no GPU resources"):
            _parse_resource_map_response(data)

    def test_gpu_memory_defaults(self):
        """If gpu_memory_gb is missing, falls back to lookup table."""
        data = {
            "instances": [
                {
                    "instance_type": "p5.48xlarge",
                    "gpu_type": "H100",
                    "gpus_per_instance": 8,
                    "vcpus": 192,
                    "quota_family": "P5",
                    "cost_per_instance_hour_usd": 98.32,
                }
            ],
            "quotas": [
                {"family": "P5", "region": "us-east-1", "market": "on_demand",
                 "baseline_vcpus": 384, "used_vcpus": 0},
            ],
        }
        rm = _parse_resource_map_response(data)
        assert rm.resources[0].gpu_memory_gb == 80.0  # H100 default
        assert rm.resources[0].interconnect == "NVLink"  # H100 default


# ---------------------------------------------------------------------------
# Backward compat: Shape A and Shape B still work
# ---------------------------------------------------------------------------

class TestShapeAB:
    def test_shape_a_list(self):
        data = [
            {
                "gpu_type": "H100",
                "instance_type": "p5.48xlarge",
                "gpus_per_instance": 8,
                "total_gpus": 64,
                "allocated_gpus": 16,
                "cost_per_instance_hour_usd": 98.32,
                "gpu_memory_gb": 80,
                "region": "us-east-1",
                "interconnect": "NVLink",
            }
        ]
        rm = _parse_resource_map_response(data)
        assert len(rm.resources) == 1
        assert rm.resources[0].gpu_type == "H100"
        assert rm.resources[0].total_gpus == 64
        assert rm.resources[0].allocated_gpus == 16

    def test_shape_b_wrapper(self):
        data = {
            "vpc_id": "vpc-abc123",
            "region": "us-west-2",
            "resources": [
                {
                    "gpu_type": "A100",
                    "instance_type": "p4d.24xlarge",
                    "gpus_per_instance": 8,
                    "total_gpus": 32,
                    "cost_per_instance_hour_usd": 32.77,
                    "gpu_memory_gb": 80,
                    "region": "us-west-2",
                    "interconnect": "NVLink",
                }
            ],
        }
        rm = _parse_resource_map_response(data)
        assert rm.vpc_id == "vpc-abc123"
        assert rm.region == "us-west-2"
        assert len(rm.resources) == 1
        assert rm.resources[0].gpu_type == "A100"


# ---------------------------------------------------------------------------
# Server endpoint tests
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def client():
    from httpx import AsyncClient, ASGITransport
    from koi.server import app

    # Manually set state to avoid needing real API keys in lifespan
    app.state.koi = MagicMock()
    app.state.koi.ensemble.model = "test-model"
    app.state.koi.oracle.perf_rag.records = []
    app.state.tracked_jobs = set()

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _mock_decision(job_id="job-test1234"):
    """Create a minimal PlacementDecision for mocking."""
    config = PlacementConfig(
        gpu_type="L40S",
        instance_type="g6e.12xlarge",
        num_gpus=16,
        num_instances=4,
        tp=4,
        pp=1,
        dp=1,
        region="us-east-1",
        engine_config=EngineConfig(
            tensor_parallel_size=4,
            pipeline_parallel_size=1,
        ),
    )
    metrics = PredictedMetrics(
        throughput_tokens_per_sec=5000.0,
        throughput_per_gpu_tokens_per_sec=312.5,
        cost_per_hour_usd=41.96,
        confidence=0.7,
        data_source=DataSource.ANALYTICAL,
    )
    return PlacementDecision(
        job_id=job_id,
        model_name="Qwen/Qwen2.5-72B-Instruct",
        recommendation=config,
        predicted_metrics=metrics,
        reasoning="Test decision",
        confidence=0.7,
    )


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "tracked_jobs" in body


@pytest.mark.asyncio
async def test_decide_valid(client):
    from koi.server import app
    decision = _mock_decision()
    app.state.koi.decide_async = AsyncMock(return_value=decision)

    resp = await client.post("/decide", json={
        "job_request": {
            "model_name": "Qwen/Qwen2.5-72B-Instruct",
            "task_type": "batch",
            "avg_input_tokens": 512,
            "avg_output_tokens": 256,
            "num_requests": 100000,
            "slo_deadline_hours": 8.0,
            "objective": "cheapest",
        },
        "resource_map": ORCA_SINGLE_FAMILY,
    })

    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == "job-test1234"
    assert body["model_name"] == "Qwen/Qwen2.5-72B-Instruct"
    assert body["recommendation"]["gpu_type"] == "L40S"
    assert "predicted_metrics" in body


@pytest.mark.asyncio
async def test_decide_empty_resources(client):
    """Empty resources should return 422."""
    resp = await client.post("/decide", json={
        "job_request": {
            "model_name": "test-model",
            "task_type": "batch",
            "avg_input_tokens": 512,
            "avg_output_tokens": 256,
        },
        "resource_map": {"instances": [], "quotas": []},
    })
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_decide_zero_candidates_422(client):
    """RuntimeError from Oracle (0 candidates) → 422."""
    from koi.server import app
    app.state.koi.decide_async = AsyncMock(
        side_effect=RuntimeError("Oracle found 0 feasible candidates.")
    )

    resp = await client.post("/decide", json={
        "job_request": {
            "model_name": "test-model",
            "task_type": "batch",
            "avg_input_tokens": 512,
            "avg_output_tokens": 256,
        },
        "resource_map": ORCA_SINGLE_FAMILY,
    })
    assert resp.status_code == 422
    assert "0 feasible candidates" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_phase2_stubs(client):
    resp = await client.post("/job/complete")
    assert resp.status_code == 501

    resp = await client.post("/reconfig")
    assert resp.status_code == 501
