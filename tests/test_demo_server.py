"""Tests for the demo backend API."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import simulation.demo_server as demo_server
from simulation.demo_server import SESSION_MANAGER, app


@pytest_asyncio.fixture
async def client():
    SESSION_MANAGER.clear()
    async def _no_koi(*args, **kwargs):
        return None

    demo_server._request_koi_decision = _no_koi
    transport = ASGITransport(app=app, raise_app_exceptions=True)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    SESSION_MANAGER.clear()


class TestDemoCatalog:
    @pytest.mark.asyncio
    async def test_demo_index_serves_html(self, client):
        resp = await client.get("/demo")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Koi Demo Simulator" in resp.text

    @pytest.mark.asyncio
    async def test_demo_static_assets_are_served(self, client):
        resp = await client.get("/demo/static/app.js")
        assert resp.status_code == 200
        assert "application/javascript" in resp.headers["content-type"] or "text/javascript" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_catalog_endpoint_returns_controls(self, client):
        resp = await client.get("/demo/catalog")
        assert resp.status_code == 200
        body = resp.json()
        assert {"models", "quota_presets", "scenarios"} <= set(body.keys())
        assert body["models"]
        assert body["quota_presets"]
        assert body["scenarios"]


class TestDemoLaunch:
    @pytest.mark.asyncio
    async def test_launch_creates_session_with_preview(self, client):
        resp = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "total_chunks": 500,
                "slo_deadline_hours": 8.0,
                "quota_preset": "aws_mixed_demo",
                "scenario": "hero_elastic",
                "cost_cap_usd": 120.0,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "created"
        assert body["session_id"].startswith("demo-")
        assert body["model"]["model_name"] == "Qwen/Qwen3-32B"
        assert body["quota"]["slug"] == "aws_mixed_demo"
        assert body["scenario"]["slug"] == "hero_elastic"
        assert body["launch_preview"]["baseline_replica_tps"] > 0
        assert body["launch_preview"]["launch_timing_s"]["total"] > 0
        assert body["resource_map"]["instances"]
        assert body["koi"]["decision"] is None

        session = await client.get(f"/demo/session/{body['session_id']}")
        assert session.status_code == 200
        assert session.json()["session_id"] == body["session_id"]

    @pytest.mark.asyncio
    async def test_launch_rejects_unknown_quota(self, client):
        resp = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "quota_preset": "nope",
                "scenario": "hero_elastic",
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_session_endpoint_returns_dynamic_runtime_state(self, client):
        launch = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "total_chunks": 500,
                "slo_deadline_hours": 8.0,
                "quota_preset": "aws_mixed_demo",
                "scenario": "hero_elastic",
            },
        )
        body = launch.json()
        session_id = body["session_id"]
        created_at = body["created_at"]
        launch_total = body["launch_preview"]["launch_timing_s"]["total"]

        launching = await client.get(f"/demo/session/{session_id}", params={"now": created_at + 0.5})
        assert launching.status_code == 200
        assert launching.json()["runtime"]["status"] == "launching"

        running = await client.get(
            f"/demo/session/{session_id}",
            params={"now": created_at + launch_total + 15},
        )
        assert running.status_code == 200
        running_body = running.json()
        assert running_body["runtime"]["status"] in {"running", "completed"}
        assert running_body["runtime"]["aggregate_tps"] > 0
        assert running_body["runtime"]["tokens_completed"] > 0

        after_pressure = await client.get(
            f"/demo/session/{session_id}",
            params={"now": created_at + launch_total + 25},
        )
        after_pressure_body = after_pressure.json()
        labels = {event["label"] for event in after_pressure_body["runtime"]["events"]}
        assert "Input spike" in labels

    @pytest.mark.asyncio
    async def test_launch_can_include_live_koi_decision_summary(self, client):
        async def _fake_koi(*args, **kwargs):
            return {
                "_decision_id": "dec-demo-1",
                "predicted_tps": 1875.0,
                "confidence": 0.91,
                "config": {
                    "gpu_type": "A100-80GB",
                    "instance_type": "p4de.24xlarge",
                    "tp": 8,
                    "pp": 1,
                },
            }

        demo_server._request_koi_decision = _fake_koi
        resp = await client.post(
            "/demo/launch",
            json={
                "model_name": "Qwen/Qwen3-32B",
                "avg_input_tokens": 800,
                "avg_output_tokens": 200,
                "quota_preset": "aws_mixed_demo",
                "scenario": "hero_elastic",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["koi"]["decision"]["_decision_id"] == "dec-demo-1"
        assert body["launch_preview"]["preferred_gpu"] == "A100-80GB"
        assert body["launch_preview"]["baseline_replica_tps"] == 1875.0
        assert body["launch_preview"]["tp"] == 8
        assert body["launch_preview"]["pp"] == 1
