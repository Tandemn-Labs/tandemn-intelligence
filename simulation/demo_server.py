"""Demo backend for the browser-based Koi + Orca simulator."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from simulation.demo_runtime import DemoSessionManager
from simulation.demo_scenarios import (
    get_quota_preset,
    get_scenario,
    quota_preset_to_resource_map,
    serialize_catalog,
)
from simulation.model_registry import resolve_model_spec
from simulation.perf_model import DemoPerfModel


app = FastAPI(title="Koi Demo Server", version="0.1")
PERF_MODEL = DemoPerfModel()
SESSION_MANAGER = DemoSessionManager()
STATIC_DIR = Path(__file__).resolve().parent / "static" / "demo"
app.mount("/demo/static", StaticFiles(directory=str(STATIC_DIR)), name="demo-static")
DEMO_KOI_URL = os.environ.get("KOI_DEMO_URL", "http://localhost:8090")


class DemoLaunchRequest(BaseModel):
    model_name: str
    avg_input_tokens: int = Field(ge=1)
    avg_output_tokens: int = Field(ge=1)
    total_chunks: int = Field(default=500, ge=1)
    slo_deadline_hours: float = Field(default=8.0, gt=0)
    quota_preset: str
    scenario: str
    cost_cap_usd: Optional[float] = Field(default=None, gt=0)
    dtype: str = "fp16"
    model_overrides: Optional[dict] = None


async def _request_koi_decision(
    req: DemoLaunchRequest,
    resource_map: dict,
) -> Optional[dict]:
    payload = {
        "job_request": {
            "model_name": req.model_name,
            "task_type": "batch",
            "avg_input_tokens": req.avg_input_tokens,
            "avg_output_tokens": req.avg_output_tokens,
            "num_requests": req.total_chunks * 10,
            "slo_deadline_hours": req.slo_deadline_hours,
            "objective": "cheapest",
        },
        "resource_map": resource_map,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(f"{DEMO_KOI_URL}/decide", json=payload)
        response.raise_for_status()
        return response.json()


@app.get("/demo/health")
async def demo_health():
    return {"status": "ok", "sessions": len(SESSION_MANAGER.sessions)}


@app.get("/demo")
async def demo_index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/demo/catalog")
async def demo_catalog():
    return serialize_catalog()


@app.post("/demo/launch")
async def demo_launch(req: DemoLaunchRequest):
    try:
        quota = get_quota_preset(req.quota_preset)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown quota preset: {req.quota_preset}") from exc

    try:
        scenario = get_scenario(req.scenario)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown scenario: {req.scenario}") from exc

    model_spec = resolve_model_spec(
        req.model_name,
        dtype=req.dtype,
        overrides=req.model_overrides,
    )

    resource_map = quota_preset_to_resource_map(req.quota_preset)
    koi_decision = None
    koi_error = None
    try:
        koi_decision = await _request_koi_decision(req, resource_map)
    except Exception as exc:
        koi_error = str(exc)

    preferred_gpu = (
        (koi_decision or {}).get("config", {}).get("gpu_type")
        or quota.instances[0].gpu_type
    )
    launch_timing = PERF_MODEL.estimate_launch_timing(
        gpu_type=preferred_gpu,
        capacity_pressure=0.8 if req.scenario == "slow_launch" else 0.2,
    )
    baseline_tps = (
        (koi_decision or {}).get("predicted_tps")
        or PERF_MODEL.estimate_replica_tps(
            model_name=req.model_name,
            gpu_type=preferred_gpu,
            tp=((koi_decision or {}).get("config", {}) or {}).get("tp", 4),
            pp=((koi_decision or {}).get("config", {}) or {}).get("pp", 1),
            input_tokens=req.avg_input_tokens,
            output_tokens=req.avg_output_tokens,
            dtype=req.dtype,
            overrides=req.model_overrides,
        )
    )

    session_id = f"demo-{uuid.uuid4().hex[:10]}"
    payload = {
        "session_id": session_id,
        "status": "created",
        "created_at": time.time(),
        "request": req.model_dump(mode="json"),
        "model": model_spec.__dict__,
        "scenario": {
            "slug": scenario.slug,
            "title": scenario.title,
            "description": scenario.description,
            "initial_replicas": scenario.initial_replicas,
            "launch_timing_multiplier": scenario.launch_timing_multiplier,
        },
        "quota": {
            "slug": quota.slug,
            "title": quota.title,
            "cloud": quota.cloud,
            "notes": quota.notes,
        },
        "resource_map": resource_map,
        "koi": {
            "configured_url": DEMO_KOI_URL,
            "decision": koi_decision,
            "error": koi_error,
        },
        "launch_preview": {
            "baseline_replica_tps": round(baseline_tps, 1),
            "launch_timing_s": {
                "searching_capacity": round(launch_timing.searching_capacity_s * scenario.launch_timing_multiplier, 1),
                "provisioning": round(launch_timing.provisioning_s * scenario.launch_timing_multiplier, 1),
                "bootstrapping": round(launch_timing.bootstrapping_s * scenario.launch_timing_multiplier, 1),
                "waiting_model_ready": round(launch_timing.waiting_model_ready_s * scenario.launch_timing_multiplier, 1),
                "total": round(launch_timing.total_seconds * scenario.launch_timing_multiplier, 1),
            },
            "preferred_gpu": preferred_gpu,
            "tp": ((koi_decision or {}).get("config", {}) or {}).get("tp", 4),
            "pp": ((koi_decision or {}).get("config", {}) or {}).get("pp", 1),
        },
    }
    return SESSION_MANAGER.create_session(payload)


@app.get("/demo/session/{session_id}")
async def demo_session(session_id: str, now: Optional[float] = None):
    if session_id not in SESSION_MANAGER.sessions:
        raise HTTPException(status_code=404, detail="unknown demo session")
    return SESSION_MANAGER.snapshot(session_id, now=now)


@app.get("/demo/stream/{session_id}")
async def demo_stream(session_id: str):
    if session_id not in SESSION_MANAGER.sessions:
        raise HTTPException(status_code=404, detail="unknown demo session")

    async def _events():
        yield ": connected\nretry: 1000\n\n"
        while True:
            snapshot = SESSION_MANAGER.snapshot(session_id)
            yield f"data: {json.dumps(snapshot)}\n\n"
            if snapshot["runtime"]["status"] == "completed":
                break
            await asyncio.sleep(1)

    return StreamingResponse(_events(), media_type="text/event-stream")
