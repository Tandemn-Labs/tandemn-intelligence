"""
koi/server.py — FastAPI HTTP service for Koi placement.

Endpoints:
  GET  /health         → service health + perfdb/model info
  POST /decide         → placement decision from job_request + resource_map
  POST /job/complete   → (Phase 2) record job completion
  POST /reconfig       → (Phase 2) monitoring-triggered reconfiguration

Usage:
  ANTHROPIC_API_KEY=sk-ant-... python -m koi.server
  ANTHROPIC_API_KEY=sk-ant-... uvicorn koi.server:app --host 0.0.0.0 --port 8090
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from koi.intake import _dict_to_job_request, _parse_resource_map_response
from koi.placement import KoiPlacement

KOI_PORT = int(os.environ.get("KOI_PORT", "8090"))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class DecideRequest(BaseModel):
    job_request: Dict[str, Any]
    resource_map: Any  # Shape A, B, or C


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Lifespan — init KoiPlacement once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.koi = KoiPlacement(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        perfdb_path=os.environ.get("KOI_PERFDB_PATH", "./perfdb"),
        data_dir=os.environ.get("KOI_DATA_DIR", "./data"),
        llm_model=os.environ.get("KOI_LLM_MODEL", "claude-opus-4-6"),
    )
    app.state.tracked_jobs: set[str] = set()
    # Phase 2: start background monitor here
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Koi Placement Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Phase 1 endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    koi: KoiPlacement = app.state.koi
    return {
        "status": "ok",
        "perfdb_entries": len(koi.oracle.rag.records) if hasattr(koi.oracle, "rag") else 0,
        "model": koi.ensemble.model,
        "tracked_jobs": len(app.state.tracked_jobs),
    }


@app.post("/decide")
async def decide(req: DecideRequest):
    koi: KoiPlacement = app.state.koi

    try:
        job_request = _dict_to_job_request(req.job_request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid job_request: {e}")

    try:
        resource_map = _parse_resource_map_response(req.resource_map)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        decision = await koi.decide_async(job_request, resource_map)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    app.state.tracked_jobs.add(decision.job_id)

    return decision.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Phase 2 stubs
# ---------------------------------------------------------------------------

@app.post("/job/complete", status_code=501)
async def job_complete():
    raise HTTPException(status_code=501, detail="Not implemented yet — Phase 2")


@app.post("/reconfig", status_code=501)
async def reconfig():
    raise HTTPException(status_code=501, detail="Not implemented yet — Phase 2")


# ---------------------------------------------------------------------------
# Entry point: python -m koi.server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=KOI_PORT)
