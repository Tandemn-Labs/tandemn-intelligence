"""
koi/server.py — FastAPI HTTP service for Koi v2.

Endpoints:
  POST /decide         → agent placement decision
  POST /job/complete   → webhook from Orca on job completion
  GET  /health         → service health
  GET  /jobs           → tracked jobs status

Usage:
  ANTHROPIC_API_KEY=sk-ant-... python -m koi.server
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import aiohttp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from koi.agent import KoiAgent
from koi.monitor import MonitoringLoop
from koi.schemas import JobRequest, MonitoringStatus
from koi.tools.memory import AgenticMemory
from koi.tools.orca_api import OrcaClient
from koi.tools.perfdb import PerfDB
from koi.tools.resources import parse_orca_resources

logger = logging.getLogger("koi.server")

KOI_PORT = int(os.environ.get("KOI_PORT", "8090"))


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class DecideRequest(BaseModel):
    job_request: Dict[str, Any]
    resource_map: Any  # Shape A, B, or C


class JobCompleteRequest(BaseModel):
    job_id: str
    status: str
    metrics: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init components
    perfdb_path = os.environ.get("KOI_PERFDB_PATH", "./perfdb/perfdb_all.csv")
    memory_path = os.environ.get("KOI_MEMORY_PATH", "./data/koi_memory.db")
    orca_url = os.environ.get("ORCA_URL", "")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    model = os.environ.get("KOI_LLM_MODEL", "claude-sonnet-4-6")

    # PerfDB
    try:
        app.state.perfdb = PerfDB(perfdb_path)
        logger.info(f"[Koi] PerfDB loaded: {app.state.perfdb.record_count} records, "
                     f"models={app.state.perfdb.models}, gpus={app.state.perfdb.gpu_types}")
    except Exception as e:
        logger.warning(f"[Koi] PerfDB load failed: {e}. Running without benchmark data.")
        app.state.perfdb = None

    # Memory
    app.state.memory = AgenticMemory(db_path=memory_path)
    logger.info(f"[Koi] Memory: {app.state.memory.decision_count()} decisions, "
                f"{app.state.memory.outcome_count()} outcomes, "
                f"{app.state.memory.rule_count()} rules")

    # Orca client
    app.state.session = aiohttp.ClientSession()
    app.state.orca = OrcaClient(orca_url, session=app.state.session) if orca_url else None
    if orca_url:
        logger.info(f"[Koi] Orca client: {orca_url}")
    else:
        logger.info("[Koi] No ORCA_URL set — running without Orca connection")

    # Agent
    app.state.agent = KoiAgent(
        perfdb=app.state.perfdb,
        memory=app.state.memory,
        orca=app.state.orca,
        api_key=api_key,
        model=model,
    )
    logger.info(f"[Koi] Agent ready (model={model})")

    # Monitor
    app.state.monitor = MonitoringLoop(
        orca=app.state.orca,
        memory=app.state.memory,
        on_trigger=app.state.agent.handle_trigger,
    )
    if app.state.orca:
        await app.state.monitor.start()
        logger.info("[Koi] Monitor started (3 async loops)")
    else:
        logger.info("[Koi] Monitor not started — no Orca connection")

    yield

    # Cleanup
    await app.state.monitor.stop()
    if app.state.session and not app.state.session.closed:
        await app.state.session.close()
    logger.info("[Koi] Shutdown complete")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Koi Placement Service", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0",
        "perfdb_records": app.state.perfdb.record_count if app.state.perfdb else 0,
        "memory_decisions": app.state.memory.decision_count(),
        "memory_outcomes": app.state.memory.outcome_count(),
        "memory_rules": app.state.memory.rule_count(),
        "tracked_jobs": len(app.state.monitor.tracked_jobs),
        "agent_model": app.state.agent.model,
        "orca_connected": app.state.orca is not None,
    }


@app.post("/decide")
async def decide(req: DecideRequest):
    """Run the Koi agent to make a placement decision."""
    agent: KoiAgent = app.state.agent
    monitor: MonitoringLoop = app.state.monitor

    # Parse job request
    try:
        from koi.schemas import TaskType, Objective
        d = req.job_request
        job_request = JobRequest(
            model_name=str(d.get("model_name", "unknown")),
            task_type=TaskType(d.get("task_type", "batch")),
            avg_input_tokens=int(d.get("avg_input_tokens", 512)),
            avg_output_tokens=int(d.get("avg_output_tokens", 256)),
            num_requests=int(d["num_requests"]) if d.get("num_requests") else None,
            slo_deadline_hours=float(d["slo_deadline_hours"]) if d.get("slo_deadline_hours") else None,
            objective=Objective(d.get("objective", "cheapest")),
            quantization=d.get("quantization"),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid job_request: {e}")

    # Parse resource map
    try:
        resource_map = parse_orca_resources(req.resource_map)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Run agent
    try:
        decision = await agent.decide(job_request, resource_map)
    except Exception as e:
        logger.error(f"[Koi] Agent error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    # Record decision in memory
    decision_id = app.state.memory.record_decision(
        job_id=decision.job_id,
        model_name=decision.model_name,
        instance_type=decision.config.instance_type,
        gpu_type=decision.config.gpu_type,
        tp=decision.config.tp, pp=decision.config.pp, dp=decision.config.dp,
        num_gpus=decision.config.num_gpus,
        predicted_tps=decision.predicted_tps,
        predicted_cost_per_hour=decision.predicted_cost_per_hour,
        predicted_total_cost=decision.predicted_total_cost,
        predicted_runtime_hours=decision.predicted_runtime_hours,
        prediction_confidence=decision.confidence,
        prediction_source=decision.data_source.value,
        slo_deadline_hours=job_request.slo_deadline_hours or 0,
        objective=job_request.objective.value,
        avg_input_tokens=job_request.avg_input_tokens,
        avg_output_tokens=job_request.avg_output_tokens,
        num_requests=job_request.num_requests,
        why_this_config=decision.reasoning[:500],
    )

    # Register in monitor
    if job_request.total_tokens and job_request.slo_deadline_hours:
        monitor.register_job(
            job_id=decision.job_id,
            config=decision.config,
            slo_deadline_hours=job_request.slo_deadline_hours,
            total_tokens=job_request.total_tokens,
            predicted_tps=decision.predicted_tps,
            decision_id=decision_id,
        )

    return decision.model_dump(mode="json")


@app.post("/job/complete")
async def job_complete(req: JobCompleteRequest):
    """Webhook from Orca when a job completes."""
    monitor: MonitoringLoop = app.state.monitor
    memory: AgenticMemory = app.state.memory

    tracker = monitor.tracked_jobs.get(req.job_id)
    if not tracker:
        logger.warning(f"[Koi] Job complete webhook for unknown job: {req.job_id}")
        # Still record what we can
        return {"status": "unknown_job", "job_id": req.job_id}

    # Record outcome
    actual_tps = req.metrics.get("avg_generation_throughput_toks_per_s")
    actual_cost_per_hour = req.metrics.get("cost_per_hour")

    if tracker.decision_id:
        outcome_id = memory.record_outcome(
            decision_id=tracker.decision_id,
            job_id=req.job_id,
            status=req.status,
            actual_tps=actual_tps,
            actual_cost_per_hour=actual_cost_per_hour,
            actual_runtime_hours=tracker.elapsed_hours,
            slo_met=req.status == "succeeded",
            slo_headroom_pct=tracker.slo_headroom_pct,
        )
        logger.info(f"[Koi] Outcome recorded: {outcome_id} for {req.job_id} ({req.status})")

    # Unregister from monitor
    monitor.unregister_job(req.job_id)

    return {"status": "recorded", "job_id": req.job_id}


@app.get("/jobs")
async def list_jobs():
    """List all tracked jobs with current status."""
    monitor: MonitoringLoop = app.state.monitor
    jobs = []
    for job_id, tracker in monitor.tracked_jobs.items():
        jobs.append({
            "job_id": job_id,
            "status": tracker.status.value,
            "gpu_type": tracker.config.gpu_type,
            "tp": tracker.config.tp,
            "pp": tracker.config.pp,
            "dp": tracker.config.dp,
            "smoothed_tps": round(tracker.smoothed_tps, 1),
            "slo_headroom_pct": round(tracker.slo_headroom_pct, 1),
            "elapsed_hours": round(tracker.elapsed_hours, 2),
            "tokens_completed": tracker.tokens_completed,
            "tokens_remaining": tracker.tokens_remaining,
        })
    return {"tracked_jobs": len(jobs), "jobs": jobs}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    # Configure logging so [Koi] messages show up
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=KOI_PORT)
