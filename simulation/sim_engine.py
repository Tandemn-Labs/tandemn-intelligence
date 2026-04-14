"""Shared simulator state and deterministic progression helpers."""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Optional


@dataclass
class SimReplica:
    replica_id: str
    phase: str = "running"
    base_tps: float = 1200.0
    gpu_type: str = "L40S"
    instance_type: str = "g6e.12xlarge"
    tp: int = 4
    pp: int = 2
    region: str = "us-east-1"
    market: str = "on_demand"
    config_index: int = 0
    started_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    warmup_seconds: float = 30.0
    wobble_pct: float = 0.10

    @property
    def tps(self) -> float:
        """Current TPS with warmup ramp and bounded noise."""
        if self.phase != "running":
            return 0.0
        elapsed = time.time() - self.started_at
        if elapsed < self.warmup_seconds:
            ramp = elapsed / self.warmup_seconds
        else:
            ramp = 1.0
        base = self.base_tps * ramp
        noise = random.gauss(1.0, self.wobble_pct)
        return max(0.0, base * noise)


@dataclass
class SimJob:
    job_id: str
    model_name: str
    replicas: Dict[str, SimReplica] = field(default_factory=dict)
    total_chunks: int = 500
    completed_chunks: int = 0
    failed_chunks: int = 0
    status: str = "running"
    slo_deadline_hours: float = 8.0
    decision_id: Optional[str] = None
    tokens_per_chunk: int = 12000
    deploy_timestamp: float = field(default_factory=time.time)


class SimState:
    """Global simulator state."""

    def __init__(self):
        self.jobs: Dict[str, SimJob] = {}
        self.koi_url: str = "http://localhost:8090"
        self._chunk_task: Optional[asyncio.Task] = None

    @property
    def primary_job(self) -> Optional[SimJob]:
        return next(iter(self.jobs.values()), None)


def aggregate_job_tps(job: SimJob, running_phases: tuple[str, ...] = ("running",)) -> float:
    return sum(replica.tps for replica in job.replicas.values() if replica.phase in running_phases)


async def advance_chunks_once(
    state: SimState,
    tick_seconds: float = 5.0,
    notify_complete: Optional[Callable[[SimJob], Awaitable[None]]] = None,
    heartbeat_at: Optional[float] = None,
) -> None:
    """Advance all running jobs by one simulator tick."""
    heartbeat_at = heartbeat_at or time.time()
    for job in list(state.jobs.values()):
        if job.status != "running":
            continue

        aggregate_tps = aggregate_job_tps(job)
        if aggregate_tps <= 0:
            continue

        tokens_this_tick = aggregate_tps * tick_seconds
        chunks_this_tick = tokens_this_tick / max(job.tokens_per_chunk, 1)
        job.completed_chunks = min(job.total_chunks, job.completed_chunks + int(chunks_this_tick))

        for replica in job.replicas.values():
            if replica.phase == "running":
                replica.last_heartbeat = heartbeat_at

        if job.completed_chunks >= job.total_chunks:
            job.status = "succeeded"
            job.completed_chunks = job.total_chunks
            if notify_complete is not None:
                await notify_complete(job)


async def advance_chunks_loop(
    state: SimState,
    notify_complete: Optional[Callable[[SimJob], Awaitable[None]]] = None,
    tick_seconds: float = 5.0,
) -> None:
    """Continuously advance jobs forever."""
    while True:
        await asyncio.sleep(tick_seconds)
        await advance_chunks_once(
            state,
            tick_seconds=tick_seconds,
            notify_complete=notify_complete,
        )
