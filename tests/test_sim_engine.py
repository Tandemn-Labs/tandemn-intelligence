"""Unit tests for the shared simulator engine."""

import pytest

from simulation.sim_engine import SimJob, SimReplica, SimState, advance_chunks_once


def _running_replica(replica_id: str, base_tps: float) -> SimReplica:
    return SimReplica(
        replica_id=replica_id,
        phase="running",
        base_tps=base_tps,
        warmup_seconds=0,
        wobble_pct=0,
    )


class TestAdvanceChunks:
    @pytest.mark.asyncio
    async def test_advances_progress_and_refreshes_heartbeats(self):
        state = SimState()
        job = SimJob(job_id="job-1", model_name="Qwen/Qwen3-32B", total_chunks=100, tokens_per_chunk=1000)
        job.replicas["job-1-r0"] = _running_replica("job-1-r0", 120.0)
        job.replicas["job-1-r1"] = _running_replica("job-1-r1", 80.0)
        state.jobs[job.job_id] = job

        await advance_chunks_once(state, tick_seconds=5.0, heartbeat_at=1234.0)

        assert job.completed_chunks == 1
        assert job.status == "running"
        assert job.replicas["job-1-r0"].last_heartbeat == 1234.0
        assert job.replicas["job-1-r1"].last_heartbeat == 1234.0

    @pytest.mark.asyncio
    async def test_marks_job_succeeded_and_notifies_on_completion(self):
        state = SimState()
        job = SimJob(
            job_id="job-2",
            model_name="Qwen/Qwen3-32B",
            total_chunks=10,
            completed_chunks=9,
            tokens_per_chunk=1000,
        )
        job.replicas["job-2-r0"] = _running_replica("job-2-r0", 200.0)
        state.jobs[job.job_id] = job

        completed = []

        async def _notify_done(done_job: SimJob):
            completed.append(done_job.job_id)

        await advance_chunks_once(
            state,
            tick_seconds=5.0,
            notify_complete=_notify_done,
            heartbeat_at=555.0,
        )

        assert job.status == "succeeded"
        assert job.completed_chunks == 10
        assert completed == ["job-2"]
