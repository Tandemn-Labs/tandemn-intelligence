from koi.runtime_state import RuntimeStateStore
from koi.schemas import EngineConfig, JobTracker, PlacementConfig


def _sample_tracker(job_id: str = "job-1", decision_id: str = "dec-1", group_id: str = "grp-1") -> JobTracker:
    return JobTracker(
        job_id=job_id,
        decision_id=decision_id,
        group_id=group_id,
        config=PlacementConfig(
            gpu_type="L40S",
            instance_type="g6e.12xlarge",
            num_gpus=4,
            num_instances=1,
            tp=4,
            pp=1,
            dp=1,
            region="us-west-2",
            market="spot",
            engine_config=EngineConfig(
                tensor_parallel_size=4,
                pipeline_parallel_size=1,
            ),
        ),
        slo_deadline_hours=2.0,
        total_tokens=12345,
        predicted_tps=900.0,
    )


def test_store_round_trips_tracked_job():
    store = RuntimeStateStore(":memory:")
    tracker = _sample_tracker()

    store.upsert_tracked_job(tracker.job_id, tracker.model_dump(mode="json"))

    loaded = store.load_tracked_jobs()
    assert set(loaded) == {"job-1"}
    assert loaded["job-1"]["group_id"] == "grp-1"
    assert loaded["job-1"]["decision_id"] == "dec-1"
    assert loaded["job-1"]["tracker"]["config"]["gpu_type"] == "L40S"
    assert loaded["job-1"]["tracker"]["total_tokens"] == 12345


def test_store_round_trips_pending_launch():
    store = RuntimeStateStore(":memory:")
    launch = {
        "group_id": "grp-1",
        "gpu_type": "L40S",
        "instance_type": "g6e.12xlarge",
        "region": "us-west-2",
        "market": "spot",
        "launched_at": 123.4,
    }

    store.upsert_pending_launch("replica-1", launch)

    loaded = store.load_pending_launches()
    assert loaded["replica-1"]["launch"] == launch


def test_store_round_trips_pending_replica_decisions():
    """Per-replica scale correlation: each replica_id maps to its own decision,
    survives restart, and can be consumed exactly once."""
    store = RuntimeStateStore(":memory:")
    store.upsert_pending_replica_decision(
        replica_id="mo-abc-v2-r0",
        decision_id="dec-a",
        scale_request_id="sr-1",
        decision={"gpu_type": "L40S", "tp": 4, "pp": 2},
    )
    store.upsert_pending_replica_decision(
        replica_id="mo-abc-v2-r1",
        decision_id="dec-a",
        scale_request_id="sr-1",
        decision={"gpu_type": "L40S", "tp": 4, "pp": 2},
    )
    store.upsert_pending_replica_decision(
        replica_id="mo-xyz-v3-r0",
        decision_id="dec-b",
        scale_request_id="sr-2",
        decision={"gpu_type": "L4", "tp": 8, "pp": 1},
    )

    loaded = store.load_pending_replica_decisions()
    assert set(loaded.keys()) == {"mo-abc-v2-r0", "mo-abc-v2-r1", "mo-xyz-v3-r0"}
    assert loaded["mo-abc-v2-r0"]["decision_id"] == "dec-a"
    assert loaded["mo-xyz-v3-r0"]["decision_id"] == "dec-b"
    assert loaded["mo-abc-v2-r0"]["scale_request_id"] == "sr-1"

    store.delete_pending_replica_decision("mo-abc-v2-r0")
    remaining = store.load_pending_replica_decisions()
    assert "mo-abc-v2-r0" not in remaining
    assert "mo-abc-v2-r1" in remaining  # sibling untouched


def test_upsert_pending_replica_decision_is_idempotent():
    store = RuntimeStateStore(":memory:")
    store.upsert_pending_replica_decision(
        replica_id="r1", decision_id="dec-a", decision={"v": 1}
    )
    store.upsert_pending_replica_decision(
        replica_id="r1", decision_id="dec-b", decision={"v": 2}
    )  # overwrite
    loaded = store.load_pending_replica_decisions()
    assert len(loaded) == 1
    assert loaded["r1"]["decision_id"] == "dec-b"
    assert loaded["r1"]["decision"] == {"v": 2}


def test_store_round_trips_ledger_reservation():
    store = RuntimeStateStore(":memory:")
    reservation = {
        "gpu_type": "L40S",
        "num_gpus": 8,
        "cloud": "aws",
        "region": "us-west-2",
        "instance_type": "g6e.24xlarge",
        "tenant_id": "default",
        "decision_id": "dec-123",
        "created_at": 111.2,
    }

    store.upsert_ledger_reservation("dec-123", reservation, expires_at=999.9)

    loaded = store.load_ledger_reservations()
    assert loaded["dec-123"]["reservation"] == reservation
    assert loaded["dec-123"]["expires_at"] == 999.9
