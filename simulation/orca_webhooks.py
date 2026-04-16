"""Shared Orca -> Koi webhook payload helpers for both simulators."""

from __future__ import annotations

import time
from typing import Any, Optional


def heartbeat_message(phase: str) -> str:
    messages = {
        "searching_capacity": "Searching quota and trying candidate capacity.",
        "provisioning": "Instances requested and provisioning is in progress.",
        "bootstrapping": "Replica booted and is finishing runtime setup.",
        "waiting_model_ready": "Replica provisioned, waiting for model_ready.",
    }
    return messages.get(phase, phase.replace("_", " "))


def build_config_attempt_payload(
    *,
    job_id: str,
    decision_id: Optional[str],
    instance_type: str,
    gpu_type: str,
    region: str,
    market: str,
    launched: bool,
    attempt_index: int = 0,
    failure_reason: str = "",
    time_to_launch: float = 0.0,
) -> dict[str, Any]:
    payload = {
        "job_id": job_id,
        "instance_type": instance_type,
        "gpu_type": gpu_type,
        "region": region,
        "market": market,
        "launched": launched,
        "attempt_index": int(attempt_index),
    }
    if decision_id:
        payload["decision_id"] = decision_id
    if launched:
        payload["time_to_launch"] = float(max(0.0, time_to_launch))
    else:
        payload["failure_reason"] = failure_reason
    return payload


def build_launching_payload(
    *,
    job_id: str,
    group_id: Optional[str],
    decision_id: Optional[str],
    gpu_type: str,
    instance_type: str,
    tp: int,
    pp: int,
    region: str,
    market: str,
    attempt_index: int = 0,
) -> dict[str, Any]:
    payload = {
        "job_id": job_id,
        "gpu_type": gpu_type,
        "instance_type": instance_type,
        "tp": int(tp),
        "pp": int(pp),
        "region": region,
        "market": market,
        "attempt_index": int(attempt_index),
    }
    if group_id:
        payload["group_id"] = group_id
    if decision_id:
        payload["decision_id"] = decision_id
    return payload


def build_launch_heartbeat_payload(
    *,
    job_id: str,
    group_id: Optional[str],
    decision_id: Optional[str],
    gpu_type: str,
    instance_type: str,
    tp: int,
    pp: int,
    region: str,
    market: str,
    phase: str,
    message: Optional[str] = None,
    attempt_index: int = 0,
    timestamp: Optional[float] = None,
) -> dict[str, Any]:
    payload = build_launching_payload(
        job_id=job_id,
        group_id=group_id,
        decision_id=decision_id,
        gpu_type=gpu_type,
        instance_type=instance_type,
        tp=tp,
        pp=pp,
        region=region,
        market=market,
        attempt_index=attempt_index,
    )
    payload.update(
        {
            "phase": phase,
            "message": message or heartbeat_message(phase),
            "timestamp": float(timestamp or time.time()),
        }
    )
    return payload


def build_started_payload(
    *,
    job_id: str,
    group_id: Optional[str],
    decision_id: Optional[str],
    gpu_type: str,
    instance_type: str,
    tp: int,
    pp: int,
    dp: int,
    region: str,
    market: str,
    slo_deadline_hours: float,
    total_tokens: int,
    predicted_tps: float,
    is_fallback: bool = False,
) -> dict[str, Any]:
    payload = {
        "job_id": job_id,
        "gpu_type": gpu_type,
        "instance_type": instance_type,
        "tp": int(tp),
        "pp": int(pp),
        "dp": int(dp),
        "region": region,
        "market": market,
        "slo_deadline_hours": float(slo_deadline_hours),
        "total_tokens": int(total_tokens),
        "predicted_tps": float(predicted_tps),
        "is_fallback": bool(is_fallback),
    }
    if group_id:
        payload["group_id"] = group_id
    if decision_id:
        payload["decision_id"] = decision_id
    return payload


def build_launch_failed_payload(
    *,
    job_id: str,
    decision_id: Optional[str],
    configs_tried: list[dict[str, Any]],
    failure_reasons: list[str],
    total_time_seconds: float = 0.0,
) -> dict[str, Any]:
    payload = {
        "job_id": job_id,
        "configs_tried": list(configs_tried),
        "failure_reasons": list(failure_reasons),
        "total_time_seconds": float(max(0.0, total_time_seconds)),
    }
    if decision_id:
        payload["decision_id"] = decision_id
    return payload


def build_replica_failed_payload(
    *,
    job_id: str,
    group_id: Optional[str],
    reason: str,
    status: str = "failed",
) -> dict[str, Any]:
    payload = {
        "job_id": job_id,
        "status": status,
        "reason": reason,
    }
    if group_id:
        payload["group_id"] = group_id
    return payload


def build_complete_payload(
    *,
    job_id: str,
    throughput_tps: float,
    status: str = "succeeded",
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": status,
        "metrics": {
            "avg_generation_throughput_toks_per_s": float(throughput_tps),
            "throughput_tokens_per_sec": float(throughput_tps),
        },
    }
