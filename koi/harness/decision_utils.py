"""Shared placement and decision conversion helpers for harness executors."""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union

from koi.schemas import DataSource, EngineConfig, JobRequest, PlacementConfig


def source_to_data_source(source: str) -> DataSource:
    if source == "VERIFIED" or source == "memory_verified":
        return DataSource.MEMORY
    if source == "PerfDB" or source == "perfdb_exact":
        return DataSource.EXACT_MATCH
    return DataSource.ANALYTICAL


def source_to_prediction_source(source: str) -> str:
    if source == "VERIFIED":
        return "memory_verified"
    if source == "PerfDB":
        return "perfdb_exact"
    return source or "analytical"


def placement_config_from_payload(
    payload: dict[str, Any],
    *,
    fallback_region: str = "unknown",
) -> PlacementConfig:
    tp = int(payload.get("tp") or 1)
    pp = int(payload.get("pp") or 1)
    dp = int(payload.get("dp") or 1)
    num_gpus = tp * pp * dp
    return PlacementConfig(
        gpu_type=str(payload.get("gpu_type") or "unknown"),
        instance_type=str(payload.get("instance_type") or "unknown"),
        num_gpus=num_gpus,
        num_instances=max(1, int(payload.get("num_instances") or -(-num_gpus // 8))),
        tp=tp,
        pp=pp,
        dp=dp,
        region=str(payload.get("region") or fallback_region or "unknown"),
        engine_config=EngineConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp),
        market=str(payload.get("market") or payload.get("planned_market") or "on_demand"),
    )


def alternative_payloads(
    packet,
    selected_action_id: str,
    *,
    action_type: Union[str, Iterable[str]],
    exclude_action_types: Optional[Iterable[str]] = None,
    limit: int = 3,
    include_action_type_in_payload: bool = False,
) -> list[dict[str, Any]]:
    """Pull alternative executor payloads from a packet.

    Args:
        action_type: a single action_type to match, or any iterable of types.
            Pass ``"*"`` (or any string equal to ``"*"``) to accept any
            action_type.
        exclude_action_types: optional types to drop even if they match.
        limit: hard cap on returned alternatives.
        include_action_type_in_payload: when True, the returned dict also
            carries an ``action_type`` field (useful when the caller mixes
            multiple recovery action types).
    """

    if isinstance(action_type, str):
        accept = {action_type}
    else:
        accept = set(action_type)
    excluded = set(exclude_action_types or ())
    accept_any = accept == {"*"}

    alternatives: list[dict[str, Any]] = []
    for option in packet.valid_actions():
        if option.action_id == selected_action_id:
            continue
        if option.action_type in excluded:
            continue
        if not accept_any and option.action_type not in accept:
            continue
        payload = packet.detail_sections.get(option.executor_payload_ref or "", {})
        entry = {
            "gpu_type": payload.get("gpu_type"),
            "instance_type": payload.get("instance_type"),
            "tp": payload.get("tp"),
            "pp": payload.get("pp"),
            "dp": payload.get("dp", 1),
            "region": payload.get("region"),
            "market": payload.get("market") or payload.get("planned_market"),
            "predicted_tps": payload.get("predicted_tps"),
            "source": payload.get("source"),
        }
        if include_action_type_in_payload:
            entry["action_type"] = option.action_type
        alternatives.append(entry)
        if len(alternatives) >= limit:
            break
    return alternatives


def reconstruct_job_request(
    *,
    decision: dict[str, Any],
    job_id: str,
    force_on_demand: bool = False,
) -> JobRequest:
    """Rebuild a ``JobRequest`` from a persisted decision row.

    Used by P1 launch recovery and P4 replica recovery so both prompts share
    one truthful translation between memory rows and the typed schema.
    """

    market = decision.get("market")
    if force_on_demand:
        market = "on_demand"
    if market not in {"spot", "on_demand"}:
        market = None
    return JobRequest(
        job_id=job_id,
        model_name=str(decision.get("model_name") or "unknown"),
        avg_input_tokens=max(1, int(decision.get("avg_input_tokens") or 1)),
        avg_output_tokens=max(1, int(decision.get("avg_output_tokens") or 1)),
        num_requests=decision.get("num_requests"),
        slo_deadline_hours=decision.get("slo_deadline_hours") or None,
        objective=decision.get("objective") or "cheapest",
        cost_roofline_usd=decision.get("cost_roofline_usd"),
        preferred_market=market,
        quantization=decision.get("quantization"),
    )
