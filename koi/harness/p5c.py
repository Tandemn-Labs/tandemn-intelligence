"""P5c chain post-mortem harness.

P5c converts a dead replica event into a structured diagnosis that future
recovery prompts can consume. It records evidence; it does not mutate cluster
state.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from koi.harness.cooloff import CooloffScope
from koi.harness.failures import classify_failure
from koi.harness.packet_tools import build_packet_read_tools
from koi.harness.prompts import HARNESS_SYSTEM_PROMPT
from koi.harness.schemas import HarnessState, TransitionPacket, TransitionType
from koi.llm import KoiToolRunner
from koi.logging_config import get_logger
from koi.schemas import MonitoringStatus
from koi.tools.memory import AgenticMemory

logger = get_logger("koi.harness.p5c")

P5C_TIMEOUT = 60.0
P5C_MAX_ITERATIONS = 2

_KNOWN_SECTIONS = (
    "failure",
    "runtime_metrics",
    "config",
    "memory",
    "quota",
    "cooloffs",
)


class P5cDiagnosis(BaseModel):
    diagnosis_code: str = Field(min_length=1)
    bottleneck: str = "unknown"
    next_fix: str = "operator_review"
    failure_scope: str = "unknown"
    event_at: float = Field(default_factory=time.time)
    avoid_until: Optional[float] = None
    hard_until: Optional[float] = None
    cooloff_key: Optional[str] = None
    cooloff_minutes: int = 0
    rationale: str = ""


def _reason_text(req: Any) -> str:
    reason_code = getattr(req, "reason_code", None)
    reason_detail = getattr(req, "reason_detail", None)
    raw = getattr(req, "reason", "") or ""
    parts = []
    if reason_code:
        parts.append(str(getattr(reason_code, "value", reason_code)))
    if reason_detail:
        parts.append(str(reason_detail))
    if raw:
        parts.append(str(raw))
    return "; ".join(parts) or "unknown"


def _scope_for_tracker(tracker: Any, *, region: str, market: str, include_topology: bool = False) -> CooloffScope:
    config = tracker.config
    return CooloffScope(
        gpu_type=str(config.gpu_type),
        instance_type=str(config.instance_type),
        region=region,
        market=market,
        tp=int(config.tp),
        pp=int(config.pp),
        dp=int(config.dp),
    )


def deterministic_diagnosis(
    *,
    req: Any,
    tracker: Any,
    failure_category: Optional[str] = None,
    region: str,
    market: str,
    actual_tps_before_death: Optional[float],
    now: Optional[float] = None,
) -> P5cDiagnosis:
    event_at = time.time() if now is None else now
    reason = _reason_text(req)
    if not failure_category:
        failure_category = classify_failure(reason)
    include_topology = failure_category == "oom"
    scope = _scope_for_tracker(
        tracker,
        region=region,
        market=market,
        include_topology=include_topology,
    )
    scope_key = scope.key(include_topology=include_topology)

    if failure_category == "spot_preemption":
        return P5cDiagnosis(
            diagnosis_code="spot_preemption",
            bottleneck="market_capacity",
            next_fix="retry_same_topology_on_demand",
            failure_scope=scope_key,
            event_at=event_at,
            avoid_until=event_at + 30 * 60,
            hard_until=event_at + 10 * 60,
            cooloff_key=scope_key,
            cooloff_minutes=30,
            rationale=f"Replica was interrupted on spot capacity: {reason}",
        )
    if failure_category in {"no_capacity", "quota"}:
        code = "quota_exhausted" if failure_category == "quota" else "no_capacity"
        return P5cDiagnosis(
            diagnosis_code=code,
            bottleneck="market_capacity",
            next_fix="switch_market_or_gpu_family",
            failure_scope=scope_key,
            event_at=event_at,
            avoid_until=event_at + 20 * 60,
            hard_until=event_at + 5 * 60,
            cooloff_key=scope_key,
            cooloff_minutes=20,
            rationale=f"Replica failed from capacity/quota pressure: {reason}",
        )
    if failure_category == "oom":
        return P5cDiagnosis(
            diagnosis_code="oom",
            bottleneck="memory_bound",
            next_fix="increase_vram_or_reduce_memory_pressure",
            failure_scope=scope_key,
            event_at=event_at,
            avoid_until=event_at + 60 * 60,
            hard_until=event_at + 20 * 60,
            cooloff_key=scope_key,
            cooloff_minutes=60,
            rationale=f"Replica likely exceeded memory capacity: {reason}",
        )
    if "heartbeat" in reason.lower() or "timeout" in reason.lower():
        return P5cDiagnosis(
            diagnosis_code="heartbeat_timeout",
            bottleneck="runtime_unhealthy",
            next_fix="replace_same_config_if_fleet_needs_capacity",
            failure_scope=scope_key,
            event_at=event_at,
            avoid_until=None,
            hard_until=None,
            cooloff_key=None,
            cooloff_minutes=0,
            rationale=(
                f"Replica stopped reporting heartbeat after previously producing "
                f"{actual_tps_before_death or 0:.0f} TPS: {reason}"
            ),
        )
    return P5cDiagnosis(
        diagnosis_code=failure_category or "unknown_failure",
        bottleneck="unknown",
        next_fix="operator_review",
        failure_scope=scope_key,
        event_at=event_at,
        rationale=f"No deterministic failure diagnosis matched: {reason}",
    )


def build_p5c_packet(
    *,
    req: Any,
    tracker: Any,
    memory: AgenticMemory,
    failure_category: str,
    region: str,
    market: str,
    actual_tps_before_death: Optional[float],
) -> TransitionPacket:
    config = tracker.config
    failure_context = {
        "replica_id": getattr(req, "replica_id", None) or getattr(req, "job_id", "unknown"),
        "group_id": getattr(req, "group_id", None),
        "reason": getattr(req, "reason", ""),
        "reason_code": str(getattr(getattr(req, "reason_code", None), "value", getattr(req, "reason_code", None)) or ""),
        "reason_detail": getattr(req, "reason_detail", None),
        "failure_category": failure_category,
        "region": region,
        "market": market,
    }
    runtime_context = {
        "actual_tps_before_death": actual_tps_before_death,
        "smoothed_tps_after_mark_failed": tracker.smoothed_tps,
        "predicted_tps": tracker.predicted_tps,
        "elapsed_hours": tracker.elapsed_hours,
        "tokens_completed": tracker.tokens_completed,
        "tokens_remaining": tracker.tokens_remaining,
        "slo_headroom_pct": tracker.slo_headroom_pct,
        "gpu_cache_usage": tracker.gpu_cache_usage,
        "gpu_sm_util": tracker.gpu_sm_util,
        "gpu_mem_bw_util": tracker.gpu_mem_bw_util,
    }
    config_context = {
        "gpu_type": config.gpu_type,
        "instance_type": config.instance_type,
        "tp": config.tp,
        "pp": config.pp,
        "dp": config.dp,
        "num_gpus": config.num_gpus,
        "region": region,
        "market": market,
    }
    try:
        memory_context = {
            "recent_outcomes": memory.query_outcomes(
                job_id=getattr(req, "group_id", None),
                limit=10,
            ),
            "failure_summary": memory.get_failure_summary(
                config.gpu_type,
                region=region if region != "unknown" else None,
                market=market if market != "unknown" else None,
            ),
            "active_cooloffs": memory.get_active_cooloffs(
                gpu_type=config.gpu_type,
                region=region if region != "unknown" else None,
                market=market if market != "unknown" else None,
            ),
        }
    except Exception as exc:
        memory_context = {"error": str(exc)}

    detail_sections = {
        "failure:chain": failure_context,
        "runtime_metrics:chain": runtime_context,
        "config:chain": config_context,
        "memory:chain": memory_context,
        "quota:chain": memory_context.get("failure_summary", {}),
        "cooloffs:chain": memory_context.get("active_cooloffs", []),
    }
    return TransitionPacket(
        packet_id=f"p5c-{getattr(req, 'job_id', 'unknown')}",
        job_id=getattr(req, "job_id", "unknown"),
        state=HarnessState.REPLICA_RECOVERY,
        transition_type=TransitionType.CHAIN_POSTMORTEM,
        job_context={
            "group_id": getattr(req, "group_id", None),
            "decision_id": tracker.decision_id,
            "status_before_event": MonitoringStatus.FAILED.value,
        },
        runtime_context=runtime_context,
        failure_context=failure_context,
        evidence_summary={
            "has_actual_tps_before_death": bool(actual_tps_before_death and actual_tps_before_death > 0),
            "failure_category": failure_category,
            "market_known": market != "unknown",
            "region_known": region != "unknown",
        },
        detail_sections=detail_sections,
        guards={
            "read_only": True,
            "write_diagnosis_only": True,
        },
    )


def render_p5c_prompt(packet: TransitionPacket) -> str:
    lines = [
        "P5C CHAIN POST-MORTEM",
        "Diagnose why this replica/chain died. Do not propose or execute cluster actions.",
        "Return a typed P5cDiagnosis. Use avoid_until only for fresh spot/no-capacity/quota/OOM scopes.",
        "",
        "FAILURE CONTEXT:",
        json.dumps(packet.failure_context, indent=2, sort_keys=True),
        "",
        "RUNTIME CONTEXT:",
        json.dumps(packet.runtime_context, indent=2, sort_keys=True),
        "",
        "EVIDENCE SUMMARY:",
        json.dumps(packet.evidence_summary, indent=2, sort_keys=True),
        "",
        "DETAIL REFS:",
        json.dumps(sorted(packet.detail_sections), indent=2),
        "",
        "Diagnosis fields must be concise and machine-readable.",
    ]
    return "\n".join(lines)


def _packet_tools(memory: AgenticMemory, packet: TransitionPacket) -> dict[str, Any]:
    tools = build_packet_read_tools(packet, known_sections=_KNOWN_SECTIONS, include_packet_sections=True)

    async def get_failure_summary(gpu_type: str, region: Optional[str] = None, market: Optional[str] = None) -> str:
        try:
            return json.dumps(
                memory.get_failure_summary(gpu_type, region=region, market=market),
                indent=2,
                default=str,
            )
        except Exception as exc:
            return f"get_failure_summary failed: {exc}"

    tools["get_failure_summary"] = get_failure_summary
    return tools


def _normalize_diagnosis(
    diagnosis: P5cDiagnosis,
    *,
    fallback: P5cDiagnosis,
) -> P5cDiagnosis:
    data = diagnosis.model_dump()
    if not data.get("failure_scope") or data["failure_scope"] == "unknown":
        data["failure_scope"] = fallback.failure_scope
    if data.get("cooloff_minutes") and not data.get("avoid_until"):
        data["avoid_until"] = diagnosis.event_at + int(data["cooloff_minutes"]) * 60
    if data.get("avoid_until") and not data.get("cooloff_key"):
        data["cooloff_key"] = data["failure_scope"]
    return P5cDiagnosis(**data)


async def run_chain_postmortem(
    *,
    agent: Any,
    req: Any,
    tracker: Any,
    memory: AgenticMemory,
    failure_category: str,
    region: str,
    market: str,
    actual_tps_before_death: Optional[float],
) -> P5cDiagnosis:
    fallback = deterministic_diagnosis(
        req=req,
        tracker=tracker,
        failure_category=failure_category,
        region=region,
        market=market,
        actual_tps_before_death=actual_tps_before_death,
    )
    # Fast-path: if no LLM is wired (test agents, deterministic mode), skip the
    # reasoner call and use the deterministic diagnosis directly. P5c is a
    # post-mortem; even without an LLM, recording the typed diagnosis and a
    # cooloff is strictly better than raw free-text only.
    model = getattr(agent, "_model", None)
    if model is None:
        return fallback

    packet = build_p5c_packet(
        req=req,
        tracker=tracker,
        memory=memory,
        failure_category=failure_category,
        region=region,
        market=market,
        actual_tps_before_death=actual_tps_before_death,
    )
    runner = KoiToolRunner(
        model=model,
        system_prompt=HARNESS_SYSTEM_PROMPT,
        tools=_packet_tools(memory, packet),
    )
    try:
        _, diagnosis = await runner.run_typed(
            render_p5c_prompt(packet),
            label="p5c",
            job_id=packet.job_id,
            max_iterations=P5C_MAX_ITERATIONS,
            timeout=P5C_TIMEOUT,
            output_type=P5cDiagnosis,
        )
        return _normalize_diagnosis(diagnosis, fallback=fallback)
    except asyncio.TimeoutError:
        logger.error("p5c_timeout", job_id=packet.job_id, timeout=P5C_TIMEOUT)
        return fallback
    except Exception as exc:
        # P5c is purely a recorder. A bad LLM response, transport error, or
        # malformed structured output should never block the replica-failed
        # webhook. Always fall back to the deterministic diagnosis.
        logger.warning(
            "p5c_reasoner_failed",
            job_id=packet.job_id,
            error=str(exc),
        )
        return fallback
