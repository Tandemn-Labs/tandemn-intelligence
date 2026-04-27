"""P1 launch recovery harness.

Phase 3 keeps the Orca-owned launch flow intact: Koi returns a recovery plan
from /job/launch-failed, and Orca decides whether to retry that plan.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import string
import time
from typing import Any, Optional

from koi.event_tap import emit_event
from koi.harness.reasoner import HarnessReasoner
from koi.harness.schemas import (
    ActionOption,
    ChosenAction,
    HarnessState,
    TransitionPacket,
    TransitionType,
    ValidatedAction,
)
from koi.harness.validator import NoValidActionError, validate_choice
from koi.logging_config import get_logger
from koi.model_features import compute_config_features, get_model_features
from koi.schemas import EngineConfig, JobRequest, PlacementConfig, ResourceMap
from koi.tools.memory import AgenticMemory
from koi.tools.resources import parse_orca_resources

logger = get_logger("koi.harness.p1")

P1_TIMEOUT = 120.0
P1_MAX_ITERATIONS = 3
MAX_MENU_OPTIONS = 8
DEFAULT_RETRY_BUDGET = 2

_FAILURE_PATTERNS = [
    (re.compile(r"spot|preempt", re.I), "spot_preemption"),
    (re.compile(r"insufficient.?capacity|no.?capacity", re.I), "no_capacity"),
    (re.compile(r"oom|out.?of.?memory|cuda.?oom", re.I), "oom"),
    (re.compile(r"quota", re.I), "quota"),
]

_KNOWN_SECTIONS = (
    "physics",
    "perfdb_exact",
    "memory_success",
    "memory_failure",
    "quota",
    "recent_failures",
    "failure",
    "executor_payload",
    "row",
)


def _action_id(index: int) -> str:
    if index < len(string.ascii_lowercase):
        return string.ascii_lowercase[index]
    return f"a{index + 1}"


def _retry_budget_limit() -> int:
    raw = os.environ.get("KOI_HARNESS_P1_RETRY_BUDGET", str(DEFAULT_RETRY_BUDGET))
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_RETRY_BUDGET


def _classify_failure(reason: str) -> str:
    for pattern, category in _FAILURE_PATTERNS:
        if pattern.search(reason or ""):
            return category
    return "unknown"


def _decision_chain(memory: AgenticMemory, decision_id: Optional[str]) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current = decision_id
    while current and current not in seen:
        seen.add(current)
        row = memory.get_decision(current)
        if not row:
            break
        chain.append(row)
        current = row.get("parent_decision_id")
    return chain


def _retry_budget_used(chain: list[dict[str, Any]]) -> int:
    return sum(1 for row in chain if row.get("triggered_by") == "launch_recovery")


def _reconstruct_job_request(
    *,
    decision: dict[str, Any],
    job_id: str,
    force_on_demand: bool,
) -> JobRequest:
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


async def _resource_map_for(
    agent: Any,
    ledger: Any = None,
    resource_map: Optional[ResourceMap] = None,
) -> Optional[ResourceMap]:
    if resource_map is not None:
        return ledger.apply_to_resource_map(resource_map) if ledger is not None else resource_map
    orca = getattr(agent, "orca", None)
    if orca is None or not hasattr(orca, "get_resources"):
        return None
    raw = await orca.get_resources()
    rm = parse_orca_resources(raw)
    if ledger is not None:
        rm = ledger.apply_to_resource_map(rm)
    return rm


def _failed_entries(req: Any, decision: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    configs = list(getattr(req, "configs_tried", []) or [])
    reasons = list(getattr(req, "failure_reasons", []) or [])
    entries: list[dict[str, Any]] = []
    for idx, cfg in enumerate(configs):
        item = dict(cfg or {})
        if decision and item.get("gpu_type") == decision.get("gpu_type"):
            item.setdefault("tp", decision.get("tp"))
            item.setdefault("pp", decision.get("pp"))
            item.setdefault("dp", decision.get("dp"))
        reason = reasons[idx] if idx < len(reasons) else "unknown"
        item["failure_reason"] = reason
        item["failure_category"] = _classify_failure(reason)
        entries.append(item)
    return entries


def _cfg_key(config: dict[str, Any], *, include_market: bool = True) -> tuple[Any, ...]:
    key: tuple[Any, ...] = (
        config.get("gpu_type"),
        config.get("instance_type"),
        int(config.get("tp") or 0),
        int(config.get("pp") or 0),
        int(config.get("dp") or 1),
    )
    if include_market:
        key = key + (config.get("market") or config.get("planned_market") or "unknown",)
    return key


def _matches_failed_same_scope(row: dict[str, Any], failed: list[dict[str, Any]]) -> bool:
    row_key = _cfg_key(row, include_market=True)
    for item in failed:
        if _cfg_key(item, include_market=True) == row_key:
            return True
        # Older Orca payloads may omit TP/PP. Treat same GPU+instance+market as
        # tried when topology is absent so we do not immediately retry it.
        if not item.get("tp") and not item.get("pp"):
            if (
                item.get("gpu_type") == row.get("gpu_type")
                and item.get("instance_type") == row.get("instance_type")
                and (item.get("market") or "unknown") == (row.get("planned_market") or row.get("market") or "unknown")
            ):
                return True
    return False


def _same_topology_different_market(row: dict[str, Any], failed: list[dict[str, Any]]) -> bool:
    row_no_market = _cfg_key(row, include_market=False)
    row_market = row.get("planned_market") or row.get("market") or "unknown"
    for item in failed:
        if _cfg_key(item, include_market=False) == row_no_market:
            return (item.get("market") or "unknown") != row_market
    return False


def _source_for_row(row: dict[str, Any], failed: list[dict[str, Any]]) -> str:
    if _same_topology_different_market(row, failed):
        return "market_alternate"
    failed_gpus = {item.get("gpu_type") for item in failed if item.get("gpu_type")}
    if row.get("gpu_type") and row.get("gpu_type") not in failed_gpus:
        return "gpu_family_alternate"
    return "topology_or_instance_alternate"


def _estimate_num_instances(row: dict[str, Any], rm: Optional[ResourceMap]) -> int:
    tp = int(row.get("tp") or 1)
    pp = int(row.get("pp") or 1)
    dp = int(row.get("dp") or 1)
    num_gpus = tp * pp * dp
    resource = rm.get_resource(str(row.get("gpu_type"))) if rm is not None else None
    if resource is None:
        return max(1, -(-num_gpus // 8))
    return max(1, -(-num_gpus // max(1, resource.gpus_per_instance)))


def _physics_for_row(req: JobRequest, rm: Optional[ResourceMap], row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    gpu_type = str(row.get("gpu_type") or "unknown")
    resource = rm.get_resource(gpu_type) if rm is not None else None
    tp = int(row.get("tp") or 1)
    pp = int(row.get("pp") or 1)
    dp = int(row.get("dp") or 1)
    num_gpus = tp * pp * dp
    hard = {
        "gpu_type": gpu_type,
        "instance_type": row.get("instance_type"),
        "capacity_ok": resource is None or num_gpus <= resource.available_gpus,
        "runtime_supported": resource is not None,
    }
    physics: dict[str, Any] = {}
    try:
        mf = get_model_features(req.model_name, dtype=req.quantization or "fp16")
        if resource is not None:
            feats = compute_config_features(
                mf,
                gpu_type=gpu_type,
                tp=tp,
                pp=pp,
                dp=dp,
                input_len=req.avg_input_tokens,
                output_len=req.avg_output_tokens,
                gpus_per_node=resource.gpus_per_instance,
                price_per_gpu_hour=resource.cost_per_gpu_hour_usd,
                gpu_memory_gb_override=resource.gpu_memory_gb,
            )
            vram_headroom_gb = float(feats.get("vram_headroom_gb", 0.0) or 0.0)
            hard.update(
                {
                    "vram_fit": vram_headroom_gb >= 8.0,
                    "vram_headroom_gb": round(vram_headroom_gb, 2),
                    "tp_heads_valid": mf.num_attention_heads % tp == 0,
                    "pp_layers_valid": mf.num_layers % pp == 0,
                    "crosses_node_boundary": bool(feats.get("crosses_node_boundary", 0)),
                }
            )
            physics = {
                "bandwidth_per_param": round(float(feats.get("bandwidth_per_param", 0.0) or 0.0), 3),
                "flops_per_param": round(float(feats.get("flops_per_param", 0.0) or 0.0), 3),
                "roofline_decode_tps": round(float(feats.get("roofline_decode_tps", 0.0) or 0.0), 1),
            }
    except Exception as exc:
        hard["physics_error"] = str(exc)
    return hard, physics


def _section_keys_for(action_id: str) -> list[str]:
    return [
        f"physics:{action_id}",
        f"perfdb_exact:{action_id}",
        f"memory_success:{action_id}",
        f"memory_failure:{action_id}",
        f"quota:{action_id}",
        f"recent_failures:{action_id}",
        f"failure:{action_id}",
        f"executor_payload:{action_id}",
        f"row:{action_id}",
    ]


def _detail_sections_for(
    *,
    agent: Any,
    memory: AgenticMemory,
    req: JobRequest,
    rm: Optional[ResourceMap],
    row: dict[str, Any],
    failed_entries: list[dict[str, Any]],
    action_id: str,
) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    gpu_type = str(row.get("gpu_type") or "unknown")
    market = row.get("planned_market") or row.get("market") or req.preferred_market or "on_demand"
    resource = rm.get_resource(gpu_type) if rm is not None else None
    region = row.get("region") or (resource.region if resource else None)
    sections[f"physics:{action_id}"] = {
        "gpu_type": gpu_type,
        "tp": int(row.get("tp") or 1),
        "pp": int(row.get("pp") or 1),
        "dp": int(row.get("dp") or 1),
        "physics": row.get("physics", {}),
        "hard_feasibility": row.get("hard_feasibility", {}),
    }

    perfdb_exact: list[dict[str, Any]] = []
    perfdb = getattr(agent, "perfdb", None)
    if perfdb is not None:
        try:
            perfdb_exact = perfdb.query(
                model_name=req.model_name,
                gpu_type=gpu_type,
                tp=int(row.get("tp") or 1),
                pp=int(row.get("pp") or 1),
                limit=10,
            ) or []
        except Exception as exc:
            perfdb_exact = [{"error": str(exc)}]
    sections[f"perfdb_exact:{action_id}"] = perfdb_exact

    try:
        memory_success = memory.query_outcomes(model_name=req.model_name, status="succeeded", limit=10) or []
    except Exception as exc:
        memory_success = [{"error": str(exc)}]
    try:
        memory_failure = memory.query_outcomes(model_name=req.model_name, status="failed", limit=10) or []
    except Exception as exc:
        memory_failure = [{"error": str(exc)}]
    sections[f"memory_success:{action_id}"] = memory_success
    sections[f"memory_failure:{action_id}"] = memory_failure

    quota_section: dict[str, Any] = {
        "gpu_type": gpu_type,
        "region": region,
        "market": market,
    }
    if resource is not None:
        quota_section.update(
            {
                "available_gpus": resource.available_gpus,
                "total_gpus": resource.total_gpus,
                "allocated_gpus": resource.allocated_gpus,
                "instance_type": resource.instance_type,
                "interconnect": resource.interconnect,
            }
        )
    try:
        quota_section["failure_summary"] = memory.get_failure_summary(
            gpu_type,
            region=region,
            market=market,
        )
    except Exception as exc:
        quota_section["failure_summary_error"] = str(exc)
    sections[f"quota:{action_id}"] = quota_section
    sections[f"recent_failures:{action_id}"] = quota_section.get("failure_summary", {})

    related_failures = [
        item for item in failed_entries if item.get("gpu_type") == gpu_type
    ]
    sections[f"failure:{action_id}"] = {
        "related_failed_attempts": related_failures,
        "all_failed_attempts": failed_entries,
    }
    sections[f"executor_payload:{action_id}"] = {
        "tool": "return_launch_recovery_plan",
        "gpu_type": gpu_type,
        "instance_type": row.get("instance_type"),
        "tp": int(row.get("tp") or 1),
        "pp": int(row.get("pp") or 1),
        "dp": int(row.get("dp") or 1),
        "market": market,
        "region": region or (rm.region if rm is not None else "unknown"),
        "predicted_tps": row.get("predicted_tps"),
        "predicted_cost_per_hour": row.get("cost_per_hour"),
        "predicted_total_cost": row.get("total_cost"),
        "predicted_runtime_hours": row.get("eta_h"),
        "source": row.get("source"),
        "prediction_source": row.get("prediction_source"),
    }
    sections[f"row:{action_id}"] = {"row": row}
    return sections


def _candidate_rows_from_cost_table(
    *,
    agent: Any,
    req: JobRequest,
    rm: ResourceMap,
    failed_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not hasattr(agent, "_build_cost_table"):
        return []
    try:
        _, rows = agent._build_cost_table(req, rm)
    except Exception as exc:
        logger.warning("p1_cost_table_failed", job_id=req.job_id, error=str(exc))
        return []

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in rows:
        row = dict(raw)
        row["planned_market"] = row.get("planned_market") or req.preferred_market or "on_demand"
        if _matches_failed_same_scope(row, failed_entries):
            continue
        source = _source_for_row(row, failed_entries)
        row["prediction_source"] = row.get("source", "unknown")
        row["source"] = source
        key = _cfg_key({**row, "market": row["planned_market"]}, include_market=True)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(row)
        if len(candidates) >= MAX_MENU_OPTIONS - 1:
            break
    return candidates


def _abort_option(rank: int, reason: str) -> ActionOption:
    action_id = _action_id(rank - 1)
    return ActionOption(
        action_id=action_id,
        action_type="abort_launch",
        summary=f"Abort launch recovery: {reason}",
        rank=rank,
        valid=True,
        evidence={"source": "policy_guard", "reason": reason},
        executor_payload_ref=f"executor_payload:{action_id}",
        detail_refs=[f"failure:{action_id}", f"executor_payload:{action_id}"],
    )


async def build_p1_packet(
    agent: Any,
    req: Any,
    memory: AgenticMemory,
    *,
    ledger: Any = None,
    resource_map: Optional[ResourceMap] = None,
    retry_budget: Optional[int] = None,
) -> TransitionPacket:
    decision_id = getattr(req, "decision_id", None)
    decision_chain = _decision_chain(memory, decision_id)
    decision = decision_chain[0] if decision_chain else None
    budget_limit = _retry_budget_limit() if retry_budget is None else retry_budget
    budget_used = _retry_budget_used(decision_chain)
    budget_remaining = max(0, budget_limit - budget_used)
    failed_entries = _failed_entries(req, decision)
    failure_categories = sorted({item["failure_category"] for item in failed_entries})
    force_on_demand = any(
        item.get("market") == "spot"
        and item.get("failure_category") in {"spot_preemption", "no_capacity", "quota"}
        for item in failed_entries
    )

    detail_sections: dict[str, Any] = {}
    options: list[ActionOption] = []
    job_context: dict[str, Any] = {"job_id": getattr(req, "job_id", "unknown")}
    rm = await _resource_map_for(agent, ledger=ledger, resource_map=resource_map)
    reconstructed: Optional[JobRequest] = None

    if decision is not None:
        reconstructed = _reconstruct_job_request(
            decision=decision,
            job_id=getattr(req, "job_id", decision.get("job_id", "unknown")),
            force_on_demand=force_on_demand,
        )
        job_context = {
            "job_id": reconstructed.job_id,
            "model_name": reconstructed.model_name,
            "objective": reconstructed.objective.value,
            "avg_input_tokens": reconstructed.avg_input_tokens,
            "avg_output_tokens": reconstructed.avg_output_tokens,
            "num_requests": reconstructed.num_requests,
            "total_tokens": reconstructed.total_tokens,
            "slo_deadline_hours": reconstructed.slo_deadline_hours,
            "required_tps": reconstructed.required_tps,
            "preferred_market": reconstructed.preferred_market,
            "cost_roofline_usd": reconstructed.cost_roofline_usd,
            "quantization": reconstructed.quantization,
            "parent_decision_id": decision_id,
        }

    if decision is not None and reconstructed is not None and rm is not None and budget_remaining > 0:
        rows = _candidate_rows_from_cost_table(
            agent=agent,
            req=reconstructed,
            rm=rm,
            failed_entries=failed_entries,
        )
        for idx, row in enumerate(rows[: MAX_MENU_OPTIONS - 1]):
            action_id = _action_id(idx)
            hard, physics = _physics_for_row(reconstructed, rm, row)
            valid = bool(row.get("meets_slo", True)) and all(
                hard.get(key, True)
                for key in ("capacity_ok", "runtime_supported", "vram_fit", "tp_heads_valid", "pp_layers_valid")
            )
            row["hard_feasibility"] = hard
            row["physics"] = physics
            sections = _detail_sections_for(
                agent=agent,
                memory=memory,
                req=reconstructed,
                rm=rm,
                row=row,
                failed_entries=failed_entries,
                action_id=action_id,
            )
            detail_sections.update(sections)
            source = row.get("source", "cost_table")
            summary = (
                f"Recover launch with {row.get('gpu_type')} TP={row.get('tp')} PP={row.get('pp')} "
                f"DP={row.get('dp', 1)} {row.get('planned_market', 'on_demand')} | "
                f"TPS={float(row.get('predicted_tps') or 0.0):.0f} | "
                f"total=${float(row.get('total_cost') or 0.0):.2f} | source={source}"
            )
            options.append(
                ActionOption(
                    action_id=action_id,
                    action_type="retry_launch",
                    summary=summary,
                    rank=idx + 1,
                    valid=valid,
                    hard_feasibility=hard,
                    performance={
                        "predicted_tps": float(row.get("predicted_tps") or 0.0),
                        "required_tps": reconstructed.required_tps,
                        "meets_slo": bool(row.get("meets_slo", True)),
                        "prediction_source": source,
                    },
                    physics=physics,
                    evidence={
                        "source": source,
                        "failure_categories": failure_categories,
                    },
                    availability=sections[f"quota:{action_id}"].get("failure_summary", {}),
                    cost={
                        "cost_per_hour": row.get("cost_per_hour"),
                        "projected_total_cost_usd": row.get("total_cost"),
                        "under_roofline": row.get("under_cost_roofline"),
                        "cost_overage_usd": row.get("cost_overage_usd"),
                    },
                    risk={
                        "fresh_failures_same_gpu": len(sections[f"failure:{action_id}"]["related_failed_attempts"]),
                    },
                    executor_payload_ref=f"executor_payload:{action_id}",
                    detail_refs=_section_keys_for(action_id),
                )
            )

    if decision is None:
        abort_reason = "missing original decision; cannot safely reconstruct launch request"
    elif budget_remaining <= 0:
        abort_reason = "retry budget exhausted"
    elif not any(o.valid and o.action_type == "retry_launch" for o in options):
        abort_reason = "no safe recovery candidate"
    else:
        abort_reason = "operator chooses not to retry"

    abort_rank = len(options) + 1
    abort = _abort_option(abort_rank, abort_reason)
    detail_sections[f"failure:{abort.action_id}"] = {
        "failed_attempts": failed_entries,
        "failure_categories": failure_categories,
        "retry_budget_remaining": budget_remaining,
    }
    detail_sections[f"executor_payload:{abort.action_id}"] = {
        "tool": "return_abort_launch_recovery",
        "reason": abort_reason,
    }
    options.append(abort)

    return TransitionPacket(
        packet_id=f"p1-{getattr(req, 'job_id', 'unknown')}",
        job_id=getattr(req, "job_id", "unknown"),
        state=HarnessState.LAUNCH_FAILED,
        transition_type=TransitionType.LAUNCH_RECOVERY,
        job_context=job_context,
        failure_context={
            "configs_tried": failed_entries,
            "failure_categories": failure_categories,
            "total_time_seconds": getattr(req, "total_time_seconds", 0.0),
        },
        policy_context={
            "retry_budget_limit": budget_limit,
            "retry_budget_used": budget_used,
            "retry_budget_remaining_before_choice": budget_remaining,
        },
        evidence_summary={
            "candidate_count": len(options),
            "valid_recovery_count": sum(1 for option in options if option.valid and option.action_type == "retry_launch"),
            "abort_available": True,
            "resource_map_available": rm is not None,
        },
        action_options=options,
        detail_sections=detail_sections,
        guards={
            "max_menu_options": MAX_MENU_OPTIONS,
            "retry_budget_enforced": True,
            "no_direct_launch_tool": True,
        },
    )


def render_p1_prompt(packet: TransitionPacket) -> str:
    lines = [
        "P1 LAUNCH RECOVERY",
        "Choose one valid action_id from the recovery menu.",
        "Prefer a safe retry/switch when retry budget remains and the candidate avoids the failed scope.",
        "Choose abort only when no safe candidate exists or retry budget is exhausted.",
        "Do not invent executable actions; Koi will only return the validated recovery plan to Orca.",
        "",
        "JOB CONTEXT:",
        json.dumps(packet.job_context, indent=2, sort_keys=True),
        "",
        "FAILURE CONTEXT:",
        json.dumps(packet.failure_context, indent=2, sort_keys=True),
        "",
        "POLICY CONTEXT:",
        json.dumps(packet.policy_context, indent=2, sort_keys=True),
        "",
        "ACTION MENU:",
    ]
    for option in packet.action_options:
        lines.append(f"{option.rank}. action_id={option.action_id} type={option.action_type} valid={option.valid}")
        lines.append(f"   {option.summary}")
        if option.hard_feasibility:
            lines.append(f"   feasibility={json.dumps(option.hard_feasibility, sort_keys=True)}")
        if option.cost:
            lines.append(f"   cost={json.dumps(option.cost, sort_keys=True)}")
        if option.availability:
            lines.append(f"   availability={json.dumps(option.availability, sort_keys=True)}")
        if option.risk:
            lines.append(f"   risk={json.dumps(option.risk, sort_keys=True)}")
    lines.extend(["", "Return your final answer as the typed ChosenAction schema."])
    return "\n".join(lines)


def _packet_tools(memory: AgenticMemory, packet: TransitionPacket) -> dict[str, Any]:
    async def list_detail_sections(action_id: str) -> str:
        option = packet.get_action(action_id)
        if option is None:
            return f"unknown action_id={action_id!r}"
        return json.dumps(option.detail_refs, indent=2)

    async def read_option_detail(action_id: str, section: str = "all") -> str:
        option = packet.get_action(action_id)
        if option is None:
            return f"unknown action_id={action_id!r}"
        if section == "all":
            return json.dumps(
                {ref: packet.detail_sections.get(ref) for ref in option.detail_refs},
                indent=2,
                default=str,
            )
        ref = f"{section}:{action_id}"
        if ref not in option.detail_refs and section not in _KNOWN_SECTIONS:
            return (
                f"unknown section={section!r} for action_id={action_id!r}; "
                f"available={option.detail_refs}"
            )
        return json.dumps({"section": ref, "data": packet.detail_sections.get(ref)}, indent=2, default=str)

    async def compare_options(action_ids: list[str], lens: str = "summary") -> str:
        selected = []
        for action_id in action_ids:
            option = packet.get_action(action_id)
            if option is None:
                continue
            selected.append(
                {
                    "action_id": option.action_id,
                    "rank": option.rank,
                    "type": option.action_type,
                    "valid": option.valid,
                    "summary": option.summary,
                    "performance": option.performance,
                    "cost": option.cost,
                    "availability": option.availability,
                    "risk": option.risk,
                    "physics": option.physics if lens == "physics" else {},
                }
            )
        return json.dumps(selected, indent=2, default=str)

    async def get_failure_summary(gpu_type: str, region: Optional[str] = None, market: Optional[str] = None) -> str:
        try:
            return json.dumps(
                memory.get_failure_summary(gpu_type, region=region, market=market),
                indent=2,
                default=str,
            )
        except Exception as exc:
            return f"get_failure_summary failed: {exc}"

    return {
        "list_detail_sections": list_detail_sections,
        "read_option_detail": read_option_detail,
        "compare_options": compare_options,
        "get_failure_summary": get_failure_summary,
    }


def _source_to_prediction_source(source: str) -> str:
    if source == "VERIFIED":
        return "memory_verified"
    if source == "PerfDB":
        return "perfdb_exact"
    return source or "analytical"


def _config_from_payload(payload: dict[str, Any], rm_region: str = "unknown") -> PlacementConfig:
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
        region=str(payload.get("region") or rm_region or "unknown"),
        engine_config=EngineConfig(tensor_parallel_size=tp, pipeline_parallel_size=pp),
        market=str(payload.get("market") or "on_demand"),
    )


def _alternative_payloads(packet: TransitionPacket, selected_action_id: str) -> list[dict[str, Any]]:
    alternatives: list[dict[str, Any]] = []
    for option in packet.valid_actions():
        if option.action_id == selected_action_id or option.action_type != "retry_launch":
            continue
        payload = packet.detail_sections.get(option.executor_payload_ref or "", {})
        alternatives.append(
            {
                "gpu_type": payload.get("gpu_type"),
                "instance_type": payload.get("instance_type"),
                "tp": payload.get("tp"),
                "pp": payload.get("pp"),
                "dp": payload.get("dp", 1),
                "region": payload.get("region"),
                "market": payload.get("market"),
                "predicted_tps": payload.get("predicted_tps"),
                "source": payload.get("source"),
            }
        )
        if len(alternatives) >= 3:
            break
    return alternatives


def _abort_plan(packet: TransitionPacket, validated: Optional[ValidatedAction] = None, reason: Optional[str] = None) -> dict[str, Any]:
    option = validated.option if validated is not None else next(
        (candidate for candidate in packet.valid_actions() if candidate.action_type == "abort_launch"),
        None,
    )
    rationale = reason or (validated.choice.rationale if validated is not None else None) or (option.summary if option else "launch recovery aborted")
    return {
        "action": "abort",
        "decision_id": None,
        "parent_decision_id": packet.job_context.get("parent_decision_id"),
        "reasoning": rationale,
        "confidence": validated.choice.confidence if validated is not None else 1.0,
        "retry_budget_remaining": packet.policy_context.get("retry_budget_remaining_before_choice", 0),
    }


def _recovery_plan_from_action(
    *,
    packet: TransitionPacket,
    memory: AgenticMemory,
    ledger: Any,
    validated: ValidatedAction,
) -> dict[str, Any]:
    option = validated.option
    if option.action_type == "abort_launch":
        return _abort_plan(packet, validated)

    payload = packet.detail_sections.get(option.executor_payload_ref or "", {})
    config = _config_from_payload(payload, packet.job_context.get("region", "unknown"))
    parent_decision_id = packet.job_context.get("parent_decision_id")
    predicted_tps = float(payload.get("predicted_tps") or 0.0)
    predicted_cost = float(payload.get("predicted_cost_per_hour") or 0.0)
    decision_id = memory.record_decision(
        job_id=packet.job_id,
        model_name=str(packet.job_context.get("model_name") or "unknown"),
        instance_type=config.instance_type,
        gpu_type=config.gpu_type,
        tp=config.tp,
        pp=config.pp,
        dp=config.dp,
        num_gpus=config.num_gpus,
        predicted_tps=predicted_tps,
        predicted_cost_per_hour=predicted_cost,
        predicted_total_cost=payload.get("predicted_total_cost"),
        predicted_runtime_hours=payload.get("predicted_runtime_hours"),
        prediction_confidence=validated.choice.confidence,
        prediction_source=_source_to_prediction_source(str(payload.get("prediction_source") or payload.get("source") or "")),
        slo_deadline_hours=float(packet.job_context.get("slo_deadline_hours") or 0.0),
        objective=str(packet.job_context.get("objective") or "cheapest"),
        avg_input_tokens=int(packet.job_context.get("avg_input_tokens") or 0),
        avg_output_tokens=int(packet.job_context.get("avg_output_tokens") or 0),
        num_requests=packet.job_context.get("num_requests"),
        quantization=packet.job_context.get("quantization"),
        triggered_by="launch_recovery",
        parent_decision_id=parent_decision_id,
        cost_roofline_usd=packet.job_context.get("cost_roofline_usd"),
        market=config.market,
    )
    if ledger is not None:
        ledger.reserve(
            decision_id=decision_id,
            gpu_type=config.gpu_type,
            num_gpus=config.num_gpus,
            region=config.region,
            instance_type=config.instance_type,
        )
    remaining_before = int(packet.policy_context.get("retry_budget_remaining_before_choice") or 0)
    rationale = validated.choice.rationale or option.summary
    if validated.fallback_used:
        rationale = f"[HARNESS FALLBACK] {rationale}"
    return {
        "action": "retry_launch",
        "decision_id": decision_id,
        "parent_decision_id": parent_decision_id,
        "action_id": option.action_id,
        "config": config.model_dump(mode="json"),
        "alternatives": _alternative_payloads(packet, option.action_id),
        "reasoning": rationale,
        "confidence": validated.choice.confidence,
        "retry_budget_remaining": max(0, remaining_before - 1),
    }


async def run_launch_recovery(
    agent: Any,
    req: Any,
    memory: AgenticMemory,
    *,
    ledger: Any = None,
    resource_map: Optional[ResourceMap] = None,
    retry_budget: Optional[int] = None,
) -> dict[str, Any]:
    t0 = time.time()
    packet = await build_p1_packet(
        agent,
        req,
        memory,
        ledger=ledger,
        resource_map=resource_map,
        retry_budget=retry_budget,
    )
    retry_actions = [
        option for option in packet.valid_actions() if option.action_type == "retry_launch"
    ]
    if not retry_actions:
        return _abort_plan(packet, reason="no valid launch recovery action")

    prompt = render_p1_prompt(packet)
    reasoner = HarnessReasoner(
        model=agent._model,
        tools=_packet_tools(memory, packet),
    )
    try:
        tool_calls, choice = await reasoner.choose(
            prompt,
            job_id=packet.job_id,
            label="p1",
            max_iterations=P1_MAX_ITERATIONS,
            timeout=P1_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("p1_timeout", job_id=packet.job_id, timeout=P1_TIMEOUT)
        choice = ChosenAction(
            action_id=retry_actions[0].action_id,
            confidence=0.3,
            rationale="P1 timed out; deterministic fallback selected top recovery candidate.",
        )
        tool_calls = 0

    try:
        validated = validate_choice(packet, choice)
    except NoValidActionError:
        return _abort_plan(packet, reason="no valid launch recovery action")

    plan = _recovery_plan_from_action(
        packet=packet,
        memory=memory,
        ledger=ledger,
        validated=validated,
    )
    emit_event(
        "harness.p1.decided",
        job_id=packet.job_id,
        action=plan.get("action"),
        action_id=validated.option.action_id,
        fallback_used=validated.fallback_used,
        tool_calls=tool_calls,
        elapsed_s=round(time.time() - t0, 2),
    )
    return plan
