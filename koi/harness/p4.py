"""P4 replica recovery harness.

Phase 5 closes the failure-handling loop:

  /job/replica-failed → P5c diagnosis → P4 recovery decision

P4 builds a bounded recovery menu (replace, switch market, migrate GPU family,
hold, abort) using shared Phase 3.5 utilities and Phase 4.5 recent-failure
ranking. It executes through existing scale primitives — no new Koi launch
endpoint.

The state stays REPLICA_RECOVERY (a distinct decision context) per the FSM
design: same execution machinery as a scale-up, but a different decision
context with its own evidence (P5c diagnosis), action vocabulary, retry
budget guard, and telemetry bucket.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Optional

from pydantic import BaseModel, Field

from koi.event_tap import emit_event
from koi.harness.decision_utils import (
    alternative_payloads,
    placement_config_from_payload,
    reconstruct_job_request,
    source_to_prediction_source,
)
from koi.harness.failures import (
    config_key,
    decision_chain as load_decision_chain,
    matches_failed_same_scope,
)
from koi.harness.feasibility import physics_for_row
from koi.harness.ids import action_id as make_action_id
from koi.harness.packet_tools import build_packet_read_tools
from koi.harness.recent_failures import annotate_and_rank_rows, recent_failure_penalty
from koi.harness.reasoner import HarnessReasoner
from koi.harness.resources import resource_map_for
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
from koi.schemas import JobRequest, PlacementConfig, ResourceMap
from koi.tools.memory import AgenticMemory

logger = get_logger("koi.harness.p4")

P4_TIMEOUT = 90.0
P4_MAX_ITERATIONS = 3
MAX_MENU_OPTIONS = 6
DEFAULT_REPLICA_RETRY_BUDGET = 3

_KNOWN_SECTIONS = (
    "physics",
    "perfdb_exact",
    "memory_success",
    "memory_failure",
    "quota",
    "recent_failures",
    "diagnosis",
    "fleet",
    "executor_payload",
    "row",
)


# ---------------------------------------------------------------------------
# Diagnosis helpers
# ---------------------------------------------------------------------------


class P4DiagnosisInput(BaseModel):
    """Subset of P5cDiagnosis P4 needs. Accepts dicts or P5cDiagnosis objects."""

    diagnosis_code: str = "unknown"
    bottleneck: str = "unknown"
    next_fix: str = "operator_review"
    failure_scope: str = "unknown"
    rationale: str = ""
    cooloff_minutes: int = 0


def _coerce_diagnosis(diagnosis: Any) -> P4DiagnosisInput:
    if diagnosis is None:
        return P4DiagnosisInput()
    if isinstance(diagnosis, P4DiagnosisInput):
        return diagnosis
    if hasattr(diagnosis, "model_dump"):
        return P4DiagnosisInput(**diagnosis.model_dump())
    if isinstance(diagnosis, dict):
        return P4DiagnosisInput(**{k: v for k, v in diagnosis.items() if k in P4DiagnosisInput.model_fields})
    return P4DiagnosisInput()


# ---------------------------------------------------------------------------
# Retry budget
# ---------------------------------------------------------------------------


def replica_retry_budget_limit() -> int:
    raw = os.environ.get("KOI_HARNESS_P4_RETRY_BUDGET", str(DEFAULT_REPLICA_RETRY_BUDGET))
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_REPLICA_RETRY_BUDGET


def replica_retry_budget_used(memory: AgenticMemory, group_id: Optional[str]) -> int:
    """Count past P4 child decisions for this group (recovery launches)."""
    if not group_id or memory is None:
        return 0
    try:
        decisions = memory.query_decisions(job_id=group_id, limit=50)
    except Exception:
        return 0
    return sum(
        1
        for d in decisions
        if d.get("triggered_by") == "replica_recovery"
    )


# ---------------------------------------------------------------------------
# Fleet snapshot
# ---------------------------------------------------------------------------


def _live_fleet_snapshot(agent: Any, group_id: Optional[str]) -> dict[str, Any]:
    monitor = getattr(agent, "monitor", None)
    if monitor is None or not group_id:
        return {"live_replicas": 0, "live_aggregate_tps": 0.0, "chains": []}
    try:
        chains = monitor.get_group_chains(group_id) or {}
    except Exception:
        chains = {}
    chain_rows: list[dict[str, Any]] = []
    aggregate_tps = 0.0
    live_count = 0
    for rid, chain in chains.items():
        status = getattr(getattr(chain, "status", None), "value", None) or str(
            getattr(chain, "status", "unknown")
        )
        smoothed = float(getattr(chain, "smoothed_tps", 0.0) or 0.0)
        if status not in {"failed", "completed", "killed", "dead"}:
            live_count += 1
            aggregate_tps += smoothed
        chain_rows.append(
            {
                "replica_id": rid,
                "status": status,
                "smoothed_tps": round(smoothed, 1),
                "predicted_tps": float(getattr(chain, "predicted_tps", 0.0) or 0.0),
                "gpu_type": getattr(getattr(chain, "config", None), "gpu_type", None),
                "tp": getattr(getattr(chain, "config", None), "tp", None),
                "pp": getattr(getattr(chain, "config", None), "pp", None),
            }
        )
    return {
        "live_replicas": live_count,
        "live_aggregate_tps": round(aggregate_tps, 1),
        "chains": chain_rows,
    }


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def _failed_entry(tracker: Any, region: str, market: str) -> dict[str, Any]:
    config = tracker.config
    return {
        "gpu_type": config.gpu_type,
        "instance_type": config.instance_type,
        "tp": int(config.tp),
        "pp": int(config.pp),
        "dp": int(config.dp),
        "region": region,
        "market": market,
    }


def _classify_source(
    row: dict[str, Any],
    failed_entry: dict[str, Any],
    diagnosis_code: str,
) -> str:
    same_gpu = row.get("gpu_type") == failed_entry.get("gpu_type")
    same_market = (
        (row.get("planned_market") or row.get("market"))
        == (failed_entry.get("market") or "")
    )
    same_topo = (
        int(row.get("tp") or 1) == int(failed_entry.get("tp") or 1)
        and int(row.get("pp") or 1) == int(failed_entry.get("pp") or 1)
    )
    if same_gpu and same_topo and not same_market:
        return "replace_market"
    if same_gpu and same_topo and same_market:
        return "replace_same"
    if not same_gpu:
        return "migrate_gpu_family"
    return "replace_alt_topology"


def _candidate_rows(
    *,
    agent: Any,
    req: JobRequest,
    rm: ResourceMap,
    failed_entry: dict[str, Any],
    diagnosis_code: str,
    force_on_demand: bool,
) -> list[dict[str, Any]]:
    if not hasattr(agent, "_build_cost_table"):
        return []
    try:
        _, rows = agent._build_cost_table(req, rm)
    except Exception as exc:
        logger.warning("p4_cost_table_failed", job_id=req.job_id, error=str(exc))
        return []

    filtered: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for raw in rows:
        row = dict(raw)
        if force_on_demand:
            row["planned_market"] = "on_demand"
        else:
            row["planned_market"] = (
                row.get("planned_market") or req.preferred_market or "on_demand"
            )
        # Always allow alternatives even if the failed scope appears — the
        # ranker will downrank, not exclude. But avoid duplicate rows.
        key = config_key(
            {**row, "market": row["planned_market"]}, include_market=True
        )
        if key in seen:
            continue
        seen.add(key)
        row["prediction_source"] = row.get("source", "unknown")
        row["source"] = _classify_source(row, failed_entry, diagnosis_code)
        filtered.append(row)
        if len(filtered) >= MAX_MENU_OPTIONS - 1:
            break
    return filtered


# ---------------------------------------------------------------------------
# Detail sections
# ---------------------------------------------------------------------------


def _section_keys_for(action_id: str) -> list[str]:
    return [
        f"physics:{action_id}",
        f"perfdb_exact:{action_id}",
        f"memory_success:{action_id}",
        f"memory_failure:{action_id}",
        f"quota:{action_id}",
        f"recent_failures:{action_id}",
        f"diagnosis:{action_id}",
        f"fleet:{action_id}",
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
    diagnosis: P4DiagnosisInput,
    fleet: dict[str, Any],
    failed_entry: dict[str, Any],
    action_id: str,
) -> dict[str, Any]:
    sections: dict[str, Any] = {}
    gpu_type = str(row.get("gpu_type") or "unknown")
    market = (
        row.get("planned_market") or row.get("market") or req.preferred_market or "on_demand"
    )
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

    perfdb = getattr(agent, "perfdb", None)
    perfdb_exact: list[dict[str, Any]] = []
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
        memory_success = memory.query_outcomes(
            model_name=req.model_name, status="succeeded", limit=10
        ) or []
    except Exception as exc:
        memory_success = [{"error": str(exc)}]
    try:
        memory_failure = memory.query_outcomes(
            model_name=req.model_name, status="failed", limit=10
        ) or []
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
            gpu_type, region=region, market=market
        )
    except Exception as exc:
        quota_section["failure_summary_error"] = str(exc)
    sections[f"quota:{action_id}"] = quota_section
    sections[f"recent_failures:{action_id}"] = {
        "failure_summary": quota_section.get("failure_summary", {}),
        "recent_failure": row.get("recent_failure"),
    }

    sections[f"diagnosis:{action_id}"] = {
        "diagnosis_code": diagnosis.diagnosis_code,
        "bottleneck": diagnosis.bottleneck,
        "next_fix": diagnosis.next_fix,
        "failure_scope": diagnosis.failure_scope,
        "rationale": diagnosis.rationale,
        "failed_entry": failed_entry,
    }
    sections[f"fleet:{action_id}"] = fleet

    sections[f"executor_payload:{action_id}"] = {
        "tool": "scale_chain_tool",
        "gpu_type": gpu_type,
        "instance_type": row.get("instance_type"),
        "tp": int(row.get("tp") or 1),
        "pp": int(row.get("pp") or 1),
        "dp": int(row.get("dp") or 1),
        "count": 1,
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


# ---------------------------------------------------------------------------
# Decision-aligned ranking + budget gate
# ---------------------------------------------------------------------------


_DIAGNOSIS_PREFERS = {
    "spot_preemption": ("replace_market", "migrate_gpu_family", "replace_alt_topology", "replace_same"),
    "no_capacity": ("replace_market", "migrate_gpu_family", "replace_alt_topology", "replace_same"),
    "quota_exhausted": ("migrate_gpu_family", "replace_market", "replace_alt_topology", "replace_same"),
    "oom": ("migrate_gpu_family", "replace_alt_topology", "replace_market", "replace_same"),
    "heartbeat_timeout": ("replace_same", "replace_market", "replace_alt_topology", "migrate_gpu_family"),
}


def _diagnosis_preference_index(diagnosis_code: str, source: str) -> int:
    pref = _DIAGNOSIS_PREFERS.get(diagnosis_code, ())
    try:
        return pref.index(source)
    except ValueError:
        return len(pref) + 1


def _rank_rows(
    rows: list[dict[str, Any]],
    diagnosis_code: str,
) -> list[dict[str, Any]]:
    """Diagnosis-aligned ordering with recent-failure penalty respected."""

    def key(row: dict[str, Any]) -> tuple:
        return (
            not bool(row.get("meets_slo", True)),
            recent_failure_penalty(row),
            _diagnosis_preference_index(diagnosis_code, str(row.get("source") or "")),
            row.get("under_cost_roofline") is False,
            float(row.get("total_cost") or 0.0),
        )

    return sorted(rows, key=key)


# ---------------------------------------------------------------------------
# Packet builder
# ---------------------------------------------------------------------------


def _abort_option(rank: int, reason: str) -> ActionOption:
    action_id = make_action_id(rank - 1)
    return ActionOption(
        action_id=action_id,
        action_type="abort_recovery",
        summary=f"Abort replica recovery: {reason}",
        rank=rank,
        valid=True,
        evidence={"source": "policy_guard", "reason": reason},
        executor_payload_ref=f"executor_payload:{action_id}",
        detail_refs=[f"diagnosis:{action_id}", f"executor_payload:{action_id}"],
    )


def _hold_option(rank: int, fleet: dict[str, Any], required_tps: Optional[float]) -> ActionOption:
    action_id = make_action_id(rank - 1)
    aggregate = float(fleet.get("live_aggregate_tps") or 0.0)
    safe = required_tps is not None and aggregate >= 1.10 * required_tps
    return ActionOption(
        action_id=action_id,
        action_type="hold_noop",
        summary=(
            f"Hold without replacement; live fleet TPS={aggregate:.0f}"
            + (f" >= 110% of required {required_tps:.0f}" if required_tps else "")
        ),
        rank=rank,
        valid=bool(safe),
        evidence={
            "source": "fleet_snapshot",
            "live_aggregate_tps": aggregate,
            "required_tps": required_tps,
        },
        executor_payload_ref=f"executor_payload:{action_id}",
        detail_refs=[f"fleet:{action_id}", f"executor_payload:{action_id}"],
    )


async def build_p4_packet(
    *,
    agent: Any,
    req: Any,
    tracker: Any,
    memory: AgenticMemory,
    diagnosis: Any,
    region: str,
    market: str,
    ledger: Any = None,
    resource_map: Optional[ResourceMap] = None,
    retry_budget: Optional[int] = None,
) -> TransitionPacket:
    job_id = getattr(req, "job_id", None) or getattr(tracker, "job_id", "unknown")
    group_id = getattr(req, "group_id", None) or getattr(tracker, "group_id", None)
    diag = _coerce_diagnosis(diagnosis)
    failed_entry = _failed_entry(tracker, region=region, market=market)

    chain = load_decision_chain(memory, getattr(tracker, "decision_id", None))
    parent_decision = chain[0] if chain else None
    budget_limit = (
        replica_retry_budget_limit() if retry_budget is None else int(retry_budget)
    )
    budget_used = replica_retry_budget_used(memory, group_id)
    budget_remaining = max(0, budget_limit - budget_used)

    force_on_demand = market == "spot" and diag.diagnosis_code in {
        "spot_preemption",
        "no_capacity",
        "quota_exhausted",
    }

    fleet = _live_fleet_snapshot(agent, group_id)

    rm = await resource_map_for(agent, ledger=ledger, resource_map=resource_map)

    detail_sections: dict[str, Any] = {}
    options: list[ActionOption] = []
    job_context: dict[str, Any] = {
        "job_id": job_id,
        "group_id": group_id,
        "parent_decision_id": getattr(tracker, "decision_id", None),
    }
    reconstructed: Optional[JobRequest] = None
    required_tps: Optional[float] = None

    if parent_decision is not None:
        reconstructed = reconstruct_job_request(
            decision=parent_decision,
            job_id=job_id,
            force_on_demand=force_on_demand,
        )
        job_context.update(
            {
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
            }
        )
        required_tps = reconstructed.required_tps

    if (
        parent_decision is not None
        and reconstructed is not None
        and rm is not None
        and budget_remaining > 0
    ):
        rows = _candidate_rows(
            agent=agent,
            req=reconstructed,
            rm=rm,
            failed_entry=failed_entry,
            diagnosis_code=diag.diagnosis_code,
            force_on_demand=force_on_demand,
        )
        rows = annotate_and_rank_rows(
            memory,
            rows,
            rm,
            default_market=reconstructed.preferred_market or "on_demand",
        )
        rows = _rank_rows(rows, diagnosis_code=diag.diagnosis_code)

        for idx, row in enumerate(rows[: MAX_MENU_OPTIONS - 2]):
            action_id = make_action_id(idx)
            physics_payload = physics_for_row(reconstructed, rm, row)
            hard = physics_payload["hard_feasibility"]
            physics = physics_payload["physics"]
            row["hard_feasibility"] = hard
            row["physics"] = physics
            valid = bool(row.get("meets_slo", True)) and all(
                hard.get(key, True)
                for key in (
                    "capacity_ok",
                    "runtime_supported",
                    "vram_fit",
                    "tp_heads_valid",
                    "pp_layers_valid",
                )
            )
            sections = _detail_sections_for(
                agent=agent,
                memory=memory,
                req=reconstructed,
                rm=rm,
                row=row,
                diagnosis=diag,
                fleet=fleet,
                failed_entry=failed_entry,
                action_id=action_id,
            )
            detail_sections.update(sections)
            source = str(row.get("source") or "replace_alt_topology")
            summary = (
                f"{source.replace('_', ' ').title()}: "
                f"{row.get('gpu_type')} TP={row.get('tp')} PP={row.get('pp')} "
                f"DP={row.get('dp', 1)} {row.get('planned_market', 'on_demand')} | "
                f"TPS={float(row.get('predicted_tps') or 0.0):.0f} | "
                f"total=${float(row.get('total_cost') or 0.0):.2f}"
            )
            risk: dict[str, Any] = {}
            if row.get("recent_failure"):
                risk["recent_failure"] = row["recent_failure"]
            options.append(
                ActionOption(
                    action_id=action_id,
                    action_type=source,
                    summary=summary,
                    rank=idx + 1,
                    valid=valid,
                    hard_feasibility=hard,
                    performance={
                        "predicted_tps": float(row.get("predicted_tps") or 0.0),
                        "required_tps": required_tps,
                        "meets_slo": bool(row.get("meets_slo", True)),
                        "prediction_source": source,
                    },
                    physics=physics,
                    evidence={
                        "source": source,
                        "diagnosis_code": diag.diagnosis_code,
                        "next_fix": diag.next_fix,
                        "recent_failure": row.get("recent_failure"),
                    },
                    availability=sections[f"quota:{action_id}"].get("failure_summary", {}),
                    cost={
                        "cost_per_hour": row.get("cost_per_hour"),
                        "projected_total_cost_usd": row.get("total_cost"),
                        "under_roofline": row.get("under_cost_roofline"),
                        "cost_overage_usd": row.get("cost_overage_usd"),
                    },
                    risk=risk,
                    executor_payload_ref=f"executor_payload:{action_id}",
                    detail_refs=_section_keys_for(action_id),
                )
            )

    # Hold/abort options always available.
    hold = _hold_option(len(options) + 1, fleet, required_tps)
    detail_sections[f"fleet:{hold.action_id}"] = fleet
    detail_sections[f"executor_payload:{hold.action_id}"] = {"tool": "noop"}
    options.append(hold)

    if parent_decision is None:
        abort_reason = "missing original decision; cannot safely build recovery menu"
    elif budget_remaining <= 0:
        abort_reason = "replica retry budget exhausted"
    elif not any(
        o.valid and o.action_type not in {"hold_noop", "abort_recovery"} for o in options
    ):
        abort_reason = "no safe replacement candidate"
    else:
        abort_reason = "operator chooses not to recover"

    abort = _abort_option(len(options) + 1, abort_reason)
    detail_sections[f"diagnosis:{abort.action_id}"] = {
        "diagnosis_code": diag.diagnosis_code,
        "rationale": diag.rationale,
        "retry_budget_remaining": budget_remaining,
    }
    detail_sections[f"executor_payload:{abort.action_id}"] = {
        "tool": "return_abort_recovery",
        "reason": abort_reason,
    }
    options.append(abort)

    return TransitionPacket(
        packet_id=f"p4-{job_id}",
        job_id=job_id,
        state=HarnessState.REPLICA_RECOVERY,
        transition_type=TransitionType.REPLICA_RECOVERY,
        job_context=job_context,
        runtime_context={"fleet": fleet},
        failure_context={
            "diagnosis_code": diag.diagnosis_code,
            "bottleneck": diag.bottleneck,
            "next_fix": diag.next_fix,
            "failure_scope": diag.failure_scope,
            "rationale": diag.rationale,
            "failed_entry": failed_entry,
        },
        policy_context={
            "retry_budget_limit": budget_limit,
            "retry_budget_used": budget_used,
            "retry_budget_remaining_before_choice": budget_remaining,
            "force_on_demand": force_on_demand,
        },
        evidence_summary={
            "candidate_count": len(options),
            "valid_replacement_count": sum(
                1
                for o in options
                if o.valid and o.action_type not in {"hold_noop", "abort_recovery"}
            ),
            "hold_safe": hold.valid,
            "resource_map_available": rm is not None,
        },
        action_options=options,
        detail_sections=detail_sections,
        guards={
            "max_menu_options": MAX_MENU_OPTIONS,
            "retry_budget_enforced": True,
            "executes_through_scale_chain": True,
        },
    )


def render_p4_prompt(packet: TransitionPacket) -> str:
    lines = [
        "P4 REPLICA RECOVERY",
        "Choose one valid action_id from the recovery menu.",
        "Use the P5c diagnosis and recent-failure evidence. SLO is hard. Cost is secondary.",
        "Prefer diagnosis-aligned repairs (e.g. spot_preemption -> on_demand, OOM -> higher VRAM).",
        "Choose hold_noop only when the live fleet still meets SLO without the failed replica.",
        "Choose abort_recovery only when retry budget is exhausted or no safe replacement exists.",
        "Do not invent executable actions; Koi will only execute the validated action_id.",
        "",
        "JOB CONTEXT:",
        json.dumps(packet.job_context, indent=2, sort_keys=True),
        "",
        "FAILURE / DIAGNOSIS CONTEXT:",
        json.dumps(packet.failure_context, indent=2, sort_keys=True),
        "",
        "POLICY CONTEXT:",
        json.dumps(packet.policy_context, indent=2, sort_keys=True),
        "",
        "FLEET SNAPSHOT:",
        json.dumps(packet.runtime_context.get("fleet", {}), indent=2, sort_keys=True),
        "",
        "ACTION MENU:",
    ]
    for option in packet.action_options:
        lines.append(
            f"{option.rank}. action_id={option.action_id} type={option.action_type} valid={option.valid}"
        )
        lines.append(f"   {option.summary}")
        if option.hard_feasibility:
            lines.append(f"   feasibility={json.dumps(option.hard_feasibility, sort_keys=True)}")
        if option.cost:
            lines.append(f"   cost={json.dumps(option.cost, sort_keys=True)}")
        if option.availability:
            lines.append(f"   availability={json.dumps(option.availability, sort_keys=True)}")
        if option.risk:
            lines.append(f"   risk={json.dumps(option.risk, sort_keys=True)}")
        if option.evidence:
            lines.append(f"   evidence={json.dumps(option.evidence, sort_keys=True)}")
    lines.extend(["", "Return your final answer as the typed ChosenAction schema."])
    return "\n".join(lines)


def _packet_tools(memory: AgenticMemory, packet: TransitionPacket) -> dict[str, Any]:
    tools = build_packet_read_tools(packet, known_sections=_KNOWN_SECTIONS)

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


# ---------------------------------------------------------------------------
# Plan synthesis + execution
# ---------------------------------------------------------------------------


def _abort_plan(
    packet: TransitionPacket,
    validated: Optional[ValidatedAction] = None,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    option = (
        validated.option
        if validated is not None
        else next(
            (
                candidate
                for candidate in packet.valid_actions()
                if candidate.action_type == "abort_recovery"
            ),
            None,
        )
    )
    rationale = (
        reason
        or (validated.choice.rationale if validated is not None else None)
        or (option.summary if option else "replica recovery aborted")
    )
    return {
        "action": "abort",
        "decision_id": None,
        "parent_decision_id": packet.job_context.get("parent_decision_id"),
        "reasoning": rationale,
        "confidence": validated.choice.confidence if validated is not None else 1.0,
        "retry_budget_remaining": packet.policy_context.get(
            "retry_budget_remaining_before_choice", 0
        ),
        "diagnosis_code": packet.failure_context.get("diagnosis_code"),
    }


def _hold_plan(packet: TransitionPacket, validated: ValidatedAction) -> dict[str, Any]:
    return {
        "action": "hold",
        "decision_id": None,
        "parent_decision_id": packet.job_context.get("parent_decision_id"),
        "reasoning": validated.choice.rationale or validated.option.summary,
        "confidence": validated.choice.confidence,
        "retry_budget_remaining": packet.policy_context.get(
            "retry_budget_remaining_before_choice", 0
        ),
        "diagnosis_code": packet.failure_context.get("diagnosis_code"),
        "fleet": packet.runtime_context.get("fleet", {}),
    }


def _record_recovery_decision(
    *,
    packet: TransitionPacket,
    memory: AgenticMemory,
    config: PlacementConfig,
    payload: dict[str, Any],
    confidence: float,
) -> str:
    return memory.record_decision(
        job_id=packet.job_id,
        model_name=str(packet.job_context.get("model_name") or "unknown"),
        instance_type=config.instance_type,
        gpu_type=config.gpu_type,
        tp=config.tp,
        pp=config.pp,
        dp=config.dp,
        num_gpus=config.num_gpus,
        predicted_tps=float(payload.get("predicted_tps") or 0.0),
        predicted_cost_per_hour=float(payload.get("predicted_cost_per_hour") or 0.0),
        predicted_total_cost=payload.get("predicted_total_cost"),
        predicted_runtime_hours=payload.get("predicted_runtime_hours"),
        prediction_confidence=confidence,
        prediction_source=source_to_prediction_source(
            str(payload.get("prediction_source") or payload.get("source") or "")
        ),
        slo_deadline_hours=float(packet.job_context.get("slo_deadline_hours") or 0.0),
        objective=str(packet.job_context.get("objective") or "cheapest"),
        avg_input_tokens=int(packet.job_context.get("avg_input_tokens") or 0),
        avg_output_tokens=int(packet.job_context.get("avg_output_tokens") or 0),
        num_requests=packet.job_context.get("num_requests"),
        quantization=packet.job_context.get("quantization"),
        triggered_by="replica_recovery",
        parent_decision_id=packet.job_context.get("parent_decision_id"),
        cost_roofline_usd=packet.job_context.get("cost_roofline_usd"),
        market=config.market,
    )


async def _execute_recovery(
    *,
    agent: Any,
    packet: TransitionPacket,
    memory: AgenticMemory,
    ledger: Any,
    validated: ValidatedAction,
) -> dict[str, Any]:
    option = validated.option
    if option.action_type == "abort_recovery":
        return _abort_plan(packet, validated)
    if option.action_type == "hold_noop":
        return _hold_plan(packet, validated)

    payload = packet.detail_sections.get(option.executor_payload_ref or "", {})
    config = placement_config_from_payload(
        payload, fallback_region=packet.job_context.get("region", "unknown")
    )

    decision_id = _record_recovery_decision(
        packet=packet,
        memory=memory,
        config=config,
        payload=payload,
        confidence=validated.choice.confidence,
    )
    if ledger is not None:
        try:
            ledger.reserve(
                decision_id=decision_id,
                gpu_type=config.gpu_type,
                num_gpus=config.num_gpus,
                region=config.region,
                instance_type=config.instance_type,
            )
        except Exception as exc:
            logger.warning(
                "p4_ledger_reserve_failed",
                job_id=packet.job_id,
                error=str(exc),
            )

    # Execute through existing scale primitive when an Orca client is wired.
    scale_result: Optional[str] = None
    if getattr(agent, "orca", None) is not None and getattr(agent, "monitor", None):
        try:
            tools = agent._build_tools(monitor=agent.monitor)
            scale_tool = tools.get("scale_chain_tool")
            if scale_tool is not None:
                use_on_demand = config.market == "on_demand"
                scale_result = await scale_tool(
                    job_id=packet.job_context.get("group_id") or packet.job_id,
                    gpu_type=config.gpu_type,
                    tp=config.tp,
                    pp=config.pp,
                    count=1,
                    on_demand=use_on_demand,
                )
        except Exception as exc:
            logger.warning(
                "p4_scale_chain_failed",
                job_id=packet.job_id,
                error=str(exc),
            )

    rationale = validated.choice.rationale or option.summary
    if validated.fallback_used:
        rationale = f"[HARNESS FALLBACK] {rationale}"

    remaining_before = int(
        packet.policy_context.get("retry_budget_remaining_before_choice") or 0
    )
    return {
        "action": option.action_type,
        "decision_id": decision_id,
        "parent_decision_id": packet.job_context.get("parent_decision_id"),
        "action_id": option.action_id,
        "config": config.model_dump(mode="json"),
        "alternatives": alternative_payloads(
            packet,
            option.action_id,
            action_type="*",
            exclude_action_types={"hold_noop", "abort_recovery"},
            include_action_type_in_payload=True,
        ),
        "reasoning": rationale,
        "confidence": validated.choice.confidence,
        "retry_budget_remaining": max(0, remaining_before - 1),
        "diagnosis_code": packet.failure_context.get("diagnosis_code"),
        "scale_result": scale_result,
    }


async def run_replica_recovery(
    *,
    agent: Any,
    req: Any,
    tracker: Any,
    memory: AgenticMemory,
    diagnosis: Any,
    region: str,
    market: str,
    ledger: Any = None,
    resource_map: Optional[ResourceMap] = None,
    retry_budget: Optional[int] = None,
) -> dict[str, Any]:
    t0 = time.time()
    packet = await build_p4_packet(
        agent=agent,
        req=req,
        tracker=tracker,
        memory=memory,
        diagnosis=diagnosis,
        region=region,
        market=market,
        ledger=ledger,
        resource_map=resource_map,
        retry_budget=retry_budget,
    )

    valid_replacements = [
        option
        for option in packet.valid_actions()
        if option.action_type not in {"hold_noop", "abort_recovery"}
    ]
    valid_holds = [
        option for option in packet.valid_actions() if option.action_type == "hold_noop"
    ]

    if not valid_replacements and not valid_holds:
        plan = _abort_plan(packet, reason="no valid replica recovery action")
        emit_event(
            "harness.p4.decided",
            job_id=packet.job_id,
            action="abort",
            elapsed_s=round(time.time() - t0, 2),
        )
        return plan

    # Fast-path: only one valid action — skip LLM.
    valid_actions = packet.valid_actions()
    if len(valid_actions) == 1:
        validated = ValidatedAction(
            choice=ChosenAction(
                action_id=valid_actions[0].action_id,
                confidence=0.5,
                rationale="P4 fast-path: single valid recovery option.",
            ),
            option=valid_actions[0],
            fallback_used=False,
        )
        plan = await _execute_recovery(
            agent=agent,
            packet=packet,
            memory=memory,
            ledger=ledger,
            validated=validated,
        )
        emit_event(
            "harness.p4.decided",
            job_id=packet.job_id,
            action=plan.get("action"),
            action_id=valid_actions[0].action_id,
            elapsed_s=round(time.time() - t0, 2),
            fast_path=True,
        )
        return plan

    model = getattr(agent, "_model", None)
    if model is None:
        # No LLM available — fall back to the top-ranked replacement.
        top = valid_replacements[0] if valid_replacements else valid_holds[0]
        validated = ValidatedAction(
            choice=ChosenAction(
                action_id=top.action_id,
                confidence=0.4,
                rationale="P4 fallback: deterministic top-ranked recovery option.",
            ),
            option=top,
            fallback_used=True,
        )
        plan = await _execute_recovery(
            agent=agent,
            packet=packet,
            memory=memory,
            ledger=ledger,
            validated=validated,
        )
        emit_event(
            "harness.p4.decided",
            job_id=packet.job_id,
            action=plan.get("action"),
            action_id=top.action_id,
            elapsed_s=round(time.time() - t0, 2),
            fallback_used=True,
        )
        return plan

    prompt = render_p4_prompt(packet)
    reasoner = HarnessReasoner(
        model=model,
        tools=_packet_tools(memory, packet),
    )
    try:
        tool_calls, choice = await reasoner.choose(
            prompt,
            job_id=packet.job_id,
            label="p4",
            max_iterations=P4_MAX_ITERATIONS,
            timeout=P4_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error("p4_timeout", job_id=packet.job_id, timeout=P4_TIMEOUT)
        top = valid_replacements[0] if valid_replacements else valid_holds[0]
        choice = ChosenAction(
            action_id=top.action_id,
            confidence=0.3,
            rationale="P4 timed out; deterministic fallback chose top recovery option.",
        )
        tool_calls = 0
    except Exception as exc:
        logger.warning("p4_reasoner_failed", job_id=packet.job_id, error=str(exc))
        top = valid_replacements[0] if valid_replacements else valid_holds[0]
        choice = ChosenAction(
            action_id=top.action_id,
            confidence=0.3,
            rationale=f"P4 reasoner error ({exc}); deterministic fallback applied.",
        )
        tool_calls = 0

    try:
        validated = validate_choice(packet, choice)
    except NoValidActionError:
        plan = _abort_plan(packet, reason="no valid replica recovery action")
        emit_event(
            "harness.p4.decided",
            job_id=packet.job_id,
            action="abort",
            elapsed_s=round(time.time() - t0, 2),
        )
        return plan

    plan = await _execute_recovery(
        agent=agent,
        packet=packet,
        memory=memory,
        ledger=ledger,
        validated=validated,
    )
    emit_event(
        "harness.p4.decided",
        job_id=packet.job_id,
        action=plan.get("action"),
        action_id=validated.option.action_id,
        elapsed_s=round(time.time() - t0, 2),
        fallback_used=validated.fallback_used,
        tool_calls=tool_calls,
    )
    return plan
