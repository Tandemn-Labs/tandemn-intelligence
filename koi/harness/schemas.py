"""Shared data models for the v0 Koi harness.

These models are intentionally generic in Phase 0. Later phases will add
transition-specific packet builders and executors around this stable shape.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class HarnessState(str, Enum):
    REQUESTED = "requested"
    LAUNCHING = "launching"
    LAUNCH_FAILED = "launch_failed"
    WARMING = "warming"
    HEALTHY = "healthy"
    AT_RISK = "at_risk"
    DEGRADED = "degraded"
    OVERPROV = "overprov"
    SCALING = "scaling"
    REPLICA_RECOVERY = "replica_recovery"
    TERMINAL_COMPLETED = "terminal_completed"
    TERMINAL_FAILED = "terminal_failed"
    TERMINAL_ABORTED = "terminal_aborted"


class TransitionType(str, Enum):
    INITIAL_PLACEMENT = "initial_placement"
    LAUNCH_RECOVERY = "launch_recovery"
    SCALE = "scale"
    REPLICA_RECOVERY = "replica_recovery"
    CHAIN_POSTMORTEM = "chain_postmortem"
    JOB_POSTMORTEM = "job_postmortem"


class EvidenceQuality(str, Enum):
    MEMORY_VERIFIED = "memory_verified"
    PERFDB_EXACT = "perfdb_exact"
    PERFDB_INTERPOLATED = "perfdb_interpolated"
    PHYSICS_PROXY = "physics_proxy"
    ANALYTICAL_ROOFLINE = "analytical_roofline"
    RUNTIME_OBSERVED = "runtime_observed"
    UNKNOWN = "unknown"


class ActionOption(BaseModel):
    """A prevalidated menu option offered to the LLM.

    The LLM chooses an ``action_id``. Executor code later resolves
    ``executor_payload_ref`` to concrete production operations.
    """

    action_id: str = Field(min_length=1)
    action_type: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    rank: int = Field(ge=1)
    valid: bool = True
    hard_feasibility: dict[str, Any] = Field(default_factory=dict)
    performance: dict[str, Any] = Field(default_factory=dict)
    physics: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    availability: dict[str, Any] = Field(default_factory=dict)
    cost: dict[str, Any] = Field(default_factory=dict)
    risk: dict[str, Any] = Field(default_factory=dict)
    executor_payload_ref: Optional[str] = None
    detail_refs: list[str] = Field(default_factory=list)


class TransitionPacket(BaseModel):
    """Common packet shape consumed by all harness prompts."""

    packet_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    tenant_id: str = "default"
    state: HarnessState
    transition_type: TransitionType
    job_context: dict[str, Any] = Field(default_factory=dict)
    runtime_context: dict[str, Any] = Field(default_factory=dict)
    failure_context: dict[str, Any] = Field(default_factory=dict)
    policy_context: dict[str, Any] = Field(default_factory=dict)
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    action_options: list[ActionOption] = Field(default_factory=list)
    detail_sections: dict[str, Any] = Field(default_factory=dict)
    guards: dict[str, Any] = Field(default_factory=dict)

    def get_action(self, action_id: str) -> Optional[ActionOption]:
        for option in self.action_options:
            if option.action_id == action_id:
                return option
        return None

    def valid_actions(self) -> list[ActionOption]:
        return sorted(
            (option for option in self.action_options if option.valid),
            key=lambda option: (option.rank, option.action_id),
        )


class ChosenAction(BaseModel):
    """Typed final choice returned by the LLM reasoner."""

    action_id: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    evidence_used: list[str] = Field(default_factory=list)
    why_not_top_choice: Optional[str] = None
    requested_more_context: bool = False


class ValidatedAction(BaseModel):
    """A choice after deterministic safety validation."""

    choice: ChosenAction
    option: ActionOption
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
