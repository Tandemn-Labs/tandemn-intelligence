"""FSM harness scaffolding for Koi's production agent loop."""

from koi.harness.fsm import derive_state, state_from_monitoring_status
from koi.harness.schemas import (
    ActionOption,
    ChosenAction,
    EvidenceQuality,
    HarnessState,
    TransitionPacket,
    TransitionType,
    ValidatedAction,
)
from koi.harness.validator import NoValidActionError, validate_choice

__all__ = [
    "ActionOption",
    "ChosenAction",
    "EvidenceQuality",
    "HarnessState",
    "NoValidActionError",
    "TransitionPacket",
    "TransitionType",
    "ValidatedAction",
    "derive_state",
    "state_from_monitoring_status",
    "validate_choice",
]
