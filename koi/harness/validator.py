"""Validation helpers for bounded harness choices."""

from __future__ import annotations

from koi.harness.schemas import ActionOption, ChosenAction, TransitionPacket, ValidatedAction


class NoValidActionError(ValueError):
    """Raised when a packet has no valid fallback action."""


def top_valid_option(packet: TransitionPacket) -> ActionOption:
    valid_options = packet.valid_actions()
    if not valid_options:
        raise NoValidActionError(f"packet {packet.packet_id!r} has no valid actions")
    return valid_options[0]


def validate_choice(packet: TransitionPacket, choice: ChosenAction) -> ValidatedAction:
    option = packet.get_action(choice.action_id)
    if option and option.valid:
        return ValidatedAction(choice=choice, option=option)

    fallback = top_valid_option(packet)
    fallback_choice = ChosenAction(
        action_id=fallback.action_id,
        confidence=min(choice.confidence, 0.3),
        rationale=(
            "Deterministic fallback: model chose an invalid or unavailable "
            f"action_id={choice.action_id!r}. Original rationale: {choice.rationale}"
        ),
        evidence_used=choice.evidence_used,
        why_not_top_choice=choice.why_not_top_choice,
        requested_more_context=choice.requested_more_context,
    )
    return ValidatedAction(
        choice=fallback_choice,
        option=fallback,
        fallback_used=True,
        fallback_reason="invalid_action_id",
    )
