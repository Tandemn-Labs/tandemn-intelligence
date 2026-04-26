"""Controller interfaces for future harness phases."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Protocol

from koi.harness.schemas import ChosenAction, TransitionPacket, ValidatedAction
from koi.harness.validator import validate_choice


class PacketBuilder(Protocol):
    async def build(self) -> TransitionPacket: ...


class ChoiceReasoner(Protocol):
    async def choose(self, prompt: str, *, job_id: str | None) -> tuple[int, ChosenAction]: ...


@dataclass
class HarnessController:
    """Phase 0 controller shell.

    Later phases will attach prompt rendering and executors. The validation
    method exists now so tests and future code share the same fallback logic.
    """

    def validate(self, packet: TransitionPacket, choice: ChosenAction) -> ValidatedAction:
        return validate_choice(packet, choice)
