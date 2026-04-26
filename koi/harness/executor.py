"""Executor interfaces for future harness phases."""

from __future__ import annotations

from typing import Protocol

from koi.harness.schemas import ValidatedAction



class ActionExecutor(Protocol):
    async def execute(self, action: ValidatedAction) -> object: ...
