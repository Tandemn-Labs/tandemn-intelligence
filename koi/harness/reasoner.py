"""LLM reasoner wrapper for harness typed choices."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from pydantic_ai.models import Model

from koi.harness.prompts import HARNESS_SYSTEM_PROMPT
from koi.harness.schemas import ChosenAction
from koi.llm import KoiToolRunner


class HarnessReasoner:
    """Small wrapper around KoiToolRunner for harness choice outputs."""

    def __init__(
        self,
        *,
        model: Model,
        tools: Optional[dict[str, Callable[..., Awaitable[Any]]]] = None,
        system_prompt: str = HARNESS_SYSTEM_PROMPT,
    ):
        self._runner = KoiToolRunner(
            model=model,
            system_prompt=system_prompt,
            tools=tools or {},
        )

    async def choose(
        self,
        prompt: str,
        *,
        job_id: Optional[str],
        label: str = "harness",
        max_iterations: int = 3,
        timeout: float = 120.0,
    ) -> tuple[int, ChosenAction]:
        return await self._runner.run_typed(
            prompt,
            label=label,
            job_id=job_id,
            max_iterations=max_iterations,
            timeout=timeout,
            output_type=ChosenAction,
        )
