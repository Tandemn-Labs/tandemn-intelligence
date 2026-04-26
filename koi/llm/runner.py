"""KoiToolRunner — wraps pydantic_ai.Agent with per-tool-call telemetry."""
import asyncio
from typing import Any, Awaitable, Callable, Optional, TypeVar

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from koi.event_tap import emit_event
from koi.logging_config import get_logger

logger = get_logger(__name__)

RunOutputT = TypeVar("RunOutputT")


class KoiToolRunner:
    def __init__(
        self,
        *,
        model: Model,
        system_prompt: str,
        tools: dict[str, Callable[..., Awaitable[Any]]],
    ):
        self._agent: Agent = Agent(model=model, system_prompt=system_prompt, retries=2)
        self._tool_names = set(tools)
        for fn in tools.values():
            self._agent.tool_plain(fn)

    def _record_tool_call(
        self,
        *,
        label: str,
        job_id: Optional[str],
        call_number: int,
        tool_name: str,
    ) -> None:
        logger.info(
            "tool_call",
            label=label,
            call_number=call_number,
            tool=tool_name,
        )
        emit_event(
            "tool_call",
            label=label,
            call_number=call_number,
            tool=tool_name,
            job_id=job_id,
        )

    async def _run_inner(
        self,
        prompt: str,
        *,
        label: str,
        job_id: Optional[str],
        max_iterations: int,
        output_type: Any = None,
    ) -> tuple[int, Any]:
        tool_calls = 0
        final_output = None
        kwargs: dict[str, Any] = {
            "usage_limits": UsageLimits(request_limit=max_iterations),
        }
        if output_type is not None:
            kwargs["output_type"] = output_type

        async with self._agent.iter(prompt, **kwargs) as run:
            async for node in run:
                if Agent.is_call_tools_node(node):
                    for part in node.model_response.parts:
                        if getattr(part, "part_kind", None) != "tool-call":
                            continue
                        tool_name = part.tool_name
                        # pydantic-ai may represent structured output as an
                        # output-tool call. Only count real Koi tools here.
                        if tool_name not in self._tool_names:
                            continue
                        tool_calls += 1
                        self._record_tool_call(
                            label=label,
                            job_id=job_id,
                            call_number=tool_calls,
                            tool_name=tool_name,
                        )
            if run.result is not None:
                final_output = run.result.output
        return tool_calls, final_output

    async def run(
        self,
        prompt: str,
        *,
        label: str,
        job_id: Optional[str],
        max_iterations: int,
        timeout: float,
    ) -> tuple[int, str]:
        async def _inner() -> tuple[int, str]:
            tool_calls, final_output = await self._run_inner(
                prompt,
                label=label,
                job_id=job_id,
                max_iterations=max_iterations,
            )
            final_text = str(final_output or "")
            return tool_calls, final_text

        return await asyncio.wait_for(_inner(), timeout)

    async def run_typed(
        self,
        prompt: str,
        *,
        label: str,
        job_id: Optional[str],
        max_iterations: int,
        timeout: float,
        output_type: type[RunOutputT],
    ) -> tuple[int, RunOutputT]:
        """Run the agent and return a pydantic-ai structured output.

        Existing ``run()`` behavior remains text-only for legacy prompts. The
        harness uses this method for final typed choices while keeping the same
        tool-call telemetry.
        """

        async def _inner() -> tuple[int, RunOutputT]:
            tool_calls, final_output = await self._run_inner(
                prompt,
                label=label,
                job_id=job_id,
                max_iterations=max_iterations,
                output_type=output_type,
            )
            if final_output is None:
                raise RuntimeError("typed runner returned no output")
            return tool_calls, final_output

        return await asyncio.wait_for(_inner(), timeout)
