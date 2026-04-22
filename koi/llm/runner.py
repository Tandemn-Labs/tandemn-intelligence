"""KoiToolRunner — wraps pydantic_ai.Agent with per-tool-call telemetry."""
import asyncio
from typing import Any, Awaitable, Callable, Optional

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from koi.event_tap import emit_event
from koi.logging_config import get_logger

logger = get_logger(__name__)


class KoiToolRunner:
    def __init__(
        self,
        *,
        model: Model,
        system_prompt: str,
        tools: dict[str, Callable[..., Awaitable[Any]]],
    ):
        self._agent: Agent = Agent(model=model, system_prompt=system_prompt, retries=2)
        for fn in tools.values():
            self._agent.tool_plain(fn)

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
            tool_calls = 0
            final_text = ""
            async with self._agent.iter(
                prompt,
                usage_limits=UsageLimits(request_limit=max_iterations),
            ) as run:
                async for node in run:
                    if Agent.is_call_tools_node(node):
                        for part in node.model_response.parts:
                            if getattr(part, "part_kind", None) == "tool-call":
                                tool_calls += 1
                                logger.info(
                                    "tool_call",
                                    label=label,
                                    call_number=tool_calls,
                                    tool=part.tool_name,
                                )
                                emit_event(
                                    "tool_call",
                                    label=label,
                                    call_number=tool_calls,
                                    tool=part.tool_name,
                                    job_id=job_id,
                                )
                if run.result is not None:
                    final_text = str(run.result.output or "")
            return tool_calls, final_text

        return await asyncio.wait_for(_inner(), timeout)
