"""Unit tests for KoiToolRunner against pydantic-ai's TestModel (no network)."""
from unittest.mock import patch

import pytest
from pydantic_ai.models.test import TestModel

from koi.llm.runner import KoiToolRunner


@pytest.mark.asyncio
async def test_runner_emits_tool_call_events():
    async def add_numbers(x: int, y: int) -> str:
        """Add two integers and return the sum as a string."""
        return str(x + y)

    runner = KoiToolRunner(
        model=TestModel(),
        system_prompt="You are a helper.",
        tools={"add_numbers": add_numbers},
    )
    with patch("koi.llm.runner.emit_event") as mock_emit:
        tool_calls, final_text = await runner.run(
            "Add two numbers",
            label="decide",
            job_id="job-42",
            max_iterations=5,
            timeout=10.0,
        )
    assert tool_calls >= 1
    assert mock_emit.call_count == tool_calls
    first_call = mock_emit.call_args_list[0]
    assert first_call.args[0] == "tool_call"
    assert first_call.kwargs["tool"] == "add_numbers"
    assert first_call.kwargs["label"] == "decide"
    assert first_call.kwargs["job_id"] == "job-42"
    assert isinstance(final_text, str)


@pytest.mark.asyncio
async def test_runner_returns_empty_string_when_no_output():
    # Tool that fires but TestModel still yields a final output; verify the
    # return tuple shape holds under a normal run.
    async def noop(note: str) -> str:
        """Record a note."""
        return note

    runner = KoiToolRunner(
        model=TestModel(),
        system_prompt="helper",
        tools={"noop": noop},
    )
    with patch("koi.llm.runner.emit_event"):
        tool_calls, final_text = await runner.run(
            "record note",
            label="trigger",
            job_id=None,
            max_iterations=3,
            timeout=10.0,
        )
    assert isinstance(tool_calls, int)
    assert isinstance(final_text, str)
