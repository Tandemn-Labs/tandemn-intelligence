"""Tests for the optional Koi event tap."""

from koi.event_tap import emit_event, read_recent_events


class TestEventTap:
    def test_emit_and_read_recent_events(self, monkeypatch, tmp_path):
        path = tmp_path / "koi-events.jsonl"
        monkeypatch.setenv("KOI_EVENT_TAP_PATH", str(path))

        emit_event("agent_deciding", job_id="demo-1", model="Qwen/Qwen3-32B")
        emit_event("tool_call", job_id="demo-1", tool="query_perfdb", call_number=1)

        events = read_recent_events(str(path))
        assert [event["event"] for event in events] == ["agent_deciding", "tool_call"]
        assert events[0]["job_id"] == "demo-1"
        assert events[1]["tool"] == "query_perfdb"

    def test_read_recent_events_ignores_bad_json_lines(self, tmp_path):
        path = tmp_path / "koi-events.jsonl"
        path.write_text('{"event":"ok","job_id":"demo-1"}\nnot-json\n{"event":"tool_call","job_id":"demo-1"}\n', encoding="utf-8")

        events = read_recent_events(str(path))
        assert [event["event"] for event in events] == ["ok", "tool_call"]
