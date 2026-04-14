"""Optional file-backed event tap for demo and debugging views."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def emit_event(event: str, **fields: Any) -> None:
    """Append one JSON event to the configured tap file, if enabled."""
    path = os.environ.get("KOI_EVENT_TAP_PATH")
    if not path:
        return

    record = {
        "event": event,
        "timestamp": time.time(),
    }
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            record[key] = value
        else:
            record[key] = str(value)

    try:
        tap_path = Path(path)
        tap_path.parent.mkdir(parents=True, exist_ok=True)
        with tap_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")))
            handle.write("\n")
    except Exception:
        # Demo diagnostics should never break the control plane.
        return


def read_recent_events(path: str, limit: int = 100) -> list[dict[str, Any]]:
    """Read and decode the most recent JSONL events from the tap file."""
    tap_path = Path(path)
    if not tap_path.exists():
        return []

    try:
        lines = tap_path.read_text(encoding="utf-8").splitlines()[-limit:]
    except Exception:
        return []

    events: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events
