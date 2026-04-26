"""Packet construction helpers shared by future harness builders."""

from __future__ import annotations

import uuid


def new_packet_id(prefix: str = "pkt") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"
