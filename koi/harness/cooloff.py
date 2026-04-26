"""Pure helpers for short-horizon failure cooloffs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CooloffScope:
    gpu_type: str
    instance_type: str
    region: str
    market: str
    tp: Optional[int] = None
    pp: Optional[int] = None
    dp: Optional[int] = None

    def key(self, *, include_topology: bool = False) -> str:
        parts = [self.gpu_type, self.instance_type, self.region, self.market]
        if include_topology:
            parts.extend([
                str(self.tp or "*"),
                str(self.pp or "*"),
                str(self.dp or "*"),
            ])
        return "|".join(parts)


@dataclass(frozen=True)
class CooloffEntry:
    scope: CooloffScope
    reason: str
    avoid_until: float
    hard_until: Optional[float] = None

    def is_active(self, now: Optional[float] = None) -> bool:
        current = time.time() if now is None else now
        return current < self.avoid_until

    def is_hard(self, now: Optional[float] = None) -> bool:
        if self.hard_until is None:
            return False
        current = time.time() if now is None else now
        return current < self.hard_until
