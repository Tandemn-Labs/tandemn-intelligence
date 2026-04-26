"""Menu helpers for harness action options."""

from __future__ import annotations

from koi.harness.schemas import ActionOption


def ranked_valid_options(options: list[ActionOption]) -> list[ActionOption]:
    return sorted(
        (option for option in options if option.valid),
        key=lambda option: (option.rank, option.action_id),
    )


def cap_menu(options: list[ActionOption], limit: int = 8) -> list[ActionOption]:
    if limit <= 0:
        return []
    return ranked_valid_options(options)[:limit]
