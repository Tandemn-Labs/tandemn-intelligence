"""Derived-state helpers for the v0 harness FSM.

Phase 0 deliberately derives state from existing production objects instead of
introducing a second FSM source of truth.
"""

from __future__ import annotations

from typing import Any, Optional

from koi.harness.schemas import HarnessState
from koi.schemas import MonitoringStatus


_MONITORING_TO_HARNESS = {
    MonitoringStatus.WARMING_UP.value: HarnessState.WARMING,
    MonitoringStatus.ON_TRACK.value: HarnessState.HEALTHY,
    MonitoringStatus.AT_RISK.value: HarnessState.AT_RISK,
    MonitoringStatus.FALLING_BEHIND.value: HarnessState.DEGRADED,
    MonitoringStatus.OVER_PROVISIONED.value: HarnessState.OVERPROV,
    MonitoringStatus.LAUNCH_FAILED.value: HarnessState.LAUNCH_FAILED,
    MonitoringStatus.COMPLETED.value: HarnessState.TERMINAL_COMPLETED,
    MonitoringStatus.FAILED.value: HarnessState.TERMINAL_FAILED,
}

_EVENT_TO_HARNESS = {
    "decide": HarnessState.REQUESTED,
    "job_launching": HarnessState.LAUNCHING,
    "job_launch_heartbeat": HarnessState.LAUNCHING,
    "launch_failed": HarnessState.LAUNCH_FAILED,
    "replica_failed": HarnessState.REPLICA_RECOVERY,
    "job_complete": HarnessState.TERMINAL_COMPLETED,
    "job_failed": HarnessState.TERMINAL_FAILED,
    "abort": HarnessState.TERMINAL_ABORTED,
}


def _get_value(source: Any, name: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _status_value(status: Any) -> Optional[str]:
    if status is None:
        return None
    return getattr(status, "value", status)


def state_from_monitoring_status(status: MonitoringStatus | str) -> HarnessState:
    value = _status_value(status)
    if value not in _MONITORING_TO_HARNESS:
        return HarnessState.AT_RISK
    return _MONITORING_TO_HARNESS[value]


def derive_state(
    *,
    event_type: Optional[str] = None,
    tracker: Any = None,
    pending_launch: Optional[dict[str, Any]] = None,
) -> HarnessState:
    """Derive the harness state from existing production state.

    Event context wins for transient recovery/terminal paths. For running jobs,
    action-in-flight overrides health and maps to ``SCALING``.
    """

    if event_type in _EVENT_TO_HARNESS:
        return _EVENT_TO_HARNESS[event_type]

    if tracker is not None:
        status = _status_value(_get_value(tracker, "status"))
        if status in {MonitoringStatus.COMPLETED.value, MonitoringStatus.FAILED.value}:
            return state_from_monitoring_status(status)
        if _get_value(tracker, "action_in_progress", False):
            return HarnessState.SCALING
        if status:
            return state_from_monitoring_status(status)

    if pending_launch:
        return HarnessState.LAUNCHING

    return HarnessState.REQUESTED
