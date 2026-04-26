import pytest

from koi.harness.cooloff import CooloffEntry, CooloffScope
from koi.harness.fsm import derive_state, state_from_monitoring_status
from koi.harness.menus import cap_menu, ranked_valid_options
from koi.harness.schemas import (
    ActionOption,
    ChosenAction,
    HarnessState,
    TransitionPacket,
    TransitionType,
)
from koi.harness.validator import NoValidActionError, validate_choice
from koi.schemas import MonitoringStatus


def _option(action_id: str, rank: int, valid: bool = True) -> ActionOption:
    return ActionOption(
        action_id=action_id,
        action_type="launch",
        summary=f"option {action_id}",
        rank=rank,
        valid=valid,
    )


def _packet(options: list[ActionOption]) -> TransitionPacket:
    return TransitionPacket(
        packet_id="pkt-test",
        job_id="job-test",
        state=HarnessState.REQUESTED,
        transition_type=TransitionType.INITIAL_PLACEMENT,
        action_options=options,
    )


def test_transition_packet_returns_ranked_valid_actions():
    packet = _packet([
        _option("slow", 3),
        _option("invalid", 1, valid=False),
        _option("fast", 1),
    ])

    assert [option.action_id for option in packet.valid_actions()] == ["fast", "slow"]
    assert packet.get_action("slow") is not None
    assert packet.get_action("missing") is None


def test_menu_helpers_filter_and_cap_valid_options():
    options = [
        _option("c", 3),
        _option("a", 1),
        _option("bad", 1, valid=False),
        _option("b", 2),
    ]

    assert [option.action_id for option in ranked_valid_options(options)] == ["a", "b", "c"]
    assert [option.action_id for option in cap_menu(options, limit=2)] == ["a", "b"]
    assert cap_menu(options, limit=0) == []


def test_state_from_monitoring_status_maps_health_states():
    assert state_from_monitoring_status(MonitoringStatus.WARMING_UP) == HarnessState.WARMING
    assert state_from_monitoring_status(MonitoringStatus.ON_TRACK) == HarnessState.HEALTHY
    assert state_from_monitoring_status(MonitoringStatus.AT_RISK) == HarnessState.AT_RISK
    assert state_from_monitoring_status(MonitoringStatus.FALLING_BEHIND) == HarnessState.DEGRADED
    assert state_from_monitoring_status(MonitoringStatus.OVER_PROVISIONED) == HarnessState.OVERPROV


def test_derive_state_prefers_transient_event_context():
    assert derive_state(event_type="decide") == HarnessState.REQUESTED
    assert derive_state(event_type="job_launch_heartbeat") == HarnessState.LAUNCHING
    assert derive_state(event_type="launch_failed") == HarnessState.LAUNCH_FAILED
    assert derive_state(event_type="replica_failed") == HarnessState.REPLICA_RECOVERY
    assert derive_state(event_type="job_complete") == HarnessState.TERMINAL_COMPLETED


def test_derive_state_maps_action_in_progress_to_scaling():
    tracker = {
        "status": MonitoringStatus.FALLING_BEHIND,
        "action_in_progress": True,
    }

    assert derive_state(tracker=tracker) == HarnessState.SCALING


def test_derive_state_keeps_terminal_tracker_states_terminal():
    assert derive_state(tracker={"status": MonitoringStatus.COMPLETED}) == HarnessState.TERMINAL_COMPLETED
    assert derive_state(tracker={"status": MonitoringStatus.FAILED}) == HarnessState.TERMINAL_FAILED


def test_validate_choice_accepts_valid_non_top_choice():
    packet = _packet([_option("top", 1), _option("second", 2)])
    choice = ChosenAction(
        action_id="second",
        confidence=0.8,
        rationale="Availability prior is better despite higher rank.",
        why_not_top_choice="Top option has fresh preemption risk.",
    )

    validated = validate_choice(packet, choice)

    assert validated.fallback_used is False
    assert validated.option.action_id == "second"
    assert validated.choice.why_not_top_choice is not None


def test_validate_choice_falls_back_to_top_valid_option():
    packet = _packet([_option("top", 1), _option("second", 2)])
    choice = ChosenAction(
        action_id="missing",
        confidence=0.9,
        rationale="Model invented an action.",
    )

    validated = validate_choice(packet, choice)

    assert validated.fallback_used is True
    assert validated.fallback_reason == "invalid_action_id"
    assert validated.option.action_id == "top"
    assert validated.choice.action_id == "top"
    assert validated.choice.confidence == 0.3


def test_validate_choice_raises_when_no_valid_action_exists():
    packet = _packet([_option("bad", 1, valid=False)])
    choice = ChosenAction(action_id="bad", confidence=0.5, rationale="No valid option.")

    with pytest.raises(NoValidActionError):
        validate_choice(packet, choice)


def test_cooloff_scope_key_and_activity():
    scope = CooloffScope(
        gpu_type="A100-80GB",
        instance_type="p4de.24xlarge",
        region="us-east-1",
        market="spot",
        tp=8,
        pp=1,
        dp=1,
    )
    entry = CooloffEntry(
        scope=scope,
        reason="spot_preemption",
        avoid_until=200.0,
        hard_until=150.0,
    )

    assert scope.key() == "A100-80GB|p4de.24xlarge|us-east-1|spot"
    assert scope.key(include_topology=True) == "A100-80GB|p4de.24xlarge|us-east-1|spot|8|1|1"
    assert entry.is_active(now=100.0) is True
    assert entry.is_hard(now=100.0) is True
    assert entry.is_hard(now=175.0) is False
    assert entry.is_active(now=250.0) is False
