"""Runtime state and streaming snapshots for demo sessions."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Optional

from simulation.demo_scenarios import due_scenario_events


class DemoSessionManager:
    def __init__(self):
        self.sessions: dict[str, dict[str, Any]] = {}

    def clear(self):
        self.sessions.clear()

    def create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        session = deepcopy(payload)
        session["runtime"] = {
            "active_replicas": payload["scenario"]["initial_replicas"],
            "baseline_replica_tps": payload["launch_preview"]["baseline_replica_tps"],
            "events": [],
            "emitted_event_ids": [],
            "status": "launching",
            "launch_phase": "searching_capacity",
            "aggregate_tps": 0.0,
            "progress_pct": 0.0,
            "tokens_completed": 0,
            "tokens_total": payload["request"]["total_chunks"] * (
                payload["request"]["avg_input_tokens"] + payload["request"]["avg_output_tokens"]
            ),
            "eta_seconds": None,
        }
        self.sessions[session["session_id"]] = session
        return session

    def snapshot(self, session_id: str, now: Optional[float] = None) -> dict[str, Any]:
        session = self.sessions[session_id]
        now = now or time.time()
        launch = session["launch_preview"]["launch_timing_s"]
        runtime = session["runtime"]

        created_at = session["created_at"]
        elapsed = max(0.0, now - created_at)
        launch_total = launch["total"]

        self._emit_due_scenario_events(session, elapsed)

        if elapsed < launch_total:
            launch_phase = self._launch_phase_for_elapsed(launch, elapsed)
            runtime["status"] = "launching"
            runtime["launch_phase"] = launch_phase
            runtime["aggregate_tps"] = 0.0
            runtime["progress_pct"] = 0.0
            runtime["tokens_completed"] = 0
            runtime["eta_seconds"] = launch_total - elapsed
            snapshot = deepcopy(session)
            snapshot["runtime"] = deepcopy(runtime)
            snapshot["runtime"]["elapsed_seconds"] = round(elapsed, 1)
            return snapshot

        aggregate_tps, active_replicas = self._aggregate_tps(session, elapsed)
        runtime["status"] = "running"
        runtime["launch_phase"] = "running"
        runtime["aggregate_tps"] = round(aggregate_tps, 1)
        runtime["active_replicas"] = active_replicas

        running_elapsed = elapsed - launch_total
        completed = min(runtime["tokens_total"], int(aggregate_tps * running_elapsed))
        runtime["tokens_completed"] = completed
        runtime["progress_pct"] = round((completed / max(runtime["tokens_total"], 1)) * 100, 2)

        remaining = max(0, runtime["tokens_total"] - completed)
        if remaining == 0:
            runtime["status"] = "completed"
            runtime["eta_seconds"] = 0.0
        elif aggregate_tps > 0:
            runtime["eta_seconds"] = round(remaining / aggregate_tps, 1)
        else:
            runtime["eta_seconds"] = None

        snapshot = deepcopy(session)
        snapshot["runtime"] = deepcopy(runtime)
        snapshot["runtime"]["elapsed_seconds"] = round(elapsed, 1)
        return snapshot

    def _emit_due_scenario_events(self, session: dict[str, Any], elapsed_seconds: float) -> None:
        runtime = session["runtime"]
        seen = runtime["emitted_event_ids"]
        due = due_scenario_events(
            session["scenario"]["slug"],
            elapsed_seconds=elapsed_seconds,
            completed_event_ids=seen,
        )
        for event in due:
            runtime["emitted_event_ids"].append(event.event_id)
            runtime["events"].append(
                {
                    "event_id": event.event_id,
                    "at_seconds": event.at_seconds,
                    "action": event.action,
                    "label": event.label,
                    "description": event.description,
                    "params": dict(event.params),
                }
            )

    @staticmethod
    def _launch_phase_for_elapsed(launch: dict[str, float], elapsed: float) -> str:
        searching = launch["searching_capacity"]
        provisioning = searching + launch["provisioning"]
        bootstrapping = provisioning + launch["bootstrapping"]
        if elapsed < searching:
            return "searching_capacity"
        if elapsed < provisioning:
            return "provisioning"
        if elapsed < bootstrapping:
            return "bootstrapping"
        return "waiting_model_ready"

    @staticmethod
    def _aggregate_tps(session: dict[str, Any], elapsed_seconds: float) -> tuple[float, int]:
        runtime = session["runtime"]
        base = float(runtime["baseline_replica_tps"])
        active_replicas = int(runtime["active_replicas"])
        cluster_tps = base * active_replicas

        for event in runtime["events"]:
            action = event["action"]
            params = event["params"]
            if action == "degrade_replica":
                if active_replicas > 0:
                    cluster_tps = max(0.0, base * max(active_replicas - 1, 0) + float(params.get("target_tps", base)))
            elif action == "restore_cluster_tps":
                cluster_tps = float(params.get("target_tps", base)) * active_replicas
            elif action == "kill_oldest_running":
                active_replicas = max(0, active_replicas - 1)
                cluster_tps = base * active_replicas

        return cluster_tps, active_replicas
