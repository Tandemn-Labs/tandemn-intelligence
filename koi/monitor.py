"""
koi/monitor.py — Runtime monitoring loop.

Responsibilities:
  - Poll live metrics from running jobs every N seconds
  - Apply Kalman filter to smooth noisy GPU metrics
  - Compute running delta = (actual - predicted)
  - Feed delta to deadband controller to decide if reconfiguration is needed
  - Emit DeltaRecord to the Refinement engine when job completes

Metrics sources (in order of preference):
  1. vLLM /metrics Prometheus endpoint
  2. Job-specific metrics API (Tandem internal)
  3. CloudWatch / GPU monitor (from results.json node*_gpu* fields)

Phase 1: stubs with clean interfaces.
Phase 2: wire up real Prometheus polling.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional

from koi.schemas import (
    DeltaRecord,
    PESComponents,
    PlacementDecision,
    PredictedMetrics,
    RuntimeMetrics,
    TaskType,
)


# ---------------------------------------------------------------------------
# Deadband states
# ---------------------------------------------------------------------------

class SLOState(str, Enum):
    GREEN = "green"          # comfortable, < 80% of SLO budget used
    YELLOW_LOW = "yellow_low"  # underprovisioned relative to SLO (<70% util)
    YELLOW_HIGH = "yellow_high"  # approaching SLO (80-100%)
    RED = "red"              # SLO violated or imminent (>100%)


# ---------------------------------------------------------------------------
# Simple Kalman filter for 1D metric smoothing
# ---------------------------------------------------------------------------

class KalmanFilter1D:
    """
    Single-variable discrete Kalman filter for smoothing noisy metric timeseries.

    State: true metric value (e.g. TPOT in ms)
    Measurement noise: R — expected variance in raw readings
    Process noise: Q — expected variance in true value changing over time

    For slowly-changing metrics (TPOT): low Q, higher R (trust model more than measurement)
    For fast-changing metrics (throughput spikes): higher Q (track changes faster)
    """

    def __init__(self, initial_value: float, R: float = 25.0, Q: float = 1.0):
        self.x = initial_value      # state estimate
        self.P = 50.0               # estimate uncertainty (high initial uncertainty)
        self.R = R                  # measurement noise variance
        self.Q = Q                  # process noise variance

    def update(self, measurement: float) -> float:
        # Predict
        self.P += self.Q

        # Update (Kalman gain)
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P

        return self.x

    @property
    def estimate(self) -> float:
        return self.x

    @property
    def uncertainty(self) -> float:
        return self.P ** 0.5  # std dev


# ---------------------------------------------------------------------------
# Deadband controller
# ---------------------------------------------------------------------------

class DeadbandController:
    """
    Two-threshold hysteresis controller.

    Prevents oscillation by requiring the metric to cross both an outer band
    (to trigger action) and an inner band (to declare "recovered").

    Thresholds are expressed as fractions of the SLO value.
    """

    def __init__(
        self,
        slo_value: float,
        green_threshold: float = 0.80,     # below this: GREEN
        yellow_high_threshold: float = 0.90,  # above this: YELLOW_HIGH
        red_threshold: float = 1.05,       # above this: RED
        yellow_low_threshold: float = 0.50,   # below this: YELLOW_LOW (overprovisioned)
    ):
        self.slo = slo_value
        self.green_t = green_threshold * slo_value
        self.yellow_high_t = yellow_high_threshold * slo_value
        self.red_t = red_threshold * slo_value
        self.yellow_low_t = yellow_low_threshold * slo_value
        self._current_state = SLOState.GREEN

    def update(self, filtered_value: float) -> SLOState:
        """
        Apply hysteresis: state only changes when crossing the appropriate threshold.
        Current state is preserved within the band (prevents oscillation).
        """
        prev = self._current_state

        if filtered_value >= self.red_t:
            self._current_state = SLOState.RED
        elif filtered_value >= self.yellow_high_t:
            # Only enter YELLOW_HIGH from GREEN or RED — not from YELLOW_LOW
            if prev in (SLOState.GREEN, SLOState.RED):
                self._current_state = SLOState.YELLOW_HIGH
            elif prev == SLOState.YELLOW_HIGH:
                pass  # stay (hysteresis)
        elif filtered_value <= self.yellow_low_t:
            self._current_state = SLOState.YELLOW_LOW
        elif filtered_value <= self.green_t:
            # Only return to GREEN if coming down from YELLOW_HIGH (hysteresis exit)
            if prev in (SLOState.GREEN, SLOState.YELLOW_LOW):
                self._current_state = SLOState.GREEN
            elif prev == SLOState.YELLOW_HIGH and filtered_value < self.green_t:
                self._current_state = SLOState.GREEN
        else:
            # In the ambiguous band (green_t to yellow_high_t): stay in current state
            pass

        return self._current_state

    @property
    def state(self) -> SLOState:
        return self._current_state


# ---------------------------------------------------------------------------
# Per-job monitor state
# ---------------------------------------------------------------------------

@dataclass
class JobMonitorState:
    """Live monitoring state for a single running job."""
    job_id: str
    decision: PlacementDecision
    start_time: float = field(default_factory=time.time)
    last_poll_time: Optional[float] = None

    # Kalman filters per metric
    kf_tpot: Optional[KalmanFilter1D] = None
    kf_throughput: Optional[KalmanFilter1D] = None

    # Deadband controllers (initialized once we have SLO values)
    deadband_tpot: Optional[DeadbandController] = None
    deadband_throughput: Optional[DeadbandController] = None

    # Current smoothed state
    current_tpot_ms: Optional[float] = None
    current_throughput_tps: Optional[float] = None
    slo_state: SLOState = SLOState.GREEN

    # History for delta computation
    raw_metrics_history: List[RuntimeMetrics] = field(default_factory=list)
    reconfiguration_count: int = 0
    action_in_progress: bool = False     # Anti-windup: suppress corrections during transition
    action_freeze_until: Optional[float] = None  # timestamp to unfreeze

    def elapsed_hours(self) -> float:
        return (time.time() - self.start_time) / 3600.0

    def is_frozen(self) -> bool:
        """Anti-windup: returns True if an action is in progress."""
        if self.action_in_progress:
            if self.action_freeze_until and time.time() > self.action_freeze_until:
                self.action_in_progress = False
                self.action_freeze_until = None
                return False
            return True
        return False

    def freeze(self, duration_seconds: float = 300.0) -> None:
        """Freeze the deadband during a reconfiguration."""
        self.action_in_progress = True
        self.action_freeze_until = time.time() + duration_seconds
        self.reconfiguration_count += 1

    def unfreeze(self) -> None:
        self.action_in_progress = False
        self.action_freeze_until = None


# ---------------------------------------------------------------------------
# Metrics source interface
# ---------------------------------------------------------------------------

class MetricsSource(ABC):
    """Abstract interface for fetching live metrics from a running job."""

    @abstractmethod
    async def fetch(self, job_id: str) -> Optional[RuntimeMetrics]:
        """Fetch latest metrics snapshot. Returns None if job not found."""
        ...


class PrometheusMetricsSource(MetricsSource):
    """
    Fetches metrics from vLLM's Prometheus /metrics endpoint.

    Expected metrics:
      vllm:num_requests_running        → concurrent_requests
      vllm:avg_generation_throughput_toks_per_s → throughput
      vllm:time_per_output_token_seconds_bucket → TPOT histogram
      vllm:time_to_first_token_seconds_bucket   → TTFT histogram
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def fetch(self, job_id: str) -> Optional[RuntimeMetrics]:
        # TODO: implement aiohttp call to self.base_url/metrics
        # Parse Prometheus text format → extract key metrics
        # Return RuntimeMetrics
        raise NotImplementedError("PrometheusMetricsSource not yet wired up")


class MockMetricsSource(MetricsSource):
    """
    Mock metrics source for testing.
    Returns synthetic metrics based on the placement decision's predictions.
    Adds configurable noise to simulate real-world variance.
    """

    def __init__(self, base_tpot: float = 27.0, noise_pct: float = 0.10):
        self.base_tpot = base_tpot
        self.noise_pct = noise_pct
        self._call_count = 0

    async def fetch(self, job_id: str) -> Optional[RuntimeMetrics]:
        import random
        self._call_count += 1
        noise = 1 + random.uniform(-self.noise_pct, self.noise_pct)
        # Simulate occasional spike at call 10-15
        if 10 <= self._call_count <= 15:
            noise *= 1.25

        return RuntimeMetrics(
            job_id=job_id,
            timestamp=datetime.utcnow(),
            throughput_tokens_per_sec=1200 * noise,
            tpot_ms=self.base_tpot * noise,
            gpu_utilization_pct=65 * noise,
            gpu_memory_used_gb=39.0,
            concurrent_requests=20,
            queue_depth=2,
        )


# ---------------------------------------------------------------------------
# Main monitor
# ---------------------------------------------------------------------------

class KoiMonitor:
    """
    Monitors all running jobs and feeds deltas to the refinement engine.

    Usage:
        monitor = KoiMonitor(metrics_source=MockMetricsSource())
        await monitor.start_job(decision)
        # runs in background, calls on_action_needed when SLO at risk
        await monitor.stop_job(job_id)
        delta_record = monitor.compute_delta(job_id)
    """

    def __init__(
        self,
        metrics_source: Optional[MetricsSource] = None,
        poll_interval_seconds: float = 30.0,
        on_action_needed: Optional[Callable[[str, SLOState, JobMonitorState], None]] = None,
    ):
        self.metrics_source = metrics_source or MockMetricsSource()
        self.poll_interval = poll_interval_seconds
        self.on_action_needed = on_action_needed
        self._jobs: Dict[str, JobMonitorState] = {}
        self._tasks: Dict[str, asyncio.Task] = {}

    async def start_job(self, decision: PlacementDecision) -> None:
        """Begin monitoring a newly deployed job."""
        predicted = decision.predicted_metrics
        state = JobMonitorState(
            job_id=decision.job_id,
            decision=decision,
        )

        # Initialize Kalman filters with predicted values as initial state
        if predicted.tpot_ms:
            state.kf_tpot = KalmanFilter1D(predicted.tpot_ms, R=30.0, Q=2.0)
            state.deadband_tpot = DeadbandController(
                slo_value=predicted.tpot_ms * 1.5,  # placeholder; real SLO set by job
            )
        state.kf_throughput = KalmanFilter1D(
            predicted.throughput_tokens_per_sec, R=10000.0, Q=500.0
        )

        self._jobs[decision.job_id] = state

        # Start polling loop in background
        task = asyncio.create_task(self._poll_loop(decision.job_id))
        self._tasks[decision.job_id] = task
        print(f"[Monitor] Started monitoring job {decision.job_id}")

    async def stop_job(self, job_id: str) -> Optional[JobMonitorState]:
        """Stop monitoring a job (call when job completes or is cancelled)."""
        if job_id in self._tasks:
            self._tasks[job_id].cancel()
            del self._tasks[job_id]
        return self._jobs.pop(job_id, None)

    def freeze_job(self, job_id: str, duration_seconds: float = 300.0) -> None:
        """Anti-windup: suppress corrections during a reconfiguration."""
        if job_id in self._jobs:
            self._jobs[job_id].freeze(duration_seconds)
            print(f"[Monitor] Anti-windup: froze corrections for {job_id} for {duration_seconds}s")

    def get_state(self, job_id: str) -> Optional[JobMonitorState]:
        return self._jobs.get(job_id)

    async def _poll_loop(self, job_id: str) -> None:
        """Background polling loop for a single job."""
        while True:
            try:
                await asyncio.sleep(self.poll_interval)
                state = self._jobs.get(job_id)
                if not state:
                    break

                metrics = await self.metrics_source.fetch(job_id)
                if not metrics:
                    continue

                state.raw_metrics_history.append(metrics)
                state.last_poll_time = time.time()

                # Apply Kalman filters
                if state.kf_tpot and metrics.tpot_ms:
                    filtered_tpot = state.kf_tpot.update(metrics.tpot_ms)
                    state.current_tpot_ms = filtered_tpot

                    # Check deadband (only if not frozen by anti-windup)
                    if not state.is_frozen() and state.deadband_tpot:
                        new_slo_state = state.deadband_tpot.update(filtered_tpot)
                        if new_slo_state != state.slo_state:
                            print(
                                f"[Monitor] Job {job_id}: SLO state "
                                f"{state.slo_state} → {new_slo_state} "
                                f"(filtered TPOT={filtered_tpot:.1f}ms)"
                            )
                            state.slo_state = new_slo_state
                            if new_slo_state in (SLOState.YELLOW_HIGH, SLOState.RED):
                                if self.on_action_needed:
                                    self.on_action_needed(job_id, new_slo_state, state)

                if state.kf_throughput:
                    state.current_throughput_tps = state.kf_throughput.update(
                        metrics.throughput_tokens_per_sec
                    )

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Monitor] Error polling job {job_id}: {e}")

    def compute_delta(self, job_id: str) -> Optional[DeltaRecord]:
        """
        Compute prediction vs actual delta for a completed or running job.
        Call this when a job completes to feed the refinement engine.
        """
        state = self._jobs.get(job_id)
        if not state or not state.raw_metrics_history:
            return None

        decision = state.decision
        predicted = decision.predicted_metrics

        # Compute actual averages from history (skip first few noisy readings)
        history = state.raw_metrics_history[2:]  # skip warmup
        if not history:
            history = state.raw_metrics_history

        avg_throughput = sum(m.throughput_tokens_per_sec for m in history) / len(history)
        avg_tpot = (
            sum(m.tpot_ms for m in history if m.tpot_ms) /
            max(1, sum(1 for m in history if m.tpot_ms))
        ) if any(m.tpot_ms for m in history) else None

        delta_throughput_pct = (
            (avg_throughput - predicted.throughput_tokens_per_sec) /
            max(predicted.throughput_tokens_per_sec, 1) * 100
        )
        delta_tpot = (
            (avg_tpot - predicted.tpot_ms) if (avg_tpot and predicted.tpot_ms) else None
        )

        cfg = decision.recommendation
        return DeltaRecord(
            vpc_id="unknown",  # set by caller from resource map
            job_id=job_id,
            model_name=decision.model_name,
            gpu_type=cfg.gpu_type,
            tp=cfg.tp,
            pp=cfg.pp,
            dp=cfg.dp,
            avg_input_tokens=0,  # set by caller from job request
            avg_output_tokens=0,
            task_type="batch",
            predicted_throughput_tps=predicted.throughput_tokens_per_sec,
            actual_throughput_tps=avg_throughput,
            predicted_tpot_ms=predicted.tpot_ms,
            actual_tpot_ms=avg_tpot,
            delta_throughput_pct=delta_throughput_pct,
            delta_tpot_ms=delta_tpot,
            prediction_data_source=predicted.data_source.value,
        )
