"""
koi/exploration.py — Active Exploration Loop.

Runs as a SEPARATE slow background process (every 30-60 minutes).
Completely independent from the placement loop (every request) and
monitor loop (every 1-5 minutes).

Purpose:
  The perf DB and delta store tell us what we KNOW. But there are regions
  of config space we've never tried. Exploration systematically probes
  these regions on suitable jobs (low-priority, relaxed SLO) to:
  1. Reduce prediction uncertainty in unknown config regions
  2. Discover new Pareto-efficient configs (expand the efficiency frontier)
  3. Validate that known-good configs are still optimal (world changes)

Strategy: UCB (Upper Confidence Bound) acquisition function
  exploration_score = predicted_PES + β × uncertainty
  β decays over time: aggressive early, conservative once landscape is mapped

Budget: typically 5-10% of decisions are exploratory
  Only activate on: low-priority jobs, generous SLO headroom, stable cluster load

Connection to OpenEvolve / AlphaEvolve:
  This is the "mutation" step in our evolutionary system.
  The Oracle + Delta Store is the "population" of known configs.
  UCB selects the most promising "mutation" (untried config region) to evaluate.
  PES is the fitness function (but it's LEARNED and EXPANDING — the key novelty).
  Unlike OpenEvolve's fixed fitness, Koi's frontier shifts as we discover better configs.
  This means exploration can never stop — the target is a moving optimum.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from koi.oracle import Oracle
from koi.refinement import DeltaStore, EfficiencyFrontier
from koi.schemas import JobRequest, OracleCandidate, PlacementConfig, TaskType


# ---------------------------------------------------------------------------
# Uncertainty estimation
# ---------------------------------------------------------------------------

def estimate_config_uncertainty(
    gpu_type: str,
    tp: int,
    pp: int,
    model_name: str,
    delta_store: DeltaStore,
) -> float:
    """
    Estimate prediction uncertainty for a config region.
    High uncertainty = unexplored, high exploration value.

    Returns a value in [0, 1] where 1 = completely unknown.
    """
    similar = delta_store.find_similar(gpu_type=gpu_type, tp=tp, pp=pp,
                                        model_name=model_name, k=10)
    if not similar:
        return 1.0  # never seen this config

    # Uncertainty decreases with more observations
    # Also increases if past delta variance was high (noisy/inconsistent)
    n = len(similar)
    if n >= 10:
        base_uncertainty = 0.1
    elif n >= 5:
        base_uncertainty = 0.3
    elif n >= 2:
        base_uncertainty = 0.5
    else:
        base_uncertainty = 0.8

    # If past deltas were large, increase uncertainty
    avg_abs_delta = sum(abs(d["delta_throughput_pct"]) for d in similar) / len(similar)
    noise_factor = min(1.0, avg_abs_delta / 20.0)  # 20% delta = max noise

    return min(1.0, base_uncertainty + noise_factor * 0.2)


# ---------------------------------------------------------------------------
# UCB acquisition function
# ---------------------------------------------------------------------------

@dataclass
class ExplorationCandidate:
    """A config region to explore, with UCB score."""
    config: PlacementConfig
    base_oracle_tps: float
    uncertainty: float
    ucb_score: float          # base_PES_estimate + β × uncertainty
    reason: str               # why this region is worth exploring


def compute_ucb_scores(
    candidates: List[OracleCandidate],
    delta_store: DeltaStore,
    beta: float = 0.5,
) -> List[ExplorationCandidate]:
    """
    Score all Oracle candidates by UCB acquisition function.
    Returns candidates sorted by UCB score (best exploration targets first).

    UCB score = estimated_PES + β × uncertainty

    β controls exploration/exploitation balance:
      β = 0.0: pure exploitation (always pick best predicted)
      β = 1.0: strong exploration (bias toward unknown regions)
      β decays over time as the landscape is mapped
    """
    results = []
    for c in candidates:
        cfg = c.config
        uncertainty = estimate_config_uncertainty(
            cfg.gpu_type, cfg.tp, cfg.pp,
            "unknown",  # model name not in config; approximate
            delta_store,
        )

        # Base PES estimate from confidence and SLO margin
        base_pes = c.metrics.confidence
        if c.slo_margin_pct and c.slo_margin_pct > 0:
            base_pes *= min(1.0, 0.5 + c.slo_margin_pct / 100)

        ucb = base_pes + beta * uncertainty

        reason = ""
        if uncertainty > 0.8:
            reason = f"Never explored: {cfg.gpu_type} TP={cfg.tp} PP={cfg.pp}"
        elif uncertainty > 0.5:
            reason = f"Few observations: {cfg.gpu_type} TP={cfg.tp} PP={cfg.pp}"
        elif base_pes > 0.85:
            reason = f"High predicted PES ({base_pes:.2f}), validating"

        results.append(ExplorationCandidate(
            config=cfg,
            base_oracle_tps=c.metrics.throughput_tokens_per_sec,
            uncertainty=uncertainty,
            ucb_score=ucb,
            reason=reason,
        ))

    results.sort(key=lambda x: x.ucb_score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Exploration budget manager
# ---------------------------------------------------------------------------

class ExplorationBudget:
    """
    Tracks exploration budget: what fraction of decisions can be exploratory.

    Budget starts high (10%) and decays as uncertainty drops.
    Resets when environment changes (new GPU type, new model, vLLM update).
    """

    def __init__(
        self,
        initial_budget_pct: float = 0.10,   # 10% of decisions exploratory
        min_budget_pct: float = 0.02,        # always at least 2%
        decay_per_decision: float = 0.001,   # slow decay
    ):
        self.budget_pct = initial_budget_pct
        self.min_budget_pct = min_budget_pct
        self.decay = decay_per_decision
        self.total_decisions = 0
        self.exploration_decisions = 0

    def should_explore(self, job_priority: int, slo_headroom_pct: float) -> bool:
        """
        Decide if the current job should be an exploration target.
        Only explore on: low-priority (≤5) and generous SLO headroom (≥50%).
        """
        if job_priority > 5:
            return False  # never explore on high-priority jobs
        if slo_headroom_pct < 50:
            return False  # don't risk SLO violations for exploration

        actual_pct = (
            self.exploration_decisions / max(self.total_decisions, 1)
        )
        return actual_pct < self.budget_pct

    def record_decision(self, was_exploration: bool) -> None:
        self.total_decisions += 1
        if was_exploration:
            self.exploration_decisions += 1
        # Decay budget
        self.budget_pct = max(
            self.min_budget_pct,
            self.budget_pct - self.decay
        )

    def reset_budget(self, reason: str = "") -> None:
        """Reset when environment changes (new hardware, new model family)."""
        self.budget_pct = 0.10
        print(f"[Exploration] Budget reset: {reason}")

    @property
    def current_utilization(self) -> float:
        return self.exploration_decisions / max(self.total_decisions, 1)


# ---------------------------------------------------------------------------
# Main exploration manager
# ---------------------------------------------------------------------------

class ExplorationManager:
    """
    Background exploration loop. Runs independently from placement decisions.

    Two modes:
      1. Passive: flag certain placements as "exploration" (use UCB-selected config)
      2. Active:  on a slow timer, identify the highest-value unexplored region
                  and recommend running a micro-benchmark there

    The key difference from normal placement:
      - Normal: pick the BEST KNOWN config for the job
      - Exploration: pick the config with HIGHEST UCB SCORE (may not be best)

    The job still has to succeed (SLO must be met), so exploration is safe.
    If the explored config fails SLO, fall back to the standard recommendation immediately.
    """

    def __init__(
        self,
        delta_store: DeltaStore,
        frontier: EfficiencyFrontier,
        exploration_interval_seconds: float = 3600.0,  # 1 hour between active scans
        beta_initial: float = 0.5,
    ):
        self.delta_store = delta_store
        self.frontier = frontier
        self.interval = exploration_interval_seconds
        self.beta = beta_initial
        self.budget = ExplorationBudget()
        self._last_scan: float = 0
        self._exploration_queue: List[ExplorationCandidate] = []

    def get_exploration_override(
        self,
        standard_candidates: List[OracleCandidate],
        job_priority: int,
        slo_headroom_pct: float,
    ) -> Optional[OracleCandidate]:
        """
        If this job qualifies for exploration, return a UCB-selected candidate
        instead of the cheapest/best one.

        Called by the placement pipeline before sending candidates to the LLM ensemble.
        Returns None if this job should NOT be an exploration target.
        """
        if not self.budget.should_explore(job_priority, slo_headroom_pct):
            return None

        ucb_candidates = compute_ucb_scores(
            standard_candidates, self.delta_store, beta=self.beta
        )

        # Find the highest UCB candidate that is NOT the standard top choice
        standard_top = standard_candidates[0] if standard_candidates else None
        for uc in ucb_candidates:
            if (not standard_top or uc.config.summary != standard_top.config.summary):
                if uc.uncertainty > 0.3:  # only explore genuinely uncertain regions
                    print(
                        f"[Exploration] UCB override: {uc.config.gpu_type} "
                        f"TP={uc.config.tp} PP={uc.config.pp} "
                        f"(uncertainty={uc.uncertainty:.2f}, UCB={uc.ucb_score:.2f})"
                        f" — {uc.reason}"
                    )
                    # Find this config in standard_candidates and return it
                    for c in standard_candidates:
                        if c.config.summary == uc.config.summary:
                            return c
                    break

        return None

    async def scan_loop(self) -> None:
        """
        Periodic background scan: identify highest-value exploration targets
        across ALL workload classes and log them for the next qualifying job.
        """
        while True:
            await asyncio.sleep(self.interval)
            try:
                self._run_scan()
            except Exception as e:
                print(f"[Exploration] Scan error: {e}")

    def _run_scan(self) -> None:
        self._last_scan = time.time()
        summary = self.delta_store.get_vpc_summary()
        n_records = summary.get("total_records", 0)

        print(
            f"[Exploration] Scan: {n_records} delta records in store. "
            f"Budget utilization: {self.budget.current_utilization:.1%}. "
            f"Beta: {self.beta:.2f}"
        )

        # Decay beta based on number of records (more data → less exploration)
        self.beta = max(0.05, 0.5 * (1.0 / (1 + n_records / 100)))
        print(f"[Exploration] Beta updated to {self.beta:.3f}")

    def decay_beta(self, n_observations: int) -> None:
        """Update exploration aggressiveness based on accumulated knowledge."""
        self.beta = max(0.05, 0.5 / (1 + n_observations / 50))
