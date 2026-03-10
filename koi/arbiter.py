"""
koi/arbiter.py — Swap Arbiter + Global Multi-Job Scheduler.

Responsibilities:
  - Maintain global cluster state (all running jobs + their current resource allocations)
  - When a job enters YELLOW/RED SLO state, evaluate if borrowing resources from
    another job (donor) would help, and whether the donor can afford it
  - Compute Net Benefit Score (NBS) for every potential (victim, donor) pair
  - For non-trivial swaps, route through LLM review
  - Apply a fairness floor: no job can be reduced below its minimum SLO-meeting config

Multi-job scheduling principle:
  "Is Job A better suited on Job B's current resources, given current SLO states and cost?"

The swap decision is framed as an MPC-extended problem:
  J_swap = SLO_recovery_benefit - transition_cost - donor_SLO_degradation_risk - opportunity_cost

Resource Adder:
  Separate decision: "Should we acquire NEW resources (cloud autoscale) rather than
  rebalancing existing ones?"
  Uses newsvendor framing: cost_of_new_capacity vs E[cost_of_SLO_violations].
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from koi.schemas import (
    GPUResource,
    JobRequest,
    PlacementConfig,
    PlacementDecision,
    PredictedMetrics,
    ResourceMap,
)


# ---------------------------------------------------------------------------
# Cluster state
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    RUNNING = "running"
    SCALING = "scaling"       # autoscale in progress
    RECONFIGURING = "reconfiguring"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RunningJob:
    """A job currently deployed in the cluster."""
    job_id: str
    model_name: str
    request: JobRequest
    decision: PlacementDecision
    status: JobStatus = JobStatus.RUNNING
    start_time: float = field(default_factory=time.time)

    # Current SLO health
    current_tpot_ms: Optional[float] = None
    current_throughput_tps: Optional[float] = None
    slo_pressure: float = 0.0      # 0=GREEN, 1.0=AT_SLO, >1=VIOLATION

    # Performance vs prediction
    actual_tpot_ms: Optional[float] = None
    actual_throughput_tps: Optional[float] = None

    # Priority (higher = more important)
    priority: int = 5               # 1 (low) to 10 (critical)

    def headroom_pct(self) -> float:
        """How much SLO headroom does this job have? Positive = good."""
        slo = self.request.slo_tpot_ms
        if slo and self.current_tpot_ms:
            return (slo - self.current_tpot_ms) / slo * 100
        return 100.0  # assume fine if no SLO

    def is_donor_candidate(self) -> bool:
        """Can this job donate resources? Only if it has >30% headroom."""
        return self.headroom_pct() > 30 and self.status == JobStatus.RUNNING

    def gpu_count(self) -> int:
        return self.decision.recommendation.num_gpus


@dataclass
class SwapProposal:
    """A proposed resource swap between two jobs."""
    victim_job_id: str          # job that needs help (SLO at risk)
    donor_job_id: str           # job that can give up resources
    resource_delta: int          # number of GPUs to transfer
    new_victim_config: PlacementConfig
    new_donor_config: PlacementConfig

    # Scoring
    nbs: float                   # Net Benefit Score
    benefit_victim: float        # SLO improvement value
    cost_donor: float            # SLO degradation risk to donor
    transition_cost: float       # downtime cost during swap
    opportunity_cost: float      # value of GPUs for future jobs

    # Metadata
    requires_llm_review: bool = False
    llm_review_reason: str = ""


# ---------------------------------------------------------------------------
# Net Benefit Score computation
# ---------------------------------------------------------------------------

def compute_nbs(
    victim: RunningJob,
    donor: RunningJob,
    resource_delta: int,
    victim_new_tpot_ms: Optional[float],
    donor_new_tpot_ms: Optional[float],
    transition_time_minutes: float = 5.0,
    gpu_cost_per_hour: float = 4.68,
) -> float:
    """
    Compute Net Benefit Score for a proposed swap.

    NBS = benefit_victim - cost_donor - transition_cost - opportunity_cost

    Positive NBS = swap is beneficial.
    NBS > 1.0 = strongly beneficial.
    NBS < 0 = harmful, do not swap.
    """
    # --- Benefit to victim ---
    # How much does this improve victim's SLO compliance?
    victim_slo = victim.request.slo_tpot_ms or 35.0  # default
    current_pressure = (victim.current_tpot_ms or victim_slo) / victim_slo
    if victim_new_tpot_ms:
        new_pressure = victim_new_tpot_ms / victim_slo
    else:
        new_pressure = current_pressure * 0.7  # rough estimate: adding GPUs helps 30%

    slo_recovery = max(0, current_pressure - new_pressure)  # 0 to ~0.3 typically
    # Weight by job priority (high-priority jobs get more benefit)
    benefit_victim = slo_recovery * victim.priority * 2.0

    # --- Cost to donor ---
    donor_slo = donor.request.slo_tpot_ms or 35.0
    donor_current = (donor.current_tpot_ms or 20.0)  # assume GREEN
    if donor_new_tpot_ms:
        donor_new_pressure = donor_new_tpot_ms / donor_slo
    else:
        donor_new_pressure = donor_current / donor_slo * 1.2  # losing GPUs hurts ~20%

    donor_headroom_loss = max(0, donor_new_pressure - donor_current / donor_slo)
    cost_donor = donor_headroom_loss * donor.priority * 1.5

    # --- Transition cost ---
    # During swap: N minutes of degraded service
    # Cost proportional to time and priority of jobs affected
    hourly_swap_cost = gpu_cost_per_hour * resource_delta
    transition_cost = (transition_time_minutes / 60.0) * hourly_swap_cost * 0.5

    # --- Opportunity cost ---
    # Small: if resources are generally available, opportunity cost is low
    opportunity_cost = 0.1  # placeholder; scale with cluster utilization

    nbs = benefit_victim - cost_donor - transition_cost - opportunity_cost
    return nbs


# ---------------------------------------------------------------------------
# Resource Adder (newsvendor)
# ---------------------------------------------------------------------------

class ResourceAdder:
    """
    Decides whether to acquire NEW GPU resources (cloud autoscale) vs
    rebalancing existing ones.

    Newsvendor framing:
      Under-provision penalty: expected SLO violation cost over horizon
      Over-provision penalty: cost of unused capacity

    Optimal order quantity Q* satisfies:
      P(demand <= Q*) = SLO_violation_cost / (SLO_violation_cost + over_provision_cost)
    """

    def __init__(
        self,
        slo_violation_cost_per_hour: float = 50.0,  # business cost of SLO miss
        provision_lead_time_minutes: float = 5.0,    # cloud instance startup
    ):
        self.slo_violation_cost = slo_violation_cost_per_hour
        self.lead_time = provision_lead_time_minutes

    def should_add_capacity(
        self,
        current_tpot_ms: float,
        slo_tpot_ms: float,
        load_trend: float,              # tokens/min/min (acceleration)
        cost_per_gpu_hour: float,
        planning_horizon_hours: float = 1.0,
    ) -> Tuple[bool, str]:
        """
        Returns (should_add, reason).

        Uses simple threshold: if projected TPOT (accounting for load trend
        and provision lead time) will exceed SLO, provision now.
        """
        # Project TPOT at lead_time minutes from now
        # TPOT scales roughly proportionally with load
        lead_time_hours = self.lead_time / 60.0
        projected_load_ratio = 1 + (load_trend * lead_time_hours)
        projected_tpot = current_tpot_ms * projected_load_ratio

        if projected_tpot >= slo_tpot_ms * 0.90:
            # Will hit SLO by the time new capacity is ready
            reason = (
                f"Projected TPOT {projected_tpot:.1f}ms will approach SLO {slo_tpot_ms}ms "
                f"in {self.lead_time:.0f}min. Provisioning now."
            )
            return True, reason

        # Cost comparison over horizon
        expected_violation_cost = (
            max(0, projected_tpot - slo_tpot_ms) / slo_tpot_ms *
            self.slo_violation_cost * planning_horizon_hours
        )
        capacity_cost = cost_per_gpu_hour * planning_horizon_hours

        if expected_violation_cost > capacity_cost:
            reason = (
                f"Expected SLO violation cost ${expected_violation_cost:.2f} > "
                f"capacity cost ${capacity_cost:.2f} over {planning_horizon_hours}h horizon."
            )
            return True, reason

        return False, "No capacity addition needed"

    def should_remove_capacity(
        self,
        current_tpot_ms: float,
        slo_tpot_ms: float,
        current_cost_per_hour: float,
        reduced_cost_per_hour: float,
        load_trend: float,
    ) -> Tuple[bool, str]:
        """Scale down if we have >40% SLO headroom and load is not growing."""
        headroom = (slo_tpot_ms - current_tpot_ms) / slo_tpot_ms

        if headroom > 0.40 and load_trend <= 0:
            savings = current_cost_per_hour - reduced_cost_per_hour
            reason = (
                f"{headroom:.0%} SLO headroom and load not growing. "
                f"Scaling down saves ${savings:.2f}/hr."
            )
            return True, reason

        return False, "Capacity reduction not warranted"


# ---------------------------------------------------------------------------
# Main Swap Arbiter
# ---------------------------------------------------------------------------

class SwapArbiter:
    """
    Global multi-job scheduler that evaluates resource rebalancing.

    Usage:
        arbiter = SwapArbiter()
        arbiter.register_job(decision, request, priority=8)
        # ... when a job hits YELLOW/RED:
        proposals = arbiter.evaluate_swaps(victim_job_id)
        if proposals:
            best = proposals[0]
            arbiter.execute_swap(best)
    """

    def __init__(
        self,
        resource_map: ResourceMap,
        fairness_floor: float = 1.0,  # min SLO headroom fraction allowed after swap
    ):
        self.resource_map = resource_map
        self.fairness_floor = fairness_floor
        self.jobs: Dict[str, RunningJob] = {}
        self.resource_adder = ResourceAdder()

    def register_job(
        self,
        decision: PlacementDecision,
        request: JobRequest,
        priority: int = 5,
    ) -> None:
        """Register a newly deployed job."""
        job = RunningJob(
            job_id=decision.job_id,
            model_name=decision.model_name,
            request=request,
            decision=decision,
            priority=priority,
        )
        self.jobs[decision.job_id] = job
        print(f"[Arbiter] Registered job {decision.job_id} (priority={priority})")

    def update_metrics(
        self,
        job_id: str,
        tpot_ms: Optional[float] = None,
        throughput_tps: Optional[float] = None,
    ) -> None:
        """Update live metrics for a running job."""
        if job_id in self.jobs:
            job = self.jobs[job_id]
            if tpot_ms is not None:
                job.current_tpot_ms = tpot_ms
            if throughput_tps is not None:
                job.current_throughput_tps = throughput_tps
            slo = job.request.slo_tpot_ms
            if slo and tpot_ms:
                job.slo_pressure = tpot_ms / slo

    def evaluate_swaps(self, victim_job_id: str) -> List[SwapProposal]:
        """
        Find all beneficial swap options for a job that needs help.
        Returns list sorted by NBS descending (best first).
        """
        victim = self.jobs.get(victim_job_id)
        if not victim:
            return []

        proposals = []

        for donor_id, donor in self.jobs.items():
            if donor_id == victim_job_id:
                continue
            if not donor.is_donor_candidate():
                continue
            if donor.priority > victim.priority + 2:
                # Don't steal from significantly higher-priority jobs
                continue

            # Try giving 1 unit of resources from donor to victim
            # (1 unit = TP×PP GPUs for a single replica)
            gpu_per_replica = victim.decision.recommendation.tp * victim.decision.recommendation.pp
            if donor.gpu_count() <= gpu_per_replica:
                continue  # donor can't spare anything

            nbs = compute_nbs(
                victim=victim,
                donor=donor,
                resource_delta=gpu_per_replica,
                victim_new_tpot_ms=None,   # TODO: call Oracle to estimate
                donor_new_tpot_ms=None,    # TODO: call Oracle to estimate
                gpu_cost_per_hour=4.68,    # TODO: from resource map
            )

            if nbs > 0:
                proposal = SwapProposal(
                    victim_job_id=victim_job_id,
                    donor_job_id=donor_id,
                    resource_delta=gpu_per_replica,
                    new_victim_config=victim.decision.recommendation,  # placeholder
                    new_donor_config=donor.decision.recommendation,    # placeholder
                    nbs=nbs,
                    benefit_victim=0,     # TODO: fill from compute_nbs internals
                    cost_donor=0,
                    transition_cost=0,
                    opportunity_cost=0,
                    # High-priority swaps or cross-model swaps get LLM review
                    requires_llm_review=(
                        victim.priority >= 8 or donor.priority >= 8 or
                        victim.model_name != donor.model_name
                    ),
                    llm_review_reason=(
                        "High-priority job or cross-model swap — requires human reasoning"
                        if victim.priority >= 8 or donor.priority >= 8
                        else ""
                    ),
                )
                proposals.append(proposal)

        proposals.sort(key=lambda p: p.nbs, reverse=True)
        if proposals:
            print(f"[Arbiter] {len(proposals)} swap proposals for job {victim_job_id}, "
                  f"best NBS={proposals[0].nbs:.2f}")
        return proposals

    def deregister_job(self, job_id: str) -> Optional[RunningJob]:
        """Remove a completed job and free its resources."""
        job = self.jobs.pop(job_id, None)
        if job:
            print(f"[Arbiter] Deregistered job {job_id}")
        return job

    def cluster_summary(self) -> str:
        """Human-readable cluster state summary."""
        lines = [f"=== Cluster State ({len(self.jobs)} running jobs) ==="]
        for jid, job in self.jobs.items():
            headroom = job.headroom_pct()
            state = "GREEN" if headroom > 20 else ("YELLOW" if headroom > 0 else "RED")
            lines.append(
                f"  {jid}: {job.model_name[:20]:20s} | "
                f"{job.gpu_count():3d} GPUs | "
                f"TPOT={job.current_tpot_ms or '?':>6} ms | "
                f"SLO headroom={headroom:+.0f}% | {state}"
            )
        return "\n".join(lines)
