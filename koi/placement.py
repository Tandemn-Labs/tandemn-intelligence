"""
koi/placement.py — KoiPlacement: main orchestrator.

Pipeline:
  JobRequest + ResourceMap
    → Oracle.get_candidates()       prune infeasible, predict metrics, sort by cost
    → KoiEnsemble.run()             3 thinkers + judge
    → PlacementDecision             structured output for Tandem CLI

Phase 2 (not yet implemented, stubs present):
    → Monitor.start()               begin tracking runtime metrics
    → Refinement.record_placement() log decision to delta store
    → Exploration loop              separate slow background process
"""

import os
import time
from typing import Optional

from koi.ensemble import KoiEnsemble
from koi.oracle import Oracle
from koi.schemas import (
    JobRequest,
    OracleCandidate,
    PlacementDecision,
    ResourceMap,
)


class KoiPlacement:
    """
    Main entry point for the Koi placement system.

    Usage:
        koi = KoiPlacement(api_key="...", perfdb_path="./perfdb")
        decision = koi.decide(request, resource_map)
        print(decision.display_summary())
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        perfdb_path: str = "./perfdb",
        llm_model: str = "claude-opus-4-6",
        max_candidates_to_llm: int = 15,
        include_non_slo_candidates: bool = False,
    ):
        """
        Args:
            api_key:                   Anthropic API key (or set ANTHROPIC_API_KEY env var)
            perfdb_path:               Path to performance database directory
            llm_model:                 Claude model for all thinkers and judge
            max_candidates_to_llm:     How many top Oracle candidates to show each thinker
            include_non_slo_candidates: If True, pass SLO-violating configs to LLMs
                                        (they may pick them if SLO is None or soft)
        """
        self.oracle = Oracle(perfdb_path=perfdb_path)
        self.ensemble = KoiEnsemble(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY", ""),
            model=llm_model,
            max_candidates_to_show=max_candidates_to_llm,
        )
        self.include_non_slo_candidates = include_non_slo_candidates

    def decide(
        self,
        request: JobRequest,
        resource_map: ResourceMap,
    ) -> PlacementDecision:
        """
        Main synchronous entry point.

        1. Oracle prunes infeasible configs and predicts metrics for all candidates.
        2. LLM ensemble (3 thinkers + judge) selects the best.
        3. Returns structured PlacementDecision.
        """
        t0 = time.time()
        print(f"\n[Koi] Starting placement for job {request.job_id} — {request.model_name}")
        print(f"[Koi] Task: {request.task_type.value}, Objective: {request.objective.value}")

        # Step 1: Oracle
        t_oracle = time.time()
        all_candidates = self.oracle.get_candidates(request, resource_map)

        if not all_candidates:
            raise RuntimeError(
                f"[Koi] Oracle returned 0 feasible candidates for {request.model_name} "
                f"on available hardware. Check resource map and model constraints."
            )

        # Filter to SLO-meeting candidates for the LLM (unless overridden)
        slo_candidates = [c for c in all_candidates if c.meets_slo]
        if not slo_candidates:
            print(
                f"[Koi] Warning: no candidates meet SLO. "
                f"Showing all {len(all_candidates)} to the ensemble."
            )
            llm_candidates = all_candidates
        elif self.include_non_slo_candidates:
            llm_candidates = all_candidates  # let LLMs see everything
        else:
            llm_candidates = slo_candidates

        print(
            f"[Koi] Oracle: {len(all_candidates)} total candidates, "
            f"{len(slo_candidates)} meet SLO. "
            f"Oracle took {time.time() - t_oracle:.2f}s"
        )

        # Step 2: LLM Ensemble
        t_llm = time.time()
        config, metrics, reasoning, confidence, thinker_proposals = self.ensemble.run_sync(
            request, resource_map, llm_candidates
        )
        print(f"[Koi] Ensemble took {time.time() - t_llm:.2f}s")

        # Build alternatives (top 3 from Oracle, excluding the chosen one)
        chosen_summary = config.summary
        alternatives = [
            c for c in llm_candidates[:6]
            if c.config.summary != chosen_summary
        ][:3]

        decision = PlacementDecision(
            job_id=request.job_id,
            model_name=request.model_name,
            recommendation=config,
            predicted_metrics=metrics,
            reasoning=reasoning,
            confidence=confidence,
            thinker_proposals=thinker_proposals,
            alternatives=alternatives,
            oracle_candidates_evaluated=len(all_candidates),
            total_llm_calls=4,  # 3 thinkers + 1 judge
        )

        total_time = time.time() - t0
        print(f"[Koi] Total placement time: {total_time:.2f}s")
        return decision

    async def decide_async(
        self,
        request: JobRequest,
        resource_map: ResourceMap,
    ) -> PlacementDecision:
        """Async version for integration with async servers."""
        t0 = time.time()
        print(f"\n[Koi] Starting placement for job {request.job_id} — {request.model_name}")

        all_candidates = self.oracle.get_candidates(request, resource_map)
        if not all_candidates:
            raise RuntimeError(f"Oracle returned 0 feasible candidates.")

        slo_candidates = [c for c in all_candidates if c.meets_slo]
        llm_candidates = slo_candidates if slo_candidates else all_candidates

        config, metrics, reasoning, confidence, thinker_proposals = await self.ensemble.run(
            request, resource_map, llm_candidates
        )

        alternatives = [
            c for c in llm_candidates[:6]
            if c.config.summary != config.summary
        ][:3]

        return PlacementDecision(
            job_id=request.job_id,
            model_name=request.model_name,
            recommendation=config,
            predicted_metrics=metrics,
            reasoning=reasoning,
            confidence=confidence,
            thinker_proposals=thinker_proposals,
            alternatives=alternatives,
            oracle_candidates_evaluated=len(all_candidates),
            total_llm_calls=4,
        )
