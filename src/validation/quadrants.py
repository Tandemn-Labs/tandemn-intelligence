"""
Four-quadrant classifier combines trajectory validity with outcome accuracy.

Outcome accuracy alone cannot tell real mechanism success from lucky success.
CUSUM checks whether the mechanism's internal V trajectory actually matched the
prediction, while the outcome check verifies whether final performance was right.

Quadrants:
    Q1 = matched  + accurate    # reliable success
    Q2 = matched  + inaccurate  # sound mechanism, bad outcome
    Q3 = diverged + accurate    # lucky-arm, punish hard
    Q4 = diverged + inaccurate  # falsified, punish harder

These labels drive confidence updates for edges and mechanisms.
"""

from collections import Counter
from enum import Enum


class Quadrant(Enum):
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


class QuadrantValidator:
    def classify_quadrant(self, cusum_result, outcome_accuracy: bool) -> Quadrant:
        """
        Definition: Combine the CUSUM trajectory axis with the outcome-
                    accuracy axis into the four-quadrant label.
                        matched and accurate         -> Q1
                        matched and not accurate     -> Q2
                        not matched and accurate     -> Q3 (lucky-arm)
                        not matched and not accurate -> Q4 (falsified)
        Usage:      Validator.s2_validate per (job, rank) deployed in
                    [t-1, t]. Returned label is stored in EvidenceStore
                    and consumed by ConfidenceService for beta evidence updates.
        Inputs:
            cusum_result     : CusumResult enum or "matched"/"diverged" string
            outcome_accuracy : True iff y_hat is within tolerance of realized y
                               (see check_outcome_accuracy)
        Outputs: Quadrant
        """
        matched = self._is_matched(cusum_result)

        if matched and outcome_accuracy:
            return Quadrant.Q1
        if matched and not outcome_accuracy:
            return Quadrant.Q2
        if not matched and outcome_accuracy:
            return Quadrant.Q3
        return Quadrant.Q4

    def aggregate_quadrant_histogram(
        self,
        evidence_store,
        window: int,
    ) -> dict[Quadrant, int]:
        """
        Definition: Count Q1/Q2/Q3/Q4 occurrences in recent DECIDED rows.
                    Excludes ICP-undecided rows so the denominator reflects
                    statistically-supported labels only.
        Usage:      agent.phase_1 ingest; dashboards; building block for
                    Q1-rate / regret computations.
        Inputs:
            evidence_store : EvidenceStore exposing get_recently_decided(window)
            window         : tick count to look back
        Outputs: Dict[Quadrant -> int]  (zero-filled for absent quadrants)
        """
        rows = evidence_store.get_recently_decided(window)
        counts = Counter(r.quadrant for r in rows)
        return {q: counts.get(q, 0) for q in Quadrant}

    def check_outcome_accuracy(
        self,
        pred_y: dict[str, float],
        obs_y: dict[str, float],
        tolerance: float,
        typical_ranges: dict[str, float],
    ) -> bool:
        """
        Definition: Per-objective relative-error check.
                        accurate iff for all j present in BOTH pred_y and obs_y:
                            abs(y_hat_j - y_j) / range_j < tolerance.
                    Objectives absent from obs_y are skipped (e.g., latency
                    metrics on batch jobs).
        Usage:      Inner helper for classify_quadrant.
        Inputs:
            pred_y         : objective -> y_hat_j (from surrogate at deploy time)
            obs_y          : objective -> realized y_j (from telemetry)
            tolerance      : relative error threshold (typical 0.15)
            typical_ranges : objective -> range_j (per-objective scale)
        Outputs: bool
        Notes:
            typical_ranges is REQUIRED. Different objectives differ by
            orders of magnitude (cost ~ 1e-7, throughput ~ 1e4); a single
            absolute tolerance cannot serve both.
        """
        if not typical_ranges:
            raise ValueError("typical_ranges is required for accuracy check")

        for obj, y_hat in pred_y.items():
            y = obs_y.get(obj)
            if y is None or y_hat is None:
                continue
            r = max(typical_ranges.get(obj, 1.0), 1e-9)
            if abs(float(y_hat) - float(y)) / r > tolerance:
                return False
        return True

    @staticmethod
    def _is_matched(cusum_result) -> bool:
        """Normalize Enum or string to a matched-bool."""
        if hasattr(cusum_result, "value"):
            return cusum_result.value == "matched"
        return cusum_result == "matched"
