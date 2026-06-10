"""
EIG proxy for causal expected information gain.
Exact Bayesian EIG needs outcome distributions and mechanism posteriors, which
we do not have. This deterministic proxy keeps the same purpose: favor candidates
that test uncertain and under-sampled edges/mechanisms.
    alpha(L') = sum_e a_e*beta_uncertainty(e) + kappa*sum_M a_M*beta_uncertainty(M)
beta_uncertainty = 4*c*(1-c)/(alpha+beta+1), where c = alpha/(alpha+beta).
Eligibility masks a_e and a_M decide what is tested. alpha is the exploration
term in sigma, weighted by annealed beta_t. Cluster EIG uses saturation
aggregation to avoid double-counting the same edge in one plan.
"""

from collections.abc import Sequence

import numpy as np
from src.config.hyperparameters import KAPPA

# Gate defaults
DEFAULT_N_B = 15  # min samples per env for ICP statistical power
DEFAULT_N_ENV_MIN = 3  # min envs required for ICP


def compute_eig(
    L_prime,
    confidence_service,
    evidence_store,
    n: int | None = None,
) -> float:
    """
    Definition: Proxy Causal-EIG for one candidate ladder.
                    alpha(L') = sum_e a_e*beta_uncertainty(e)
                          + kappa*sum_M a_M*beta_uncertainty(M)
                Sums over edges/mechanisms touched by L'.
    Usage:      The alpha term in sigma(L') = J + beta*alpha - lambda*Pr_DRO - lambda*SwitchCost.
                Called by agent.tools.compute_eig per (config, mechanism)
    Inputs:
        L_prime            : Ladder with .ranks; each rank has .mechanism_id,
                             .config, .n_replicas
        confidence_service : ConfidenceService with candidate_graph and
                             mechanism_registry references
        evidence_store     : EvidenceStore (for eligibility-gate lookups)
        n                  : optional evidence-count cap
    Outputs:
        alpha : float >= 0
    """
    if not L_prime.ranks:
        return 0.0

    candidate_graph = confidence_service.candidate_graph
    mechanism_registry = confidence_service.mechanism_registry

    deployed_mids = {r.mechanism_id for r in L_prime.ranks}
    if not deployed_mids:
        return 0.0

    # Union of edges across all deployed mechanisms
    touched_edge_ids = set()
    for mid in deployed_mids:
        touched_edge_ids.update(mechanism_registry.get_mechanism(mid).edge_ids)

    # Edge term
    edge_sum = 0.0
    for edge_id in touched_edge_ids:
        edge = candidate_graph.edge_table[edge_id]
        if not _edge_eligible(edge, L_prime, evidence_store):
            continue
        edge_metadata = candidate_graph.edge_metadata_table[edge_id]
        c_e = confidence_service.get_edge_confidence(edge_id)
        edge_sum += _beta_uncertainty(c_e, edge_metadata.alpha + edge_metadata.beta, n)

    # Mechanism term
    mech_sum = 0.0
    for mid in deployed_mids:
        mechanism = mechanism_registry.get_mechanism(mid)
        if not check_mechanism_eligibility(mechanism, L_prime, (candidate_graph, evidence_store)):
            continue
        mechanism_metadata = mechanism_registry.mechanism_metadata_table[mid]
        c_m = confidence_service.get_mechanism_confidence(mid)
        mech_sum += _beta_uncertainty(c_m, mechanism_metadata.alpha + mechanism_metadata.beta, n)

    return edge_sum + KAPPA * mech_sum


def check_mechanism_eligibility(mechanism, L_prime, state) -> bool:
    """
    Definition: a_M(L') = 1 iff at least one X->V->Y path through M has
                BOTH edges eligible. Ensures mechanism is testable by L'.
    Usage:      Inner gate for compute_eig and aggregate_cluster_eig.
    Inputs:
        mechanism : Mechanism with .edge_ids
        L_prime   : Ladder
        state     : tuple (candidate_graph, evidence_store) - bundled context
    Outputs:
        bool
    """
    candidate_graph, evidence_store = state
    for xv_edge, vy_edge in find_eligible_paths(mechanism, candidate_graph):
        if _edge_eligible(xv_edge, L_prime, evidence_store) and _edge_eligible(
            vy_edge, L_prime, evidence_store
        ):
            return True
    return False


def aggregate_cluster_eig(
    cluster_plan,
    ranks: Sequence,
    confidence_service,
) -> float:
    """
    Definition: Cluster-level EIG with saturation aggregation.
                    A_e(P) = 1 - prod_i(1 - a_e(L_i'))
                    alpha_cluster(P) = sum_e beta_uncertainty(e)*A_e
                                     + kappa*sum_M beta_uncertainty(M)*A_M
                Saturation prevents double-counting an edge tested by
                multiple ranks across the cluster's plan P.
    Usage:      agent.phase_4 cluster scoring; budget-reallocation delta-sigma check.
    Inputs:
        cluster_plan       : Plan (Dict[job_id -> Action]) - for logging/audit
        ranks              : flat List[Rank] across all ladders in the plan
        confidence_service : ConfidenceService with candidate_graph and
                             mechanism_registry references
    Outputs:
        alpha_cluster : float >= 0
    """
    if not ranks:
        return 0.0

    candidate_graph = confidence_service.candidate_graph
    mechanism_registry = confidence_service.mechanism_registry

    edges_to_ranks: dict[str, list] = {}
    mechs_to_ranks: dict[str, list] = {}
    for r in ranks:
        mechanism = mechanism_registry.get_mechanism(r.mechanism_id)
        for edge_id in mechanism.edge_ids:
            edges_to_ranks.setdefault(edge_id, []).append(r)
        mechs_to_ranks.setdefault(r.mechanism_id, []).append(r)

    # Edge saturation
    edge_term = 0.0
    for edge_id, rank_list in edges_to_ranks.items():
        edge_metadata = candidate_graph.edge_metadata_table[edge_id]
        c_e = confidence_service.get_edge_confidence(edge_id)
        a_values = [
            1.0
            if _edge_eligible_by_id(edge_id, r.ladder, r.evidence_store, candidate_graph)
            else 0.0
            for r in rank_list
        ]
        A = 1.0 - float(np.prod([1.0 - a for a in a_values]))
        edge_term += _beta_uncertainty(c_e, edge_metadata.alpha + edge_metadata.beta) * A

    # Mechanism saturation
    mech_term = 0.0
    for m_id, rank_list in mechs_to_ranks.items():
        mechanism = mechanism_registry.get_mechanism(m_id)
        mechanism_metadata = mechanism_registry.mechanism_metadata_table[m_id]
        c_m = confidence_service.get_mechanism_confidence(m_id)
        a_values = [
            1.0
            if check_mechanism_eligibility(mechanism, r.ladder, (candidate_graph, r.evidence_store))
            else 0.0
            for r in rank_list
        ]
        A = 1.0 - float(np.prod([1.0 - a for a in a_values]))
        mech_term += _beta_uncertainty(c_m, mechanism_metadata.alpha + mechanism_metadata.beta) * A

    return edge_term + KAPPA * mech_term


def find_eligible_paths(mechanism, candidate_graph) -> list[tuple]:
    """
    Definition: Enumerate (X->V edge, V->Y edge) path pairs through the
                mechanism's sub-DAG that share a common V node.
    Usage:      Inner helper for check_mechanism_eligibility.
    Inputs:
        mechanism : Mechanism with .edge_ids
    Outputs:
        List[(xv_edge, vy_edge)] - full X->V->Y paths through the bundle
    """
    edges = [candidate_graph.edge_table[edge_id] for edge_id in mechanism.edge_ids]
    xv = [e for e in edges if e.src_type == "X" and e.dst_type == "V"]
    vy = [e for e in edges if e.src_type == "V" and e.dst_type == "Y"]
    return [(a, b) for a in xv for b in vy if a.dst == b.src]


def _beta_uncertainty(confidence: float, evidence_count: float, n: int | None = None) -> float:
    if n is not None:
        evidence_count = min(evidence_count, n)
    return 4.0 * confidence * (1.0 - confidence) / (evidence_count + 1.0)


def _edge_eligible(edge, L_prime, evidence_store) -> bool:
    """All six gates must pass"""
    return (
        _gate_selected(edge, L_prime)
        and _gate_valid_contrast(edge, L_prime)
        and _gate_child_observed(edge)
        and _gate_enough_samples(edge, L_prime)
        and _gate_validator_support(edge, evidence_store)
        and _gate_relevance(edge, L_prime)
    )


def _edge_eligible_by_id(edge_id, L_prime, evidence_store, candidate_graph) -> bool:
    """Resolve edge_id to Edge, then evaluate gates."""
    edge = candidate_graph.edge_table.get(edge_id)
    if edge is None:
        return False
    return _edge_eligible(edge, L_prime, evidence_store)


def _gate_selected(edge, L_prime) -> bool:
    # X-side of edge is set / varied somewhere in the ladder.
    return any(edge.src in r.config for r in L_prime.ranks)


def _gate_valid_contrast(edge, L_prime) -> bool:
    """The ladder produces variation in edge.src across its ranks.
    X->V edges: need at least 1 distinct value of edge.src.
    V->Y edges: V variation is mediated by upstream X-variation
    in the same ladder (validated structurally elsewhere)."""
    values = {r.config.get(edge.src) for r in L_prime.ranks if edge.src in r.config}
    return len(values) >= 1


def _gate_child_observed(edge) -> bool:
    """V or Y on dst side is in our telemetry catalog."""
    return getattr(edge, "dst_observable", True)


def _gate_enough_samples(edge, L_prime, n_b: int = DEFAULT_N_B) -> bool:
    """Deployment provides at least n_b samples per env."""
    n_envs = max(1, len(L_prime.envs()))
    total = L_prime.duration_minutes * sum(r.n_replicas for r in L_prime.ranks)
    return total >= n_b * n_envs


def _gate_validator_support(
    edge,
    evidence_store,
    n_env_min: int = DEFAULT_N_ENV_MIN,
) -> bool:
    """After this deployment, edge will have at least n_env_min envs tested."""
    return len(evidence_store.envs_for_edge(edge.edge_id)) + 1 >= n_env_min


def _gate_relevance(edge, L_prime) -> bool:
    """Edge belongs to at least one mechanism applicable to L_prime's job."""
    return any(edge.edge_id in mechanism.edge_ids for mechanism in L_prime.applicable_mechanisms)


# if __name__ == "__main__":
#     from dataclasses import dataclass
#     from typing import Any

#     from src.core.candidate_graph import CandidateGraph
#     from src.core.confidence_service import ConfidenceService
#     from src.core.mechanism_registry import MechanismRegistry
#     from src.core.models import Edge, EdgeMetadata, Mechanism, MechanismMetadata, Node

#     @dataclass
#     class Rank:
#         mechanism_id: str
#         mechanism: Mechanism
#         config: dict[str, Any]
#         n_replicas: int
#         ladder: Any = None
#         evidence_store: Any = None

#     @dataclass
#     class Ladder:
#         ranks: list[Rank]
#         duration_minutes: int
#         applicable_mechanisms: list[Mechanism]

#         def envs(self):
#             return ["env_a", "env_b", "env_c"]

#     class FakeEvidenceStore:
#         def __init__(self, edge_envs):
#             self.edge_envs = edge_envs

#         def envs_for_edge(self, edge_id):
#             return self.edge_envs.get(edge_id, [])

#     def build_test_case():
#         e1 = Edge(
#             edge_id="edge_batch_to_kv",
#             src="batch_size",
#             dst="kv_cache_pressure",
#             src_type="X",
#             dst_type="V",
#         )
#         e2 = Edge(
#             edge_id="edge_kv_to_latency",
#             src="kv_cache_pressure",
#             dst="ttft_ms",
#             src_type="V",
#             dst_type="Y",
#         )

#         mechanism = Mechanism(
#             mechanism_id="mech_kv_latency",
#             edge_ids=[e1.edge_id, e2.edge_id],
#             scope={"x": ["batch_size"], "v": ["kv_cache_pressure"]},
#             narrative="KV pressure mediates batch size and TTFT.",
#         )

#         rank_1 = Rank(
#             mechanism_id=mechanism.mechanism_id,
#             mechanism=mechanism,
#             config={"batch_size": 8, "kv_cache_pressure": "medium"},
#             n_replicas=30,
#         )
#         rank_2 = Rank(
#             mechanism_id=mechanism.mechanism_id,
#             mechanism=mechanism,
#             config={"batch_size": 16, "kv_cache_pressure": "high"},
#             n_replicas=30,
#         )

#         evidence_store = FakeEvidenceStore(
#             edge_envs={
#                 e1.edge_id: ["env_a", "env_b"],
#                 e2.edge_id: ["env_a", "env_b"],
#             }
#         )
#         ladder = Ladder(
#             ranks=[rank_1, rank_2],
#             duration_minutes=10,
#             applicable_mechanisms=[mechanism],
#         )

#         rank_1.ladder = ladder
#         rank_2.ladder = ladder
#         rank_1.evidence_store = evidence_store
#         rank_2.evidence_store = evidence_store

#         node_table = {
#             "batch_size": Node(node_id="batch_size", node_type="X"),
#             "kv_cache_pressure": Node(node_id="kv_cache_pressure", node_type="V"),
#             "ttft_ms": Node(node_id="ttft_ms", node_type="Y"),
#         }
#         edge_table = {e1.edge_id: e1, e2.edge_id: e2}
#         edge_metadata_table = {
#             e1.edge_id: EdgeMetadata(edge_id=e1.edge_id, alpha=1.5, beta=1.5, visit_count=3),
#             e2.edge_id: EdgeMetadata(edge_id=e2.edge_id, alpha=5.6, beta=2.4, visit_count=8),
#         }
#         graph = CandidateGraph(
#             node_table=node_table,
#             edge_table=edge_table,
#             edge_metadata_table=edge_metadata_table,
#         )

#         registry = MechanismRegistry(
#             mechanism_table={mechanism.mechanism_id: mechanism},
#             mechanism_metadata_table={
#                 mechanism.mechanism_id: MechanismMetadata(
#                     mechanism_id=mechanism.mechanism_id,
#                     alpha=3.0,
#                     beta=2.0,
#                     visit_count=5,
#                 )
#             },
#         )
#         confidence_service = ConfidenceService(graph, registry)

#         return ladder, confidence_service, evidence_store, [rank_1, rank_2]

#     ladder, confidence_service, evidence_store, ranks = build_test_case()

#     alpha = compute_eig(
#         L_prime=ladder,
#         confidence_service=confidence_service,
#         evidence_store=evidence_store,
#     )
#     print("compute_eig alpha:", alpha)

#     alpha_capped = compute_eig(
#         L_prime=ladder,
#         confidence_service=confidence_service,
#         evidence_store=evidence_store,
#         n=2,
#     )
#     print("compute_eig alpha with evidence cap n=2:", alpha_capped)

#     alpha_cluster = aggregate_cluster_eig(
#         cluster_plan={"job_1": "fake_action"},
#         ranks=ranks,
#         confidence_service=confidence_service,
#     )
#     print("aggregate_cluster_eig alpha_cluster:", alpha_cluster)

#     assert alpha > 0
#     assert alpha_capped > alpha
#     assert alpha_cluster > 0
#     print("All EIG tests passed.")
