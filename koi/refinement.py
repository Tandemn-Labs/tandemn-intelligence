"""
koi/refinement.py — Evolutionary refinement engine.

This is what makes Koi get smarter over time. Three independent learning channels:

  Channel 1 — Per-VPC Delta Store (SQLite)
      Stores (config, workload, predicted, actual, delta) for every completed job.
      The Oracle uses this as a RAG corpus: similar past deltas → LLM correction estimate.
      Effect: Oracle predictions improve for this specific cluster's characteristics.

  Channel 2 — Policy Memory (ChromaDB vector store)
      Stores high-level job outcomes: "Qwen-72B batch on A100 TP=4, PES=0.87, stable"
      The LLM ensemble retrieves similar past decisions as few-shot examples.
      Effect: LLM thinkers make better first-shot proposals for known workload patterns.

  Channel 3 — PES Tracker (SQLite)
      Computes and stores PES = α×CER + β×PER + γ×SS for every completed job.
      Tracks the regret curve and efficiency frontier.
      Effect: system-level metric for measuring improvement over time.

The Policy Learner (evolutionary step):
      Periodically runs RAG correction: retrieves k similar delta records → asks Claude
      to reason about what systematic corrections to apply to Oracle predictions.
      This is the "in-context learning" step — deltas become the model's long-term memory.

Publishing angle (OpenEvolve/AlphaEvolve connection):
      Unlike traditional evolutionary systems (fixed fitness function, discrete mutations),
      Koi's fitness function (PES frontier) is itself learned and expands over time.
      The population is the set of (config, workload) → outcome mappings in policy memory.
      Mutations are LLM-proposed config changes guided by delta analysis.
      Selection is PES-based with the expanding frontier as reference.
      This is a meta-evolutionary system with a living fitness function.
"""

import json
import os
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from koi.schemas import (
    DeltaRecord,
    JobRequest,
    OracleCandidate,
    PESComponents,
    PlacementDecision,
    TaskType,
)


# ---------------------------------------------------------------------------
# PES computation
# ---------------------------------------------------------------------------

# Weights by task type (from structure_expl.txt)
PES_WEIGHTS = {
    TaskType.BATCH: {"alpha": 0.50, "beta": 0.35, "gamma": 0.15},
    TaskType.ONLINE: {"alpha": 0.40, "beta": 0.30, "gamma": 0.30},
}


def compute_pes(
    job_id: str,
    task_type: TaskType,
    # CER inputs
    actual_cost_usd: float,
    best_known_slo_meeting_cost_usd: Optional[float],
    slo_met: bool,
    # PER inputs
    actual_throughput_tps: float,
    roofline_peak_tps: float,
    # SS inputs
    time_in_final_config_hours: float,
    total_job_hours: float,
    num_reconfigurations: int = 0,
) -> PESComponents:
    """
    Compute the Placement Efficiency Score for a completed job.

    PES = α×CER + β×PER + γ×SS

    CER: did we find the cheapest SLO-meeting config? (0 if SLO missed)
    PER: of the GPU capability allocated, how much was used productively?
    SS:  how much of job time was spent in the final (good) config?
    """
    weights = PES_WEIGHTS.get(task_type, PES_WEIGHTS[TaskType.BATCH])

    # CER — Cost Efficiency Ratio
    if not slo_met or actual_cost_usd <= 0:
        cer = 0.0
    elif best_known_slo_meeting_cost_usd and best_known_slo_meeting_cost_usd > 0:
        cer = min(1.0, best_known_slo_meeting_cost_usd / actual_cost_usd)
    else:
        cer = 0.7  # no reference yet → moderate score (frontier is loose)

    # PER — Physical Efficiency Ratio
    if roofline_peak_tps > 0:
        per = min(1.0, actual_throughput_tps / roofline_peak_tps)
    else:
        per = 0.5  # unknown

    # SS — Stability Score
    if total_job_hours > 0:
        ss = min(1.0, time_in_final_config_hours / total_job_hours)
    else:
        ss = 1.0  # no reconfigs by definition

    composite = weights["alpha"] * cer + weights["beta"] * per + weights["gamma"] * ss

    return PESComponents(
        job_id=job_id,
        cer=cer,
        per=per,
        ss=ss,
        composite=composite,
        task_type=task_type.value,
        alpha=weights["alpha"],
        beta=weights["beta"],
        gamma=weights["gamma"],
        num_reconfigurations=num_reconfigurations,
        slo_violations_count=0 if slo_met else 1,
        total_job_hours=total_job_hours,
    )


# ---------------------------------------------------------------------------
# Delta Store (SQLite)
# ---------------------------------------------------------------------------

class DeltaStore:
    """
    Per-VPC SQLite store for prediction-vs-actual delta records.

    This is Channel 1: the RAG corpus for Oracle correction.
    Each row is one completed job's prediction error.

    Query pattern: find k most similar past runs and ask the LLM
    what correction to apply to the current Oracle prediction.
    """

    def __init__(self, db_path: str = "./data/delta_store.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deltas (
                    record_id TEXT PRIMARY KEY,
                    vpc_id TEXT,
                    job_id TEXT,
                    model_name TEXT,
                    gpu_type TEXT,
                    tp INTEGER,
                    pp INTEGER,
                    dp INTEGER,
                    avg_input_tokens INTEGER,
                    avg_output_tokens INTEGER,
                    task_type TEXT,
                    predicted_throughput_tps REAL,
                    actual_throughput_tps REAL,
                    predicted_tpot_ms REAL,
                    actual_tpot_ms REAL,
                    delta_throughput_pct REAL,
                    delta_tpot_ms REAL,
                    cluster_gpu_utilization_pct REAL,
                    prediction_data_source TEXT,
                    timestamp TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_gpu_tp_pp
                ON deltas (gpu_type, tp, pp, model_name)
            """)
            conn.commit()

    def insert(self, record: DeltaRecord) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO deltas VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                record.record_id, record.vpc_id, record.job_id,
                record.model_name, record.gpu_type,
                record.tp, record.pp, record.dp,
                record.avg_input_tokens, record.avg_output_tokens,
                record.task_type,
                record.predicted_throughput_tps, record.actual_throughput_tps,
                record.predicted_tpot_ms, record.actual_tpot_ms,
                record.delta_throughput_pct, record.delta_tpot_ms,
                record.cluster_gpu_utilization_pct,
                record.prediction_data_source,
                record.timestamp.isoformat(),
            ))
            conn.commit()

    def find_similar(
        self,
        gpu_type: str,
        tp: int,
        pp: int,
        model_name: Optional[str] = None,
        avg_input_tokens: Optional[int] = None,
        avg_output_tokens: Optional[int] = None,
        k: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve k most similar past delta records for RAG correction.
        Similarity: exact GPU+TP+PP, optionally same model, sorted by recency.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if model_name:
                rows = conn.execute("""
                    SELECT * FROM deltas
                    WHERE gpu_type = ? AND tp = ? AND pp = ?
                    AND (model_name = ? OR model_name LIKE ?)
                    ORDER BY timestamp DESC LIMIT ?
                """, (gpu_type, tp, pp, model_name, f"%{model_name.split('/')[-1][:10]}%", k)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM deltas
                    WHERE gpu_type = ? AND tp = ? AND pp = ?
                    ORDER BY timestamp DESC LIMIT ?
                """, (gpu_type, tp, pp, k)).fetchall()

        return [dict(row) for row in rows]

    def get_vpc_summary(self, vpc_id: Optional[str] = None) -> Dict[str, Any]:
        """Return aggregate stats for the VPC delta store."""
        with sqlite3.connect(self.db_path) as conn:
            where = "WHERE vpc_id = ?" if vpc_id else ""
            params = (vpc_id,) if vpc_id else ()
            row = conn.execute(f"""
                SELECT
                    COUNT(*) as total_records,
                    AVG(ABS(delta_throughput_pct)) as avg_abs_delta_throughput_pct,
                    AVG(ABS(delta_tpot_ms)) as avg_abs_delta_tpot_ms,
                    COUNT(DISTINCT model_name) as num_models,
                    COUNT(DISTINCT gpu_type) as num_gpu_types
                FROM deltas {where}
            """, params).fetchone()
        return dict(row) if row else {}


# ---------------------------------------------------------------------------
# Policy Memory (ChromaDB)
# ---------------------------------------------------------------------------

class PolicyMemory:
    """
    Vector store for job placement decisions and their outcomes.
    Used as few-shot context for the LLM ensemble.

    Each document = one completed job's story:
      "Model X, task Y, config Z was placed. Predicted TPOT 27ms, actual 29ms.
       PES=0.85. Key lesson: TP=4 on L40S PCIe adds ~2ms overhead vs NVLink estimate."

    At placement time: retrieve k most similar past stories → inject into LLM prompt.
    The LLM sees examples of what worked and what didn't for similar workloads.
    """

    def __init__(self, persist_dir: str = "./data/policy_memory"):
        self.persist_dir = persist_dir
        self._collection = None
        self._available = False
        self._records: List[Dict] = []  # in-memory fallback
        self._init()

    def _init(self) -> None:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = client.get_or_create_collection(
                name="koi_policy_memory",
                metadata={"description": "Koi placement decisions and outcomes"}
            )
            self._available = True
            print(f"[PolicyMemory] ChromaDB initialized at {self.persist_dir}")
        except ImportError:
            print("[PolicyMemory] ChromaDB not installed — using in-memory fallback (not persistent)")
        except Exception as e:
            print(f"[PolicyMemory] ChromaDB init failed: {e} — using in-memory fallback")

    def add_outcome(
        self,
        decision: PlacementDecision,
        pes: PESComponents,
        actual_metrics: Optional[Dict] = None,
        key_lesson: str = "",
    ) -> None:
        """
        Store a completed job's outcome in policy memory.
        The document is a natural-language summary that the LLM can reason over.
        """
        cfg = decision.recommendation
        pred = decision.predicted_metrics

        # Build natural language document
        doc = (
            f"Job {decision.job_id}: model={decision.model_name}, "
            f"placement={cfg.gpu_type} TP={cfg.tp} PP={cfg.pp} DP={cfg.dp} "
            f"({cfg.num_gpus} GPUs, {cfg.num_instances}x {cfg.instance_type}). "
            f"Predicted throughput={pred.throughput_tokens_per_sec:.0f} tok/s"
        )
        if actual_metrics:
            doc += f", actual={actual_metrics.get('throughput_tps', '?'):.0f} tok/s"
        doc += (
            f". PES={pes.composite:.2f} (CER={pes.cer:.2f} PER={pes.per:.2f} SS={pes.ss:.2f}). "
            f"Reconfigs={pes.num_reconfigurations}. "
        )
        if key_lesson:
            doc += f"Key lesson: {key_lesson}"

        metadata = {
            "job_id": decision.job_id,
            "model_name": decision.model_name,
            "gpu_type": cfg.gpu_type,
            "tp": cfg.tp,
            "pp": cfg.pp,
            "dp": cfg.dp,
            "pes_composite": pes.composite,
            "task_type": pes.task_type,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if self._available and self._collection is not None:
            self._collection.add(
                documents=[doc],
                metadatas=[metadata],
                ids=[decision.job_id],
            )
        else:
            self._records.append({"doc": doc, "metadata": metadata})

    def retrieve_similar(
        self,
        model_name: str,
        gpu_type: Optional[str] = None,
        task_type: Optional[str] = None,
        k: int = 5,
    ) -> List[str]:
        """
        Retrieve k most relevant past decisions as natural-language strings.
        Called by the ensemble to inject few-shot examples into LLM prompts.
        """
        query = f"model={model_name}"
        if gpu_type:
            query += f" gpu={gpu_type}"
        if task_type:
            query += f" task={task_type}"

        if self._available and self._collection is not None:
            try:
                results = self._collection.query(
                    query_texts=[query],
                    n_results=min(k, self._collection.count()),
                )
                return results["documents"][0] if results["documents"] else []
            except Exception:
                pass

        # In-memory fallback: simple keyword match
        relevant = [
            r["doc"] for r in self._records
            if model_name.split("/")[-1][:10].lower() in r["doc"].lower()
            or (gpu_type and gpu_type.lower() in r["doc"].lower())
        ]
        return relevant[:k]

    def count(self) -> int:
        if self._available and self._collection is not None:
            return self._collection.count()
        return len(self._records)


# ---------------------------------------------------------------------------
# Efficiency Frontier tracker
# ---------------------------------------------------------------------------

class EfficiencyFrontier:
    """
    Tracks the Pareto frontier of (cost, throughput) for each workload class.

    The frontier expands as the system discovers cheaper/better configs.
    CER denominator (best known SLO-meeting cost) is sourced from here.

    Workload class = (model_family, task_type, approx_input_len, approx_output_len)
    """

    def __init__(self, db_path: str = "./data/frontier.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS frontier (
                    workload_class TEXT,
                    gpu_type TEXT,
                    tp INTEGER, pp INTEGER, dp INTEGER,
                    cost_per_hour_usd REAL,
                    throughput_tps REAL,
                    slo_met INTEGER,
                    pes_composite REAL,
                    last_updated TEXT,
                    PRIMARY KEY (workload_class, gpu_type, tp, pp, dp)
                )
            """)
            conn.commit()

    def _workload_class(
        self, model_name: str, task_type: str, input_len: int, output_len: int
    ) -> str:
        """Bucket into workload class for frontier comparison."""
        family = model_name.split("/")[-1][:15]
        # Bucket I/O lengths to reduce fragmentation
        input_bucket = round(input_len / 512) * 512
        output_bucket = round(output_len / 256) * 256
        return f"{family}_{task_type}_{input_bucket}in_{output_bucket}out"

    def update(
        self,
        model_name: str,
        task_type: str,
        input_len: int,
        output_len: int,
        gpu_type: str,
        tp: int, pp: int, dp: int,
        cost_per_hour: float,
        throughput_tps: float,
        slo_met: bool,
        pes: float,
    ) -> None:
        wc = self._workload_class(model_name, task_type, input_len, output_len)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO frontier VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (wc, gpu_type, tp, pp, dp, cost_per_hour, throughput_tps,
                  1 if slo_met else 0, pes, datetime.utcnow().isoformat()))
            conn.commit()

    def get_best_known_cost(
        self,
        model_name: str,
        task_type: str,
        input_len: int,
        output_len: int,
    ) -> Optional[float]:
        """Return cheapest known SLO-meeting cost for this workload class (CER denominator)."""
        wc = self._workload_class(model_name, task_type, input_len, output_len)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("""
                SELECT MIN(cost_per_hour_usd) FROM frontier
                WHERE workload_class = ? AND slo_met = 1
            """, (wc,)).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def get_frontier_configs(self, workload_class: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM frontier WHERE workload_class = ? AND slo_met = 1
                ORDER BY cost_per_hour_usd ASC
            """, (workload_class,)).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Policy Learner (RAG-based LLM correction)
# ---------------------------------------------------------------------------

class PolicyLearner:
    """
    The evolutionary learning step — Channel 1+2 combined.

    Given a new Oracle query, retrieves similar past delta records and asks
    the LLM to reason about what correction to apply.

    This is the "in-context learning" mechanism:
    deltas accumulated in DeltaStore become the model's long-term memory.
    The LLM generalizes from these examples without any weight updates.

    Connection to OpenEvolve/AlphaEvolve:
    - Population: delta records in DeltaStore
    - Fitness signal: PES improvement over time
    - Evolution: LLM proposes corrections → system validates → adds to population
    - No fixed fitness function: frontier expands as better configs are discovered
    """

    def __init__(
        self,
        delta_store: DeltaStore,
        policy_memory: PolicyMemory,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-6",
    ):
        self.delta_store = delta_store
        self.policy_memory = policy_memory
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model

    def get_rag_correction_prompt(
        self,
        gpu_type: str,
        tp: int,
        pp: int,
        model_name: str,
        oracle_prediction_tps: float,
        oracle_prediction_tpot_ms: Optional[float],
        k: int = 6,
    ) -> Optional[str]:
        """
        Build the RAG correction context for injecting into the LLM ensemble prompt.
        Returns a natural-language correction hint, or None if no delta history.
        """
        similar_deltas = self.delta_store.find_similar(
            gpu_type=gpu_type, tp=tp, pp=pp,
            model_name=model_name, k=k,
        )
        if not similar_deltas:
            return None

        lines = [
            f"VPC DELTA HISTORY ({len(similar_deltas)} similar past runs on this cluster):",
        ]
        for d in similar_deltas:
            delta_sign = "+" if d["delta_throughput_pct"] > 0 else ""
            lines.append(
                f"  [{d['timestamp'][:10]}] {d['model_name'][:20]} "
                f"{d['gpu_type']} TP={d['tp']} PP={d['pp']}: "
                f"Oracle predicted {d['predicted_throughput_tps']:.0f} tok/s, "
                f"actual was {d['actual_throughput_tps']:.0f} tok/s "
                f"({delta_sign}{d['delta_throughput_pct']:.1f}% delta)"
            )
            if d.get("delta_tpot_ms"):
                delta_t = d["delta_tpot_ms"]
                lines.append(f"    TPOT: predicted {d.get('predicted_tpot_ms','?')} ms, "
                             f"actual was {d.get('actual_tpot_ms','?')} ms "
                             f"({'+'  if delta_t>0 else ''}{delta_t:.1f}ms delta)")

        lines.append(
            f"\nCurrent Oracle prediction: {oracle_prediction_tps:.0f} tok/s"
            + (f", {oracle_prediction_tpot_ms:.1f}ms TPOT" if oracle_prediction_tpot_ms else "")
        )
        lines.append(
            "Apply any learned correction from the history above when evaluating this prediction."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main refinement engine
# ---------------------------------------------------------------------------

class KoiRefinement:
    """
    Ties together DeltaStore, PolicyMemory, EfficiencyFrontier, and PolicyLearner.

    Called by KoiPlacement:
      - on placement: check policy memory for similar past decisions
      - on job completion: record delta, update PES, update frontier
    """

    def __init__(self, data_dir: str = "./data"):
        os.makedirs(data_dir, exist_ok=True)
        self.delta_store = DeltaStore(db_path=f"{data_dir}/delta_store.db")
        self.policy_memory = PolicyMemory(persist_dir=f"{data_dir}/policy_memory")
        self.frontier = EfficiencyFrontier(db_path=f"{data_dir}/frontier.db")
        self.learner = PolicyLearner(
            delta_store=self.delta_store,
            policy_memory=self.policy_memory,
        )
        print(f"[Refinement] Initialized. Policy memory: {self.policy_memory.count()} entries")

    def get_context_for_ensemble(
        self,
        request: JobRequest,
        gpu_type: str,
        tp: int,
        pp: int,
        oracle_tps: float,
        oracle_tpot: Optional[float] = None,
    ) -> Optional[str]:
        """
        Returns VPC delta history + policy memory examples as a string
        to inject into the LLM ensemble prompt.
        """
        # Get RAG correction hint from delta history
        correction_hint = self.learner.get_rag_correction_prompt(
            gpu_type=gpu_type, tp=tp, pp=pp,
            model_name=request.model_name,
            oracle_prediction_tps=oracle_tps,
            oracle_prediction_tpot_ms=oracle_tpot,
        )

        # Get similar past decisions from policy memory
        past_decisions = self.policy_memory.retrieve_similar(
            model_name=request.model_name,
            gpu_type=gpu_type,
            task_type=request.task_type.value,
            k=3,
        )

        parts = []
        if correction_hint:
            parts.append(correction_hint)
        if past_decisions:
            parts.append("\nSIMILAR PAST DECISIONS:")
            for d in past_decisions:
                parts.append(f"  {d}")

        return "\n".join(parts) if parts else None

    def record_completion(
        self,
        decision: PlacementDecision,
        request: JobRequest,
        delta_record: DeltaRecord,
        actual_throughput_tps: float,
        slo_met: bool,
        total_hours: float,
        time_in_final_config: float,
        roofline_peak_tps: float,
    ) -> PESComponents:
        """
        Called when a job completes. Records delta, computes PES, updates frontier.
        """
        # 1. Store delta
        delta_record.vpc_id = "unknown"  # caller should set this
        delta_record.avg_input_tokens = request.avg_input_tokens
        delta_record.avg_output_tokens = request.avg_output_tokens
        delta_record.task_type = request.task_type.value
        self.delta_store.insert(delta_record)

        # 2. Compute PES
        cfg = decision.recommendation
        best_cost = self.frontier.get_best_known_cost(
            request.model_name, request.task_type.value,
            request.avg_input_tokens, request.avg_output_tokens,
        )
        pes = compute_pes(
            job_id=decision.job_id,
            task_type=request.task_type,
            actual_cost_usd=decision.predicted_metrics.cost_per_hour_usd * total_hours,
            best_known_slo_meeting_cost_usd=(
                best_cost * total_hours if best_cost else None
            ),
            slo_met=slo_met,
            actual_throughput_tps=actual_throughput_tps,
            roofline_peak_tps=roofline_peak_tps,
            time_in_final_config_hours=time_in_final_config,
            total_job_hours=total_hours,
        )

        print(
            f"[Refinement] Job {decision.job_id} PES={pes.composite:.3f} "
            f"(CER={pes.cer:.2f} PER={pes.per:.2f} SS={pes.ss:.2f})"
        )

        # 3. Update efficiency frontier
        self.frontier.update(
            model_name=request.model_name,
            task_type=request.task_type.value,
            input_len=request.avg_input_tokens,
            output_len=request.avg_output_tokens,
            gpu_type=cfg.gpu_type,
            tp=cfg.tp, pp=cfg.pp, dp=cfg.dp,
            cost_per_hour=decision.predicted_metrics.cost_per_hour_usd,
            throughput_tps=actual_throughput_tps,
            slo_met=slo_met,
            pes=pes.composite,
        )

        # 4. Add to policy memory
        key_lesson = self._derive_lesson(delta_record, pes)
        self.policy_memory.add_outcome(
            decision=decision, pes=pes, key_lesson=key_lesson,
            actual_metrics={"throughput_tps": actual_throughput_tps},
        )

        return pes

    def _derive_lesson(self, delta: DeltaRecord, pes: PESComponents) -> str:
        """Generate a one-line lesson from the delta record for policy memory."""
        lessons = []
        if abs(delta.delta_throughput_pct) > 15:
            direction = "over" if delta.delta_throughput_pct > 0 else "under"
            lessons.append(
                f"Oracle {direction}predicted throughput by {abs(delta.delta_throughput_pct):.0f}% "
                f"for {delta.gpu_type} TP={delta.tp} PP={delta.pp}"
            )
        if delta.delta_tpot_ms and abs(delta.delta_tpot_ms) > 5:
            lessons.append(
                f"TPOT was {delta.delta_tpot_ms:+.0f}ms vs prediction"
            )
        if pes.composite > 0.85:
            lessons.append("Excellent placement — use as reference for similar jobs")
        elif pes.composite < 0.5:
            lessons.append("Poor placement — avoid this config for this workload class")
        return ". ".join(lessons) if lessons else "No significant anomalies"
