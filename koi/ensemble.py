"""
koi/ensemble.py — Multi-LLM ensemble: 3 thinkers + 1 judge.

All three thinkers use claude-opus-4-6 but with different system-prompt personas
that bias them toward different optimization objectives. They run in parallel.
The judge (also claude-opus-4-6) synthesizes all proposals into a final structured decision.

Thinkers:
  - Sagan   : Cost minimizer — cheapest config that meets SLO
  - Turing  : SLO guardian  — reliable margin, no surprises
  - Hopper  : HW efficiency — maximize GPU utilization, avoid waste

Judge:
  - Koi     : Synthesizes all three, asks follow-up if needed, outputs final structured decision

Note on C-PMI: The original aws_magic.py used Qwen3-0.6B for C-PMI tie-breaking.
In Koi, the Claude judge replaces this. C-PMI can be added back as a fast
pre-filter if API cost becomes a concern (TODO: plug in open-source model).
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from anthropic import AsyncAnthropic

from koi.schemas import (
    JobRequest,
    OracleCandidate,
    PlacementConfig,
    PlacementDecision,
    PredictedMetrics,
    ResourceMap,
    TaskType,
    ThinkerProposal,
)

# ---------------------------------------------------------------------------
# Thinker persona definitions
# ---------------------------------------------------------------------------

THINKER_PERSONAS = {
    "Sagan": {
        "persona": "Cost optimizer — minimize total spend while meeting SLO",
        "system_prompt": """You are Sagan, an expert GPU infrastructure cost optimizer.

Your PRIMARY goal is to find the CHEAPEST configuration that meets the SLO.
Your philosophy: right-size everything. Never overprovision. GPU-hours wasted are money burned.

When evaluating options:
- Focus on total cost (runtime_hours × cost_per_hour)
- Prefer simpler configs (lower TP/PP) when cost is equal — less overhead
- Be skeptical of expensive GPU types when cheaper ones can do the job
- DP (data parallelism / replicas) scales linearly — only add replicas if needed for SLO
- Remember: the SLO is a constraint, not a target. Meeting it by 1% is as good as 50%.

You must return your response as valid JSON matching the schema provided.""",
    },
    "Turing": {
        "persona": "SLO guardian — reliable execution with safety margin",
        "system_prompt": """You are Turing, an expert in distributed ML systems reliability.

Your PRIMARY goal is to ensure the job meets its SLO with COMFORTABLE MARGIN (≥20% headroom).
Your philosophy: predictions are uncertain. Build in safety buffers. The cost of missing an SLO
vastly exceeds the cost of modest overprovisioning.

When evaluating options:
- Look for configs with >20% SLO headroom to absorb prediction errors and noisy neighbors
- Prefer configs with known-good empirical data (higher confidence) over analytical estimates
- Be cautious about PP > 1 (pipeline bubbles add variance) and high TP on slow interconnect
- Consider the uncertainty: if confidence is 0.4, treat the predicted metrics as optimistic
- A config that meets SLO by 50% is strictly better than one that meets it by 5%

You must return your response as valid JSON matching the schema provided.""",
    },
    "Hopper": {
        "persona": "Hardware efficiency expert — maximize GPU utilization",
        "system_prompt": """You are Hopper, an expert in GPU utilization and hardware efficiency.

Your PRIMARY goal is to maximize hardware utilization — minimize waste from pipeline bubbles,
communication overhead, and memory bandwidth underuse.
Your philosophy: a config where GPUs run at 40% utilization is fundamentally broken.
Find configs where the hardware is the bottleneck, not the software.

When evaluating options:
- Prefer TP configurations where TP divides model architecture cleanly (avoids padding waste)
- Be suspicious of high PP on small models (pipeline bubbles dominate)
- Check if the model is memory-bandwidth bound or compute bound and choose accordingly
- For large models (>70B): TP=4 or TP=8 on NVLink is ideal; PCIe TP=8 adds too much comm overhead
- Higher tokens/GPU/sec indicates better hardware fit
- The 'throughput_per_gpu' metric is your north star

You must return your response as valid JSON matching the schema provided.""",
    },
}

JUDGE_SYSTEM_PROMPT = """You are Koi, an expert LLM infrastructure placement judge.

You will receive:
1. A job request (model, workload, SLO, objective)
2. A list of feasible configurations from the Oracle (with predicted metrics)
3. Proposals from three independent advisors: Sagan (cost), Turing (reliability), Hopper (efficiency)

Your job is to synthesize these proposals into ONE final placement decision.

Reasoning process:
1. Identify where advisors agree — consensus is usually correct
2. Where they disagree, understand WHY (cost vs safety vs utilization trade-off)
3. Weight their reasoning by: job priority, SLO strictness, objective (cheapest/fastest/balanced)
4. If the objective is "cheapest": lean toward Sagan unless Turing raises a concrete risk
5. If the objective is "fastest": lean toward Turing's safety margin approach
6. If the objective is "balanced": synthesize all three

You must return your response as valid JSON matching the schema provided.
Be decisive. Do not hedge. One clear winner."""

# ---------------------------------------------------------------------------
# Response schemas (for structured output parsing)
# ---------------------------------------------------------------------------

THINKER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_candidate_idx": {
            "type": "integer",
            "description": "0-based index into the candidates list"
        },
        "reasoning": {
            "type": "string",
            "description": "Full chain-of-thought reasoning for this choice (3-6 sentences)"
        },
        "key_concerns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "1-3 specific concerns about this placement"
        },
        "confidence_in_choice": {
            "type": "number",
            "description": "0.0 to 1.0 — how confident are you in this choice?"
        }
    },
    "required": ["chosen_candidate_idx", "reasoning", "key_concerns", "confidence_in_choice"]
}

JUDGE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_candidate_idx": {
            "type": "integer",
            "description": "0-based index into the candidates list for the final recommendation"
        },
        "reasoning": {
            "type": "string",
            "description": "Synthesis of advisor proposals — why this config wins (4-8 sentences)"
        },
        "confidence": {
            "type": "number",
            "description": "Overall confidence in this placement decision (0.0 to 1.0)"
        },
        "advisor_agreement": {
            "type": "string",
            "enum": ["full", "majority", "split"],
            "description": "Did advisors agree? full=all same, majority=2/3, split=all different"
        }
    },
    "required": ["chosen_candidate_idx", "reasoning", "confidence", "advisor_agreement"]
}

# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_candidates_table(candidates: List[OracleCandidate], max_show: int = 15) -> str:
    """Format candidates as a numbered table for LLM context."""
    # Show only top N (already sorted cheapest first by Oracle)
    show = candidates[:max_show]
    lines = [f"Top {len(show)} feasible configurations (0-indexed, sorted cheapest first):\n"]

    for i, c in enumerate(show):
        cfg = c.config
        met = c.metrics
        slo_str = f"✓ +{c.slo_margin_pct:.0f}% headroom" if c.meets_slo else f"✗ {c.slo_margin_pct:.0f}% short"
        runtime_str = f"{met.estimated_runtime_hours:.2f}h" if met.estimated_runtime_hours else "N/A"
        cost_str = f"${met.total_cost_usd:.2f}" if met.total_cost_usd else f"${met.cost_per_hour_usd:.2f}/hr"
        tpot_str = f"{met.tpot_ms:.0f}ms" if met.tpot_ms else "N/A"

        lines.append(
            f"[{i:2d}] {cfg.gpu_type:8s} TP={cfg.tp} PP={cfg.pp} DP={cfg.dp} | "
            f"{cfg.num_gpus:3d} GPUs ({cfg.num_instances}x {cfg.instance_type}) | "
            f"TPS={met.throughput_tokens_per_sec:6.0f} | "
            f"TPS/GPU={met.throughput_per_gpu_tokens_per_sec:5.1f} | "
            f"TPOT={tpot_str:8s} | "
            f"Runtime={runtime_str:6s} | "
            f"Cost={cost_str:10s} | "
            f"Conf={met.confidence:.0%} ({met.data_source.value[:6]}) | "
            f"SLO: {slo_str}"
        )

    return "\n".join(lines)


def _build_job_context(request: JobRequest, resource_map: ResourceMap) -> str:
    """Summarize the job request for LLM context."""
    available = [(r.gpu_type, r.available_gpus) for r in resource_map.resources if r.available_gpus > 0]
    lines = [
        f"JOB REQUEST:",
        f"  model         : {request.model_name}",
        f"  task_type     : {request.task_type.value}",
        f"  avg_input_len : {request.avg_input_tokens} tokens",
        f"  avg_output_len: {request.avg_output_tokens} tokens",
        f"  prefill/decode: {request.prefill_decode_ratio:.1f}x",
    ]
    if request.num_requests:
        lines.append(f"  num_requests  : {request.num_requests:,} rows")
        lines.append(f"  total_tokens  : {request.total_tokens:,}")
    if request.slo_deadline_hours:
        lines.append(f"  SLO deadline  : {request.slo_deadline_hours}h")
    if request.slo_tpot_ms:
        lines.append(f"  SLO TPOT      : {request.slo_tpot_ms}ms")
    if request.slo_ttft_ms:
        lines.append(f"  SLO TTFT      : {request.slo_ttft_ms}ms")
    lines += [
        f"  objective     : {request.objective.value}",
        f"",
        f"AVAILABLE RESOURCES (VPC {resource_map.vpc_id}, region {resource_map.region}):",
    ]
    for gpu_type, avail in available:
        res = resource_map.get_resource(gpu_type)
        if res:
            lines.append(
                f"  {gpu_type:8s}: {avail:3d} available GPUs "
                f"(${res.cost_per_gpu_hour_usd:.2f}/GPU/hr, {res.gpu_memory_gb}GB VRAM, "
                f"{res.interconnect})"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single thinker call
# ---------------------------------------------------------------------------

async def _call_thinker(
    client: AsyncAnthropic,
    thinker_name: str,
    persona_config: Dict[str, str],
    job_context: str,
    candidates_table: str,
    candidates: List[OracleCandidate],
    model: str = "claude-opus-4-6",
) -> Optional[ThinkerProposal]:
    """
    Call one thinker persona. Returns ThinkerProposal or None on failure.
    """
    user_prompt = f"""{job_context}

{candidates_table}

Based on your persona as {thinker_name} ({persona_config['persona']}),
choose the BEST configuration from the list above.

Return ONLY valid JSON with this exact structure:
{{
  "chosen_candidate_idx": <integer, 0-based index>,
  "reasoning": "<your detailed reasoning, 3-6 sentences>",
  "key_concerns": ["<concern 1>", "<concern 2>"],
  "confidence_in_choice": <float 0.0-1.0>
}}"""

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=512,
            system=persona_config["system_prompt"],
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = response.content[0].text.strip()

        # Parse JSON — strip markdown code fences if present
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        data = json.loads(content)
        idx = int(data["chosen_candidate_idx"])

        # Clamp idx to valid range
        idx = max(0, min(idx, len(candidates) - 1))
        chosen = candidates[idx]

        return ThinkerProposal(
            thinker_name=thinker_name,
            thinker_persona=persona_config["persona"],
            chosen_candidate_idx=idx,
            config=chosen.config,
            metrics=chosen.metrics,
            reasoning=data.get("reasoning", ""),
            key_concerns=data.get("key_concerns", []),
            confidence_in_choice=float(data.get("confidence_in_choice", 0.7)),
        )

    except Exception as e:
        print(f"[Ensemble] {thinker_name} failed: {e}")
        # Fallback: pick first SLO-meeting candidate
        fallback_idx = next(
            (i for i, c in enumerate(candidates) if c.meets_slo), 0
        )
        fallback = candidates[fallback_idx]
        return ThinkerProposal(
            thinker_name=thinker_name,
            thinker_persona=persona_config["persona"],
            chosen_candidate_idx=fallback_idx,
            config=fallback.config,
            metrics=fallback.metrics,
            reasoning=f"Fallback due to API error: {e}. Chose cheapest SLO-meeting config.",
            key_concerns=["API call failed — using fallback"],
            confidence_in_choice=0.3,
        )


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------

async def _call_judge(
    client: AsyncAnthropic,
    proposals: List[ThinkerProposal],
    job_context: str,
    candidates_table: str,
    candidates: List[OracleCandidate],
    model: str = "claude-opus-4-6",
) -> Tuple[int, str, float, str]:
    """
    Judge synthesizes the three proposals.
    Returns (chosen_idx, reasoning, confidence, advisor_agreement).
    """
    proposals_text = ""
    for p in proposals:
        proposals_text += (
            f"\n{p.thinker_name} ({p.thinker_persona}):\n"
            f"  Chose: config [{p.chosen_candidate_idx}] — "
            f"{p.config.gpu_type} TP={p.config.tp} PP={p.config.pp} DP={p.config.dp} "
            f"({p.config.num_gpus} GPUs)\n"
            f"  Reasoning: {p.reasoning}\n"
            f"  Concerns: {', '.join(p.key_concerns)}\n"
            f"  Confidence: {p.confidence_in_choice:.0%}\n"
        )

    user_prompt = f"""{job_context}

{candidates_table}

ADVISOR PROPOSALS:
{proposals_text}

Synthesize these proposals and select the FINAL placement configuration.
Consider: job objective={proposals[0].config if proposals else 'unknown'}, SLO requirements, advisor disagreements.

Return ONLY valid JSON:
{{
  "chosen_candidate_idx": <integer, 0-based index>,
  "reasoning": "<synthesis reasoning, 4-8 sentences>",
  "confidence": <float 0.0-1.0>,
  "advisor_agreement": "<full|majority|split>"
}}"""

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=768,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = response.content[0].text.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        data = json.loads(content)
        idx = max(0, min(int(data["chosen_candidate_idx"]), len(candidates) - 1))
        return (
            idx,
            data.get("reasoning", "Judge selected this configuration."),
            float(data.get("confidence", 0.7)),
            data.get("advisor_agreement", "majority"),
        )

    except Exception as e:
        print(f"[Ensemble] Judge failed: {e}")
        # Fallback: pick majority vote among thinkers
        votes: Dict[int, int] = {}
        for p in proposals:
            votes[p.chosen_candidate_idx] = votes.get(p.chosen_candidate_idx, 0) + 1
        best_idx = max(votes, key=lambda k: votes[k])
        return (
            best_idx,
            f"Judge fallback due to error: {e}. Using majority vote from thinkers.",
            0.5,
            "majority",
        )


# ---------------------------------------------------------------------------
# Main ensemble entry point
# ---------------------------------------------------------------------------

class KoiEnsemble:
    """
    Runs the full 3-thinker + judge pipeline.

    Usage:
        ensemble = KoiEnsemble(api_key="sk-ant-...")
        decision = await ensemble.run(request, resource_map, candidates)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-6",
        max_candidates_to_show: int = 15,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_candidates_to_show = max_candidates_to_show

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

    async def run(
        self,
        request: JobRequest,
        resource_map: ResourceMap,
        candidates: List[OracleCandidate],
    ) -> Tuple[PlacementConfig, PredictedMetrics, str, float, List[ThinkerProposal]]:
        """
        Run 3 thinkers in parallel → judge → return final decision components.

        Returns:
            (config, metrics, reasoning, confidence, thinker_proposals)
        """
        if not candidates:
            raise ValueError("No feasible candidates — Oracle returned empty list")

        client = AsyncAnthropic(api_key=self.api_key)
        job_context = _build_job_context(request, resource_map)
        candidates_table = _build_candidates_table(candidates, self.max_candidates_to_show)

        print(f"[Ensemble] Running 3 thinkers in parallel on {len(candidates)} candidates...")

        # Run all three thinkers concurrently
        thinker_tasks = [
            _call_thinker(
                client, name, persona, job_context, candidates_table, candidates, self.model
            )
            for name, persona in THINKER_PERSONAS.items()
        ]
        proposals_raw = await asyncio.gather(*thinker_tasks)
        proposals = [p for p in proposals_raw if p is not None]

        # Log thinker choices
        for p in proposals:
            print(
                f"[Ensemble] {p.thinker_name:6s} → [{p.chosen_candidate_idx:2d}] "
                f"{p.config.gpu_type} TP={p.config.tp} PP={p.config.pp} DP={p.config.dp} "
                f"({p.config.num_gpus} GPUs, conf={p.confidence_in_choice:.0%})"
            )

        # Run judge
        print(f"[Ensemble] Running judge...")
        chosen_idx, reasoning, confidence, agreement = await _call_judge(
            client, proposals, job_context, candidates_table, candidates, self.model
        )

        chosen = candidates[chosen_idx]
        print(
            f"[Ensemble] Judge → [{chosen_idx}] {chosen.config.gpu_type} "
            f"TP={chosen.config.tp} PP={chosen.config.pp} DP={chosen.config.dp} "
            f"(conf={confidence:.0%}, advisors={agreement})"
        )

        return chosen.config, chosen.metrics, reasoning, confidence, proposals

    def run_sync(
        self,
        request: JobRequest,
        resource_map: ResourceMap,
        candidates: List[OracleCandidate],
    ) -> Tuple[PlacementConfig, PredictedMetrics, str, float, List[ThinkerProposal]]:
        """Synchronous wrapper for non-async callers."""
        return asyncio.run(self.run(request, resource_map, candidates))
