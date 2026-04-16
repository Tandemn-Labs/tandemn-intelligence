"""Model resolution for the demo simulator.

The demo prefers known registry entries, can use Hugging Face config metadata
when available, and still falls back to heuristics when it has to run offline.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from koi.tools.physics import (
    ModelFeatures as RawModelFeatures,
    _KNOWN_MODELS,
    _fetch_hf_config,
    _infer_from_name,
    get_model_features,
)


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    source: str
    dtype: str
    num_params_billions: float
    active_params_billions: float
    model_size_gb: float
    active_expert_ratio: float
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    num_kv_heads: int
    vocab_size: int
    is_moe: bool
    num_experts: int
    active_experts: int
    architecture_family: str


def _is_known_model(model_name: str) -> bool:
    lower = model_name.lower()
    return any(
        lower == key.lower() or lower in key.lower() or key.lower() in lower
        for key in _KNOWN_MODELS
    )


def _materialize_features(
    model_name: str,
    dtype: str = "fp16",
    overrides: Optional[Mapping[str, Any]] = None,
) -> tuple[RawModelFeatures, str]:
    if _is_known_model(model_name):
        base = get_model_features(model_name, dtype=dtype)
        source = "registry"
    else:
        if overrides:
            base = _infer_from_name(model_name, dtype)
            source = "heuristic"
        else:
            base = _fetch_hf_config(model_name, dtype=dtype)
            source = "huggingface" if base is not None else "heuristic"
            if base is None:
                base = _infer_from_name(model_name, dtype)

    if not overrides:
        return base, source

    payload = {
        "model_name": model_name,
        "num_params_billions": base.num_params_billions,
        "num_layers": base.num_layers,
        "hidden_dim": base.hidden_dim,
        "num_attention_heads": base.num_attention_heads,
        "num_kv_heads": base.num_kv_heads,
        "vocab_size": base.vocab_size,
        "is_moe": base.is_moe,
        "num_experts": base.num_experts,
        "active_experts": base.active_experts,
        "architecture_family": base.architecture_family,
        "dtype": base.dtype,
    }
    payload.update(dict(overrides))
    source = "override"
    return RawModelFeatures(**payload), source


def resolve_model_spec(
    model_name: str,
    *,
    dtype: str = "fp16",
    overrides: Optional[Mapping[str, Any]] = None,
) -> ModelSpec:
    """Resolve any model string into a demo-usable ModelSpec."""
    features, source = _materialize_features(
        model_name, dtype=dtype, overrides=overrides
    )
    active_expert_ratio = (
        features.active_experts / features.num_experts
        if features.is_moe and features.num_experts > 0
        else 1.0
    )
    active_params = features.num_params_billions * active_expert_ratio
    return ModelSpec(
        model_name=model_name,
        source=source,
        dtype=features.dtype,
        num_params_billions=features.num_params_billions,
        active_params_billions=active_params,
        model_size_gb=features.model_size_gb,
        active_expert_ratio=active_expert_ratio,
        num_layers=features.num_layers,
        hidden_dim=features.hidden_dim,
        num_attention_heads=features.num_attention_heads,
        num_kv_heads=features.num_kv_heads,
        vocab_size=features.vocab_size,
        is_moe=features.is_moe,
        num_experts=features.num_experts,
        active_experts=features.active_experts,
        architecture_family=features.architecture_family,
    )
