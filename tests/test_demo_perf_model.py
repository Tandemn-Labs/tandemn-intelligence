"""Tests for the demo-only model registry and performance model."""

import simulation.model_registry as model_registry
from koi.tools.physics import ModelFeatures as PhysicsModelFeatures
from simulation.model_registry import resolve_model_spec
from simulation.perf_model import DemoPerfModel


class TestModelRegistry:
    def test_known_model_resolves_from_registry(self):
        spec = resolve_model_spec("Qwen/Qwen3-32B")
        assert spec.source == "registry"
        assert spec.num_params_billions == 32
        assert spec.architecture_family == "qwen"
        assert spec.active_params_billions == 32

    def test_unknown_model_with_overrides_is_supported(self, monkeypatch):
        def _should_not_fetch(*args, **kwargs):
            raise AssertionError("override path should not hit Hugging Face")

        monkeypatch.setattr(model_registry, "_fetch_hf_config", _should_not_fetch)
        spec = resolve_model_spec(
            "acme/Custom-13B-Instruct",
            overrides={
                "num_params_billions": 13,
                "num_layers": 40,
                "hidden_dim": 5120,
                "num_attention_heads": 40,
                "num_kv_heads": 8,
                "vocab_size": 64000,
                "architecture_family": "custom",
            },
        )
        assert spec.source == "override"
        assert spec.num_params_billions == 13
        assert spec.num_layers == 40
        assert spec.architecture_family == "custom"
        assert spec.model_size_gb > 0

    def test_unknown_model_can_resolve_from_huggingface_config(self, monkeypatch):
        monkeypatch.setattr(
            model_registry,
            "_fetch_hf_config",
            lambda model_name, dtype="fp16": PhysicsModelFeatures(
                model_name=model_name,
                num_params_billions=34.5,
                num_layers=48,
                hidden_dim=6144,
                num_attention_heads=48,
                num_kv_heads=8,
                vocab_size=128000,
                is_moe=False,
                architecture_family="custom",
                dtype=dtype,
            ),
        )

        spec = resolve_model_spec("org/Random-34B-Instruct")

        assert spec.source == "huggingface"
        assert spec.model_name == "org/Random-34B-Instruct"
        assert spec.num_params_billions == 34.5
        assert spec.architecture_family == "custom"
        assert spec.model_size_gb > 0


class TestDemoPerfModel:
    def test_gpu_ordering_is_reasonable(self):
        model = DemoPerfModel(prefer_perfdb=False)
        common = dict(
            model_name="Qwen/Qwen3-32B",
            tp=4,
            pp=1,
            input_tokens=800,
            output_tokens=200,
        )
        l40s = model.estimate_replica_tps(gpu_type="L40S", **common)
        a100 = model.estimate_replica_tps(gpu_type="A100-80GB", **common)
        h100 = model.estimate_replica_tps(gpu_type="H100", **common)

        assert h100 > a100 > l40s

    def test_nvlink_tp_scales_better_than_pcie(self):
        model = DemoPerfModel(prefer_perfdb=False)
        a100_tp4 = model.estimate_replica_tps(
            model_name="Qwen/Qwen2.5-72B-Instruct",
            gpu_type="A100-80GB",
            tp=4,
            pp=1,
            input_tokens=1024,
            output_tokens=512,
        )
        a100_tp8 = model.estimate_replica_tps(
            model_name="Qwen/Qwen2.5-72B-Instruct",
            gpu_type="A100-80GB",
            tp=8,
            pp=1,
            input_tokens=1024,
            output_tokens=512,
        )
        l40s_tp4 = model.estimate_replica_tps(
            model_name="Qwen/Qwen2.5-72B-Instruct",
            gpu_type="L40S",
            tp=4,
            pp=1,
            input_tokens=1024,
            output_tokens=512,
        )
        l40s_tp8 = model.estimate_replica_tps(
            model_name="Qwen/Qwen2.5-72B-Instruct",
            gpu_type="L40S",
            tp=8,
            pp=1,
            input_tokens=1024,
            output_tokens=512,
        )

        assert a100_tp8 > a100_tp4
        assert l40s_tp8 > l40s_tp4
        assert (a100_tp8 / a100_tp4) > (l40s_tp8 / l40s_tp4)

    def test_pipeline_parallelism_has_overhead(self):
        model = DemoPerfModel(prefer_perfdb=False)
        pp1 = model.estimate_replica_tps(
            model_name="Qwen/Qwen3-32B",
            gpu_type="L40S",
            tp=4,
            pp=1,
            input_tokens=512,
            output_tokens=256,
        )
        pp2 = model.estimate_replica_tps(
            model_name="Qwen/Qwen3-32B",
            gpu_type="L40S",
            tp=4,
            pp=2,
            input_tokens=512,
            output_tokens=256,
        )

        assert pp1 > pp2
