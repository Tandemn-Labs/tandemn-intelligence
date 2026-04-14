"""Demo-only performance modeling for the realistic simulator profile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from koi.tools.perfdb import PerfDB
from koi.tools.physics import lookup_gpu_spec

from simulation.model_registry import ModelSpec, resolve_model_spec


@dataclass(frozen=True)
class LaunchTiming:
    searching_capacity_s: float
    provisioning_s: float
    bootstrapping_s: float
    waiting_model_ready_s: float

    @property
    def total_seconds(self) -> float:
        return (
            self.searching_capacity_s
            + self.provisioning_s
            + self.bootstrapping_s
            + self.waiting_model_ready_s
        )


class LegacyPerfModel:
    """Current mock_orca-style fixed baseline behavior."""

    def estimate_replica_tps(
        self,
        *,
        base_tps: float = 1200.0,
        **_: object,
    ) -> float:
        return float(base_tps)

    def estimate_launch_timing(self, **_: object) -> LaunchTiming:
        return LaunchTiming(
            searching_capacity_s=1.0,
            provisioning_s=3.0,
            bootstrapping_s=1.0,
            waiting_model_ready_s=1.0,
        )


class DemoPerfModel:
    """PerfDB-seeded, architecture-aware demo throughput model."""

    def __init__(self, perfdb_path: Optional[str] = None, prefer_perfdb: bool = True):
        self.prefer_perfdb = prefer_perfdb
        self.perfdb = None
        path = perfdb_path or str(Path(__file__).resolve().parents[1] / "perfdb" / "perfdb_all.csv")
        perf_path = Path(path)
        if prefer_perfdb and perf_path.exists():
            self.perfdb = PerfDB(str(perf_path))

    def resolve_model(
        self,
        model_name: str,
        *,
        dtype: str = "fp16",
        overrides: Optional[dict] = None,
    ) -> ModelSpec:
        return resolve_model_spec(model_name, dtype=dtype, overrides=overrides)

    def estimate_replica_tps(
        self,
        *,
        model_name: str,
        gpu_type: str,
        tp: int,
        pp: int,
        input_tokens: int,
        output_tokens: int,
        dtype: str = "fp16",
        overrides: Optional[dict] = None,
    ) -> float:
        spec = self.resolve_model(model_name, dtype=dtype, overrides=overrides)

        if self.perfdb is not None:
            perfdb_tps = self._estimate_from_perfdb(
                spec=spec,
                gpu_type=gpu_type,
                tp=tp,
                pp=pp,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            if perfdb_tps is not None:
                return perfdb_tps

        return self._estimate_from_physics(
            spec=spec,
            gpu_type=gpu_type,
            tp=tp,
            pp=pp,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def estimate_launch_timing(
        self,
        *,
        gpu_type: str,
        market: str = "on_demand",
        capacity_pressure: float = 0.0,
    ) -> LaunchTiming:
        market_penalty = 0.9 if market == "spot" else 0.0
        gpu_penalty = {
            "H100": 0.8,
            "A100": 0.6,
            "L40S": 0.4,
            "L4": 0.3,
            "A10G": 0.3,
        }.get(gpu_type, 0.5)
        pressure = max(0.0, min(capacity_pressure, 1.0))
        return LaunchTiming(
            searching_capacity_s=1.2 + pressure * 2.0 + market_penalty,
            provisioning_s=2.0 + gpu_penalty,
            bootstrapping_s=1.4 + gpu_penalty * 0.6,
            waiting_model_ready_s=1.3 + gpu_penalty * 0.4,
        )

    def _estimate_from_perfdb(
        self,
        *,
        spec: ModelSpec,
        gpu_type: str,
        tp: int,
        pp: int,
        input_tokens: int,
        output_tokens: int,
    ) -> Optional[float]:
        rows = self.perfdb.query(
            model_name=spec.model_name,
            gpu_type=gpu_type,
            tp=tp,
            pp=pp,
            limit=20,
        )
        if not rows:
            return None

        target_ratio = input_tokens / max(output_tokens, 1)

        def score(row: dict) -> tuple[float, float]:
            row_input = row.get("input_len") or input_tokens
            row_output = row.get("output_len") or output_tokens
            row_ratio = row_input / max(row_output, 1)
            ratio_gap = abs(row_ratio - target_ratio)
            total_gap = abs(row_input - input_tokens) + abs(row_output - output_tokens)
            return ratio_gap, total_gap

        best = min(rows, key=score)
        base_tps = float(best.get("throughput_tps") or 0.0)
        if base_tps <= 0:
            return None

        row_input = best.get("input_len") or input_tokens
        row_output = best.get("output_len") or output_tokens
        io_pressure_row = self._io_pressure(row_input, row_output)
        io_pressure_target = self._io_pressure(input_tokens, output_tokens)
        adjusted = base_tps * (io_pressure_row / max(io_pressure_target, 0.1))
        return max(25.0, adjusted)

    def _estimate_from_physics(
        self,
        *,
        spec: ModelSpec,
        gpu_type: str,
        tp: int,
        pp: int,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        gpu = lookup_gpu_spec(gpu_type)
        active_params = max(spec.active_params_billions, 0.1)
        bandwidth_term = gpu["bandwidth_gbps"] * 0.60
        flops_term = gpu["fp16_tflops"] * 1.40
        raw_capacity = (bandwidth_term + flops_term) / active_params

        tp_scale = self._tp_scale(tp, interconnect=str(gpu.get("interconnect", "PCIe")))
        pp_efficiency = 1.0 / (1.0 + 0.09 * max(pp - 1, 0))
        io_pressure = self._io_pressure(input_tokens, output_tokens)
        size_penalty = min(1.25, max(0.65, 80.0 / max(spec.model_size_gb, 1.0)))

        tps = 14.0 * raw_capacity * tp_scale * pp_efficiency * size_penalty / io_pressure
        return max(20.0, tps)

    @staticmethod
    def _tp_scale(tp: int, *, interconnect: str) -> float:
        tp = max(tp, 1)
        if interconnect == "NVLink":
            return tp ** 0.82
        if tp <= 4:
            return tp ** 0.78
        return (4 ** 0.78) * ((tp / 4) ** 0.25)

    @staticmethod
    def _io_pressure(input_tokens: int, output_tokens: int) -> float:
        return 1.0 + (input_tokens / 2048.0) * 0.25 + (output_tokens / 1024.0) * 0.55
