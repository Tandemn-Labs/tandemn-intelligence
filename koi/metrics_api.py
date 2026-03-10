"""
koi/metrics_api.py — Metrics data fetching layer.

Koi does NOT do any monitoring instrumentation. It only FETCHES metrics
that are already being collected by the Tandem infrastructure.

Implementations:
  TandemMetricsAPISource  — hits the Tandem internal metrics API (primary)
  VLLMPrometheusSource    — hits vLLM's /metrics Prometheus endpoint directly
  MockMetricsSource       — deterministic fake data for testing (in monitor.py)

To add a new source:
  1. Subclass MetricsSource from koi.monitor
  2. Implement async fetch(job_id) → Optional[RuntimeMetrics]
  3. Pass instance to KoiMonitor(metrics_source=...)

Expected metrics fields (all optional except throughput and gpu_utilization):
  throughput_tokens_per_sec  — total output tokens/sec across all DP replicas
  tpot_ms                    — median time-per-output-token
  ttft_ms                    — median time-to-first-token
  gpu_utilization_pct        — avg SM utilization across GPUs
  gpu_memory_used_gb         — avg VRAM used per GPU
  gpu_memory_bw_pct          — avg memory bandwidth utilization
  concurrent_requests        — number of in-flight requests
  queue_depth                — requests waiting to be processed
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from koi.monitor import MetricsSource
from koi.schemas import RuntimeMetrics


# ---------------------------------------------------------------------------
# Tandem internal metrics API
# ---------------------------------------------------------------------------

class TandemMetricsAPISource(MetricsSource):
    """
    Fetches job metrics from the Tandem internal metrics API.

    Expected endpoint:
        GET {base_url}/jobs/{job_id}/metrics
        Returns JSON with fields matching RuntimeMetrics schema.

    Set TANDEM_METRICS_API_URL in environment or pass base_url directly.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: float = 10.0,
    ):
        self.base_url = (
            base_url
            or os.environ.get("TANDEM_METRICS_API_URL", "http://localhost:8080")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("TANDEM_METRICS_API_KEY", "")
        self.timeout = timeout_seconds

    async def fetch(self, job_id: str) -> Optional[RuntimeMetrics]:
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required: pip install aiohttp")

        url = f"{self.base_url}/jobs/{job_id}/metrics"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as resp:
                    if resp.status == 404:
                        return None  # job not found / not yet started
                    resp.raise_for_status()
                    data: Dict[str, Any] = await resp.json()
            return _parse_tandem_response(job_id, data)
        except Exception as e:
            print(f"[MetricsAPI] Failed to fetch metrics for {job_id}: {e}")
            return None


def _parse_tandem_response(job_id: str, data: Dict[str, Any]) -> RuntimeMetrics:
    """
    Parse Tandem API response into RuntimeMetrics.
    Handles both flat and nested response formats.

    Flat:
        {"throughput_tokens_per_sec": 1200, "tpot_ms": 27.5, ...}

    Nested:
        {"metrics": {"throughput": 1200, "latency": {"tpot_p50": 27.5}}, "gpu": {...}}
    """
    # Try flat format first
    if "throughput_tokens_per_sec" in data:
        return RuntimeMetrics(
            job_id=job_id,
            timestamp=datetime.utcnow(),
            throughput_tokens_per_sec=float(data["throughput_tokens_per_sec"]),
            tpot_ms=data.get("tpot_ms") or data.get("tpot_p50_ms"),
            ttft_ms=data.get("ttft_ms") or data.get("ttft_p50_ms"),
            gpu_utilization_pct=float(data.get("gpu_utilization_pct", 0)),
            gpu_memory_used_gb=float(data.get("gpu_memory_used_gb", 0)),
            gpu_memory_bw_pct=data.get("gpu_memory_bw_pct"),
            concurrent_requests=data.get("concurrent_requests"),
            queue_depth=data.get("queue_depth"),
        )

    # Try nested format
    metrics = data.get("metrics", data)
    gpu = data.get("gpu", {})
    latency = metrics.get("latency", {})

    throughput = (
        metrics.get("throughput_tokens_per_sec")
        or metrics.get("throughput")
        or metrics.get("output_tokens_per_sec", 0)
    )
    tpot = (
        latency.get("tpot_p50_ms")
        or latency.get("tpot_ms")
        or metrics.get("tpot_ms")
    )
    ttft = (
        latency.get("ttft_p50_ms")
        or latency.get("ttft_ms")
        or metrics.get("ttft_ms")
    )
    gpu_util = (
        gpu.get("utilization_pct")
        or gpu.get("sm_pct_avg")
        or metrics.get("gpu_utilization_pct", 0)
    )
    gpu_mem = (
        gpu.get("memory_used_gb")
        or gpu.get("mem_gb_avg")
        or metrics.get("gpu_memory_used_gb", 0)
    )

    return RuntimeMetrics(
        job_id=job_id,
        timestamp=datetime.utcnow(),
        throughput_tokens_per_sec=float(throughput),
        tpot_ms=float(tpot) if tpot else None,
        ttft_ms=float(ttft) if ttft else None,
        gpu_utilization_pct=float(gpu_util),
        gpu_memory_used_gb=float(gpu_mem),
        gpu_memory_bw_pct=gpu.get("membw_pct_avg"),
        concurrent_requests=metrics.get("concurrent_requests"),
        queue_depth=metrics.get("queue_depth"),
    )


# ---------------------------------------------------------------------------
# vLLM Prometheus endpoint (alternative source)
# ---------------------------------------------------------------------------

class VLLMPrometheusSource(MetricsSource):
    """
    Fetches metrics directly from vLLM's built-in Prometheus /metrics endpoint.

    vLLM exposes these gauges:
      vllm:num_requests_running{...}
      vllm:avg_generation_throughput_toks_per_s{...}
      vllm:gpu_cache_usage_perc{...}

    And histograms:
      vllm:time_per_output_token_seconds_bucket{...}
      vllm:time_to_first_token_seconds_bucket{...}

    Pass a dict mapping job_id → endpoint_url when multiple jobs run on different ports.
    """

    def __init__(self, job_endpoints: Optional[Dict[str, str]] = None):
        """
        job_endpoints: {"job-abc123": "http://node1:8000", "job-def456": "http://node2:8000"}
        If None, uses VLLM_METRICS_URL env var as a single endpoint for all jobs.
        """
        self.endpoints = job_endpoints or {}
        self._default_url = os.environ.get("VLLM_METRICS_URL", "http://localhost:8000")

    def register_job(self, job_id: str, endpoint_url: str) -> None:
        """Register a job's vLLM endpoint URL after deployment."""
        self.endpoints[job_id] = endpoint_url

    async def fetch(self, job_id: str) -> Optional[RuntimeMetrics]:
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp is required: pip install aiohttp")

        url = self.endpoints.get(job_id, self._default_url)
        metrics_url = f"{url.rstrip('/')}/metrics"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    metrics_url,
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as resp:
                    resp.raise_for_status()
                    text = await resp.text()
            return _parse_prometheus_text(job_id, text)
        except Exception as e:
            print(f"[VLLMPrometheus] Failed for {job_id} at {metrics_url}: {e}")
            return None


def _parse_prometheus_text(job_id: str, text: str) -> RuntimeMetrics:
    """
    Parse Prometheus text format into RuntimeMetrics.
    Extracts key vLLM metrics by line scanning.
    """
    values: Dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        # e.g.: vllm:avg_generation_throughput_toks_per_s{model_name="..."} 1234.5
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            key_full, val_str = parts
            try:
                val = float(val_str)
            except ValueError:
                continue
            # Strip labels to get bare metric name
            key = key_full.split("{")[0].strip()
            values[key] = val

    throughput = values.get("vllm:avg_generation_throughput_toks_per_s", 0)
    concurrent = values.get("vllm:num_requests_running", 0)
    gpu_cache_pct = values.get("vllm:gpu_cache_usage_perc", 0)

    # TPOT: extract from histogram (p50 approximation from bucket counts)
    # vllm:time_per_output_token_seconds_bucket is a cumulative histogram
    # For now, use a simple heuristic: if throughput > 0, rough estimate
    tpot_ms = (1000.0 / throughput * max(1, concurrent)) if throughput > 0 else None

    return RuntimeMetrics(
        job_id=job_id,
        timestamp=datetime.utcnow(),
        throughput_tokens_per_sec=throughput,
        tpot_ms=tpot_ms,
        gpu_utilization_pct=0,   # not directly in vLLM metrics, use GPU monitor
        gpu_memory_used_gb=0,    # same
        gpu_memory_bw_pct=gpu_cache_pct,
        concurrent_requests=int(concurrent),
    )
