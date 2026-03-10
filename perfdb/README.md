# perfdb/ — Performance Database

The Oracle loads all data from this directory at startup.
Two formats are supported and can coexist:

---

## Format 1: results.json (our experiment format)

A JSON array of benchmark run records. Each entry must have at minimum:

```json
[
  {
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "tp": 4,
    "pp": 4,
    "max_input_length": 128,
    "max_output_length": 128,
    "total_tokens_per_sec": 1196.8,
    "instance_type": "4x g6e.12xlarge",
    "price_per_hour": 18.72,
    "total_gpus": 16,
    "benchmark_target_concurrency": 81
  }
]
```

Optional but useful:
- `tpot_ms_p50` — median time per output token (ms)
- `ttft_ms_p50` — median time to first token (ms)
- `requests_per_sec`
- `gpu_monitor_*` — per-GPU utilization stats

---

## Format 2: data.csv (canonical schema format)

See `schema.md` in the project root for full column reference.
Load with: `pd.read_csv("data.csv", keep_default_na=False, na_values=[""])`

Minimum required columns:
- `model_name`, `gpu_model`, `tp`, `pp`, `dp`
- `input_len_tokens_fixed`, `output_len_tokens_fixed`
- `tokens_per_sec_total`, `gpu_count_total`
- `price_per_instance_hour_usd` (optional but needed for cost predictions)

---

## Profiling data coverage (in progress)

Models:
- Qwen/Qwen3-32B
- Qwen/Qwen2.5-72B-Instruct
- Qwen/Qwen3-235B-A22B
- deepseek-ai/DeepSeek-R1-Distill-Llama-70B

GPU types: A10G, L40S, A100, H100

TP/PP configs:
- TP=2/4, PP=2/4
- TP=4, PP=2/3/4
- TP=8, PP=1/2/3

I/O lengths (input/output tokens):
- 128/128, 128/2048, 512/256, 512/1024
- 1024/512, 1024/4096, 2048/512, 4096/256
- 4096/1024, 8192/256, 8192/512, 16384/2048

---

## Oracle interpolation fallback order

1. Exact match (same model, GPU, TP, PP) — scale I/O length
2. Same GPU + TP + PP, different model — scale by param count
3. Same model + TP + PP, different GPU — scale by bandwidth/FLOPS ratio
4. Pure analytical (roofline model) — no nearby data needed
5. VPC delta correction — applied on top of whichever layer matched
