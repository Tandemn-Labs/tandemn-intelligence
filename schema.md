# Canonical Data Schema

Schema reference for `data.csv` produced by `build_canonical.py`.

## NaN vs "None" Semantics

- **Empty cell (NaN)** — value is unknown or was not recorded by this source.
- **Literal string `"None"`** — feature was explicitly disabled or not used in
  the experiment. When reading with pandas, use
  `pd.read_csv("data.csv", keep_default_na=False, na_values=[""])` to preserve
  this distinction.

## p90 / p95 Note

Vidur and solver report p90 latencies, not p95. These are stored in the `_p95`
columns. The column names say "p95" but for `vidur` and `solver` rows the actual
percentile is p90.

---

## Data Sources

| `data_source` | `data_source_type` | Origin | Rows |
|---|---|---|---|
| `dynamo_swept` | measured | NVIDIA Dynamo pre-swept profiling (H100 SXM, H200) | ~202 |
| `dynamo_test` | measured | NVIDIA Dynamo test profiling (H200, 8B model) | ~58 |
| `solver` | analytical | Roofline-based placement solver (AWS g5/g6e instances) | ~190 |
| `vidur` | simulated | Vidur discrete-event simulator (A100, sarathi scheduler) | ~13 |
| `our_experiment` | measured | Our vLLM 0.10.0 runs on AWS (L40S, A10G, L4) | ~69 |
| `our_experiment_perfdb` | measured | Our vLLM 0.10.0 L40S profiling (wider IO range) | ~73 |
| `splitwise` | measured | SplitwiseSim DGX profiling (A100, H100) from ISCA'24 | ~1260 |

---

## Column Reference

### Identifiers

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `data_source` | string | — | Which experiment produced this row | 100% |
| `data_source_type` | string | — | `measured`, `analytical`, or `simulated` | 100% |
| `model_name` | string | — | HuggingFace model ID (e.g. `meta-llama/Llama-2-70b-hf`) | 100% |
| `model_architecture` | string | — | HF architecture class (e.g. `LlamaForCausalLM`) | 100% |
| `precision` | string | — | Weight precision: `fp8`, `fp16` | 100% |
| `params_billion` | float | billions | Model parameter count from HF config | 100% |

### Parallelism

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `tp` | int | — | Tensor parallelism degree | 100% |
| `pp` | int | — | Pipeline parallelism degree | 100% |
| `dp` | float | — | Data parallelism degree | dynamo_swept only |

### Hardware

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `gpu_model` | string | — | Canonical GPU name: `H100_SXM`, `H200`, `A100`, `A10G`, `L40S`, `L4`, `H100` | 100% |
| `gpu_count_total` | float | — | Total GPUs used (= tp * pp for most sources) | 99% |
| `gpu_mem_gb` | float | GB | VRAM per GPU | 90% |
| `num_nodes` | float | — | Number of physical nodes | our_experiment only |
| `gpus_per_node` | float | — | GPUs allocated per node | our_experiment only |
| `interconnect` | string | — | `NVLink` or `PCIe` | 100% |

### Cloud

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `cloud` | string | — | Cloud provider (currently only `aws`) | solver, our_experiment, our_experiment_perfdb |
| `region` | string | — | Cloud region (e.g. `us-east-1`) | our_experiment, our_experiment_perfdb |
| `instance_type` | string | — | AWS instance type or DGX label. Heterogeneous solver configs use `g5.12xlarge#0,g6e.4xlarge#1` format. Multi-node uses `3x g5.12xlarge` format. | 82% |
| `price_per_instance_hour_usd` | float | USD/hr | Hourly on-demand cost across all nodes | solver, our_experiment |

### Runtime

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `runtime_stack` | string | — | Serving framework and version (e.g. `vllm 0.10.0`, `vidur (sarathi scheduler)`, `splitwise-sim profiling`) | 87% |

### Workload

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `task_type` | string | — | `prefill`, `decode`, `batched`, `online` | 90% |
| `request_pattern` | string | — | `offline_batch`, `queries_per_second` | vidur, our_experiment, our_experiment_perfdb |
| `num_requests` | float | — | Number of requests in the benchmark run | vidur, our_experiment |
| `max_num_seqs` | float | — | Maximum concurrent sequences / batch size cap | dynamo_swept, solver, splitwise |

### Input / Output Lengths

All four variants are set to the same value when the source provides a single
fixed length. When the source provides a distribution, `_min`/`_max`/`_avg`
would differ (not the case in current data).

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `input_len_tokens_min` | float | tokens | Minimum input sequence length | 100% |
| `input_len_tokens_max` | float | tokens | Maximum input sequence length | 100% |
| `input_len_tokens_avg` | float | tokens | Average input sequence length | 100% |
| `input_len_tokens_fixed` | float | tokens | Fixed input length (when constant) | 100% |
| `output_len_tokens_min` | float | tokens | Minimum output sequence length | 86% |
| `output_len_tokens_max` | float | tokens | Maximum output sequence length | 86% |
| `output_len_tokens_avg` | float | tokens | Average output sequence length | 86% |
| `output_len_tokens_fixed` | float | tokens | Fixed output length (when constant) | 86% |

### Throughput

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `tokens_per_sec_total` | float | tokens/s | Total throughput across all GPUs | 100% |
| `tokens_per_sec_per_gpu` | float | tokens/s/GPU | Per-GPU throughput (= total / gpu_count) | 99% |
| `tokens_per_sec_prefill` | float | tokens/s | Prefill-phase throughput (input tokens processed per second) | 85% |
| `tokens_per_sec_decode` | float | tokens/s | Decode-phase throughput (output tokens generated per second) | 93% |

### Latency

All latency columns are in **milliseconds**. Vidur and solver values were
converted from seconds (x1000). Splitwise and dynamo values were already in ms.

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `ttft_ms_p50` | float | ms | Time to first token, median | 72% |
| `ttft_ms_p95` | float | ms | Time to first token, p95 (p90 for vidur/solver) | 1% |
| `ttft_ms_p99` | float | ms | Time to first token, p99 | 1% |
| `tpot_ms_p50` | float | ms | Time per output token, median | 79% |
| `tpot_ms_p95` | float | ms | Time per output token, p95 (p90 for vidur/solver) | 1% |
| `tpot_ms_p99` | float | ms | Time per output token, p99 | 1% |
| `e2e_ms_p50` | float | ms | End-to-end request latency, median | 68% |
| `e2e_ms_p95` | float | ms | End-to-end latency, p95 (p90 for vidur/solver) | 1% |
| `e2e_ms_p99` | float | ms | End-to-end latency, p99 | 1% |

### Cost

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `total_cost_usd` | float | USD | Total dollar cost for the benchmark run | solver, our_experiment |
| `cost_per_1m_tokens_total_usd` | float | USD/1M tokens | Cost per million total tokens | solver, our_experiment |
| `cost_per_1m_tokens_prefill_usd` | float | USD/1M tokens | Derived: `price_per_hour / (prefill_tps * 3.6)` | solver, our_experiment |
| `cost_per_1m_tokens_decode_usd` | float | USD/1M tokens | Derived: `price_per_hour / (decode_tps * 3.6)` | solver, our_experiment |

### Feature Flags

These columns use the "None" string convention: `"None"` means the feature was
**explicitly disabled**, while an empty cell (NaN) means it is **unknown**.

| Column | Type | Description | Coverage |
|---|---|---|---|
| `is_lmcache` | string | LMCache usage | our_experiment, our_experiment_perfdb |
| `is_continuous_batching` | string | Continuous batching / chunked prefill | our_experiment, our_experiment_perfdb |
| `kv_offload_target` | string | KV cache offload target (CPU/disk) | our_experiment, our_experiment_perfdb |
| `cuda_graphs` | string | CUDA graph optimization | our_experiment, our_experiment_perfdb |
| `spec_decode` | string | Speculative decoding | our_experiment, our_experiment_perfdb |

### Derived (Existing)

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `prefill_decode_ratio` | float | — | `input_len_avg / output_len_avg` (where both exist). Equivalent to `io_ratio`. | 86% |
| `batch_size` | float | — | Batch size used in the run. For vidur this stores the QPS. | 78% |

### Model Config

| Column | Type | Description | Coverage |
|---|---|---|---|
| `model_config_json` | string | Full HuggingFace `config.json` as compact JSON string. Parse with `json.loads()`. Contains architecture details: `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`, `vocab_size`, `max_position_embeddings`, etc. | 100% |

### Derived (Model Structure)

Extracted from `model_config_json` (HuggingFace config.json).

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `is_moe` | bool | — | True if the model is a Mixture-of-Experts architecture (has `num_local_experts > 0` in HF config) | ~100% |
| `num_experts_active` | int | — | Number of active experts per token (`num_experts_per_tok` from HF config). NaN for dense models. | MoE models only |
| `vocab_size` | int | tokens | Vocabulary size from HF config | ~100% |
| `attention_heads_per_kv_head` | float | — | GQA group size = `num_attention_heads / num_key_value_heads`. Value > 1 indicates grouped-query attention. | ~100% |

### Derived (Sizing)

Computed from `params_billion`, `precision`, `gpu_mem_gb`, `gpu_count_total`.

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `model_size_gb` | float | GB | Weight footprint: `params_billion * bytes_per_param` (2 for fp16, 1 for fp8) | ~99% |
| `params_per_gpu` | float | billions | `params_billion / gpu_count_total` | ~99% |
| `model_fits_single_gpu` | bool | — | True if `model_size_gb <= gpu_mem_gb` | ~90% |
| `vram_headroom_gb` | float | GB | `(gpu_mem_gb * gpu_count_total) - model_size_gb`. VRAM remaining after weights — the budget available for KV cache, CUDA graphs, activations, and framework overhead. Negative means weights alone exceed total VRAM. | ~90% |

### Derived (Hardware)

Looked up from `GPU_SPECS` using `gpu_model`. NaN for heterogeneous GPU configs (comma-separated solver rows).

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `gpu_bandwidth_gbps` | float | GB/s | Memory bandwidth from GPU_SPECS | ~95% |
| `gpu_tflops_fp16` | float | TFLOPS | FP16 tensor core TFLOPS from GPU_SPECS | ~95% |
| `gpu_generation` | string | — | Architecture generation (e.g. `Hopper`, `Ampere`, `Ada Lovelace`) | ~95% |

### Derived (Efficiency Ratios)

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `bandwidth_per_param` | float | GB/s/B-params | `gpu_bandwidth_gbps * tp / params_billion` — memory bandwidth available per billion params | ~95% |
| `flops_per_param` | float | TFLOPS/B-params | `gpu_tflops_fp16 * tp / params_billion` — compute available per billion params | ~95% |
| `kv_heads_per_tp` | float | — | `num_key_value_heads / tp`. Values < 1 indicate KV head replication across TP ranks. | ~100% |

### Derived (Topology)

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `crosses_node_boundary` | bool | — | True if `num_nodes > 1`. After topology inference, populated for all sources. | ~100% |

### Derived (Cost)

| Column | Type | Unit | Description | Coverage |
|---|---|---|---|---|
| `price_per_gpu_hour_usd` | float | USD/GPU/hr | `price_per_instance_hour_usd / gpu_count_total` | ~14% |

### Notes

- **`io_ratio`** is equivalent to the existing `prefill_decode_ratio` column. No separate column is created.
- **`num_nodes`** and **`gpus_per_node`** already exist in the schema. `compute_derived()` now infers values for all sources (dynamo → DGX 8 GPU/node, solver → from instance type, vidur → from gpu_count, splitwise → DGX 8 GPU/node).
- **Heterogeneous GPU configs** (solver rows with comma-separated `gpu_model` like `"L40S,A10G"`) produce NaN for all GPU-spec-derived columns (`gpu_bandwidth_gbps`, `gpu_tflops_fp16`, `gpu_generation`, and downstream ratios).

---

## Source-to-Column Coverage Matrix

Columns populated per source (empty = NaN for all rows from that source):

| Column | dynamo_swept | dynamo_test | solver | vidur | our_experiment | our_experiment_perfdb | splitwise |
|---|---|---|---|---|---|---|---|
| dp | x | | | | | | |
| cloud | | | x | | x | x | |
| region | | | | | x | x | |
| instance_type | | | x | x | x | | x |
| price_per_instance_hour_usd | | | x | | x | | |
| num_nodes | x | x | x | x | x | x | x |
| gpus_per_node | x | x | x | x | x | x | x |
| request_pattern | | | | x | x | x | |
| num_requests | | | | x | x | | |
| output_len_tokens_* | | | x | x | x | x | x |
| ttft_ms_p50 | x | x | | x | | | x |
| tpot_ms_p50 | x | x | | x | | | x |
| e2e_ms_p50 | | | | x | | | x |
| *_p95, *_p99 | | | | x | | | |
| total_cost_usd | | | x | | x | | |
| cost_per_1m_tokens_* | | | x | | x | | |
| is_lmcache (+ other flags) | | | | | x | x | |
| batch_size | | | x | x | | | x |
| is_moe | x | x | x | x | x | x | x |
| vocab_size | x | x | x | x | x | x | x |
| attention_heads_per_kv_head | x | x | x | x | x | x | x |
| model_size_gb | x | x | x | x | x | x | x |
| params_per_gpu | x | x | x | x | x | x | x |
| model_fits_single_gpu | x | x | | x | x | x | x |
| vram_headroom_gb | x | x | | x | x | x | x |
| gpu_bandwidth_gbps | x | x | ~partial | x | x | x | x |
| gpu_tflops_fp16 | x | x | ~partial | x | x | x | x |
| gpu_generation | x | x | ~partial | x | x | x | x |
| bandwidth_per_param | x | x | ~partial | x | x | x | x |
| flops_per_param | x | x | ~partial | x | x | x | x |
| kv_heads_per_tp | x | x | x | x | x | x | x |
| crosses_node_boundary | x | x | x | x | x | x | x |
| price_per_gpu_hour_usd | | | x | | x | | |
