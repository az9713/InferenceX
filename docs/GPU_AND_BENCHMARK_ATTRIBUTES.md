# GPU Hardware & Benchmark Attributes Reference

A comprehensive reference for all GPU hardware platforms and benchmark
configuration/output attributes used in InferenceX.

---

## Table of Contents

1. [GPU Hardware Platforms](#1-gpu-hardware-platforms)
   - [NVIDIA GPUs](#11-nvidia-gpus)
   - [AMD GPUs](#12-amd-gpus)
   - [GPU Attribute Glossary](#13-gpu-attribute-glossary)
2. [Benchmark Configuration Attributes](#2-benchmark-configuration-attributes)
   - [Top-Level Config Fields](#21-top-level-config-fields)
   - [Sequence Length Fields](#22-sequence-length-fields)
   - [Search Space Fields (Single-Node)](#23-search-space-fields-single-node)
   - [Search Space Fields (Multi-Node)](#24-search-space-fields-multi-node)
   - [Matrix Entry Fields](#25-matrix-entry-fields)
3. [Benchmark Output Attributes](#3-benchmark-output-attributes)
   - [Identity Fields](#31-identity-fields)
   - [Parallelism Fields (Single-Node)](#32-parallelism-fields-single-node)
   - [Parallelism Fields (Multi-Node)](#33-parallelism-fields-multi-node)
   - [Throughput Metrics](#34-throughput-metrics)
   - [Latency Metrics](#35-latency-metrics)
   - [Interactivity Metrics](#36-interactivity-metrics)
   - [Eval Metrics](#37-eval-metrics)
4. [Runner Naming Conventions](#4-runner-naming-conventions)

---

## 1. GPU Hardware Platforms

### 1.1 NVIDIA GPUs

#### H100

| Attribute | Value |
|-----------|-------|
| **Vendor** | NVIDIA |
| **Architecture** | Hopper |
| **Process Node** | TSMC 4N |
| **GPU Memory** | 80 GB HBM3 |
| **Memory Bandwidth** | 3.35 TB/s |
| **FP8 Tensor Performance** | ~3,958 TFLOPS |
| **FP4 Support** | No |
| **Interconnect** | NVLink 4.0 (900 GB/s bidirectional) |
| **Form Factor** | SXM5 |
| **Typical Config** | 8x GPUs per node (DGX H100) |
| **InferenceX Runner** | `h100` |
| **Notes** | Previous generation, widely deployed in datacenters. The baseline GPU for many LLM inference comparisons. |

#### H200

| Attribute | Value |
|-----------|-------|
| **Vendor** | NVIDIA |
| **Architecture** | Hopper |
| **Process Node** | TSMC 4N |
| **GPU Memory** | 141 GB HBM3e |
| **Memory Bandwidth** | 4.8 TB/s |
| **FP8 Tensor Performance** | ~3,958 TFLOPS |
| **FP4 Support** | No |
| **Interconnect** | NVLink 4.0 (900 GB/s bidirectional) |
| **Form Factor** | SXM5 |
| **Typical Config** | 8x GPUs per node |
| **InferenceX Runner** | `h200` |
| **Notes** | Same Hopper compute as H100 but with ~76% more memory and ~43% higher bandwidth. Enables larger models or larger batch sizes without changes to parallelism. |

#### B200

| Attribute | Value |
|-----------|-------|
| **Vendor** | NVIDIA |
| **Architecture** | Blackwell |
| **Process Node** | TSMC 4NP |
| **GPU Memory** | 192 GB HBM3e |
| **Memory Bandwidth** | 8 TB/s |
| **FP8 Tensor Performance** | ~9,000 TFLOPS |
| **FP4 Tensor Performance** | ~18,000 TFLOPS |
| **Interconnect** | NVLink 5.0 (1.8 TB/s bidirectional) |
| **Form Factor** | SXM |
| **Typical Config** | 8x GPUs per node (DGX B200) |
| **InferenceX Runners** | `b200` (Docker workloads), `b200-trt` (TensorRT-LLM), `b200-multinode-slurm` (Slurm multi-node) |
| **Notes** | Current-generation NVIDIA datacenter GPU. Native FP4 support enables significantly higher throughput for quantized models. The primary benchmarking target in InferenceX. |

#### B300

| Attribute | Value |
|-----------|-------|
| **Vendor** | NVIDIA |
| **Architecture** | Blackwell (Ultra) |
| **Process Node** | TSMC 4NP |
| **GPU Memory** | 288 GB HBM3e |
| **Memory Bandwidth** | ~12 TB/s |
| **FP4 Tensor Performance** | ~Improved over B200 |
| **Interconnect** | NVLink 5.0 |
| **Form Factor** | SXM |
| **InferenceX Runner** | `b300` |
| **Notes** | Next-generation Blackwell Ultra with more memory and higher bandwidth than B200. |

#### GB200

| Attribute | Value |
|-----------|-------|
| **Vendor** | NVIDIA |
| **Architecture** | Blackwell + Grace CPU |
| **GPU Memory** | 192 GB HBM3e per GPU |
| **Memory Bandwidth** | 8 TB/s per GPU |
| **Interconnect** | NVLink 5.0 with NVLink Switch (multi-node NVLink domain) |
| **Form Factor** | Grace Blackwell Superchip (rack-scale) |
| **Typical Config** | NVLink-connected rack of GPUs spanning multiple nodes |
| **InferenceX Runner** | `gb200` |
| **Notes** | A "superchip" pairing a Grace ARM CPU with a Blackwell GPU. Multiple GB200s are connected via NVLink Switch into a single logical NVLink domain, enabling multi-node tensor parallelism without PCIe/network bottlenecks. |

#### GB300

| Attribute | Value |
|-----------|-------|
| **Vendor** | NVIDIA |
| **Architecture** | Blackwell Ultra + Grace CPU |
| **GPU Memory** | 288 GB HBM3e per GPU |
| **Interconnect** | NVLink 5.0 with NVLink Switch |
| **Form Factor** | Grace Blackwell Ultra Superchip (rack-scale) |
| **InferenceX Runner** | `gb300` |
| **Notes** | Next-generation of GB200 with Blackwell Ultra GPUs. More memory per GPU and improved interconnect. |

### 1.2 AMD GPUs

#### MI300X

| Attribute | Value |
|-----------|-------|
| **Vendor** | AMD |
| **Architecture** | CDNA 3 |
| **Process Node** | TSMC 5nm/6nm (chiplet) |
| **GPU Memory** | 192 GB HBM3 |
| **Memory Bandwidth** | 5.3 TB/s |
| **FP8 Tensor Performance** | ~2,600 TFLOPS |
| **FP4 Support** | Via software (MXFP4) |
| **Interconnect** | AMD Infinity Fabric |
| **Form Factor** | OAM |
| **Typical Config** | 8x GPUs per node |
| **InferenceX Runner** | `mi300x` |
| **Notes** | AMD's flagship datacenter GPU. Uses chiplet design with 8 XCDs (Accelerated Compute Dies). Very high memory bandwidth makes it competitive for memory-bound LLM inference workloads. |

#### MI325X

| Attribute | Value |
|-----------|-------|
| **Vendor** | AMD |
| **Architecture** | CDNA 3 |
| **Process Node** | TSMC 5nm/6nm (chiplet) |
| **GPU Memory** | 256 GB HBM3e |
| **Memory Bandwidth** | 6.0 TB/s |
| **FP8 Tensor Performance** | ~2,600 TFLOPS |
| **Interconnect** | AMD Infinity Fabric |
| **Form Factor** | OAM |
| **Typical Config** | 8x GPUs per node |
| **InferenceX Runner** | `mi325x` |
| **Notes** | Upgraded MI300X with more memory (256 GB vs 192 GB) and higher bandwidth HBM3e. Same compute die, so FP8 performance is similar, but the extra memory allows larger batch sizes or models. |

#### MI355X

| Attribute | Value |
|-----------|-------|
| **Vendor** | AMD |
| **Architecture** | CDNA 4 |
| **Process Node** | TSMC 3nm |
| **GPU Memory** | 288 GB HBM3e |
| **Memory Bandwidth** | ~8 TB/s |
| **FP8 Tensor Performance** | ~Significant improvement over MI300X |
| **FP4 Support** | Native MXFP4 |
| **Interconnect** | AMD Infinity Fabric (next-gen) |
| **Form Factor** | OAM |
| **Typical Config** | 8x GPUs per node |
| **InferenceX Runners** | `mi355x`, `mi355x-disagg` (disaggregated prefill/decode) |
| **Notes** | AMD's latest-generation datacenter GPU. New CDNA 4 architecture with native FP4 support, substantially higher compute and memory bandwidth than MI300X/MI325X. |

### 1.3 GPU Attribute Glossary

These are the key hardware attributes that affect LLM inference performance:

| Attribute | Description |
|-----------|-------------|
| **Architecture** | The GPU microarchitecture generation (e.g., Hopper, Blackwell, CDNA 3). Determines the instruction set, tensor core design, and supported precisions. |
| **Process Node** | The semiconductor manufacturing process (e.g., TSMC 4N, 5nm). Smaller nodes generally enable higher transistor density, clock speeds, and power efficiency. |
| **GPU Memory (VRAM)** | The total high-bandwidth memory on the GPU, measured in GB. Determines the maximum model size, KV cache capacity, and batch size that fit on the GPU. For LLM inference, more memory means more concurrent requests can be served. |
| **HBM Generation** | The type of High Bandwidth Memory (HBM3 vs HBM3e). HBM3e is faster and more power-efficient than HBM3. Memory type directly impacts how fast model weights and KV cache can be read. |
| **Memory Bandwidth** | The rate at which data can be read from/written to GPU memory, measured in TB/s. This is often the primary bottleneck for LLM inference because generating each token requires reading the model weights from memory. Higher bandwidth = higher throughput for memory-bound workloads (which most LLM decode steps are). |
| **FP8 Tensor Performance** | The GPU's peak throughput for FP8 (8-bit floating point) matrix operations, measured in TFLOPS. FP8 is the standard precision for efficient LLM inference. Higher FP8 TFLOPS means more compute-bound operations (like prefill/prompt processing) run faster. |
| **FP4 Tensor Performance** | The GPU's peak throughput for FP4 (4-bit floating point) operations. FP4 halves the memory needed for model weights compared to FP8, effectively doubling the achievable throughput for memory-bandwidth-bound operations. Only supported on Blackwell (NVIDIA) and CDNA 4 (AMD). |
| **Interconnect** | The high-speed link connecting GPUs within a node (NVLink for NVIDIA, Infinity Fabric for AMD). Measured in GB/s bidirectional. Faster interconnect enables more efficient tensor parallelism because GPUs must exchange activations at every layer. |
| **NVLink Switch** | A technology in GB200/GB300 systems that extends NVLink across multiple nodes, creating a single NVLink domain. This eliminates the network bottleneck that normally occurs in multi-node tensor parallelism. |
| **Form Factor** | The physical module format (SXM, OAM, Superchip). Determines what server platforms the GPU is compatible with. SXM is NVIDIA's high-performance form factor; OAM is AMD's equivalent. |
| **Typical Config** | The standard number of GPUs per server node. Most datacenter GPU nodes have 8 GPUs connected via NVLink or Infinity Fabric. |

---

## 2. Benchmark Configuration Attributes

These fields appear in the master config files (`.github/configs/nvidia-master.yaml`
and `amd-master.yaml`) and control how benchmarks are generated and run.

### 2.1 Top-Level Config Fields

| Field | Type | Description |
|-------|------|-------------|
| `image` | string | Docker container image that packages the inference engine and its dependencies. Format: `registry/repo:tag` (e.g., `lmsysorg/sglang:v0.5.8`). The image must contain the framework binary and all required libraries. |
| `model` | string | The full model identifier, typically a Hugging Face model path (e.g., `deepseek-ai/DeepSeek-R1-0528`). This is passed to the inference engine to specify which model weights to download and load. |
| `model-prefix` | string | A short human-readable code for the model (e.g., `dsr1` for DeepSeek-R1, `gptoss` for GPT-OSS-120B). Used in config key naming and result identification. |
| `precision` | string | The numerical precision for model weights and computation. Values: `fp8` (8-bit float) or `fp4` (4-bit float). Lower precision reduces memory usage and increases throughput but may reduce model accuracy. |
| `framework` | string | The inference engine software. Values: `sglang`, `vllm`, `trt` (TensorRT-LLM), `atom` (AMD), `dynamo-trt`, `dynamo-sglang`, `sglang-disagg`. Each engine has different optimization strategies and hardware support. |
| `runner` | string | The GPU hardware type to run on. Maps to physical machines in `runners.yaml`. Examples: `b200`, `h200`, `mi355x`. |
| `multinode` | boolean | Whether this config requires multiple physical server nodes. `false` = single machine with 1-8 GPUs. `true` = Slurm-orchestrated multi-node with disaggregated prefill/decode. |
| `disagg` | boolean | Whether to use disaggregated inference, where prefill (input processing) and decode (token generation) run on separate GPU pools. Only valid when `multinode: true`. Default: `false`. |

### 2.2 Sequence Length Fields

These appear under `seq-len-configs` in the master config.

| Field | Type | Description |
|-------|------|-------------|
| `isl` | integer | **Input Sequence Length** — the number of tokens in the input prompt. Standard values: `1024` or `8192`. Longer inputs require more prefill compute and KV cache memory. |
| `osl` | integer | **Output Sequence Length** — the number of tokens the model generates in response. Standard values: `1024` or `8192`. Longer outputs mean more autoregressive decode steps. |
| `search-space` | list | A list of parallelism and concurrency configurations to benchmark. Each entry defines a specific combination of TP, EP, and concurrency to test. |

The three standard sequence length combinations and what they simulate:

| Combo | ISL | OSL | Real-World Analog |
|-------|-----|-----|-------------------|
| `1k1k` | 1024 | 1024 | Short chat conversation |
| `1k8k` | 1024 | 8192 | Short prompt with long response (code generation, reasoning) |
| `8k1k` | 8192 | 1024 | Long document with short summary |

### 2.3 Search Space Fields (Single-Node)

These appear inside `search-space` entries for single-node configs.

| Field | Type | Description |
|-------|------|-------------|
| `tp` | integer | **Tensor Parallelism** — the number of GPUs across which the model is sharded. Each GPU holds a slice of every layer. Common values: `4` or `8`. Higher TP reduces per-GPU memory but adds inter-GPU communication overhead. |
| `ep` | integer | **Expert Parallelism** — for Mixture-of-Experts (MoE) models, the number of GPUs across which experts are distributed. Relevant for models like DeepSeek-R1 that use MoE architecture. Default: not set (uses framework default, typically 1). |
| `dp-attn` | boolean | **Data-Parallel Attention** — when `true`, the attention computation is replicated across GPUs instead of being tensor-parallelized. Can improve throughput for MoE models where the attention layer is relatively small. Default: framework-dependent. |
| `spec-decoding` | string | **Speculative Decoding** method. `none` = standard autoregressive decoding. `mtp` = Multi-Token Prediction (model predicts multiple tokens at once). `draft_model` = uses a smaller draft model to propose tokens that the main model verifies. Speculative decoding can significantly improve decode throughput. |
| `conc-start` | integer | Starting concurrency value for geometric sweep. Combined with `conc-end`, generates a series: start, start×2, start×4, ..., end. |
| `conc-end` | integer | Ending concurrency value for geometric sweep. |
| `conc-list` | list[int] | Explicit list of concurrency values to test. Use instead of `conc-start`/`conc-end` when you need specific non-geometric values. Cannot be combined with `conc-start`/`conc-end`. |

### 2.4 Search Space Fields (Multi-Node)

Multi-node configs replace `tp`/`ep`/`dp-attn` with separate `prefill` and
`decode` worker configurations.

| Field | Type | Description |
|-------|------|-------------|
| `prefill` | object | Configuration for the **prefill workers** — the GPUs that process input prompts. Contains `num-worker`, `tp`, `ep`, `dp-attn`, and optional `additional-settings`. |
| `decode` | object | Configuration for the **decode workers** — the GPUs that generate output tokens autoregressively. Same sub-fields as `prefill`. |
| `num-worker` | integer | Number of worker instances for this phase (prefill or decode). Each worker uses `tp` GPUs, so total GPUs = `num-worker × tp`. |
| `additional-settings` | list[string] | Extra environment variables or config file paths passed to the runner. Typically used for Dynamo recipe files (e.g., `CONFIG_FILE=recipes/...`). |
| `batch-size` | integer | Maximum batch size per worker (framework-specific). |
| `max-num-tokens` | integer | Maximum total tokens in flight per worker (framework-specific). |

### 2.5 Matrix Entry Fields

These are computed fields that appear in the generated benchmark matrix (the
output of `generate_sweep_configs.py`).

| Field | Type | Description |
|-------|------|-------------|
| `conc` | integer or list | The specific concurrency value(s) for this benchmark run. Expanded from `conc-start`/`conc-end` or taken directly from `conc-list`. |
| `max-model-len` | integer | Maximum sequence length the engine should allocate memory for. Computed as `isl + osl`. Determines KV cache memory allocation. |
| `exp-name` | string | Experiment name — a unique identifier for this specific benchmark run. Format: `<config-key>_tp<N>_isl<N>_osl<N>`. Used for result file naming and artifact tracking. |
| `run-eval` | boolean | Whether to run accuracy evaluation (lm-eval) after the throughput benchmark. Only selected for representative configurations (highest/lowest TP, highest concurrency, 1k8k only). |

---

## 3. Benchmark Output Attributes

These fields appear in the result JSON files produced by `process_result.py`
after a benchmark completes. Each file represents one benchmark run (one
specific combination of hardware, framework, precision, parallelism, and
concurrency).

### 3.1 Identity Fields

These fields identify what was benchmarked.

| Field | Type | Description |
|-------|------|-------------|
| `hw` | string | The hardware/runner type (e.g., `b200`, `h200`, `mi355x`). Taken from the `RUNNER_TYPE` environment variable. |
| `image` | string | The Docker image used for this run. |
| `model` | string | Full model identifier (e.g., `deepseek-ai/DeepSeek-R1-0528`). |
| `infmax_model_prefix` | string | Short model code (e.g., `dsr1`). Used for filtering and grouping results. |
| `framework` | string | The inference engine used (e.g., `sglang`, `trt`). |
| `precision` | string | Numerical precision (`fp8` or `fp4`). |
| `spec_decoding` | string | Speculative decoding method used (`none`, `mtp`, or `draft_model`). |
| `disagg` | boolean | Whether disaggregated inference was used. |
| `isl` | integer | Input sequence length in tokens. |
| `osl` | integer | Output sequence length in tokens. |
| `conc` | integer | The concurrency level (number of simultaneous requests). |
| `is_multinode` | boolean | Whether this was a multi-node benchmark. |

### 3.2 Parallelism Fields (Single-Node)

Present when `is_multinode` is `false`.

| Field | Type | Description |
|-------|------|-------------|
| `tp` | integer | Tensor parallelism degree — how many GPUs the model is sharded across. Total GPUs used = `tp`. |
| `ep` | integer | Expert parallelism degree for MoE models. |
| `dp_attention` | string | Whether data-parallel attention was enabled (`"true"` or `"false"`). |

### 3.3 Parallelism Fields (Multi-Node)

Present when `is_multinode` is `true`.

| Field | Type | Description |
|-------|------|-------------|
| `prefill_tp` | integer | Tensor parallelism degree for each prefill worker. |
| `prefill_ep` | integer | Expert parallelism degree for each prefill worker. |
| `prefill_dp_attention` | string | Data-parallel attention setting for prefill workers. |
| `prefill_num_workers` | integer | Number of prefill worker instances. |
| `num_prefill_gpu` | integer | Total GPUs dedicated to prefill (`prefill_num_workers × prefill_tp`). |
| `decode_tp` | integer | Tensor parallelism degree for each decode worker. |
| `decode_ep` | integer | Expert parallelism degree for each decode worker. |
| `decode_dp_attention` | string | Data-parallel attention setting for decode workers. |
| `decode_num_workers` | integer | Number of decode worker instances. |
| `num_decode_gpu` | integer | Total GPUs dedicated to decode (`decode_num_workers × decode_tp`). |

### 3.4 Throughput Metrics

All throughput metrics are **higher is better**.

| Field | Unit | Description |
|-------|------|-------------|
| `tput_per_gpu` | tokens/sec/GPU | **Total throughput per GPU.** (input tokens + output tokens) divided by the number of GPUs. This is the primary throughput metric for comparing configurations. For single-node: `total_token_throughput / tp`. For multi-node: `total_token_throughput / (prefill_gpus + decode_gpus)`. |
| `output_tput_per_gpu` | tokens/sec/GPU | **Output throughput per GPU.** Only counts generated output tokens. For single-node: `output_throughput / tp`. For multi-node: `output_throughput / decode_gpus` (since decode workers generate the output). |
| `input_tput_per_gpu` | tokens/sec/GPU | **Input throughput per GPU.** Measures how fast input prompts are processed. Computed as `(total_throughput - output_throughput) / num_gpus`. For multi-node: divided by `prefill_gpus` (since prefill workers process input). |

### 3.5 Latency Metrics

All latency metrics are **lower is better**. Raw benchmark output is in
milliseconds (`_ms` suffix); `process_result.py` converts them to seconds.

| Field | Unit | Description |
|-------|------|-------------|
| `mean_ttft` | seconds | **Mean Time To First Token.** Average time from sending a request to receiving the first output token. Dominated by prefill time (processing the entire input prompt). Affected by input length, model size, TP degree, and queuing at high concurrency. |
| `median_ttft` | seconds | **Median TTFT.** The 50th percentile TTFT — half of requests are faster, half are slower. Less sensitive to outliers than mean. |
| `p99_ttft` | seconds | **99th Percentile TTFT.** The worst-case TTFT for 99% of requests. Important for SLA compliance — measures tail latency. |
| `std_ttft` | seconds | **Standard deviation of TTFT.** Measures consistency. Lower std means more predictable first-token latency. |
| `mean_tpot` | seconds | **Mean Time Per Output Token.** Average time between consecutive output tokens during the decode phase. This is the core "streaming speed" metric — determines how fast text appears to users. Affected by memory bandwidth, batch size, and decode efficiency. |
| `median_tpot` | seconds | **Median TPOT.** The 50th percentile time per output token. |
| `p99_tpot` | seconds | **99th Percentile TPOT.** Worst-case decode speed for 99% of requests. |
| `std_tpot` | seconds | **Standard deviation of TPOT.** Measures decode speed consistency. |
| `mean_e2el` | seconds | **Mean End-to-End Latency.** Average total time from sending a request to receiving the complete response (all output tokens). Roughly: `TTFT + (TPOT × output_tokens)`. This is the metric users feel most directly — how long until they have the full answer. |
| `median_e2el` | seconds | **Median End-to-End Latency.** |
| `p99_e2el` | seconds | **99th Percentile End-to-End Latency.** |
| `std_e2el` | seconds | **Standard deviation of E2EL.** |
| `mean_itl` | seconds | **Mean Inter-Token Latency.** Average time between any two consecutive tokens (similar to TPOT but may include first token). |
| `median_itl` | seconds | **Median ITL.** |
| `p99_itl` | seconds | **99th Percentile ITL.** |
| `std_itl` | seconds | **Standard deviation of ITL.** |

### 3.6 Interactivity Metrics

Derived from TPOT. All interactivity metrics are **higher is better**.

| Field | Unit | Description |
|-------|------|-------------|
| `mean_intvty` | tokens/sec | **Mean Interactivity.** Computed as `1000 / mean_tpot_ms` (i.e., `1 / mean_tpot` in seconds). Represents how many tokens per second appear to the user during streaming. Think of it as "typing speed" — higher values mean text appears faster. |
| `median_intvty` | tokens/sec | **Median Interactivity.** |
| `p99_intvty` | tokens/sec | **P99 Interactivity.** Note: since this is derived from p99 TPOT (the slowest), p99 interactivity represents the *lowest* streaming speed experienced by 99% of requests. |

### 3.7 Eval Metrics

These appear in eval result files (`agg_eval_all.json`) for runs where
`run-eval: true`.

| Field | Range | Description |
|-------|-------|-------------|
| `score` | 0.0–1.0 | Primary accuracy score from lm-eval. Task-dependent (e.g., percentage of math problems solved correctly). A score of `0.85` means 85% accuracy. |
| `em_strict` | 0.0–1.0 | **Strict exact match.** The model's answer must match the reference exactly, including formatting. |
| `em_flexible` | 0.0–1.0 | **Flexible exact match.** Uses looser matching rules (e.g., extracting numbers from text). Generally higher than `em_strict`. |

---

## 4. Runner Naming Conventions

Physical runner node names in `runners.yaml` follow the pattern:
`<gpu>-<provider>_<index>`

| Component | Description | Examples |
|-----------|-------------|----------|
| `<gpu>` | GPU type | `b200`, `h200`, `mi300x`, `mi355x`, `gb200` |
| `<provider>` | Infrastructure provider or cluster identifier | `nv` (NVIDIA), `nb` (Nebius), `cw` (CoreWeave), `cr` (Crusoe), `amd` (AMD), `amds` (AMD staging), `dgxc` (DGX Cloud) |
| `<index>` | Node instance number | `0`, `1`, `2`, ... |

Examples:
- `b200-nv_0` — First B200 node from NVIDIA
- `h200-cw_1` — Second H200 node from CoreWeave
- `mi355x-amds_3` — Fourth MI355X node from AMD staging

The runner type (used in configs) is derived by stripping the provider suffix:
`b200-nv_0` maps to runner scripts via `launch_b200-nv.sh`.
