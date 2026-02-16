# InferenceX User Guide

A complete guide for users who want to understand, run, and analyze InferenceX
benchmarks. No deep technical experience required.

---

## Table of Contents

1. [What is InferenceX?](#1-what-is-inferencex)
2. [Key Concepts Explained Simply](#2-key-concepts-explained-simply)
3. [Viewing Benchmark Results](#3-viewing-benchmark-results)
4. [Understanding the Dashboard](#4-understanding-the-dashboard)
5. [Understanding Benchmark Metrics](#5-understanding-benchmark-metrics)
6. [Running Benchmarks via GitHub Actions](#6-running-benchmarks-via-github-actions)
7. [Reading Benchmark Results from CI](#7-reading-benchmark-results-from-ci)
8. [Triggering New Benchmarks](#8-triggering-new-benchmarks)
9. [Filtering and Comparing Results](#9-filtering-and-comparing-results)
10. [Understanding Configurations](#10-understanding-configurations)
11. [Frequently Asked Questions](#11-frequently-asked-questions)

---

## 1. What is InferenceX?

InferenceX is a benchmarking system that answers: **"How fast can this AI model
generate text on this GPU using this software?"**

Imagine you want to run an AI chatbot. You need to choose:
- **Hardware**: Which GPU? (like choosing between a sports car and a truck)
- **Software**: Which inference engine? (like choosing the transmission)
- **Settings**: How many users at once? (like choosing highway speed)

InferenceX tests all these combinations automatically every night and publishes
the results so you can make informed decisions.

### What It Measures

For each combination of hardware + software + settings, InferenceX measures:
- **Throughput**: How many tokens per second per GPU (speed of generation)
- **Latency**: How long you wait for the first token and each subsequent token
- **Accuracy**: Whether the model still gives correct answers after optimization

---

## 2. Key Concepts Explained Simply

### 2.1 Models

InferenceX currently benchmarks two models:

| Code | Full Name | What It Is |
|------|-----------|------------|
| `dsr1` | DeepSeek-R1-0528 | A large reasoning model (671 billion parameters) |
| `gptoss` | GPT-OSS-120B | An open-source GPT model (120 billion parameters) |

### 2.2 GPUs (Hardware)

| GPU | Vendor | Generation | Typical Use |
|-----|--------|-----------|-------------|
| B200 | NVIDIA | Blackwell | Latest NVIDIA datacenter GPU |
| B300 | NVIDIA | Blackwell | Next-gen NVIDIA |
| H100 | NVIDIA | Hopper | Previous generation, widely deployed |
| H200 | NVIDIA | Hopper | H100 with more memory |
| GB200 | NVIDIA | Blackwell | Multi-node NVLink rack |
| GB300 | NVIDIA | Blackwell | Next-gen multi-node |
| MI300X | AMD | CDNA3 | AMD's flagship datacenter GPU |
| MI325X | AMD | CDNA3 | MI300X with more memory |
| MI355X | AMD | CDNA4 | AMD's latest generation |

> For detailed GPU specifications (memory, bandwidth, tensor performance,
> interconnect) and a glossary of hardware attributes, see
> [GPU & Benchmark Attributes Reference](GPU_AND_BENCHMARK_ATTRIBUTES.md).

### 2.3 Inference Engines (Software)

| Engine | What It Does |
|--------|-------------|
| **SGLang** | Fast open-source engine with advanced scheduling |
| **vLLM** | Popular open-source engine with PagedAttention |
| **TensorRT-LLM** | NVIDIA's optimized engine |
| **ATOM** | AMD's optimized engine |
| **Dynamo** | NVIDIA's distributed inference framework |

### 2.4 Precision (FP4 vs FP8)

AI models store numbers at different precision levels:
- **FP8**: 8-bit floating point - good balance of speed and accuracy
- **FP4**: 4-bit floating point - faster but less precise

Lower precision = faster inference but potentially lower accuracy. InferenceX
measures both to show the tradeoff.

### 2.5 Sequence Lengths

Benchmarks test different input/output length combinations:

| Code | Input Tokens | Output Tokens | Simulates |
|------|-------------|---------------|-----------|
| `1k1k` | 1,024 | 1,024 | Short chat conversation |
| `1k8k` | 1,024 | 8,192 | Short prompt, long response (like code generation) |
| `8k1k` | 8,192 | 1,024 | Long document, short summary |

### 2.6 Concurrency

Concurrency means "how many users are being served at the same time." Higher
concurrency tests how well the system handles multiple simultaneous requests.
Typically tested at: 4, 8, 16, 32, 64, 128 concurrent users.

### 2.7 Tensor Parallelism (TP)

Large models don't fit on a single GPU. Tensor parallelism splits the model
across multiple GPUs. `tp=8` means the model is split across 8 GPUs working
together.

---

## 3. Viewing Benchmark Results

### 3.1 Live Dashboard

Visit **https://inferencemax.ai/** to see the latest results in an interactive
dashboard. You can filter by hardware, framework, model, and sequence length.

### 3.2 GitHub Actions

Every benchmark run uploads results as GitHub Actions artifacts. Navigate to:

1. Go to the GitHub repository
2. Click "Actions" tab
3. Click on a workflow run
4. Scroll to "Artifacts" section
5. Download `results_bmk` for benchmark results

---

## 4. Understanding the Dashboard

The dashboard at inferencemax.ai shows benchmark results as interactive charts.

**Key chart types:**
- **Throughput vs Concurrency**: Shows how throughput scales with more users
- **Latency vs Concurrency**: Shows latency tradeoffs at different loads
- **Hardware Comparison**: Side-by-side GPU performance comparison

---

## 5. Understanding Benchmark Metrics

### 5.1 Throughput Metrics

| Metric | Unit | What It Means |
|--------|------|--------------|
| `tput_per_gpu` | tokens/sec/GPU | Total tokens processed per GPU per second |
| `output_tput_per_gpu` | tokens/sec/GPU | Output tokens generated per GPU per second |
| `input_tput_per_gpu` | tokens/sec/GPU | Input tokens processed per GPU per second |

**Higher is better** for all throughput metrics.

### 5.2 Latency Metrics

| Metric | Unit | What It Means |
|--------|------|--------------|
| `mean_ttft` | seconds | Average time until the first token appears |
| `p99_ttft` | seconds | 99th percentile TTFT (worst-case for most users) |
| `mean_tpot` | seconds | Average time between each output token |
| `mean_e2el` | seconds | Average total time for a complete request |

**Lower is better** for all latency metrics.

### 5.3 Interactivity

| Metric | Unit | What It Means |
|--------|------|--------------|
| `mean_intvty` | tokens/sec | How fast text appears to the user (1/TPOT) |

**Higher is better.** Think of it as the "typing speed" of the AI.

### 5.4 Eval Metrics (Accuracy)

| Metric | Range | What It Means |
|--------|-------|--------------|
| `score` | 0-1 | Primary accuracy metric (e.g., % of math problems correct) |
| `em_strict` | 0-1 | Strict exact match (answers must be formatted correctly) |
| `em_flexible` | 0-1 | Flexible matching (looser number extraction) |

**Higher is better.** A score of 0.85 means the model got 85% of test problems right.

---

## 6. Running Benchmarks via GitHub Actions

### 6.1 Using the E2E Test Workflow (Manual Trigger)

1. Go to GitHub > Actions > "End-to-End Tests"
2. Click "Run workflow"
3. In the text box, enter a command like:
   ```
   full-sweep --single-node --model-prefix dsr1 --runner-type b200 --max-conc 16 --config-files .github/configs/nvidia-master.yaml
   ```
4. Click "Run workflow"
5. Wait for results (typically 30-180 minutes per benchmark)

### 6.2 Using the Sweep Workflow (Automatic)

1. Edit `perf-changelog.yaml` to add your benchmark entries
2. Create a pull request
3. Add the `sweep-enabled` label to the PR
4. The system automatically runs all specified benchmarks

---

## 7. Reading Benchmark Results from CI

### 7.1 Download Results

```bash
# Install GitHub CLI if you haven't
# https://cli.github.com

# Find the run ID from the Actions tab URL
# e.g., https://github.com/.../actions/runs/12345 -> RUN_ID=12345

# Download benchmark results
gh run download 12345 --repo SemiAnalysisAI/InferenceX -n results_bmk -D ./results
```

### 7.2 View Summary

```bash
# Count total results
cat ./results/agg_bmk.json | python3 -c "import json,sys; print(len(json.load(sys.stdin)))"

# Pretty summary table (requires jq - install with your package manager)
cat ./results/agg_bmk.json | jq -r '
  .[] | [.hw, .framework, .precision,
  "\(.isl)/\(.osl)", (.tput_per_gpu | round)]
  | @tsv' | column -t
```

### 7.3 Download Eval Results

```bash
gh run download 12345 --repo SemiAnalysisAI/InferenceX -n eval_results_all -D ./evals

cat ./evals/agg_eval_all.json | jq -r '
  .[] | [.hw, .framework, .task, (.score * 100 | round | . / 100)]
  | @tsv' | column -t
```

---

## 8. Triggering New Benchmarks

### 8.1 What Triggers Benchmarks

Benchmarks run when `perf-changelog.yaml` changes. This file acts as a log
of what should be re-benchmarked and why.

### 8.2 Adding a Benchmark Trigger

Edit `perf-changelog.yaml` and add at the end:

```yaml
- config-keys:
    - dsr1-fp8-b200-sglang       # Which config to benchmark
  description:
    - "Test after SGLang update"  # Why
  pr-link: https://github.com/.../pull/XXX
```

**Wildcards work:**
```yaml
- config-keys:
    - dsr1-fp8-*-sglang           # All dsr1-fp8 SGLang configs
    - gptoss*                      # All GPT-OSS configs
```

### 8.3 Skipping Benchmarks

Add `[skip-sweep]` to your commit message:
```
Update documentation [skip-sweep]
```

---

## 9. Filtering and Comparing Results

### 9.1 Compare GPUs

```bash
# B200 vs H200 throughput
cat ./results/agg_bmk.json | jq -r '
  [.[] | select(.hw == "b200" or .hw == "h200")
  | select(.isl == 1024 and .osl == 1024)]
  | group_by(.hw)
  | .[] | .[0].hw + ": avg " + ([.[].tput_per_gpu] | add / length | round | tostring) + " tok/s/gpu"'
```

### 9.2 Compare Frameworks

```bash
# SGLang vs TRT on B200
cat ./results/agg_bmk.json | jq '[
  .[] | select(.hw == "b200") | select(.isl == 1024 and .osl == 1024)
  | {framework, conc, tput: (.tput_per_gpu | round)}
]'
```

### 9.3 Find Best Configuration

```bash
# Best throughput for DSR1 on any hardware
cat ./results/agg_bmk.json | jq '
  [.[] | select(.infmax_model_prefix == "dsr1")]
  | max_by(.tput_per_gpu)
  | {hw, framework, precision, tp: .tp, conc, tput: (.tput_per_gpu | round)}'
```

---

## 10. Understanding Configurations

### 10.1 Config Entry Names

Config names follow a pattern: `<model>-<precision>-<gpu>-<framework>`

Examples:
- `dsr1-fp8-b200-sglang` = DeepSeek R1, FP8 precision, NVIDIA B200, SGLang
- `gptoss-fp4-mi355x-atom` = GPT-OSS, FP4 precision, AMD MI355X, ATOM

### 10.2 Reading a Config Entry

```yaml
dsr1-fp8-b200-sglang:
  image: lmsysorg/sglang:v0.5.8    # Container image (the software package)
  model: deepseek-ai/DeepSeek-R1   # The AI model being benchmarked
  model-prefix: dsr1                # Short code
  runner: b200                      # Hardware type
  precision: fp8                    # Number precision
  framework: sglang                 # Inference engine
  multinode: false                  # Single machine
  seq-len-configs:
  - isl: 1024                      # Input: 1024 tokens
    osl: 1024                      # Output: 1024 tokens
    search-space:
    - { tp: 8, conc-start: 4, conc-end: 64 }
      # This means: test with 8 GPUs, at concurrency 4, 8, 16, 32, 64
```

---

## 11. Frequently Asked Questions

### Q: How often do benchmarks run?
A: Every night, and whenever `perf-changelog.yaml` changes on the main branch.

### Q: How long does a benchmark take?
A: Each individual benchmark takes 5-30 minutes. A full sweep with many
configurations can take several hours, but they run in parallel across GPUs.

### Q: Can I run benchmarks locally?
A: The benchmarks require specific GPU hardware (B200, H200, MI355X, etc.)
that is typically only available in datacenters. You can validate configurations
locally, but actual benchmark execution requires the hardware.

### Q: What does "per GPU" mean in throughput?
A: Total throughput divided by the number of GPUs used. If a model uses 8 GPUs
with TP=8 and achieves 10,000 tokens/sec total, that's 1,250 tokens/sec/GPU.

### Q: Why do some configs have `conc-list` instead of `conc-start/conc-end`?
A: `conc-start/conc-end` generates a geometric series (4, 8, 16, 32, 64).
`conc-list` allows specific values that don't follow a pattern (e.g., [6, 875, 1214]).
Multi-node configs often use `conc-list` because optimal concurrency depends on
the specific prefill/decode worker configuration.

### Q: What's the difference between single-node and multi-node?
A: Single-node runs the entire model on one machine (1-8 GPUs). Multi-node
spreads the model across multiple machines, typically splitting "prefill"
(processing input) and "decode" (generating output) onto separate nodes for
better throughput at scale.

### Q: How are eval entries selected?
A: The system picks 2 representative configs per group: highest TP with
highest concurrency, and lowest TP with highest concurrency. This covers
the best and worst parallelism scenarios. Only 1k8k sequence lengths are
evaluated.

### Q: What if a benchmark fails?
A: Failed benchmarks don't block other benchmarks (fail-fast is disabled).
Server logs are uploaded as artifacts for debugging. The success rate is
calculated and reported.
