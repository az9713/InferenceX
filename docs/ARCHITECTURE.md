# InferenceX Architecture Guide

This document provides a comprehensive architectural overview of InferenceX, from
high-level system design down to individual component details. All diagrams are
ASCII-based and self-contained.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Stage 1: Configuration & Validation](#3-stage-1-configuration--validation)
4. [Stage 2: CI/CD Orchestration](#4-stage-2-cicd-orchestration)
5. [Stage 3: Benchmark Execution](#5-stage-3-benchmark-execution)
6. [Stage 4: Result Processing & Publishing](#6-stage-4-result-processing--publishing)
7. [Data Flow Diagrams](#7-data-flow-diagrams)
8. [Validation Architecture](#8-validation-architecture)
9. [File-Level Component Map](#9-file-level-component-map)
10. [Multi-Node Disaggregated Architecture](#10-multi-node-disaggregated-architecture)
11. [Evaluation Subsystem](#11-evaluation-subsystem)

---

## 1. System Overview

InferenceX is an automated benchmarking pipeline that measures how fast different
LLM inference engines (vLLM, SGLang, TensorRT-LLM, ATOM) run on different GPU
hardware (NVIDIA B200/H200/H100/GB200, AMD MI300X/MI325X/MI355X). It runs
nightly and on every code change, publishing results to a live dashboard.

Think of it as a four-stage assembly line:

```
  +------------------+    +----------------+    +--------------------+    +-------------------+
  |  STAGE 1         |    |  STAGE 2       |    |  STAGE 3           |    |  STAGE 4          |
  |  Configuration   |--->|  CI/CD         |--->|  Benchmark         |--->|  Result            |
  |  & Validation    |    |  Orchestration |    |  Execution         |    |  Processing        |
  +------------------+    +----------------+    +--------------------+    +-------------------+
                                                                                    |
        YAML configs          GitHub Actions        GPU hardware                    v
        Pydantic models        Workflow dispatch     Docker containers        Dashboard at
        Python CLI             Matrix strategy       Bash scripts             inferencemax.ai
```

**What each stage does:**

- **Stage 1** reads YAML config files, validates them, and generates a matrix of
  benchmark jobs (which model, which GPU, which settings).
- **Stage 2** is GitHub Actions. It takes that matrix and dispatches one job per
  benchmark configuration to the appropriate GPU hardware.
- **Stage 3** runs on actual GPU machines. It starts an inference server inside a
  Docker container, sends benchmark traffic to it, and measures performance.
- **Stage 4** collects all results, aggregates them into JSON, generates summary
  tables, and triggers a deployment to the live dashboard.

---

## 2. High-Level Architecture

```
                         +----------------------------+
                         |     perf-changelog.yaml    |
                         |  (trigger: what changed?)  |
                         +-------------+--------------+
                                       |
                                       | git push / PR
                                       v
+-------------------------------+    +----------------------------+
| .github/configs/              |    | utils/process_changelog.py |
|   nvidia-master.yaml          |--->| (diff changelog, resolve   |
|   amd-master.yaml             |    |  wildcards, call sweeps)   |
|   runners.yaml                |    +-------------+--------------+
+-------------------------------+                  |
                                                   v
                               +-----------------------------------+
                               | utils/matrix_logic/               |
                               |   generate_sweep_configs.py       |
                               |   validation.py                   |
                               | (validate input, generate matrix, |
                               |  validate output)                 |
                               +----------------+------------------+
                                                |
                                    JSON matrix (list of job configs)
                                                |
                                                v
                          +---------------------------------------------+
                          |  .github/workflows/run-sweep.yml            |
                          |  (orchestrator: split by seq-len & node     |
                          |   type, dispatch to template workflows)     |
                          +----+--------+--------+--------+--------+---+
                               |        |        |        |        |
                          1k1k-SN  1k8k-SN  8k1k-SN  1k1k-MN  1k8k-MN ...
                               |        |        |        |        |
                               v        v        v        v        v
                 +-------------------+  +-------------------+
                 | benchmark-tmpl.yml|  | benchmark-        |
                 | (single-node)     |  | multinode-tmpl.yml|
                 +--------+----------+  +---------+---------+
                          |                       |
              +-----------+-----------+           |
              |           |           |           |
              v           v           v           v
         +--------+  +--------+  +--------+  +--------+
         |Runner A|  |Runner B|  |Runner C|  |Slurm   |
         |b200-nv |  |h200-cw |  |mi355x  |  |Cluster |
         +---+----+  +---+----+  +---+----+  +---+----+
             |            |          |            |
             v            v          v            v
      runners/       runners/    runners/     runners/
      launch_*.sh    launch_*.sh launch_*.sh  launch_*-slurm.sh
             |            |          |            |
             v            v          v            v
      benchmarks/    benchmarks/ benchmarks/  benchmarks/
      {model}_{prec}_{gpu}[_{fw}].sh
             |            |          |            |
             v            v          v            v
      +----------------------------------------------+
      | utils/bench_serving/benchmark_serving.py     |
      | (load generator: sends requests, measures    |
      |  throughput, latency, TTFT, TPOT, E2EL)      |
      +---------------------+------------------------+
                            |
                   {result}.json per job
                            |
                            v
      +----------------------------------------------+
      |  utils/process_result.py                     |
      |  (normalize metrics, compute per-GPU stats)  |
      +---------------------+------------------------+
                            |
                  agg_{result}.json per job
                            |
                            v
      +----------------------------------------------+
      |  collect-results.yml / collect-evals.yml     |
      |  (download all artifacts, aggregate)         |
      +---------------------+------------------------+
                            |
                            v
      +----------------------------------------------+
      |  Dashboard deployment (Vercel)               |
      |  https://inferencemax.ai/                    |
      +----------------------------------------------+
```

---

## 3. Stage 1: Configuration & Validation

### 3.1 Configuration Files

The system's source of truth is a set of YAML files:

```
.github/configs/
    +-- nvidia-master.yaml    <-- All NVIDIA benchmark definitions
    +-- amd-master.yaml       <-- All AMD benchmark definitions
    +-- runners.yaml          <-- Maps GPU types to physical nodes
    +-- CONFIGS.md            <-- Documentation of config format
```

**Master Config Structure (single-node example):**

```yaml
dsr1-fp8-b200-sglang:           # <-- Entry name (model-prec-gpu-framework)
  image: lmsysorg/sglang:v0.5.8 #     Docker image to use
  model: deepseek-ai/DeepSeek-R1 #    Model to benchmark
  model-prefix: dsr1             #     Short code (maps to benchmark script)
  runner: b200                   #     GPU type (must exist in runners.yaml)
  precision: fp8                 #     Numerical precision
  framework: sglang              #     Inference engine
  multinode: false               #     Single-node or multi-node
  seq-len-configs:               #     List of input/output length combos
  - isl: 1024                   #       Input sequence length
    osl: 1024                   #       Output sequence length
    search-space:                #       Parallelism + concurrency combos
    - { tp: 8, conc-start: 4, conc-end: 64 }    # TP=8, conc 4->8->16->32->64
    - { tp: 4, ep: 4, conc-start: 4, conc-end: 64 }
```

**Runners Config:**

```yaml
b200:                        # GPU type
- 'b200-dgxc_1'             # Physical node names
- 'b200-dgxc_2'             # (GitHub Actions self-hosted runners)
- 'b200-nv_0'
- 'b200-nv_1'
```

**Entry naming convention:** `<model-prefix>-<precision>-<gpu>-<framework>`
For example: `dsr1-fp4-b200-sglang`, `gptoss-fp4-mi355x-atom`

### 3.2 Config Generation CLI

**File:** `utils/matrix_logic/generate_sweep_configs.py`

This is a Python CLI tool with three subcommands:

```
generate_sweep_configs.py
    |
    +-- full-sweep            Generate full benchmark matrix with filters
    |     |-- --config-files   (required) YAML config files
    |     |-- --single-node / --multi-node  (required, mutually exclusive)
    |     |-- --model-prefix   Filter by model (e.g., dsr1, gptoss)
    |     |-- --framework      Filter by engine (e.g., sglang, trt)
    |     |-- --precision      Filter by precision (e.g., fp4, fp8)
    |     |-- --runner-type    Filter by GPU (e.g., b200, mi355x)
    |     |-- --seq-lens       Filter by sequence lengths (1k1k, 1k8k, 8k1k)
    |     |-- --max-conc       Cap maximum concurrency
    |     |-- --max-tp         Cap tensor parallelism
    |     +-- --run-evals      Mark eval entries
    |
    +-- runner-model-sweep    Test all nodes for a GPU type
    |     |-- --runner-type    (required) Which GPU type
    |     +-- --single-node / --multi-node
    |
    +-- test-config           Generate matrix for specific config keys
          +-- --config-keys    (required) Named configs to expand
```

**How it works (full-sweep):**

```
  Master YAML        Filters (CLI args)        Output JSON
  +---------+        +----------+              +------------+
  | entry-1 |        | --model  |              | [{image:..,|
  | entry-2 |------->| --prec   |------+------>|   model:..,|
  | entry-3 |  load  | --fw     | for  | each  |   tp:8,   |
  | ...     |  +     | --runner | each | conc   |   conc:4},|
  +---------+  valid | --seq    | bmk  | value  |  {..},..] |
               ate   +----------+      |        +------------+
                                       |
                     For single-node configs, concurrency is expanded:
                     conc-start=4, conc-end=64, step=2x
                     produces: [4, 8, 16, 32, 64] -> one matrix entry each
```

### 3.3 Validation Layer

**File:** `utils/matrix_logic/validation.py`

The validation uses Pydantic V2 with strict mode (`extra='forbid'`). This means:
- Any unknown field raises an error (catches typos)
- Any missing required field raises an error
- Type mismatches raise errors

There are two layers of validation:

```
                    INPUT VALIDATION                    OUTPUT VALIDATION
                    (master configs)                    (matrix entries)

    .github/configs/*.yaml                    JSON matrix for workflows
            |                                         |
            v                                         v
  +---------------------------+             +---------------------------+
  | SingleNodeMasterConfig-   |             | SingleNodeMatrixEntry     |
  |   Entry                   |             |   image, model, runner,   |
  |   image, model, runner,   |             |   tp, ep, conc, isl, osl  |
  |   seq-len-configs[...]    |             |   max-model-len, exp-name |
  +---------------------------+             +---------------------------+
  | MultiNodeMasterConfig-    |             | MultiNodeMatrixEntry      |
  |   Entry                   |             |   + prefill{tp,ep,dp-attn}|
  |   + multinode: true       |             |   + decode{tp,ep,dp-attn} |
  |   + prefill/decode workers|             |   + conc as list          |
  +---------------------------+             +---------------------------+
```

**Key Pydantic models hierarchy:**

```
  SingleNodeMasterConfigEntry
      +-- SingleNodeSeqLenConfig (list)
            +-- SingleNodeSearchSpaceEntry (list)
                  Fields: tp, ep?, dp-attn?, conc-start/conc-end OR conc-list

  MultiNodeMasterConfigEntry
      +-- MultiNodeSeqLenConfig (list)
            +-- MultiNodeSearchSpaceEntry (list)
                  Fields: prefill(WorkerConfig), decode(WorkerConfig),
                          conc-start/conc-end OR conc-list

  WorkerConfig
      Fields: num-worker, tp, ep, dp-attn, additional-settings[]
```

---

## 4. Stage 2: CI/CD Orchestration

### 4.1 Workflow Architecture

```
                    perf-changelog.yaml
                          |
                    push to main
                    or PR with 'sweep-enabled' label
                          |
                          v
              +------------------------+
              |  run-sweep.yml         |
              |  (Main orchestrator)   |
              +---+---+---+---+---+---+
                  |   |   |   |   |   |
    +-------------+   |   |   |   |   +----------------+
    |                 |   |   |   |                    |
    v                 v   v   v   v                    v
sweep-SN-1k1k   SN-1k8k SN-8k1k MN-1k1k MN-1k8k   MN-8k1k
    |                 |   |   |   |                    |
    | (each uses strategy.matrix)                      |
    v                 v   v   v   v                    v
benchmark-tmpl.yml .................... benchmark-multinode-tmpl.yml
    |                                                  |
    | (runs on self-hosted GPU runners)                |
    v                                                  v
+--------------------+                   +--------------------+
| 1. Resource cleanup|                   | 1. Resource cleanup|
| 2. Checkout code   |                   | 2. Checkout code   |
| 3. Launch runner   |                   | 3. Launch Slurm    |
| 4. Process result  |                   | 4. Process result  |
| 5. Upload artifact |                   | 5. Upload artifact |
| 6. Upload eval     |                   | 6. Upload eval     |
| 7. Cleanup         |                   | 7. Cleanup         |
+--------------------+                   +--------------------+
    |                                                  |
    +--------------------------------------------------+
    |
    v
collect-results.yml        collect-evals.yml
    |                            |
    v                            v
results_bmk artifact       eval_results_all artifact
    |
    v
calc-success-rate    trigger-vercel-deploy
```

### 4.2 The Trigger Mechanism

The system is triggered by changes to `perf-changelog.yaml`:

```
perf-changelog.yaml (what triggers benchmarks)
+--------------------------------------------------------------+
| - config-keys:                                               |
|     - dsr1-fp8-*-sglang         <-- Wildcard patterns        |
|   description:                                               |
|     - "Update SGLang to v0.5.8" <-- Human-readable change    |
|   pr-link: https://github.com/.../pull/204                   |
|                                                              |
| - config-keys:                                               |
|     - gptoss-fp4-b200-vllm      <-- Exact config key         |
|   description:                                               |
|     - "Extend concurrency to 128"                            |
|   pr-link: https://github.com/.../pull/209                   |
+--------------------------------------------------------------+
```

**How `process_changelog.py` works:**

```
  1. Git diff perf-changelog.yaml (base..head)
  2. Extract only ADDED lines (deletions are rejected)
  3. Parse added YAML as changelog entries
  4. For each entry:
     a. Resolve config-keys against master configs (expand wildcards)
     b. Call generate_sweep_configs.py test-config with resolved keys
     c. Collect all generated matrix entries
  5. Split into single_node / multi_node buckets, keyed by seq-len
  6. Validate final structure with ChangelogMatrixEntry Pydantic model
  7. Output JSON for run-sweep.yml to consume
```

### 4.3 How runner/benchmark scripts are selected

The workflow template (`benchmark-tmpl.yml`) constructs the launch command:

```
  Runner name from GitHub Actions: e.g., "b200-nv_0"
                    |
                    | Strip suffix after first underscore:
                    | "b200-nv_0" --> "b200-nv"
                    v
  runners/launch_b200-nv.sh        <-- Runner launcher
                    |
                    | Inside the launcher, construct benchmark script name:
                    | MODEL_CODE = EXP_NAME up to first underscore
                    |   e.g., "dsr1_1k1k" --> "dsr1"
                    | FRAMEWORK_SUFFIX = "_trt" if framework=trt, else ""
                    | SPEC_SUFFIX = "_mtp" if spec-decoding=mtp, else ""
                    v
  benchmarks/{MODEL_CODE}_{PRECISION}_{GPU}{FRAMEWORK_SUFFIX}{SPEC_SUFFIX}.sh
  e.g., benchmarks/dsr1_fp8_b200.sh
```

---

## 5. Stage 3: Benchmark Execution

### 5.1 Single-Node Execution Flow

```
  Runner Machine (self-hosted GitHub Actions runner with GPU)
  +-------------------------------------------------------------+
  |                                                             |
  |  1. runners/launch_b200-nv.sh                               |
  |     +-- Allocate GPU via Slurm (salloc)                     |
  |     +-- Import Docker image as Enroot squash file            |
  |     +-- Launch container with srun                           |
  |         +-- Mount: workspace + HuggingFace cache             |
  |         +-- Run: benchmarks/dsr1_fp8_b200.sh                |
  |                                                             |
  |  2. benchmarks/dsr1_fp8_b200.sh                              |
  |     +-- source benchmark_lib.sh                              |
  |     +-- check_env_vars (MODEL, TP, CONC, ISL, OSL, ...)     |
  |     +-- Download model (hf download)                         |
  |     +-- Start inference server (sglang/vllm/trt)             |
  |     |     Server starts in background, logs to server.log    |
  |     +-- wait_for_server_ready (poll /health endpoint)        |
  |     +-- run_benchmark_serving                                |
  |     |     +-- Calls benchmark_serving.py with:               |
  |     |     |   - random dataset                               |
  |     |     |   - input-len, output-len                        |
  |     |     |   - max-concurrency                              |
  |     |     |   - request-rate=inf (as fast as possible)       |
  |     |     +-- Outputs: {RESULT_FILENAME}.json                |
  |     +-- If RUN_EVAL=true:                                    |
  |           +-- run_eval (lm-eval harness)                     |
  |           +-- append_lm_eval_summary                         |
  |                                                             |
  +-------------------------------------------------------------+
```

### 5.2 Key Functions in benchmark_lib.sh

```
benchmarks/benchmark_lib.sh
    |
    +-- check_env_vars(VAR1, VAR2, ...)
    |     Exits with error if any env var is unset
    |
    +-- wait_for_server_ready(--port, --server-log, --server-pid)
    |     Polls http://0.0.0.0:PORT/health until 200 OK
    |     Monitors server process, exits if server dies
    |
    +-- run_benchmark_serving(--model, --port, --backend, ...)
    |     Calls utils/bench_serving/benchmark_serving.py
    |     Monitors server health during benchmark
    |     Returns benchmark exit code
    |
    +-- run_eval(--framework, --port, ...)
    |     Dispatches to run_lm_eval() for lm-eval framework
    |
    +-- run_lm_eval(--port, --task, --num-fewshot, ...)
    |     Installs lm-eval dependencies
    |     Patches lm-eval for reasoning tokens + TRT compatibility
    |     Runs lm_eval against OpenAI-compatible endpoint
    |
    +-- append_lm_eval_summary()
          Writes meta_env.json with benchmark context
          Moves eval artifacts to workspace root
```

### 5.3 Benchmark Client

**File:** `utils/bench_serving/benchmark_serving.py`

This is the load generator (originally upstreamed from vLLM). It:
1. Generates random prompts of specified input length
2. Sends them to the inference server at max concurrency
3. Measures: throughput, TTFT, TPOT, ITL, E2EL
4. Writes results to JSON

```
  benchmark_serving.py
      |
      +-- Generate random prompts (--dataset-name random)
      +-- Warmup requests (--num-warmups = 2 * concurrency)
      +-- Send benchmark requests:
      |     +-- num-prompts = concurrency * 10
      |     +-- max-concurrency = CONC
      |     +-- request-rate = inf (no throttling)
      +-- Collect metrics:
      |     +-- total_token_throughput (input + output tokens/sec)
      |     +-- output_throughput (output tokens/sec)
      |     +-- mean/median/p99 TTFT (time to first token)
      |     +-- mean/median/p99 TPOT (time per output token)
      |     +-- mean/median/p99 E2EL (end-to-end latency)
      +-- Write to {result_filename}.json
```

---

## 6. Stage 4: Result Processing & Publishing

### 6.1 Result Processing Flow

```
  Raw benchmark JSON                Processed JSON
  ({RESULT_FILENAME}.json)          (agg_{RESULT_FILENAME}.json)
       |                                  |
       v                                  v
  +-------------------+           +-------------------+
  | process_result.py |           | collect_results.py|
  | Per-job:          |           | All jobs:         |
  | - Read raw JSON   |           | - Download all    |
  | - Read env vars   |           |   agg_*.json      |
  | - Compute:        |           | - Merge into      |
  |   tput_per_gpu    |           |   single array    |
  |   output_tput/gpu |           | - Write           |
  |   input_tput/gpu  |           |   agg_bmk.json    |
  | - Convert ms->s   |           +-------------------+
  | - Compute intvty  |                  |
  |   (1000/TPOT)     |                  v
  +-------------------+           +-------------------+
                                  | summarize.py      |
                                  | - Read all JSON   |
                                  | - Group by config |
                                  | - Output markdown |
                                  |   summary tables  |
                                  +-------------------+
```

### 6.2 Key Metrics Computed

```
  process_result.py computes:
  +------------------------------------------+----------------------------------+
  | Metric                                   | Formula                          |
  +------------------------------------------+----------------------------------+
  | tput_per_gpu (total tokens/sec/GPU)      | total_throughput / num_gpus      |
  | output_tput_per_gpu                      | output_throughput / num_gpus     |
  | input_tput_per_gpu                       | (total - output) / num_gpus     |
  | *_ttft, *_tpot, *_e2el (in seconds)     | raw_ms_value / 1000             |
  | *_intvty (interactivity, tokens/sec)     | 1000 / tpot_ms                  |
  +------------------------------------------+----------------------------------+

  For single-node: num_gpus = TP
  For multi-node:  total_tput / (prefill_gpus + decode_gpus)
                   output_tput / decode_gpus
                   input_tput  / prefill_gpus
```

### 6.3 Publishing

After all results are collected:

```
  collect-results.yml
       |
       +-- Download all bmk_* artifacts
       +-- Run summarize.py -> GitHub Step Summary (markdown table)
       +-- Run collect_results.py -> agg_bmk.json
       +-- Upload results_bmk artifact
       |
       v
  trigger-vercel-deploy (only on main branch, all jobs successful)
       |
       +-- POST to Vercel deploy hook
       +-- Dashboard updates at inferencemax.ai
```

---

## 7. Data Flow Diagrams

### 7.1 Complete Data Flow

```
  Developer edits perf-changelog.yaml
       |
       v
  Git push to main (or PR with 'sweep-enabled' label)
       |
       v
  +-------------------------------------------------------------+
  |                   GitHub Actions                             |
  |                                                              |
  |  run-sweep.yml                                               |
  |    +-- setup job (ubuntu-latest)                             |
  |    |     +-- process_changelog.py                            |
  |    |     |     +-- git diff base..head perf-changelog.yaml   |
  |    |     |     +-- parse added entries                       |
  |    |     |     +-- resolve wildcards against master configs  |
  |    |     |     +-- generate_sweep_configs.py test-config     |
  |    |     |     +-- validate with ChangelogMatrixEntry        |
  |    |     +-- Output: search-space-config (JSON)              |
  |    |                                                         |
  |    +-- sweep-single-node-{1k1k,1k8k,8k1k} jobs              |
  |    |     +-- strategy.matrix from search-space-config        |
  |    |     +-- Each job runs benchmark-tmpl.yml on GPU runner  |
  |    |                                                         |
  |    +-- sweep-multi-node-{1k1k,1k8k,8k1k} jobs               |
  |    |     +-- Each job runs benchmark-multinode-tmpl.yml      |
  |    |                                                         |
  |    +-- collect-results job                                   |
  |    |     +-- Download all artifacts                          |
  |    |     +-- summarize.py -> Step Summary                    |
  |    |     +-- collect_results.py -> agg_bmk.json              |
  |    |                                                         |
  |    +-- collect-evals job                                     |
  |    |     +-- collect_eval_results.py -> agg_eval_all.json    |
  |    |                                                         |
  |    +-- calc-success-rate job                                 |
  |    |     +-- calc_success_rate.py -> run_stats.json          |
  |    |                                                         |
  |    +-- trigger-vercel-deploy job                             |
  |          +-- curl Vercel deploy hook                          |
  +-------------------------------------------------------------+
```

### 7.2 File I/O Flow

```
  INPUT FILES                           OUTPUT FILES
  +--------------------------+          +---------------------------------+
  | .github/configs/         |          | {RESULT_FILENAME}.json          |
  |   nvidia-master.yaml     |---+      |   (raw benchmark results)       |
  |   amd-master.yaml        |   |      |                                 |
  |   runners.yaml           |   |      | agg_{RESULT_FILENAME}.json      |
  +--------------------------+   |      |   (processed per-GPU metrics)    |
                                 |      |                                 |
  perf-changelog.yaml  ---------+----> | agg_bmk.json                    |
                                       |   (all results merged)           |
                                       |                                 |
                                       | agg_eval_all.json               |
                                       |   (all eval results merged)      |
                                       |                                 |
                                       | run_stats.json                  |
                                       |   (success/failure counts)       |
                                       |                                 |
                                       | server.log                      |
                                       |   (inference server logs)        |
                                       |                                 |
                                       | meta_env.json                   |
                                       |   (eval context metadata)        |
                                       +---------------------------------+
```

---

## 8. Validation Architecture

The system uses a "validate both ends" strategy:

```
  +------------------+     +-----------------------+     +-------------------+
  | Master YAML      |     | generate_sweep_       |     | Workflow          |
  | configs          |     | configs.py            |     | Templates         |
  |                  |     |                       |     |                   |
  | (human-edited)   |     | (Python logic)        |     | (GitHub Actions)  |
  +--------+---------+     +-----------+-----------+     +--------+----------+
           |                           |                          |
           v                           v                          v
  +------------------+     +-----------------------+     +-------------------+
  | INPUT VALIDATION |     | MATRIX GENERATION     |     | NO VALIDATION     |
  | Pydantic models: |     | Iterate configs,      |     | NEEDED            |
  | SingleNode/Multi |---->| apply filters,        |---->| All inputs are    |
  | NodeMasterConfig |     | expand concurrency,   |     | guaranteed valid  |
  | Entry            |     | validate each entry   |     | by Pydantic       |
  +------------------+     +-----------------------+     +-------------------+
         STOPS                    STOPS bad                  RUNTIME IS
         bad configs              matrix entries              SAFE
         EARLY                    EARLY
```

**Design principles:**
1. No defaults in output validation - missing values must fail
2. `extra='forbid'` - unknown fields are rejected (catches typos)
3. Strict typing - `Literal["mtp", "draft_model", "none"]` for enums
4. Concurrency validation - either `conc-list` OR `conc-start/conc-end`, never both

---

## 9. File-Level Component Map

```
InferenceX/
|
+-- .github/
|   +-- configs/
|   |   +-- nvidia-master.yaml      # NVIDIA benchmark definitions
|   |   +-- amd-master.yaml         # AMD benchmark definitions
|   |   +-- runners.yaml            # GPU type -> node mapping
|   |   +-- CONFIGS.md              # Config format documentation
|   |
|   +-- workflows/
|       +-- run-sweep.yml           # Main orchestrator (triggered by changelog)
|       +-- benchmark-tmpl.yml      # Single-node benchmark template
|       +-- benchmark-multinode-tmpl.yml  # Multi-node benchmark template
|       +-- e2e-tests.yml           # Manual/callable test workflow
|       +-- collect-results.yml     # Aggregate benchmark results
|       +-- collect-evals.yml       # Aggregate eval results
|       +-- test-matrix-logic.yml   # Unit test CI for matrix_logic/
|       +-- README.md               # Workflow usage documentation
|
+-- benchmarks/
|   +-- benchmark_lib.sh            # SHARED: all benchmark utilities
|   +-- dsr1_fp8_b200.sh            # Example: DSR1 FP8 on B200 (SGLang)
|   +-- dsr1_fp8_b200_trt.sh        # Example: DSR1 FP8 on B200 (TensorRT)
|   +-- dsr1_fp8_b200_trt_mtp.sh    # Example: with MTP speculative decoding
|   +-- gptoss_fp4_b200.sh          # Example: GPT-OSS FP4 on B200
|   +-- ...                         # ~30 scripts, one per model/gpu/framework
|
+-- runners/
|   +-- launch_b200-nv.sh           # NVIDIA B200 (Slurm + Enroot)
|   +-- launch_b200-dgxc.sh         # NVIDIA B200 DGX Cloud (Docker)
|   +-- launch_b200-dgxc-slurm.sh   # NVIDIA B200 DGX Cloud (Slurm multinode)
|   +-- launch_h200-cw.sh           # NVIDIA H200 (CoreWeave)
|   +-- launch_mi300x-amd.sh        # AMD MI300X (Docker)
|   +-- launch_mi355x-amds.sh       # AMD MI355X (Slurm)
|   +-- ...                         # ~19 scripts, one per node type
|
+-- utils/
|   +-- matrix_logic/
|   |   +-- generate_sweep_configs.py   # CLI: config -> matrix generation
|   |   +-- validation.py              # Pydantic models for all schemas
|   |   +-- test_validation.py         # Unit tests for validation
|   |   +-- test_generate_sweep_configs.py  # Unit tests for sweep generation
|   |
|   +-- bench_serving/
|   |   +-- benchmark_serving.py       # Load generator (from vLLM)
|   |   +-- backend_request_func.py    # Backend-specific request functions
|   |   +-- benchmark_utils.py         # Benchmark utility functions
|   |
|   +-- evals/
|   |   +-- gsm8k.yaml                # GSM8K eval task definition
|   |   +-- gpqa_diamond.yaml         # GPQA Diamond eval task definition
|   |   +-- utils.py                  # Eval utility functions
|   |   +-- EVALS.md                  # Eval documentation
|   |
|   +-- process_result.py            # Per-job result normalization
|   +-- process_changelog.py         # Changelog diff -> matrix generation
|   +-- collect_eval_results.py      # Merge all eval results
|   +-- collect_results.py           # Merge all benchmark results
|   +-- summarize.py                 # Generate markdown summary tables
|   +-- calc_success_rate.py         # Calculate job success rates
|   +-- constants.py                 # Paths to master configs, scripts
|   +-- test_process_result.py       # Tests for result processing
|
+-- experimental/                    # WIP experimental code (not production)
|
+-- perf-changelog.yaml             # THE TRIGGER: what configs to benchmark
+-- AGENTS.md                       # AI agent guidance
+-- README.md                       # Project overview
+-- LICENSE                         # Apache 2.0
```

---

## 10. Multi-Node Disaggregated Architecture

Multi-node configs split inference across multiple machines:

```
  +------------------+         +------------------+
  |  PREFILL NODES   |         |  DECODE NODES    |
  |  (process input) |         |  (generate output)|
  |                  |         |                  |
  |  num-worker: 1   |  KV     |  num-worker: 5   |
  |  tp: 4           |  cache  |  tp: 8           |
  |  ep: 4           |-------->|  ep: 8           |
  |  dp-attn: true   |  xfer   |  dp-attn: true   |
  +------------------+         +------------------+
```

**Config structure for multi-node:**

```yaml
dsr1-fp4-b200-dynamo-trt:
  multinode: true
  disagg: true
  framework: dynamo-trt
  seq-len-configs:
  - isl: 1024
    osl: 1024
    search-space:
    - conc-list: [1214]
      prefill:
        num-worker: 1          # Number of prefill nodes
        tp: 4                  # Tensor parallelism per node
        ep: 4                  # Expert parallelism per node
        dp-attn: true          # Data-parallel attention
        additional-settings:
        - "CONFIG_FILE=recipes/trtllm/b200-fp4/1k1k/mtp/ctx1_gen2.yaml"
      decode:
        num-worker: 2          # Number of decode nodes
        tp: 8
        ep: 8
        dp-attn: true
```

**Throughput calculation for multi-node:**

```
  total_tput_per_gpu = total_throughput / (prefill_gpus + decode_gpus)
  output_tput_per_gpu = output_throughput / decode_gpus
  input_tput_per_gpu = (total - output) / prefill_gpus
```

---

## 11. Evaluation Subsystem

Evals verify that inference optimizations don't degrade model accuracy.

```
  Benchmark Matrix
  +-------------------+
  | Entry 1 (no eval) |
  | Entry 2 (no eval) |
  | Entry 3 (EVAL)    |<-- Marked by mark_eval_entries()
  | Entry 4 (no eval) |
  | Entry 5 (EVAL)    |<-- Marked by mark_eval_entries()
  | ...               |
  +-------------------+
        |
        v (for eval entries only)
  +-----------------------------------+
  | After throughput benchmark:       |
  | 1. Install lm-eval dependencies   |
  | 2. Patch lm-eval:                 |
  |    - Handle reasoning_content     |
  |    - TRT compatibility            |
  | 3. Run: lm_eval --model           |
  |    local-chat-completions          |
  |    --tasks gsm8k                   |
  | 4. Write meta_env.json            |
  | 5. Upload eval artifacts           |
  +-----------------------------------+
        |
        v
  collect-evals.yml
  +-----------------------------------+
  | 1. Download all eval_* artifacts  |
  | 2. collect_eval_results.py:       |
  |    - Parse results JSON           |
  |    - Extract score, em_strict,    |
  |      em_flexible, n_eff           |
  |    - Merge with meta_env.json     |
  | 3. Output: agg_eval_all.json      |
  | 4. Publish table to Step Summary  |
  +-----------------------------------+
```

**Eval selection policy** (`mark_eval_entries()` in `generate_sweep_configs.py`):
- Only considers 1k8k sequence length (isl=1024, osl=8192)
- Only considers single-node entries (must have top-level `tp` field)
- For each unique (model, runner, framework, precision, isl, osl, spec-decoding, dp-attn):
  - Marks the entry with **highest TP + highest concurrency**
  - Marks the entry with **lowest TP + highest concurrency** (if different from above)
- This gives 2 eval points per configuration group: best parallelism and worst parallelism
