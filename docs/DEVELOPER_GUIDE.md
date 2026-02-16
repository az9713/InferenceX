# InferenceX Developer Guide

A step-by-step guide for developers new to the InferenceX codebase. No prior
experience with the specific technologies is assumed. If you can write Python
and use a terminal, you can follow this guide.

---

## Table of Contents

1. [Prerequisites & Setup](#1-prerequisites--setup)
2. [Understanding the Technology Stack](#2-understanding-the-technology-stack)
3. [Project Structure Walkthrough](#3-project-structure-walkthrough)
4. [How the System Works End-to-End](#4-how-the-system-works-end-to-end)
5. [Running Tests Locally](#5-running-tests-locally)
6. [Working with Configuration Files](#6-working-with-configuration-files)
7. [Working with the Config Generator CLI](#7-working-with-the-config-generator-cli)
8. [Adding a New Benchmark Configuration](#8-adding-a-new-benchmark-configuration)
9. [Writing Benchmark Scripts](#9-writing-benchmark-scripts)
10. [Writing Runner Launch Scripts](#10-writing-runner-launch-scripts)
11. [Working with the Validation System](#11-working-with-the-validation-system)
12. [Working with GitHub Actions Workflows](#12-working-with-github-actions-workflows)
13. [Working with Evals](#13-working-with-evals)
14. [Result Processing Pipeline](#14-result-processing-pipeline)
15. [Debugging Guide](#15-debugging-guide)
16. [Contributing Checklist](#16-contributing-checklist)
17. [Glossary](#17-glossary)

---

## 1. Prerequisites & Setup

### 1.1 What You Need Installed

| Tool | Version | Why | Install |
|------|---------|-----|---------|
| Python | 3.11+ | Core automation language | https://python.org/downloads |
| pip | (comes with Python) | Install Python packages | Included |
| Git | Any recent | Version control | https://git-scm.com |
| gh (GitHub CLI) | Any recent | Interact with GitHub | https://cli.github.com |
| A text editor | Any | Edit YAML, Python, Bash | VS Code recommended |

### 1.2 Clone the Repository

```bash
# Clone the repo
git clone https://github.com/SemiAnalysisAI/InferenceX.git
cd InferenceX
```

### 1.3 Install Python Dependencies

```bash
# Install core dependencies
pip install pydantic pyyaml pytest

# For result processing
pip install tabulate pandas numpy

# For the MCP server (optional, for Claude Code integration)
pip install -r .claude/requirements-mcp.txt
```

### 1.4 Verify Your Setup

```bash
# Run the unit tests - if these pass, your environment is good
cd utils
python -m pytest matrix_logic/ -v
cd ..
```

You should see output like:
```
test_validation.py::test_... PASSED
test_generate_sweep_configs.py::test_... PASSED
...
```

---

## 2. Understanding the Technology Stack

If you're coming from C/C++/Java, here's what each technology does and why
it's used:

### 2.1 Python (the main language)

All automation, config generation, and result processing is in Python. If you
know any programming language, Python will feel familiar. Key differences from
C/Java:
- No compilation step - just run `python script.py`
- Indentation matters (replaces `{}` braces)
- Type hints are optional but used here: `def foo(x: int) -> str:`

### 2.2 YAML (configuration format)

YAML is like JSON but more human-readable. It uses indentation instead of
braces. Here's a comparison:

```yaml
# YAML (what we use)
name: my-config
values:
  - item1
  - item2
settings:
  key: value
```

```json
// JSON equivalent
{
  "name": "my-config",
  "values": ["item1", "item2"],
  "settings": {"key": "value"}
}
```

**Important YAML rules:**
- Indentation must use spaces (not tabs)
- Lists start with `- ` (dash + space)
- Key-value pairs use `: ` (colon + space)
- Strings usually don't need quotes unless they contain special characters

### 2.3 Pydantic (validation library)

Pydantic is a Python library that validates data structure and types. Think of
it like a strict `struct` in C that rejects invalid data:

```python
# This is a Pydantic model (like a strict struct)
class PersonConfig(BaseModel):
    name: str           # Must be a string
    age: int            # Must be an integer
    active: bool        # Must be true/false

# This succeeds:
PersonConfig(name="Alice", age=30, active=True)

# This raises ValidationError (age is not an int):
PersonConfig(name="Alice", age="thirty", active=True)
```

In InferenceX, Pydantic validates both the YAML config files and the generated
benchmark matrix to catch errors before they reach expensive GPU hardware.

### 2.4 Bash (shell scripts)

Bash scripts (`*.sh` files) run on Linux machines. They're used for:
- Starting inference servers (Docker containers)
- Running benchmarks
- Managing GPU resources (via Slurm)

If you're not familiar with Bash, the key things to know:
- `#!/usr/bin/env bash` at the top means "run this with Bash"
- `export VAR=value` sets an environment variable
- `$VAR` reads an environment variable
- `command &` runs something in the background
- `source file.sh` includes another script's functions

### 2.5 GitHub Actions (CI/CD)

GitHub Actions automates running code when things change. Workflow files are
in `.github/workflows/` and written in YAML. Key concepts:
- **Workflow**: A YAML file that defines when and what to run
- **Job**: A set of steps that run on one machine
- **Step**: A single command or action
- **Runner**: The machine that runs the job (self-hosted GPU servers)
- **Artifact**: A file uploaded/downloaded between jobs
- **Matrix strategy**: Run the same job with different parameters

### 2.6 Docker / Enroot (containers)

Containers package an inference engine with all its dependencies into a
portable image. Think of it as a lightweight virtual machine:
- `docker run image_name` starts a container from an image
- Enroot is NVIDIA's alternative to Docker, designed for GPU clusters
- The `IMAGE` field in configs specifies which container to use

### 2.7 Slurm (GPU job scheduler)

Slurm manages GPU resources on clusters. Commands you'll see:
- `salloc` - Allocate GPUs
- `srun` - Run a command on allocated GPUs
- `scancel` - Cancel a job
- `squeue` - List running jobs

---

## 3. Project Structure Walkthrough

```
InferenceX/
|
|-- .github/configs/          # CONFIGURATION: What to benchmark
|   |-- nvidia-master.yaml    #   Every NVIDIA benchmark definition
|   |-- amd-master.yaml       #   Every AMD benchmark definition
|   +-- runners.yaml          #   Maps GPU types to physical machines
|
|-- .github/workflows/        # ORCHESTRATION: How benchmarks run in CI
|   |-- run-sweep.yml         #   Main workflow (triggered by changelog)
|   |-- e2e-tests.yml         #   Manual test workflow
|   |-- benchmark-tmpl.yml    #   Template for single-node benchmarks
|   |-- benchmark-multinode-tmpl.yml  # Template for multi-node
|   |-- collect-results.yml   #   Aggregate benchmark results
|   +-- collect-evals.yml     #   Aggregate eval results
|
|-- benchmarks/               # EXECUTION: Scripts that run benchmarks
|   |-- benchmark_lib.sh      #   Shared utilities (source this first)
|   |-- dsr1_fp8_b200.sh      #   One script per model/gpu/framework combo
|   +-- gptoss_fp4_mi355x.sh  #   ...etc
|
|-- runners/                  # INFRASTRUCTURE: Scripts that start containers
|   |-- launch_b200-nv.sh     #   One script per physical node type
|   +-- launch_mi355x-amds.sh #   ...etc
|
|-- utils/                    # TOOLS: Python utilities
|   |-- matrix_logic/         #   Config generation + validation
|   |-- bench_serving/        #   Benchmark load generator
|   |-- evals/                #   Eval task definitions
|   |-- process_result.py     #   Per-job result normalization
|   |-- process_changelog.py  #   Changelog diff -> matrix generation
|   +-- summarize.py          #   Generate markdown summary tables
|
+-- perf-changelog.yaml       # TRIGGER: What changed -> what to re-benchmark
```

---

## 4. How the System Works End-to-End

Here's what happens when someone makes a change:

**Step 1: Developer edits `perf-changelog.yaml`**

They add an entry saying "I updated SGLang to v0.5.8, re-benchmark these configs":

```yaml
- config-keys:
    - dsr1-fp8-b200-sglang
  description:
    - "Update SGLang to v0.5.8"
  pr-link: https://github.com/.../pull/XXX
```

**Step 2: Push to GitHub / Create PR**

The push triggers `.github/workflows/run-sweep.yml`.

**Step 3: setup job runs on ubuntu-latest**

- `process_changelog.py` computes git diff on `perf-changelog.yaml`
- Extracts only the new entries added in this commit
- Resolves config keys against master configs (expanding wildcards like `dsr1-fp8-*`)
- Calls `generate_sweep_configs.py test-config` to generate the full matrix
- Outputs a JSON matrix of benchmark jobs

**Step 4: Matrix jobs dispatch to GPU runners**

The JSON matrix is split by sequence length (1k1k, 1k8k, 8k1k) and node type
(single-node, multi-node). Each combination dispatches to the appropriate
workflow template.

**Step 5: Each benchmark job runs on a GPU machine**

1. Runner launch script starts the container (`runners/launch_*.sh`)
2. Benchmark script runs inside the container (`benchmarks/*.sh`)
3. Inference server starts, waits for health check
4. Benchmark client sends load, measures performance
5. Result JSON is written and uploaded as an artifact

**Step 6: Results are collected and published**

- `collect-results.yml` downloads all artifacts, merges them
- `summarize.py` generates markdown tables
- On main branch, a Vercel deploy is triggered to update the dashboard

---

## 5. Running Tests Locally

### 5.1 Run All Unit Tests

```bash
cd utils
python -m pytest matrix_logic/ -v
```

### 5.2 Run a Specific Test File

```bash
cd utils
python -m pytest matrix_logic/test_validation.py -v
```

### 5.3 Run a Single Test

```bash
cd utils
python -m pytest matrix_logic/test_validation.py::test_single_node_valid -v
```

### 5.4 Run Tests with Extra Output

```bash
cd utils
python -m pytest matrix_logic/ -v -s  # -s shows print statements
```

### 5.5 What the Tests Cover

- `test_validation.py` - Tests that Pydantic models accept valid configs and
  reject invalid ones (missing fields, wrong types, extra fields)
- `test_generate_sweep_configs.py` - Tests that the CLI generates correct
  matrix entries with proper filtering and concurrency expansion
- `test_process_result.py` - Tests result normalization logic

---

## 6. Working with Configuration Files

### 6.1 Master Config Files

**Location:** `.github/configs/nvidia-master.yaml` and `amd-master.yaml`

Each entry follows the naming convention: `<model>-<precision>-<gpu>-<framework>`

**Single-node entry example:**

```yaml
dsr1-fp8-b200-sglang:              # Entry name
  image: lmsysorg/sglang:v0.5.8    # Docker image
  model: deepseek-ai/DeepSeek-R1   # HuggingFace model
  model-prefix: dsr1                # Short code -> maps to benchmark script
  runner: b200                      # GPU type (from runners.yaml)
  precision: fp8                    # Numerical precision
  framework: sglang                 # Inference engine
  multinode: false                  # Single-node
  seq-len-configs:
  - isl: 1024                      # Input sequence length
    osl: 1024                      # Output sequence length
    search-space:
    - { tp: 8, conc-start: 4, conc-end: 64 }
      # tp=8: Use 8-way tensor parallelism
      # conc-start/end: Test concurrencies 4, 8, 16, 32, 64
    - { tp: 4, ep: 4, dp-attn: true, conc-start: 4, conc-end: 64 }
      # tp=4 with expert parallelism=4 and data-parallel attention
```

**Multi-node entry example:**

```yaml
dsr1-fp4-b200-dynamo-trt:
  image: nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.8.1
  model: deepseek-r1-fp4
  model-prefix: dsr1
  runner: b200-multinode-slurm
  precision: fp4
  framework: dynamo-trt
  multinode: true                   # Multi-node!
  disagg: true                      # Disaggregated prefill/decode
  seq-len-configs:
  - isl: 1024
    osl: 1024
    search-space:
    - conc-list: [1214]             # Specific concurrency values (not range)
      prefill:                      # Prefill node configuration
        num-worker: 1               # Number of prefill nodes
        tp: 4                       # Tensor parallelism per node
        ep: 4                       # Expert parallelism
        dp-attn: true
        additional-settings:
        - "CONFIG_FILE=recipes/..."  # External recipe file
      decode:                       # Decode node configuration
        num-worker: 2
        tp: 8
        ep: 8
        dp-attn: true
```

### 6.2 Runners Config

**Location:** `.github/configs/runners.yaml`

Maps GPU types to physical GitHub Actions runner names:

```yaml
b200:                    # GPU type (used in master config "runner" field)
- 'b200-dgxc_1'         # Physical runner node names
- 'b200-dgxc_2'
- 'b200-nv_0'
- 'b200-nv_1'

mi355x:
- 'mi355x-amds_0'
- 'mi355x-amds_1'
```

### 6.3 Performance Changelog

**Location:** `perf-changelog.yaml` (repository root)

This file drives which benchmarks run. Add entries here to trigger benchmarks:

```yaml
- config-keys:
    - dsr1-fp8-b200-sglang         # Exact config key
  description:
    - "Update SGLang image to v0.5.8"
  pr-link: https://github.com/.../pull/204

- config-keys:
    - dsr1-fp8-*-vllm              # Wildcard: matches all vLLM dsr1-fp8 configs
  description:
    - "Upgrade vLLM to v0.13.0"
  pr-link: https://github.com/.../pull/XXX
```

**Rules:**
- Only additions are allowed (deletions are rejected by `process_changelog.py`)
- Config keys can use `*` wildcards
- Each entry must have `config-keys`, `description`, and `pr-link`
- The file must end with a newline character

---

## 7. Working with the Config Generator CLI

The CLI tool `utils/matrix_logic/generate_sweep_configs.py` is the bridge
between YAML configs and GitHub Actions matrix jobs.

### 7.1 full-sweep: Generate All Matching Configs

```bash
# Generate ALL single-node configs (this will be a large matrix)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --single-node

# Filter by model
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --single-node \
  --model-prefix dsr1

# Combine multiple filters
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml .github/configs/amd-master.yaml \
  --single-node \
  --model-prefix dsr1 \
  --framework sglang \
  --precision fp8 \
  --runner-type b200 \
  --seq-lens 1k1k

# Limit concurrency for quick testing
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --single-node \
  --max-conc 16 \
  --max-tp 4
```

### 7.2 runner-model-sweep: Validate All Nodes

```bash
# Test all H200 nodes with all H200 configs
python utils/matrix_logic/generate_sweep_configs.py runner-model-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --runner-config .github/configs/runners.yaml \
  --runner-type h200 \
  --single-node
```

### 7.3 test-config: Expand Specific Config Keys

```bash
# Generate matrix for specific config keys
python utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files .github/configs/nvidia-master.yaml \
  --config-keys dsr1-fp8-b200-sglang dsr1-fp4-b200-sglang
```

### 7.4 Understanding the Output

The CLI outputs a JSON array. Each element is one benchmark job:

```json
[
  {
    "image": "lmsysorg/sglang:v0.5.8",
    "model": "deepseek-ai/DeepSeek-R1-0528",
    "model-prefix": "dsr1",
    "precision": "fp8",
    "framework": "sglang",
    "runner": "b200",
    "isl": 1024,
    "osl": 1024,
    "tp": 8,
    "ep": 1,
    "dp-attn": false,
    "conc": 4,
    "max-model-len": 2248,
    "exp-name": "dsr1_1k1k",
    "disagg": false,
    "run-eval": false,
    "spec-decoding": "none"
  },
  {
    "...same but conc": 8
  },
  {
    "...same but conc": 16
  }
]
```

Notice how `conc-start: 4, conc-end: 64` expands into separate entries for
each concurrency level (4, 8, 16, 32, 64).

---

## 8. Adding a New Benchmark Configuration

This is the most common development task. Follow these steps exactly:

### Step 1: Choose Your Parameters

Decide:
- **Model**: Which LLM? (e.g., `deepseek-ai/DeepSeek-R1-0528`)
- **Model prefix**: Short code (e.g., `dsr1`)
- **GPU**: Which hardware? (e.g., `b200`, `mi355x`)
- **Framework**: Which engine? (e.g., `sglang`, `vllm`, `trt`)
- **Precision**: `fp4` or `fp8`
- **Docker image**: The container image with the engine

### Step 2: Add Entry to Master Config

Edit `.github/configs/nvidia-master.yaml` (or `amd-master.yaml` for AMD):

```yaml
# Add at the end of the file
dsr1-fp8-b300-sglang:
  image: lmsysorg/sglang:v0.5.8-cu130
  model: deepseek-ai/DeepSeek-R1-0528
  model-prefix: dsr1
  runner: b300
  precision: fp8
  framework: sglang
  multinode: false
  seq-len-configs:
  - isl: 1024
    osl: 1024
    search-space:
    - { tp: 8, conc-start: 4, conc-end: 64 }
  - isl: 1024
    osl: 8192
    search-space:
    - { tp: 8, conc-start: 4, conc-end: 64 }
  - isl: 8192
    osl: 1024
    search-space:
    - { tp: 8, conc-start: 4, conc-end: 64 }
```

### Step 3: Add Runner (if new GPU type)

If your GPU type doesn't exist in `runners.yaml`, add it:

```yaml
b300:
- 'b300-nv_0'
```

### Step 4: Create Benchmark Script (if needed)

If no `benchmarks/dsr1_fp8_b300.sh` exists, create one:

```bash
#!/usr/bin/env bash

source "$(dirname "$0")/benchmark_lib.sh"

# Validate required environment variables
check_env_vars MODEL TP CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME EP_SIZE

# Download the model
hf download "$MODEL"

# Server configuration
SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

# Start the inference server
python3 -m sglang.launch_server \
  --model-path=$MODEL \
  --host=0.0.0.0 --port=$PORT \
  --tensor-parallel-size=$TP \
  --kv-cache-dtype fp8_e4m3 \
  --ep-size $EP_SIZE \
  --quantization fp8 \
  > $SERVER_LOG 2>&1 &

SERVER_PID=$!

# Wait for server to be ready (polls /health endpoint)
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# Install benchmark dependencies
pip install -q datasets pandas

# Run the benchmark
run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir /workspace/

# Run evaluation if requested
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
```

### Step 5: Create Runner Script (if needed)

If no `runners/launch_b300-nv.sh` exists, create one modeled on an existing
runner script.

### Step 6: Validate Your Config

```bash
# This will fail if your config has errors
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --single-node \
  --runner-type b300
```

### Step 7: Add Performance Changelog Entry

Add to `perf-changelog.yaml`:

```yaml
- config-keys:
    - dsr1-fp8-b300-sglang
  description:
    - "Add DSR1 FP8 B300 SGLang benchmark configuration"
  pr-link: https://github.com/SemiAnalysisAI/InferenceX/pull/XXX
```

### Step 8: Run Tests

```bash
cd utils
python -m pytest matrix_logic/ -v
```

### Step 9: Create PR

Create a pull request. Add the `sweep-enabled` label to trigger benchmarks.

---

## 9. Writing Benchmark Scripts

### 9.1 Script Naming Convention

```
benchmarks/{MODEL_CODE}_{PRECISION}_{GPU}[_{FRAMEWORK}][_{SPEC}].sh
```

Examples:
- `dsr1_fp8_b200.sh` - DSR1 FP8 on B200 (default framework: SGLang)
- `dsr1_fp8_b200_trt.sh` - DSR1 FP8 on B200 with TensorRT-LLM
- `dsr1_fp8_b200_trt_mtp.sh` - Same but with MTP speculative decoding
- `gptoss_fp4_mi355x_atom.sh` - GPT-OSS FP4 on MI355X with ATOM

### 9.2 Script Template

Every benchmark script follows this pattern:

```bash
#!/usr/bin/env bash

# 1. Source shared utilities
source "$(dirname "$0")/benchmark_lib.sh"

# 2. Validate environment variables
check_env_vars MODEL TP CONC ISL OSL RANDOM_RANGE_RATIO RESULT_FILENAME EP_SIZE

# 3. (Optional) Show GPU info
nvidia-smi  # or rocm-smi for AMD

# 4. Download model
hf download "$MODEL"

# 5. Configure and start inference server
SERVER_LOG=/workspace/server.log
PORT=${PORT:-8888}

python3 -m sglang.launch_server \
  --model-path=$MODEL \
  --host=0.0.0.0 --port=$PORT \
  --tensor-parallel-size=$TP \
  [... framework-specific args ...] \
  > $SERVER_LOG 2>&1 &
SERVER_PID=$!

# 6. Wait for server
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# 7. Run benchmark
run_benchmark_serving \
    --model "$MODEL" --port "$PORT" --backend vllm \
    --input-len "$ISL" --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$((CONC * 10))" --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" --result-dir /workspace/

# 8. (Optional) Run evals
if [ "${RUN_EVAL}" = "true" ]; then
    run_eval --framework lm-eval --port "$PORT" --concurrent-requests $CONC
    append_lm_eval_summary
fi
```

### 9.3 Available Environment Variables

These are set by the workflow templates and passed to your script:

| Variable | Example | Description |
|----------|---------|-------------|
| `MODEL` | `deepseek-ai/DeepSeek-R1-0528` | HuggingFace model path |
| `MODEL_PREFIX` | `dsr1` | Short model code |
| `TP` | `8` | Tensor parallelism |
| `EP_SIZE` | `4` | Expert parallelism |
| `DP_ATTENTION` | `true` | Data-parallel attention flag |
| `CONC` | `32` | Benchmark concurrency |
| `ISL` | `1024` | Input sequence length |
| `OSL` | `8192` | Output sequence length |
| `MAX_MODEL_LEN` | `9392` | Maximum model context length |
| `PRECISION` | `fp8` | Numerical precision |
| `FRAMEWORK` | `sglang` | Inference framework |
| `IMAGE` | `lmsysorg/sglang:v0.5.8` | Docker image |
| `RANDOM_RANGE_RATIO` | `0.8` | Randomization ratio for prompts |
| `RESULT_FILENAME` | `dsr1_1k1k_fp8_sglang_tp8...` | Output filename |
| `RUN_EVAL` | `true` | Whether to run accuracy evals |
| `SPEC_DECODING` | `none` | Speculative decoding mode |

### 9.4 Important Rules

1. **Do NOT create new directories in `/workspace/`** during the benchmark.
   Files are fine, but new directories can cause issues.
2. Always use `check_env_vars` to validate inputs at the start.
3. Always start the server in the background with `&` and capture `$!` as PID.
4. Always use `wait_for_server_ready` before running benchmarks.
5. The result file must be at `/workspace/{RESULT_FILENAME}.json`.

---

## 10. Writing Runner Launch Scripts

### 10.1 Script Naming Convention

```
runners/launch_{RUNNER_NODE_PREFIX}.sh
```

The runner node name (e.g., `b200-nv_0`) is stripped to get the prefix
(`b200-nv`), which maps to the script (`launch_b200-nv.sh`).

### 10.2 Two Types of Launchers

**Docker-based** (simpler):
```bash
#!/usr/bin/bash
docker run --gpus all --rm \
  -v $GITHUB_WORKSPACE:/workspace \
  -v $HF_HUB_CACHE:/mnt/hf_hub_cache \
  --network host \
  -e MODEL -e TP -e CONC -e ISL -e OSL ... \
  $IMAGE \
  bash /workspace/benchmarks/${MODEL_CODE}_${PRECISION}_${GPU}.sh
```

**Slurm + Enroot** (for GPU clusters):
```bash
#!/usr/bin/bash
# 1. Allocate GPU(s)
salloc --partition=$PARTITION --gres=gpu:$TP --exclusive

# 2. Import Docker image as Enroot squash file
srun enroot import -o $SQUASH_FILE docker://$IMAGE

# 3. Run benchmark inside container
srun --container-image=$SQUASH_FILE \
     --container-mounts=$GITHUB_WORKSPACE:/workspace/ \
     bash benchmarks/{script}.sh

# 4. Cleanup
scancel $JOB_ID
```

---

## 11. Working with the Validation System

### 11.1 How Validation Works

The system validates data at two points (see `utils/matrix_logic/validation.py`):

1. **Input validation**: When master config YAML files are loaded
2. **Output validation**: When matrix entries are generated

Both use Pydantic models with `extra='forbid'` to catch any unexpected fields.

### 11.2 Adding a New Config Field

If you need to add a new field to the config format:

**Step 1:** Add the field name to the `Fields` enum:
```python
# In validation.py
class Fields(Enum):
    ...
    MY_NEW_FIELD = 'my-new-field'  # kebab-case for YAML
```

**Step 2:** Add to the appropriate Pydantic model(s):
```python
class SingleNodeSearchSpaceEntry(BaseModel):
    ...
    my_new_field: Optional[int] = Field(
        default=None, alias=Fields.MY_NEW_FIELD.value)
```

**Step 3:** Update the matrix generation in `generate_sweep_configs.py`:
```python
entry = {
    ...
    Fields.MY_NEW_FIELD.value: bmk.get(Fields.MY_NEW_FIELD.value),
}
```

**Step 4:** Update the output validation model:
```python
class SingleNodeMatrixEntry(BaseModel):
    ...
    my_new_field: Optional[int] = Field(alias=Fields.MY_NEW_FIELD.value)
```

**Step 5:** Add tests in `test_validation.py`.

**Step 6:** Update the workflow template (`benchmark-tmpl.yml`) to accept and
pass through the new field.

---

## 12. Working with GitHub Actions Workflows

### 12.1 Running E2E Tests Manually

Go to GitHub Actions -> "End-to-End Tests" -> "Run workflow" and enter:

```
full-sweep --single-node --model-prefix dsr1 --runner-type b200 --max-conc 16 --config-files .github/configs/nvidia-master.yaml
```

### 12.2 Triggering Benchmarks via PR

1. Create a PR that modifies `perf-changelog.yaml`
2. Add the `sweep-enabled` label to the PR
3. The `run-sweep.yml` workflow will trigger automatically

### 12.3 Skipping Benchmarks

Include `[skip-sweep]` in your commit message to skip benchmarks on push to main.

### 12.4 Understanding Workflow Template Inputs

The workflow templates (`benchmark-tmpl.yml`, `benchmark-multinode-tmpl.yml`)
accept inputs that exactly match the fields in the matrix entries. These are
validated by Pydantic, so if something changes in the matrix, update both the
Pydantic model and the workflow template.

---

## 13. Working with Evals

### 13.1 What Evals Do

Evals run accuracy checks (e.g., GSM8K math problems) after throughput
benchmarks to ensure inference optimizations don't degrade model quality.

### 13.2 When Evals Run

Evals are off by default. They run when:
- `--run-evals` or `--evals-only` flag is passed to the config generator
- The system automatically selects 2 representative configs per group:
  - Highest TP with highest concurrency
  - Lowest TP with highest concurrency
- Only for 1k8k sequence length

### 13.3 Adding a New Eval Task

1. Create a YAML task file in `utils/evals/` following the lm-eval format
2. Set `EVAL_TASK=<your_task>` when running
3. Update `utils/collect_eval_results.py` if new metrics need extraction

---

## 14. Result Processing Pipeline

### 14.1 Per-Job Processing

`utils/process_result.py` runs after each benchmark:

1. Reads raw `{RESULT_FILENAME}.json` from benchmark_serving.py
2. Reads environment variables for context (GPU type, framework, etc.)
3. Computes per-GPU metrics:
   - `tput_per_gpu = total_throughput / num_gpus`
   - Converts millisecond metrics to seconds
   - Computes interactivity (`1000 / TPOT_ms`)
4. Writes `agg_{RESULT_FILENAME}.json`

### 14.2 Aggregation

`collect-results.yml` downloads all `agg_*.json` artifacts and merges them
into a single `agg_bmk.json` array.

### 14.3 Querying Results

```bash
# Download results from a CI run
gh run download <RUN_ID> --repo SemiAnalysisAI/InferenceX -n results_bmk -D ./results

# Summary table
cat ./results/agg_bmk.json | jq -r '
  .[] | [.hw, .framework, .precision, (.tput_per_gpu | round)]
  | @tsv' | column -t

# Filter by hardware
cat ./results/agg_bmk.json | jq '[.[] | select(.hw == "b200")]'
```

---

## 15. Debugging Guide

### 15.1 Config Validation Errors

**Symptom:** `ValueError: Master config entry 'X' failed validation`

**Fix:** Check your YAML against the Pydantic model in `validation.py`.
Common issues:
- Missing required fields
- Extra fields (typos)
- Wrong types (string instead of int)
- Both `conc-list` and `conc-start/conc-end` specified

### 15.2 Server Won't Start

**Symptom:** `Server died before becoming healthy`

**Fix:** Check `server.log` in the uploaded artifacts. Common issues:
- Model doesn't fit in GPU memory (reduce TP or use smaller model)
- Docker image doesn't have the right framework version
- Port conflict

### 15.3 Benchmark Returns No Results

**Symptom:** `Benchmark result {file}.json not found`

**Fix:** The benchmark script didn't write its output. Check:
- Is `RESULT_FILENAME` set correctly?
- Is `result-dir` set to `/workspace/`?
- Did the benchmark client crash? Check server.log.

### 15.4 CI Not Triggering

**Symptom:** No workflow runs after pushing

**Fix:**
- For PRs: Is the `sweep-enabled` label added?
- Does the commit modify `perf-changelog.yaml`?
- Is `[skip-sweep]` in the commit message?
- Is the PR a draft? Drafts are skipped.

---

## 16. Contributing Checklist

Before submitting a PR:

- [ ] Config entries follow naming convention: `<model>-<precision>-<gpu>-<framework>`
- [ ] Config validates: `python utils/matrix_logic/generate_sweep_configs.py full-sweep --config-files ... --single-node`
- [ ] Unit tests pass: `python -m pytest utils/matrix_logic/ -v`
- [ ] `perf-changelog.yaml` entry added (if benchmarks should run)
- [ ] `perf-changelog.yaml` ends with a newline
- [ ] Benchmark script follows the template pattern (source benchmark_lib.sh, check_env_vars, etc.)
- [ ] No new directories created in `/workspace/` during benchmarks
- [ ] Conventional commit message used

---

## 17. Glossary

| Term | Meaning |
|------|---------|
| **TP** | Tensor Parallelism - splitting a model across GPUs |
| **EP** | Expert Parallelism - splitting MoE experts across GPUs |
| **DP-Attn** | Data-Parallel Attention - parallelizing attention computation |
| **Concurrency** | Number of simultaneous requests sent to the server |
| **ISL** | Input Sequence Length (number of input tokens) |
| **OSL** | Output Sequence Length (number of output tokens) |
| **TTFT** | Time To First Token - latency before first token appears |
| **TPOT** | Time Per Output Token - latency per generated token |
| **E2EL** | End-to-End Latency - total time for a complete request |
| **Interactivity** | 1/TPOT - tokens per second perceived by user |
| **MoE** | Mixture of Experts - model architecture with specialized sub-networks |
| **MTP** | Multi-Token Prediction - speculative decoding technique |
| **Disaggregated** | Separating prefill and decode onto different nodes |
| **Prefill** | Processing the input prompt (compute-heavy) |
| **Decode** | Generating output tokens one at a time (memory-heavy) |
| **Enroot** | NVIDIA's container runtime for GPU clusters |
| **Slurm** | Job scheduler for GPU clusters |
| **KV Cache** | Key-Value cache for attention, stored in GPU memory |
| **Framework** | Inference engine (vLLM, SGLang, TensorRT-LLM, ATOM) |
