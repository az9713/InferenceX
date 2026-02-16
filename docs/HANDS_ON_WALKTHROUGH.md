# InferenceX Hands-On Walkthrough: What You Ran and What It Means

This document is a detailed line-by-line explanation of every command you can run
on a local machine (no GPU required), what each part does, and how to interpret
every piece of the output. Written for someone who wants to extract maximum
learning from this codebase without access to datacenter GPUs.

---

## Table of Contents

1. [The Big Picture First](#1-the-big-picture-first)
2. [Command 1: Installing Dependencies](#2-command-1-installing-dependencies)
3. [Command 2: Running the Test Suite](#3-command-2-running-the-test-suite)
4. [Command 3: Generating a Benchmark Matrix](#4-command-3-generating-a-benchmark-matrix)
5. [Tracing Config to Output: A Complete Walkthrough](#5-tracing-config-to-output-a-complete-walkthrough)
6. [More Commands to Try](#6-more-commands-to-try)
7. [What Happens After (On Real GPUs)](#7-what-happens-after-on-real-gpus)
8. [Glossary](#8-glossary)

---

## 1. The Big Picture First

InferenceX is a benchmarking system that measures how fast AI models can generate
text on different GPUs using different software engines. Think of it like a car
testing lab:

```
  Car Testing Analogy              InferenceX Equivalent
  ─────────────────────            ─────────────────────
  Which car model?            -->  Which AI model? (DeepSeek-R1, GPT-OSS)
  Which engine type?          -->  Which inference engine? (SGLang, vLLM, TRT)
  Which test track?           -->  Which GPU? (B200, H200, MI355X)
  How many cars at once?      -->  How many users at once? (concurrency: 4, 8, 16...)
  Miles per hour (speed)      -->  Tokens per second (throughput)
  0-to-60 time (responsiveness) -> Time to first token (latency)
```

The codebase has a 4-stage pipeline. **You can only run Stage 1 locally**
(shown with *), but that's where the interesting logic lives:

```
  * Stage 1: YAML Config --> Python CLI --> JSON job matrix
    Stage 2: GitHub Actions reads JSON, dispatches jobs
    Stage 3: GPU machines run benchmarks in Docker containers
    Stage 4: Results aggregated to dashboard (inferencemax.ai)
```

Everything you ran on your laptop is Stage 1: turning YAML configs into a list
of benchmark jobs (JSON). On a real system, GitHub Actions would take that JSON
and send each job to the right GPU machine.

---

## 2. Command 1: Installing Dependencies

### What You Typed

```bash
pip install pydantic pyyaml pytest
```

### Breaking It Down

| Part | What It Does |
|------|-------------|
| `pip` | Python's package installer. Like `apt-get` for Linux or `npm` for JavaScript. It downloads and installs third-party libraries from PyPI (Python Package Index, a public repository of Python packages). |
| `install` | The pip subcommand that means "download and install these packages". |
| `pydantic` | A data validation library. InferenceX uses it to enforce strict rules on config files. If someone puts `tp: "eight"` instead of `tp: 8`, Pydantic catches it immediately. Think of it as a strict type-checker for data structures. (Used in `utils/matrix_logic/validation.py`) |
| `pyyaml` | A library for reading/writing YAML files. YAML is the human-readable config format used throughout InferenceX (like `.github/configs/nvidia-master.yaml`). It's similar to JSON but easier for humans to read and write. |
| `pytest` | A testing framework. Runs all test files that start with `test_` and reports pass/fail for each test function. |

### What the Output Meant

```
Collecting pydantic
  Using cached pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
```

- "Collecting" = pip is resolving what version to download
- "Using cached" = you already downloaded this package before (it's in pip's cache, so it doesn't re-download from the internet)
- `.whl` = "wheel" file, a pre-built Python package (no compilation needed)
- `py3-none-any` = works on Python 3, any OS, any CPU architecture

```
Installing collected packages: typing-extensions, pyyaml, pygments, pluggy,
  packaging, iniconfig, colorama, annotated-types, typing-inspection, pytest,
  pydantic-core, pydantic
```

You asked for 3 packages, but 12 got installed. That's because each package has
its own dependencies:

```
  You requested          What it pulled in (dependencies)
  ─────────────          ────────────────────────────────
  pydantic          -->  pydantic-core, annotated-types, typing-extensions,
                         typing-inspection
  pyyaml            -->  (none - self-contained)
  pytest            -->  pluggy, packaging, iniconfig, colorama, pygments
```

This is normal. Libraries build on other libraries. `pydantic-core` is the fast
Rust-compiled engine that Pydantic V2 uses under the hood. `colorama` makes
colored terminal output work on Windows.

---

## 3. Command 2: Running the Test Suite

### What You Typed

```bash
cd utils && python -m pytest matrix_logic/ -v && cd ..
```

### Breaking It Down

| Part | What It Does |
|------|-------------|
| `cd utils` | Change directory into `utils/`. The tests import modules using relative paths, so you need to be in this directory. |
| `&&` | "Run the next command ONLY if the previous one succeeded." If `cd utils` fails, pytest won't run. This is a safety chain. |
| `python -m pytest` | Run pytest as a Python module. The `-m` flag means "run this module as a script." This is more reliable than just `pytest` because it ensures you're using the same Python that has pytest installed. |
| `matrix_logic/` | The directory containing the test files. pytest will automatically discover all files matching `test_*.py` in this directory. |
| `-v` | "Verbose" mode. Without this, pytest just shows dots (`.` = pass, `F` = fail). With `-v`, it shows the full name of each test. |
| `&& cd ..` | Go back to the project root after tests complete. |

### Interpreting the Output

**Header:**
```
platform win32 -- Python 3.13.5, pytest-9.0.2, pluggy-1.6.0
```
- `platform win32` = you're on Windows
- `Python 3.13.5` = your Python version
- `pytest-9.0.2` = testing framework version
- `pluggy-1.6.0` = pytest's plugin system version

**Test discovery:**
```
cachedir: .pytest_cache
rootdir: <project-root>\utils\matrix_logic
configfile: pytest.ini
collected 115 items
```
- `collected 115 items` = pytest found 115 individual test functions to run
- `pytest.ini` = a config file that tells pytest how to behave (e.g., which
  directories to search)

**Individual tests:**
```
matrix_logic\test_generate_sweep_configs.py::TestSeqLenMappings::test_seq_len_stoi_values PASSED [  0%]
```

This line reads as:
```
  File                              Class                Method                Result  Progress
  ────                              ─────                ──────                ──────  ────────
  test_generate_sweep_configs.py :: TestSeqLenMappings :: test_seq_len_stoi_values  PASSED  [  0%]
```

- **File**: `test_generate_sweep_configs.py` - tests for the config generator
- **Class**: `TestSeqLenMappings` - a group of related tests about sequence length mappings
- **Method**: `test_seq_len_stoi_values` - one specific test that verifies the string-to-integer mapping works: `"1k1k"` --> `(1024, 1024)`
- **PASSED**: the test's assertions were all true
- **[  0%]**: progress through all 115 tests

### What the Two Test Files Cover

**`test_generate_sweep_configs.py`** (54 tests) - Tests the config generation logic:

| Test Class | What It Tests | Example |
|-----------|--------------|---------|
| `TestSeqLenMappings` | String <--> number conversion | `"1k1k"` = `(1024, 1024)` |
| `TestSeqLenToStr` | Reverse conversion | `(1024, 8192)` = `"1k8k"` |
| `TestGenerateFullSweepSingleNode` | Main config expansion | One YAML entry → many JSON jobs |
| `TestGenerateFullSweepMultiNode` | Multi-machine configs | Prefill/decode worker setups |
| `TestGenerateRunnerModelSweepConfig` | Runner validation | Each GPU node gets tested |
| `TestEdgeCases` | Special configs | Expert parallelism, speculative decoding |
| `TestArgumentDefaults` | CLI argument parsing | Default values for optional flags |

**`test_validation.py`** (61 tests) - Tests the Pydantic validation layer:

| Test Class | What It Tests | Example |
|-----------|--------------|---------|
| `TestFieldsEnum` | Field name constants | `Fields.TP.value == "tp"` |
| `TestWorkerConfig` | Multi-node worker setup | Prefill/decode configs |
| `TestSingleNodeMatrixEntry` | Output JSON validation | Every field has correct type |
| `TestMultiNodeMatrixEntry` | Multi-node output | Must have prefill AND decode |
| `TestValidateMatrixEntry` | Validation wrapper | Invalid data raises ValueError |
| `TestSingleNodeSearchSpaceEntry` | Config input validation | conc-start must be <= conc-end |
| `TestMultiNodeSearchSpaceEntry` | Multi-node config input | Must have worker definitions |
| `TestSeqLenConfigs` | Sequence length blocks | isl/osl with search-space list |
| `TestMasterConfigEntries` | Top-level config entries | All required fields present |
| `TestValidateMasterConfig` | Full config validation | Catches unknown/missing fields |
| `TestValidateRunnerConfig` | Runner file validation | GPU types map to node lists |
| `TestLoadConfigFiles` | File loading | Merge multiple YAML files |
| `TestLoadRunnerFile` | Runner file loading | Parse runners.yaml |

**Final line:**
```
============================= 115 passed in 1.29s =============================
```
All 115 tests passed in 1.29 seconds. If any test had failed, you'd see `FAILED`
in red with a detailed traceback showing what was expected vs. what was actually
returned.

---

## 4. Command 3: Generating a Benchmark Matrix

### What You Typed

```bash
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
    --config-files .github/configs/nvidia-master.yaml \
    --single-node \
    --model-prefix dsr1 \
    --runner-type b200 \
    --seq-lens 1k1k \
    | python -m json.tool
```

### Breaking Down Every Part

**`python utils/matrix_logic/generate_sweep_configs.py`**

Runs the main config generator Python script. This is the heart of Stage 1.
The script is located at `utils/matrix_logic/generate_sweep_configs.py` (931
lines). It uses Python's `argparse` module to parse command-line arguments,
similar to how a C program might use `getopt`.

**`full-sweep`**

This is a **subcommand**. The script has 3 subcommands:

| Subcommand | Purpose |
|-----------|---------|
| `full-sweep` | Generate ALL benchmark jobs matching your filters. This is what the nightly CI runs. |
| `runner-model-sweep` | Generate ONE job per physical GPU node. Used to verify that all machines in a cluster are working. |
| `test-config` | Generate jobs for specific named configs (e.g., `dsr1-fp8-b200-sglang`). Used for debugging one config. |

**`--config-files .github/configs/nvidia-master.yaml`**

Tells the script which YAML configuration file(s) to read. This file is 1800+
lines and contains every NVIDIA benchmark configuration. There's also an
`amd-master.yaml` for AMD GPUs. You can pass multiple files:
`--config-files nvidia-master.yaml amd-master.yaml`.

**`--single-node`**

This is a **required mutually exclusive flag**. You must choose exactly one:

| Flag | What It Selects |
|------|----------------|
| `--single-node` | Configs where `multinode: false` (model runs on one machine, 1-8 GPUs) |
| `--multi-node` | Configs where `multinode: true` (model spans multiple machines) |

**`--model-prefix dsr1`**

A **filter**. Only include configs whose `model-prefix` field matches `dsr1`
(which is the short code for DeepSeek-R1-0528). Without this, you'd get configs
for ALL models (dsr1 AND gptoss).

Available model prefixes: `dsr1` (DeepSeek-R1), `gptoss` (GPT-OSS-120B)

**`--runner-type b200`**

Another **filter**. Only include configs whose `runner` field matches `b200`.
Without this, you'd get configs for ALL GPU types (b200, h100, h200, mi355x, etc.)

Available runner types include: `b200`, `b200-trt`, `h100`, `h200`, `mi300x`,
`mi325x`, `mi355x`, `gb200`, and more.

**`--seq-lens 1k1k`**

Another **filter**. Only include sequence length configurations matching `1k1k`
(input=1024 tokens, output=1024 tokens). Without this, you'd get all sequence
lengths: 1k1k, 1k8k, 8k1k.

| Code | Input Length | Output Length | Real-world Analogy |
|------|-------------|---------------|-------------------|
| `1k1k` | 1,024 tokens | 1,024 tokens | Short chat conversation |
| `1k8k` | 1,024 tokens | 8,192 tokens | Short question, long detailed answer |
| `8k1k` | 8,192 tokens | 1,024 tokens | Long document, short summary |

**`| python -m json.tool`**

The `|` (pipe) takes the output from the first command and feeds it to the second
command. `python -m json.tool` is Python's built-in JSON pretty-printer. The raw
output is one giant line of compressed JSON. The pipe formats it with indentation
so humans can read it.

Without the pipe, you'd see:
```
[{"image":"lmsysorg/sglang:v0.5.6-cu129-amd64","model":"nvidia/DeepSeek-R1-0528-FP4-V2",...},{"image":...}]
```

With the pipe, you get the nicely indented version you saw.

### Interpreting the JSON Output

Your command produced **25 JSON objects** in an array. Each object is one
**benchmark job** - a single test that would be dispatched to a GPU machine.
Let's examine the first one:

```json
{
    "image": "lmsysorg/sglang:v0.5.6-cu129-amd64",
    "model": "nvidia/DeepSeek-R1-0528-FP4-V2",
    "model-prefix": "dsr1",
    "precision": "fp4",
    "framework": "sglang",
    "runner": "b200",
    "isl": 1024,
    "osl": 1024,
    "tp": 4,
    "conc": 4,
    "max-model-len": 2248,
    "ep": 4,
    "dp-attn": false,
    "spec-decoding": "none",
    "exp-name": "dsr1_1k1k",
    "disagg": false,
    "run-eval": false
}
```

**Field-by-field explanation:**

| Field | Value | What It Means |
|-------|-------|--------------|
| `image` | `lmsysorg/sglang:v0.5.6-cu129-amd64` | **Docker container image.** This is the exact software package that runs on the GPU machine. `lmsysorg/sglang` is the SGLang project, `v0.5.6` is the version, `cu129` means CUDA 12.9 (NVIDIA's GPU programming toolkit), `amd64` means 64-bit x86 CPU architecture. Think of it as a frozen snapshot of an operating system + all required software. |
| `model` | `nvidia/DeepSeek-R1-0528-FP4-V2` | **The AI model to load.** This is a HuggingFace model ID. The GPU machine downloads this model (hundreds of gigabytes) and loads it into GPU memory. `FP4-V2` means this version is pre-quantized to 4-bit precision. |
| `model-prefix` | `dsr1` | **Short code for the model.** Used in file naming and filtering. `dsr1` = DeepSeek-R1. |
| `precision` | `fp4` | **Number precision.** `fp4` = 4-bit floating point. Each number in the model's neural network weights is stored using only 4 bits instead of the standard 16 or 32. This makes it faster (less data to move) but slightly less accurate. |
| `framework` | `sglang` | **The inference engine.** SGLang is the software that actually runs the AI model and serves requests. It handles GPU memory management, request scheduling, batching multiple users together, etc. |
| `runner` | `b200` | **Which GPU type** this job should run on. NVIDIA B200 is a Blackwell-generation datacenter GPU. |
| `isl` | `1024` | **Input Sequence Length.** Each request sends 1,024 tokens of input. A "token" is roughly 3/4 of a word, so ~768 words of input. |
| `osl` | `1024` | **Output Sequence Length.** The model generates 1,024 tokens of output per request. |
| `tp` | `4` | **Tensor Parallelism.** The model is split across 4 GPUs. DeepSeek-R1 has 671 billion parameters - it's too large for one GPU. TP=4 means each GPU holds 1/4 of the model, and they communicate to process each request together. |
| `conc` | `4` | **Concurrency.** The load generator sends 4 simultaneous requests to the server. This simulates 4 users chatting with the AI at the same time. Higher concurrency = more realistic but higher latency per user. |
| `max-model-len` | `2248` | **Maximum context window.** Calculated as `isl + osl + 200` = `1024 + 1024 + 200` = `2248`. The extra 200 tokens is a safety buffer. This tells the inference engine how much GPU memory to allocate for the key-value cache (the "working memory" the model uses while generating). |
| `ep` | `4` | **Expert Parallelism.** DeepSeek-R1 is a Mixture-of-Experts (MoE) model - it has many "expert" sub-networks and routes each input to the most relevant ones. EP=4 means the expert weights are distributed across 4 GPUs. |
| `dp-attn` | `false` | **Data-Parallel Attention.** When true, the attention computation (the most memory-intensive part) is replicated across GPUs rather than split. False means standard tensor-parallel attention. |
| `spec-decoding` | `none` | **Speculative Decoding.** An optimization where a small "draft" model predicts multiple tokens ahead, then the big model verifies them in one pass. `none` means this optimization is disabled. `mtp` means Multi-Token Prediction is enabled. |
| `exp-name` | `dsr1_1k1k` | **Experiment name.** Used in result file names and the dashboard. Format: `{model-prefix}_{seq-len}`. |
| `disagg` | `false` | **Disaggregated inference.** When true, the "thinking" phase (prefill) and "speaking" phase (decode) run on separate machines. False = both phases on the same machine. |
| `run-eval` | `false` | **Run accuracy evaluation.** When true, after the speed benchmark, the system also runs math/reasoning tests (GSM8K, GPQA) to verify the model still gives correct answers after optimization. |

### Why 25 Jobs? Tracing the Expansion

Your command matched **3 YAML config entries** in `nvidia-master.yaml`. Here's
exactly how each one expanded:

**Config 1: `dsr1-fp4-b200-sglang`** (line 1652 in nvidia-master.yaml)

```yaml
dsr1-fp4-b200-sglang:
  image: lmsysorg/sglang:v0.5.6-cu129-amd64
  model: nvidia/DeepSeek-R1-0528-FP4-V2
  precision: fp4
  framework: sglang
  runner: b200
  multinode: false
  seq-len-configs:
  - isl: 1024           # <-- matches your --seq-lens 1k1k filter
    osl: 1024
    search-space:
    - { tp: 4, ep: 4, conc-start: 4, conc-end: 128 }   # 4,8,16,32,64,128 = 6 jobs
    - { tp: 8, ep: 8, conc-start: 4, conc-end: 128 }   # 4,8,16,32,64,128 = 6 jobs
```

`conc-start: 4, conc-end: 128` expands geometrically (doubling each time):
4 → 8 → 16 → 32 → 64 → 128 = **6 values**

Two search-space entries × 6 concurrency values = **12 jobs** from this config.

These are jobs 1-12 in your output (all with `precision: fp4`).

**Config 2: `dsr1-fp8-b200-sglang`** (line 1762)

```yaml
dsr1-fp8-b200-sglang:
  image: lmsysorg/sglang:v0.5.6-cu129-amd64
  model: deepseek-ai/DeepSeek-R1-0528        # Different model (not pre-quantized)
  precision: fp8                               # FP8 instead of FP4
  framework: sglang
  runner: b200
  seq-len-configs:
  - isl: 1024
    osl: 1024
    search-space:
    - { tp: 8, ep: 1, conc-start: 4, conc-end: 64 }    # 4,8,16,32,64 = 5 jobs
```

One search-space entry × 5 concurrency values = **5 jobs** from this config.

Note: FP8 only has `tp: 8` (needs all 8 GPUs because FP8 uses more memory per
parameter than FP4). Also `ep: 1` means expert parallelism is not used - the
MoE routing happens within each GPU's tensor-parallel slice.

These are jobs 13-17 in your output (with `precision: fp8`, `spec-decoding: none`).

**Config 3: `dsr1-fp8-b200-sglang-mtp`** (line 1785)

```yaml
dsr1-fp8-b200-sglang-mtp:
  image: lmsysorg/sglang:v0.5.8-cu130-amd64    # Newer SGLang version!
  model: deepseek-ai/DeepSeek-R1-0528
  precision: fp8
  framework: sglang
  runner: b200
  seq-len-configs:
  - isl: 1024
    osl: 1024
    search-space:
    - { tp: 8, ep: 1, conc-start: 4, conc-end: 512, spec-decoding: mtp }
      #  4,8,16,32,64,128,256,512 = 8 jobs
```

One search-space entry × 8 concurrency values = **8 jobs** from this config.

Note: This config enables speculative decoding (`mtp`) and tests much higher
concurrency (up to 512 simultaneous users). It also uses a newer SGLang version
(`v0.5.8`) that supports the MTP feature.

These are jobs 18-25 in your output (with `spec-decoding: mtp`).

**Total: 12 + 5 + 8 = 25 jobs.** ✓

### What Patterns to Notice in the Output

**1. FP4 vs FP8 uses different models:**
- FP4: `nvidia/DeepSeek-R1-0528-FP4-V2` (pre-quantized by NVIDIA)
- FP8: `deepseek-ai/DeepSeek-R1-0528` (original model, quantized at runtime)

**2. FP4 can use fewer GPUs:**
- FP4 has entries with `tp: 4` AND `tp: 8`
- FP8 only has `tp: 8`
- FP4 compresses the model more, so it fits in fewer GPUs

**3. MTP enables higher concurrency:**
- Without MTP: max concurrency is 64-128
- With MTP: max concurrency is 512
- Speculative decoding makes each request faster, so the server can handle more

**4. Same exp-name across all entries:**
- All 25 jobs have `exp-name: dsr1_1k1k`
- Results get grouped by experiment name on the dashboard
- The dashboard then shows how throughput/latency changes across concurrency levels

---

## 5. Tracing Config to Output: A Complete Walkthrough

To understand how the Python code turns YAML into JSON, here's the exact
execution path:

### Step 1: Parse Command-Line Arguments

File: `generate_sweep_configs.py`, line 901

```python
args = parser.parse_args()
```

Your arguments become an `argparse.Namespace` object:
```python
args.command = "full-sweep"
args.config_files = [".github/configs/nvidia-master.yaml"]
args.single_node = True
args.multi_node = False
args.model_prefix = ["dsr1"]
args.runner_type = "b200"
args.seq_lens = ["1k1k"]
# All optional filters you didn't specify are None
```

### Step 2: Load and Validate the YAML Config

File: `validation.py`, function `load_config_files()`

```python
all_config_data = load_config_files(args.config_files)
```

This:
1. Opens `nvidia-master.yaml`
2. Parses it with PyYAML (turns YAML text into Python dictionaries)
3. **Validates every entry** using Pydantic models
   - Each entry is checked against `SingleNodeMasterConfigEntry` or
     `MultiNodeMasterConfigEntry` depending on the `multinode` field
   - Unknown fields → error (Pydantic's `extra='forbid'`)
   - Wrong types → error (e.g., `tp: "eight"` instead of `tp: 8`)
   - Missing required fields → error
4. Returns a dict of ALL configs (hundreds of entries)

### Step 3: Filter Configs

File: `generate_sweep_configs.py`, function `generate_full_sweep()`

The function loops through all configs and applies your filters:

```python
for config_name, config in all_config_data.items():
    # Skip multinode configs (you passed --single-node)
    if config[Fields.MULTINODE.value] == True:
        continue

    # Skip non-matching model prefix
    if args.model_prefix and config[Fields.MODEL_PREFIX.value] not in args.model_prefix:
        continue   # Skips gptoss configs

    # Skip non-matching runner type
    if args.runner_type and not config[Fields.RUNNER.value].startswith(args.runner_type):
        continue   # Skips h100, h200, mi355x, etc.
```

After filtering: only `dsr1-fp4-b200-sglang`, `dsr1-fp8-b200-sglang`, and
`dsr1-fp8-b200-sglang-mtp` survive.

### Step 4: Expand Concurrency Ranges

For each surviving config, for each matching sequence length, for each
search-space entry, the code expands concurrency:

```python
# From conc-start: 4, conc-end: 128
conc = conc_start  # 4
while conc <= conc_end:
    # Create one JSON entry with this concurrency value
    matrix_values.append({...})
    conc *= 2  # Double: 4 → 8 → 16 → 32 → 64 → 128
```

### Step 5: Validate Output

Each generated JSON entry is validated by `validate_matrix_entry()`:

```python
validate_matrix_entry(entry, multinode=False)
```

This uses `SingleNodeMatrixEntry` Pydantic model to verify:
- All required fields are present
- No extra/unknown fields
- All values have correct types
- This catches bugs in the config generator itself

### Step 6: Print as JSON

```python
print(json.dumps(matrix_values))
```

The 25 entries are serialized to a JSON array and printed to stdout.
Your `| python -m json.tool` pipe then pretty-prints it.

---

## 6. More Commands to Try

### 6.1 Count Jobs Without Seeing the Full JSON

```bash
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
    --config-files .github/configs/nvidia-master.yaml \
    --single-node \
    --model-prefix dsr1 \
    --runner-type b200 \
    --seq-lens 1k1k \
    | python -c "import json,sys; data=json.load(sys.stdin); print(f'{len(data)} jobs')"
```

What this does:
- `python -c "..."` runs a one-liner Python script
- `json.load(sys.stdin)` reads the JSON from the pipe
- `len(data)` counts the array elements
- Expected output: `25 jobs`

### 6.2 See ALL Sequence Lengths (Remove the Filter)

```bash
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
    --config-files .github/configs/nvidia-master.yaml \
    --single-node \
    --model-prefix dsr1 \
    --runner-type b200 \
    | python -c "import json,sys; data=json.load(sys.stdin); print(f'{len(data)} jobs')"
```

This removes `--seq-lens 1k1k`, so you get 1k1k + 1k8k + 8k1k. Expect many
more jobs.

### 6.3 See What AMD Configs Look Like

```bash
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
    --config-files .github/configs/amd-master.yaml \
    --single-node \
    --runner-type mi355x \
    --seq-lens 1k1k \
    --max-conc 16 \
    | python -m json.tool
```

New flag: `--max-conc 16` caps the concurrency at 16. So instead of going
4, 8, 16, 32, 64... it stops at 16. This gives fewer, more manageable results.

### 6.4 Compare Frameworks on One GPU

```bash
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
    --config-files .github/configs/nvidia-master.yaml \
    --single-node \
    --model-prefix dsr1 \
    --runner-type b200 \
    --seq-lens 1k1k \
    --max-conc 4 \
    | python -c "
import json, sys
data = json.load(sys.stdin)
for entry in data:
    print(f'{entry[\"framework\"]:10s} prec={entry[\"precision\"]} tp={entry[\"tp\"]} ep={entry[\"ep\"]} spec={entry[\"spec-decoding\"]}')
"
```

This shows all frameworks available for B200 at concurrency 4. `--max-conc 4`
gives you just one concurrency level per search-space entry so you can compare
framework configurations side-by-side.

### 6.5 Count Jobs for the ENTIRE NVIDIA Suite

```bash
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
    --config-files .github/configs/nvidia-master.yaml \
    --single-node \
    | python -c "import json,sys; data=json.load(sys.stdin); print(f'{len(data)} total single-node NVIDIA jobs')"
```

This removes ALL filters. You'll see how many individual benchmark jobs the
nightly CI runs across all NVIDIA hardware. It's a large number.

### 6.6 See Which Configs Would Get Accuracy Evals

```bash
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
    --config-files .github/configs/nvidia-master.yaml \
    --single-node \
    --model-prefix dsr1 \
    --runner-type b200 \
    --run-evals \
    | python -c "
import json, sys
data = json.load(sys.stdin)
evals = [e for e in data if e.get('run-eval')]
print(f'{len(data)} total jobs, {len(evals)} with evals')
for e in evals:
    print(f'  tp={e[\"tp\"]} conc={e[\"conc\"]} isl={e[\"isl\"]} osl={e[\"osl\"]} spec={e[\"spec-decoding\"]}')
"
```

New flag: `--run-evals` marks certain entries with `run-eval: true`. The eval
policy is: only 1k8k sequence length, highest TP + highest concurrency, lowest
TP + highest concurrency. This minimizes eval cost while still catching accuracy
issues.

### 6.7 Explore a Single YAML Config in Detail

```bash
python -c "
import yaml
with open('.github/configs/nvidia-master.yaml') as f:
    configs = yaml.safe_load(f)
# Look at one specific config
import json
print(json.dumps(configs['dsr1-fp4-b200-sglang'], indent=2))
"
```

This reads the raw YAML and shows you exactly what the config generator reads as
input - before any expansion happens.

### 6.8 List All Config Names Matching a Pattern

```bash
python -c "
import yaml
with open('.github/configs/nvidia-master.yaml') as f:
    configs = yaml.safe_load(f)
for name in sorted(configs.keys()):
    if 'b200' in name and 'dsr1' in name:
        c = configs[name]
        print(f'{name:40s} framework={c[\"framework\"]:10s} multinode={c[\"multinode\"]}')
"
```

Shows all B200 + DeepSeek-R1 configs, including multi-node ones your previous
command filtered out.

### 6.9 View the Runner Definitions

```bash
python -c "
import yaml
with open('.github/configs/runners.yaml') as f:
    runners = yaml.safe_load(f)
for gpu_type, nodes in runners.items():
    print(f'{gpu_type:30s} {len(nodes)} nodes: {nodes[:3]}...' if len(nodes) > 3 else f'{gpu_type:30s} {len(nodes)} nodes: {nodes}')
"
```

Shows which physical machines exist for each GPU type. Each node name is a
self-hosted GitHub Actions runner label.

### 6.10 Run a Single Specific Test

```bash
cd utils && python -m pytest matrix_logic/test_generate_sweep_configs.py::TestGenerateFullSweepSingleNode::test_basic_sweep_generation -v && cd ..
```

The `::` syntax lets you run just one test. Useful for debugging.

---

## 7. What Happens After (On Real GPUs)

For completeness, here's what happens with the JSON you generated, even though
you can't run these stages locally:

### Stage 2: GitHub Actions Reads the JSON

The workflow file `.github/workflows/run-sweep.yml` runs
`generate_sweep_configs.py` and feeds the JSON into a **matrix strategy**:

```yaml
strategy:
  matrix:
    include: ${{ fromJSON(needs.setup.outputs.matrix) }}
```

GitHub Actions creates a separate **parallel job** for each of the 25 entries.
So 25 GPU machines start working simultaneously.

### Stage 3: Each GPU Machine Runs a Benchmark

For each job, the workflow:
1. Starts a Docker container with the specified `image`
2. Downloads the AI model specified in `model`
3. Launches the inference server (SGLang/vLLM/TRT)
4. Waits for the server to be ready (health check polling)
5. Runs `bench_serving.py` - a load generator that sends `conc` simultaneous
   requests, each with `isl` input tokens, expecting `osl` output tokens
6. Collects metrics: throughput, TTFT, TPOT, E2EL
7. Optionally runs accuracy evals (GSM8K math problems, GPQA reasoning)

### Stage 4: Results Published

Results are:
- Saved as JSON artifacts in GitHub Actions
- Aggregated by `utils/process_result.py`
- Summarized in markdown tables by `utils/summarize.py`
- Published to the live dashboard at https://inferencemax.ai/

---

## 8. Glossary

| Term | Definition |
|------|-----------|
| **Token** | The smallest unit of text for an AI model. Roughly 3/4 of an English word. "Hello world" ≈ 2 tokens. |
| **Throughput** | Speed of text generation, measured in tokens per second per GPU. Higher = better. |
| **TTFT** | Time To First Token. How long a user waits before the first word appears. Lower = better. |
| **TPOT** | Time Per Output Token. Time between each subsequent word appearing. Lower = better. |
| **E2EL** | End-to-End Latency. Total time from request to complete response. Lower = better. |
| **Concurrency** | Number of simultaneous users/requests. Higher concurrency = more realistic production load. |
| **Tensor Parallelism (TP)** | Splitting a model across multiple GPUs. TP=8 means 8 GPUs each hold 1/8 of the model. |
| **Expert Parallelism (EP)** | Splitting the "expert" sub-networks of a Mixture-of-Experts model across GPUs. |
| **FP4 / FP8** | Floating-point precision: 4-bit or 8-bit. Lower precision = faster but potentially less accurate. |
| **MoE** | Mixture of Experts. A model architecture where only a subset of parameters are active per token, enabling larger models with less compute. DeepSeek-R1 uses this. |
| **MTP** | Multi-Token Prediction (speculative decoding). The model predicts multiple tokens at once and verifies them in a single pass, improving throughput. |
| **Disaggregated** | Separating the "understanding" phase (prefill) from the "generating" phase (decode) onto different machines for better resource utilization. |
| **Pydantic** | A Python library for data validation. Enforces type safety and schema rules on data structures. Used here to catch config errors before expensive GPU time is wasted. |
| **Docker** | A containerization system. Packages software + all dependencies into a reproducible "container" that runs identically on any machine. Each inference engine runs in its own Docker container. |
| **YAML** | A human-readable data format. Used for all config files. Indentation-based (like Python), unlike JSON which uses braces. |
| **GitHub Actions** | GitHub's CI/CD (Continuous Integration / Continuous Deployment) system. Runs automated workflows when code changes. InferenceX uses it to orchestrate benchmark runs across GPU machines. |
| **Matrix Strategy** | A GitHub Actions feature that runs the same job multiple times with different parameters. InferenceX uses this to run many benchmark configs in parallel. |
| **HuggingFace** | A platform that hosts AI models. Model IDs like `deepseek-ai/DeepSeek-R1-0528` reference models on huggingface.co. |

---

*This document was generated to help extract maximum learning from the InferenceX
codebase on a local machine without GPU access.*
