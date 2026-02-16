# InferenceX Quick Start Guide

Get started with InferenceX in minutes. This guide walks you through 12
hands-on use cases, from simple to intermediate. Each one gives you a
quick win and teaches a specific skill.

---

## Prerequisites

Before starting, make sure you have:

```bash
# 1. Clone the repo
git clone https://github.com/SemiAnalysisAI/InferenceX.git
cd InferenceX

# 2. Install Python dependencies
pip install pydantic pyyaml pytest

# 3. Verify everything works
cd utils && python -m pytest matrix_logic/ -v && cd ..
```

You also need the GitHub CLI (`gh`) for use cases involving CI results:
```bash
# Install: https://cli.github.com
gh auth login
```

---

## Use Case 1: Validate Your Setup

**Goal:** Confirm that the codebase and your Python environment work correctly.

**What you'll learn:** How to run the test suite.

```bash
cd utils
python -m pytest matrix_logic/ -v
```

**Expected output:**
```
test_validation.py::test_... PASSED
test_generate_sweep_configs.py::test_... PASSED
...
X passed in Y.YYs
```

If all tests pass, your environment is ready.

---

## Use Case 2: See All Available GPU Runners

**Goal:** List all GPU hardware types and their physical nodes.

**What you'll learn:** How the runners.yaml config works.

```bash
# View the runners configuration
cat .github/configs/runners.yaml
```

**What you'll see:** A YAML file mapping GPU types (b200, h200, mi355x, etc.)
to lists of physical machine names. Each machine name corresponds to a
self-hosted GitHub Actions runner.

**Try this:** Count how many nodes exist per GPU type:
```bash
python3 -c "
import yaml
with open('.github/configs/runners.yaml') as f:
    runners = yaml.safe_load(f)
for gpu_type, nodes in runners.items():
    print(f'{gpu_type}: {len(nodes)} nodes')
"
```

---

## Use Case 3: List All Benchmark Configurations

**Goal:** See every benchmark config defined in the system.

**What you'll learn:** How master config files are structured.

```bash
# List all NVIDIA config names
python3 -c "
import yaml
with open('.github/configs/nvidia-master.yaml') as f:
    configs = yaml.safe_load(f)
for name in sorted(configs.keys()):
    c = configs[name]
    print(f\"{name}: {c['framework']} on {c['runner']} ({c['precision']})\")"
```

```bash
# List all AMD config names
python3 -c "
import yaml
with open('.github/configs/amd-master.yaml') as f:
    configs = yaml.safe_load(f)
for name in sorted(configs.keys()):
    c = configs[name]
    print(f\"{name}: {c['framework']} on {c['runner']} ({c['precision']})\")"
```

---

## Use Case 4: Generate a Simple Benchmark Matrix

**Goal:** Generate the JSON matrix for a single model/GPU combination.

**What you'll learn:** How the config generator CLI works.

```bash
# Generate matrix for DeepSeek R1 on B200 with SGLang, single-node, 1k1k only
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --single-node \
  --model-prefix dsr1 \
  --framework sglang \
  --runner-type b200 \
  --seq-lens 1k1k \
  | python3 -m json.tool
```

**What you'll see:** A JSON array where each element is one benchmark job.
Notice how `conc-start: 4, conc-end: 64` expands into separate entries for
concurrency values 4, 8, 16, 32, 64.

---

## Use Case 5: Count How Many Jobs a Sweep Generates

**Goal:** Understand the scale of a benchmark sweep.

**What you'll learn:** How filter flags reduce the number of jobs.

```bash
# Count ALL single-node NVIDIA jobs (this will be a large number)
echo "All NVIDIA single-node jobs:"
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --single-node \
  | python3 -c "import json,sys; data=json.load(sys.stdin); print(f'  {len(data)} jobs')"

# Count with filters applied
echo "Only DSR1 FP8 B200 SGLang 1k1k jobs:"
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --single-node \
  --model-prefix dsr1 \
  --framework sglang \
  --runner-type b200 \
  --seq-lens 1k1k \
  | python3 -c "import json,sys; data=json.load(sys.stdin); print(f'  {len(data)} jobs')"
```

---

## Use Case 6: Understand a Single Matrix Entry

**Goal:** Read and understand every field in a generated benchmark job.

**What you'll learn:** What each field means and how it's used.

```bash
# Generate one entry and pretty-print it
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --single-node \
  --model-prefix dsr1 \
  --framework sglang \
  --runner-type b200 \
  --seq-lens 1k1k \
  --max-conc 4 \
  --max-tp 8 \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
if data:
    entry = data[0]
    for key, value in entry.items():
        print(f'  {key:20s} = {value}')
"
```

**Field-by-field explanation:**
- `image` - Docker container with the inference engine
- `model` - HuggingFace model identifier
- `model-prefix` - Short code (dsr1 = DeepSeek R1)
- `precision` - fp4 or fp8
- `framework` - Which inference engine
- `runner` - Which GPU type
- `isl` / `osl` - Input/output sequence lengths
- `tp` - Tensor parallelism (how many GPUs)
- `ep` - Expert parallelism
- `dp-attn` - Data-parallel attention enabled?
- `conc` - Concurrency level for this run
- `max-model-len` - Maximum context length (isl + osl + 200)
- `exp-name` - Experiment name (used in result filenames)
- `disagg` - Disaggregated inference?
- `run-eval` - Run accuracy eval after benchmark?
- `spec-decoding` - Speculative decoding mode

---

## Use Case 7: Compare Frameworks for One GPU

**Goal:** See which frameworks are configured for a specific GPU.

**What you'll learn:** How to filter by runner type and compare frameworks.

```bash
# What frameworks run on B200?
python3 -c "
import yaml
with open('.github/configs/nvidia-master.yaml') as f:
    configs = yaml.safe_load(f)
frameworks = set()
for name, cfg in configs.items():
    if cfg['runner'].startswith('b200'):
        frameworks.add(cfg['framework'])
print('Frameworks on B200:', sorted(frameworks))
"
```

```bash
# Generate matrix for ALL frameworks on B200 (dsr1, 1k1k only)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --single-node \
  --model-prefix dsr1 \
  --runner-type b200 \
  --seq-lens 1k1k \
  --max-conc 16 \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
from collections import Counter
fw_counts = Counter(e['framework'] for e in data)
for fw, count in sorted(fw_counts.items()):
    print(f'  {fw}: {count} benchmark jobs')
"
```

---

## Use Case 8: Validate a Config Change

**Goal:** Check that a YAML config change is valid before committing.

**What you'll learn:** How the validation system catches errors.

```bash
# First, try generating from the existing config (should succeed)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/amd-master.yaml \
  --single-node \
  --runner-type mi355x \
  --max-conc 4 \
  > /dev/null && echo "Validation PASSED"

# Now let's see what happens with an invalid config
# (Don't worry, we won't modify any files)
python3 -c "
import yaml, tempfile, subprocess
# Load and corrupt a config
with open('.github/configs/amd-master.yaml') as f:
    config = yaml.safe_load(f)
first_key = list(config.keys())[0]
config[first_key]['unknown-field'] = 'this should fail'
# Write to temp file and try to validate
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
    yaml.dump(config, tmp)
    tmp_path = tmp.name
result = subprocess.run(
    ['python', 'utils/matrix_logic/generate_sweep_configs.py', 'full-sweep',
     '--config-files', tmp_path, '--single-node'],
    capture_output=True, text=True)
if result.returncode != 0:
    print('Validation FAILED (as expected):')
    # Show just the key error line
    for line in result.stderr.split('\\n')[:3]:
        print(f'  {line}')
"
```

**What you'll learn:** The Pydantic validation catches unknown fields, missing
fields, and type mismatches before any benchmarks run.

---

## Use Case 9: Explore Multi-Node Configurations

**Goal:** See how multi-node disaggregated configs differ from single-node.

**What you'll learn:** The prefill/decode split architecture.

```bash
# Generate a multi-node config
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --multi-node \
  --max-conc 100 \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
if data:
    entry = data[0]
    print('Multi-node entry:')
    print(f'  Framework: {entry[\"framework\"]}')
    print(f'  Runner: {entry[\"runner\"]}')
    print(f'  Disagg: {entry[\"disagg\"]}')
    print(f'  Concurrency values: {entry[\"conc\"]}')
    if 'prefill' in entry:
        p = entry['prefill']
        print(f'  Prefill: {p[\"num-worker\"]} workers, TP={p[\"tp\"]}, EP={p[\"ep\"]}')
    if 'decode' in entry:
        d = entry['decode']
        print(f'  Decode: {d[\"num-worker\"]} workers, TP={d[\"tp\"]}, EP={d[\"ep\"]}')
else:
    print('No multi-node configs found')
"
```

---

## Use Case 10: See Which Configs Get Evals

**Goal:** Understand how eval entries are automatically selected.

**What you'll learn:** The eval selection policy (highest/lowest TP with highest concurrency).

```bash
# Generate configs with eval marking
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --single-node \
  --model-prefix dsr1 \
  --framework sglang \
  --runner-type b200 \
  --run-evals \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
eval_entries = [e for e in data if e.get('run-eval', False)]
non_eval = [e for e in data if not e.get('run-eval', False)]
print(f'Total entries: {len(data)}')
print(f'With eval: {len(eval_entries)}')
print(f'Without eval: {len(non_eval)}')
print()
for e in eval_entries:
    print(f'  EVAL: tp={e[\"tp\"]} conc={e[\"conc\"]} isl={e[\"isl\"]} osl={e[\"osl\"]}')
"
```

**Key insight:** Evals only run on 1k8k sequence length, and only at the
extremes of tensor parallelism. This minimizes eval cost while covering the
range of parallelism settings.

---

## Use Case 11: Generate an Evals-Only Subset

**Goal:** Generate only the benchmark entries that include accuracy evaluation.

**What you'll learn:** How `--evals-only` filters the matrix.

```bash
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --single-node \
  --model-prefix dsr1 \
  --evals-only \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'Eval-only entries: {len(data)}')
for e in data:
    print(f'  {e[\"framework\"]:10s} tp={e[\"tp\"]} conc={e[\"conc\"]} {e[\"runner\"]} {e[\"isl\"]}/{e[\"osl\"]}')
"
```

---

## Use Case 12: Inspect the Performance Changelog

**Goal:** See what recent benchmarks have been triggered and why.

**What you'll learn:** How the changelog drives the benchmarking pipeline.

```bash
# Show the last 5 changelog entries
python3 -c "
import yaml
with open('perf-changelog.yaml') as f:
    entries = yaml.safe_load(f)
print(f'Total changelog entries: {len(entries)}')
print()
for entry in entries[-5:]:
    keys = ', '.join(entry['config-keys'])
    desc = entry['description'][0] if entry['description'] else 'No description'
    print(f'  Keys: {keys}')
    print(f'  Why:  {desc}')
    print(f'  PR:   {entry[\"pr-link\"]}')
    print()
"
```

---

## What's Next?

Now that you've completed these use cases, you can:

1. **Read the full docs:**
   - `docs/ARCHITECTURE.md` - System architecture with diagrams
   - `docs/DEVELOPER_GUIDE.md` - Complete developer reference
   - `docs/USER_GUIDE.md` - Full user documentation

2. **Try advanced tasks:**
   - Add a new benchmark configuration (see Developer Guide, Section 8)
   - Trigger a benchmark run via PR (see User Guide, Section 6)
   - Analyze CI results with jq (see User Guide, Section 7)

3. **Explore the code:**
   - `utils/matrix_logic/validation.py` - Understand the validation system
   - `benchmarks/benchmark_lib.sh` - Understand shared benchmark utilities
   - `.github/workflows/run-sweep.yml` - Understand CI orchestration
