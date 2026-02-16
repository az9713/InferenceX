# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

InferenceX is an open-source automated benchmarking system that continuously tracks LLM inference performance across hardware platforms (NVIDIA B200/B300/H100/H200/GB200/GB300, AMD MI300X/MI325X/MI355X) and software stacks (vLLM, SGLang, TensorRT-LLM, ATOM). Results publish to https://inferencemax.ai/.

## Commands

### Run tests
```bash
cd utils && python -m pytest matrix_logic/ -v
```

### Generate benchmark configs
```bash
# Full sweep (--single-node or --multi-node is required)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --single-node

# With filters (all combinable)
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml .github/configs/amd-master.yaml \
  --single-node \
  --model-prefix dsr1 --framework sglang --precision fp8 --runner-type b200 \
  --seq-lens 1k1k --max-conc 64 --max-tp 8

# With evals marked
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml --single-node --run-evals

# Evals-only subset
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml --single-node --evals-only

# Test specific config keys
python utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files .github/configs/nvidia-master.yaml \
  --config-keys dsr1-fp8-b200-sglang

# Runner validation sweep
python utils/matrix_logic/generate_sweep_configs.py runner-model-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --runner-config .github/configs/runners.yaml \
  --runner-type h200 --single-node
```

### Process results
```bash
python utils/process_result.py
python utils/summarize.py <results_directory>
```

## Architecture

Four-stage pipeline: **config generation** -> **CI dispatch** -> **benchmark execution** -> **result processing**.

### Config generation (`utils/matrix_logic/`)
- `generate_sweep_configs.py`: CLI with subcommands `full-sweep`, `runner-model-sweep`, `test-config`. Reads master YAML configs, applies filters, expands concurrency ranges, outputs JSON matrix.
- `validation.py`: Pydantic V2 models (strict mode, `extra='forbid'`). Validates both inputs (master configs via `SingleNodeMasterConfigEntry`/`MultiNodeMasterConfigEntry`) and outputs (matrix entries via `SingleNodeMatrixEntry`/`MultiNodeMatrixEntry`). Field aliases for kebab-case YAML: `Field(alias="model-prefix")`.
- Concurrency expansion: `conc-start`/`conc-end` with step=2x produces geometric series. `conc-list` for explicit values.
- Eval selection: `mark_eval_entries()` selects highest/lowest TP with highest concurrency, 1k8k only.

### CI/CD (`.github/workflows/`)
- `run-sweep.yml`: Main entry point. Triggered by changes to `perf-changelog.yaml`. Uses `process_changelog.py` to diff, resolve wildcards, generate matrix. Splits jobs by seq-len (1k1k/1k8k/8k1k) and node type (single/multi).
- `benchmark-tmpl.yml`: Single-node template. Runs on self-hosted GPU runners. Steps: cleanup, checkout, launch runner script, process result, upload artifacts.
- `benchmark-multinode-tmpl.yml`: Multi-node template with Slurm orchestration.
- `e2e-tests.yml`: Manual dispatch workflow. Takes CLI command as input.
- `collect-results.yml` / `collect-evals.yml`: Download artifacts, aggregate, publish summary.
- PRs need `sweep-enabled` label. Use `[skip-sweep]` in commit messages to skip.

### Benchmark execution
- `runners/launch_{node-prefix}.sh`: Per-node launchers. Docker-based or Slurm+Enroot. Runner name → script mapping: strip suffix after first `_` (e.g., `b200-nv_0` → `launch_b200-nv.sh`).
- `benchmarks/{model}_{prec}_{gpu}[_{fw}][_{spec}].sh`: Per-config scripts. All `source benchmark_lib.sh`. Pattern: check_env_vars → download model → start server → wait_for_server_ready → run_benchmark_serving → optional run_eval.
- `utils/bench_serving/benchmark_serving.py`: Load generator (from vLLM). Random dataset, request-rate=inf, measures throughput/TTFT/TPOT/E2EL.
- `benchmarks/benchmark_lib.sh`: Shared functions - `check_env_vars()`, `wait_for_server_ready()`, `run_benchmark_serving()`, `run_eval()`, `run_lm_eval()`, `append_lm_eval_summary()`.

### Result processing (`utils/`)
- `process_result.py`: Reads env vars + raw JSON. Computes `tput_per_gpu = total_throughput / num_gpus`. Converts ms→s. Computes interactivity (1000/TPOT).
- `process_changelog.py`: Git diffs `perf-changelog.yaml`, rejects deletions, resolves wildcards via `get_config_keys_from_master()`, calls `generate_sweep_configs.py test-config`.
- `collect_eval_results.py`: Merges eval artifacts into `agg_eval_all.json`.
- `summarize.py`: Generates markdown summary tables.
- `constants.py`: Paths - `MASTER_CONFIGS`, `RUNNER_CONFIG`, `GENERATE_SWEEPS_PY_SCRIPT`.

### Master configs (`.github/configs/`)
- `nvidia-master.yaml` / `amd-master.yaml`: Source of truth. Entry naming: `<model>-<precision>-<gpu>-<framework>`. Fields: image, model, model-prefix, runner, precision, framework, multinode, seq-len-configs with search-space.
- `runners.yaml`: GPU type → list of physical node names.
- `perf-changelog.yaml` (repo root): Triggers benchmarks. Entries have config-keys (wildcards allowed), description, pr-link. Only additions allowed.

## Key Conventions

- **Python**: Type hints (`list[str]`, `Optional[int]`), Pydantic with `extra='forbid'`, field aliases for YAML kebab-case. No defaults in output validation models.
- **YAML**: Kebab-case field names (`model-prefix`, `conc-start`, `dp-attn`). No `*` wildcards in master config keys.
- **Bash**: Parameters via environment variables. Source `benchmark_lib.sh`. No new directories in `/workspace/` during benchmarks.
- **Git**: Conventional commits. `[skip-sweep]` skips benchmarks. PRs need `sweep-enabled` label.

## Supported Values

| Dimension | Values |
|-----------|--------|
| Models | `dsr1` (DeepSeek-R1-0528), `gptoss` (GPT-OSS-120B) |
| Precisions | `fp4`, `fp8` |
| Frameworks | `sglang`, `trt`, `vllm`, `atom`, `dynamo-trt`, `dynamo-sglang`, `sglang-disagg` |
| NVIDIA runners | `b200`, `b200-trt`, `b200-multinode-slurm`, `b300`, `h100`, `h100-multinode-slurm`, `h200`, `gb200`, `gb300` |
| AMD runners | `mi300x`, `mi325x`, `mi355x`, `mi355x-disagg` |
| Seq lengths | `1k1k` (1024/1024), `1k8k` (1024/8192), `8k1k` (8192/1024) |
| Spec decoding | `none`, `mtp`, `draft_model` |

## Key Metrics

| Field | Description |
|-------|-------------|
| `tput_per_gpu` | Total throughput per GPU (tokens/sec) |
| `output_tput_per_gpu` | Output token throughput per GPU |
| `mean_ttft` / `p99_ttft` | Time to first token |
| `mean_tpot` | Time per output token |
| `mean_e2el` | End-to-end latency |
| `*_intvty` | Interactivity (1000/TPOT, tokens/sec) |

## Common Tasks

### Adding a benchmark config
1. Add entry to `.github/configs/nvidia-master.yaml` or `amd-master.yaml`
2. Create benchmark script in `benchmarks/` if needed (naming: `{model}_{prec}_{gpu}[_{fw}].sh`)
3. Create runner script in `runners/` if new node type
4. Validate: `python utils/matrix_logic/generate_sweep_configs.py full-sweep --config-files .github/configs/nvidia-master.yaml --single-node`
5. Add entry to `perf-changelog.yaml`
6. Run tests: `cd utils && python -m pytest matrix_logic/ -v`

### Updating Docker images
1. Update image tag in `.github/configs/*-master.yaml`
2. Add `perf-changelog.yaml` entry with config-key wildcards (e.g., `dsr1-fp8-*-vllm`)

### Fetching results from CI
```bash
gh run download <RUN_ID> --repo SemiAnalysisAI/InferenceX -n results_bmk -D ./results
# Use jq with rounding - raw JSON is large
cat ./results/agg_bmk.json | jq -r '.[] | [.hw, .framework, .precision, (.tput_per_gpu | round)] | @tsv' | column -t
```

## Multi-Node Disaggregated Configs

For `dynamo-sglang`/`dynamo-trt` entries: `multinode: true`, `disagg: true`, separate `prefill`/`decode` sections with `num-worker`, `tp`, `ep`, `dp-attn`, `additional-settings`. Recipes from [srtslurm](https://github.com/ishandhanani/srt-slurm) referenced via `CONFIG_FILE=recipes/...`. Throughput: total/(prefill_gpus+decode_gpus), output/decode_gpus, input/prefill_gpus.

## MCP Server

`.claude/mcp/server.py` provides an MCP server for querying vLLM/SGLang/TensorRT-LLM source code. Configured in `.mcp.json`. Dependencies: `.claude/requirements-mcp.txt`.
