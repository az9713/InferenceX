# InferenceX - Unofficial Clone with Added Documentation

> **This is an unofficial clone of [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX).**
> The original InferenceX™ is an open-source LLM inference benchmarking system by [SemiAnalysis](https://semianalysis.com/).
> All original code, configurations, and benchmark results belong to the original project.
>
> **What this clone adds:** Comprehensive documentation generated with the help of
> Claude Code, including a detailed hands-on walkthrough that traces how YAML
> benchmark configurations expand into individual benchmark jobs. For example,
> running the config generator for DeepSeek-R1 on NVIDIA B200 with SGLang at
> 1k1k sequence length produces 25 benchmark jobs from 3 YAML config entries -
> and the walkthrough explains every field, every flag, and every expansion step.
> No original code was modified.

---

> [!IMPORTANT]
> Only [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) repo contains the Official InferenceX™ result, all other forks & repos are Unofficial. The benchmark setup & quality of machines/clouds in unofficial repos may be differ leading to subpar benchmarking. Unofficial must be explicitly labelled as Unofficial.
> Forks may not remove this disclaimer

---

## Added Documentation

| Document | For | Description |
|----------|-----|-------------|
| [GPU and AI Primer](docs/GPU_AND_AI_PRIMER.md) | Everyone | Demystifies GPU hardware, AI training/inference, and all NVIDIA jargon used in InferenceX |
| [Hands-On Walkthrough](docs/HANDS_ON_WALKTHROUGH.md) | Everyone | Line-by-line explanation of every command you can run locally (no GPU needed), with full interpretation of outputs |
| [Quick Start Guide](docs/QUICKSTART.md) | Everyone | 12 hands-on use cases to get started fast |
| [User Guide](docs/USER_GUIDE.md) | Users | Understanding and analyzing benchmark results |
| [Developer Guide](docs/DEVELOPER_GUIDE.md) | Developers | Step-by-step guide for contributing code |
| [Architecture Guide](docs/ARCHITECTURE.md) | Developers | System architecture with ASCII diagrams |

---

## Original README

*Everything below is from the original [SemiAnalysisAI/InferenceX](https://github.com/SemiAnalysisAI/InferenceX) project.*

# InferenceX™, Open Source Inference Frequent Benchmarking

InferenceX™ (formerly InferenceMAX) runs our suite of benchmarks every night, continually re-benchmarking the world's most popular open-source inference frameworks used by major token factories and models to track real performance in real time. As these software stacks improve, InferenceX™ captures that progress in near real-time, providing a live indicator of inference performance progress. A live dashboard is available for free publicly at https://inferencemax.ai/.

[Full Article Write Up for InferenceMAXv1](https://newsletter.semianalysis.com/p/inferencemax-open-source-inference)


<img width="1627" height="1022" alt="CleanShot 2026-02-04 at 15 26 09" src="https://github.com/user-attachments/assets/65110e16-7590-424f-884d-12876d9e8f3e" />


## Why?

InferenceMAX™, an open-source, under Apache2 license, automated benchmark designed to move at the same rapid speed as the software ecosystem itself, is built to address this challenge.

LLM Inference performance is driven by two pillars, hardware and software. While hardware innovation drives step jumps in performance every year through the release of new GPUs/XPUs and new systems, software evolves every single day, delivering continuous performance gains on top of these step jumps. Speed is the Moat

AI software like SGLang, vLLM, TensorRT-LLM, CUDA, ROCm and achieve this continuous improvement in performance through kernel-level optimizations, distributed inference strategies, and scheduling innovations that increase the pareto frontier of performance in incremental releases that can be just days apart.

This pace of software advancement creates a challenge: benchmarks conducted at a fixed point in time quickly go stale and do not represent the performance that can be achieved with the latest software packages.

## Quick Start

### Prerequisites

- Python 3.11+
- pip (comes with Python)
- Git

### Setup

```bash
git clone https://github.com/SemiAnalysisAI/InferenceX.git
cd InferenceX
pip install pydantic pyyaml pytest
```

### Verify Installation

```bash
cd utils && python -m pytest matrix_logic/ -v && cd ..
```

### Generate a Benchmark Matrix

```bash
# See what benchmarks would run for DeepSeek R1 on NVIDIA B200
python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files .github/configs/nvidia-master.yaml \
  --single-node \
  --model-prefix dsr1 \
  --runner-type b200 \
  --seq-lens 1k1k \
  | python3 -m json.tool
```

See the [Quick Start Guide](docs/QUICKSTART.md) for 12 hands-on use cases.

## How It Works

InferenceX is a four-stage pipeline:

```
  YAML Configs  -->  CI/CD Dispatch  -->  GPU Benchmark  -->  Results Dashboard
  (what to test)     (orchestration)      (execution)         (inferencemax.ai)
```

1. **Configuration**: YAML files in `.github/configs/` define every benchmark (model, GPU, framework, settings)
2. **Orchestration**: GitHub Actions reads `perf-changelog.yaml`, generates a matrix of jobs, dispatches them
3. **Execution**: Self-hosted GPU runners start inference servers in containers and measure performance
4. **Publishing**: Results are aggregated and deployed to the live dashboard

## Hardware & Software Coverage

| Hardware | Frameworks |
|----------|-----------|
| NVIDIA B200, B300, H100, H200, GB200, GB300 | SGLang, vLLM, TensorRT-LLM, Dynamo |
| AMD MI300X, MI325X, MI355X | SGLang, vLLM, ATOM |

| Models | Precisions | Sequence Lengths |
|--------|-----------|-----------------|
| DeepSeek-R1-0528 (`dsr1`) | FP4, FP8 | 1k1k, 1k8k, 8k1k |
| GPT-OSS-120B (`gptoss`) | FP4, FP8 | 1k1k, 1k8k, 8k1k |

## Key Metrics

| Metric | Description |
|--------|-------------|
| `tput_per_gpu` | Total throughput per GPU (tokens/sec) |
| `output_tput_per_gpu` | Output tokens generated per GPU per second |
| `mean_ttft` / `p99_ttft` | Time to first token (mean and 99th percentile) |
| `mean_tpot` | Time per output token |
| `mean_e2el` | End-to-end latency |

## Documentation

| Document | For | Description |
|----------|-----|-------------|
| [GPU and AI Primer](docs/GPU_AND_AI_PRIMER.md) | Everyone | Demystifies GPU hardware, AI training/inference, and all NVIDIA jargon |
| [Hands-On Walkthrough](docs/HANDS_ON_WALKTHROUGH.md) | Everyone | Line-by-line command explanation with output interpretation |
| [Quick Start Guide](docs/QUICKSTART.md) | Everyone | 12 hands-on use cases to get started fast |
| [User Guide](docs/USER_GUIDE.md) | Users | Understanding and analyzing benchmark results |
| [Developer Guide](docs/DEVELOPER_GUIDE.md) | Developers | Step-by-step guide for contributing code |
| [Architecture Guide](docs/ARCHITECTURE.md) | Developers | System architecture with ASCII diagrams |
| [AGENTS.md](AGENTS.md) | AI Agents | Guidance for AI agents working with the codebase |
| [Workflow README](.github/workflows/README.md) | Developers | How to use and test CI/CD workflows |
| [Config Format](.github/configs/CONFIGS.md) | Developers | Master config YAML format specification |
| [Evals Documentation](utils/evals/EVALS.md) | Developers | Accuracy evaluation system |

## Project Structure

```
├── .github/configs/          # Benchmark definitions (YAML)
│   ├── nvidia-master.yaml    # All NVIDIA configurations
│   ├── amd-master.yaml       # All AMD configurations
│   └── runners.yaml          # GPU type -> node mapping
├── .github/workflows/        # CI/CD workflows
│   ├── run-sweep.yml         # Main orchestrator
│   ├── benchmark-tmpl.yml    # Single-node benchmark template
│   └── e2e-tests.yml         # Manual test workflow
├── benchmarks/               # Benchmark execution scripts
│   ├── benchmark_lib.sh      # Shared utilities
│   └── {model}_{prec}_{gpu}[_{fw}].sh
├── runners/                  # GPU launcher scripts
│   └── launch_{node-type}.sh
├── utils/                    # Python utilities
│   ├── matrix_logic/         # Config generation & validation
│   ├── bench_serving/        # Benchmark load generator
│   └── evals/                # Eval task definitions
└── perf-changelog.yaml       # Benchmark trigger (edit this to run benchmarks)
```

## Contributing

1. Add benchmark config to `.github/configs/nvidia-master.yaml` or `amd-master.yaml`
2. Create/update benchmark script in `benchmarks/` if needed
3. Add entry to `perf-changelog.yaml`
4. Validate: `python utils/matrix_logic/generate_sweep_configs.py full-sweep --config-files .github/configs/nvidia-master.yaml --single-node`
5. Run tests: `cd utils && python -m pytest matrix_logic/ -v`
6. Create PR with `sweep-enabled` label

See the [Developer Guide](docs/DEVELOPER_GUIDE.md) for detailed instructions.

## Acknowledgements & Supporters
Thank you to Lisa Su and Anush Elangovan for providing the MI355X and CDNA3 GPUs for this free and open-source project. We want to recognize the many AMD contributors for their responsiveness and for debugging, optimizing, and validating performance across AMD GPUs.
We're also grateful to Jensen Huang and Ian Buck for supporting this open source with access to a GB200 NVL72 rack (through OCI) and B200 GPUs. Thank you to the many NVIDIA contributors from the NVIDIA inference team, NVIDIA Dynamo team.

We also want to recognize the SGLang, vLLM, and TensorRT-LLM maintainers for building a world-class software stack and open sourcing it to the entire world.
Finally, we're grateful to Crusoe, CoreWeave, Nebius, TensorWave, Oracle and TogetherAI for supporting open-source innovation through compute resources, enabling this.

## SemiAnalysis is Hiring

We are looking for an engineer to join our special projects team. This is a unique opportunity to work on high-visibility special projects such as InferenceMAX™ with support from many industry leaders and CEOs. If you're passionate about performance engineering, system reliability, and want to work at the intersection of hardware and software, this is a rare chance to make industry wide impact.
What you'll work on:
- Building and running large-scale benchmarks across multiple vendors (AMD, NVIDIA, TPU, Trainium, etc.)
- Designing reproducible CI/CD pipelines to automate benchmarking workflows
- Ensuring reliability and scalability of systems used by industry partners

What we're looking for:
- Strong skills in Python
- Background in Site Reliability Engineering (SRE) or systems-level problem solving
- Experience with CI/CD pipelines and modern DevOps practices
- Curiosity about GPUs, TPUs, Trainium, multi-cloud, and performance benchmarking
Link to apply: https://app.dover.com/apply/SemiAnalysis/2a9c8da5-6d59-4ac8-8302-3877345dbce1

## License

Apache 2.0
