# GPU and AI Primer: The Explainer for the Explainers

This document exists because the other InferenceX docs assume you already know
what a GPU is, what "inference" means, what "FP8" refers to, and why anyone would
split a model across 8 chips. If those assumptions don't hold for you, start here.

This is written for someone with general computing knowledge (you know what a CPU
is, you've written some code) but no background in AI hardware or deep learning.

---

## Table of Contents

1. [Why GPUs? A CPU Person's Introduction](#1-why-gpus-a-cpu-persons-introduction)
2. [Training vs Inference: Two Very Different Jobs](#2-training-vs-inference-two-very-different-jobs)
3. [What Is a Large Language Model, Physically?](#3-what-is-a-large-language-model-physically)
4. [NVIDIA GPU Generations: The Family Tree](#4-nvidia-gpu-generations-the-family-tree)
5. [GPU Anatomy: What's Inside the Chip](#5-gpu-anatomy-whats-inside-the-chip)
6. [Number Precision: FP32, FP16, FP8, FP4](#6-number-precision-fp32-fp16-fp8-fp4)
7. [GPU Memory: HBM and Why It Matters](#7-gpu-memory-hbm-and-why-it-matters)
8. [Connecting GPUs: NVLink, NVSwitch, and PCIe](#8-connecting-gpus-nvlink-nvswitch-and-pcie)
9. [How Inference Actually Works: Prefill and Decode](#9-how-inference-actually-works-prefill-and-decode)
10. [Parallelism: Why One GPU Is Never Enough](#10-parallelism-why-one-gpu-is-never-enough)
11. [Mixture of Experts: The DeepSeek-R1 Architecture](#11-mixture-of-experts-the-deepseek-r1-architecture)
12. [Speculative Decoding: Predicting the Future](#12-speculative-decoding-predicting-the-future)
13. [The KV Cache: The Model's Working Memory](#13-the-kv-cache-the-models-working-memory)
14. [Inference Engines: SGLang, vLLM, TensorRT-LLM](#14-inference-engines-sglang-vllm-tensorrt-llm)
15. [Benchmarking: What InferenceX Actually Measures](#15-benchmarking-what-inferencex-actually-measures)
16. [Infrastructure: Docker, Slurm, Enroot](#16-infrastructure-docker-slurm-enroot)
17. [Putting It All Together: Reading a Config Entry](#17-putting-it-all-together-reading-a-config-entry)
18. [NVIDIA Product Name Decoder Ring](#18-nvidia-product-name-decoder-ring)
19. [AMD Equivalent Concepts](#19-amd-equivalent-concepts)
20. [Further Reading](#20-further-reading)

---

## 1. Why GPUs? A CPU Person's Introduction

A CPU (like Intel Core i7) is designed to do one thing very fast. It has a few
powerful cores (4-16 typically) that each handle complex, sequential tasks.

A GPU (like NVIDIA B200) is designed to do thousands of things at once. It has
thousands of small cores that each do simple math, but they all work in parallel.

```
  CPU (few cores, very fast each)       GPU (thousands of cores, modest each)
  ┌──────┐                              ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
  │ Core │  ← complex, sequential       │·│·│·│·│·│·│·│·│·│·│·│·│·│·│·│·│
  │  1   │                              ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  ├──────┤                              │·│·│·│·│·│·│·│·│·│·│·│·│·│·│·│·│
  │ Core │                              ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │  2   │                              │·│·│·│·│·│·│·│·│·│·│·│·│·│·│·│·│
  ├──────┤                              ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │ Core │                              │·│·│·│·│·│·│·│·│·│·│·│·│·│·│·│·│
  │  3   │  ← good at "if/else" logic   ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
  ├──────┤                              │·│·│·│·│·│·│·│·│·│·│·│·│·│·│·│·│
  │ Core │                              └─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘
  │  4   │                                ↑ good at "do the same math on
  └──────┘                                  10,000 numbers at once"
```

AI models are fundamentally massive matrix multiplications. A matrix multiply is
the same operation repeated millions of times on different numbers - exactly what
GPUs are built for. This is why AI runs on GPUs, not CPUs.

**Analogy:** A CPU is one brilliant chef who can cook any dish. A GPU is 10,000
line cooks who can only chop vegetables, but they chop 10,000 vegetables per
second. Neural networks need a lot of vegetable-chopping.

---

## 2. Training vs Inference: Two Very Different Jobs

These are the two phases of an AI model's life:

### Training (Teaching)

- Feed the model billions of text examples
- The model adjusts its internal numbers (parameters) to get better at
  predicting the next word
- Takes weeks to months on hundreds/thousands of GPUs
- Costs millions of dollars
- Done once (or rarely) by the model creator (OpenAI, DeepSeek, Meta, etc.)
- **InferenceX does NOT do training**

### Inference (Using)

- The trained model is deployed on GPUs to answer user questions
- The model's parameters are frozen (read-only) - it's not learning anymore
- A user sends text in, the model generates text out
- Takes milliseconds to seconds per request
- Done billions of times per day across all AI services
- **InferenceX benchmarks this**

```
  Training (done once, very expensive)
  ┌─────────────┐      ┌─────────┐      ┌─────────────┐
  │ Trillions   │ ---> │ Months  │ ---> │  Trained    │
  │ of words    │      │ on 1000s│      │  model      │
  │ (internet)  │      │ of GPUs │      │  (frozen)   │
  └─────────────┘      └─────────┘      └──────┬──────┘
                                               │
  Inference (done forever, per-request)        │
  ┌─────────────┐      ┌─────────┐      ┌─────▼──────┐
  │ User's      │ ---> │ Seconds │ ---> │  Response   │
  │ question    │      │ on 1-8  │      │  text       │
  │             │      │ GPUs    │      │             │
  └─────────────┘      └─────────┘      └─────────────┘
```

**Why inference performance matters:** If an AI service has 100 million users,
even small speed improvements save enormous amounts of money and make the user
experience better. A model that's 20% faster at inference needs 20% fewer GPUs
to serve the same users, saving millions per year.

---

## 3. What Is a Large Language Model, Physically?

An LLM like DeepSeek-R1 is, at its core, a very large collection of numbers
called **parameters** (also called weights). These numbers encode everything
the model has learned.

| Model | Parameters | Size on Disk (FP16) | Size on Disk (FP8) |
|-------|-----------|--------------------|--------------------|
| GPT-2 (old, small) | 1.5 billion | ~3 GB | ~1.5 GB |
| Llama 3 70B | 70 billion | ~140 GB | ~70 GB |
| DeepSeek-R1 | 671 billion | ~1,340 GB | ~670 GB |

DeepSeek-R1 at FP8 precision is about **670 GB** of numbers. A single NVIDIA
B200 GPU has 192 GB of memory. So you need at minimum 4 GPUs just to hold the
model, and realistically 8 GPUs to leave room for computation.

Each parameter is a decimal number (like 0.0023 or -1.447). During inference,
the model multiplies your input text through layers of these numbers, eventually
producing probabilities for what the next word should be. It picks the most
likely word, appends it, and repeats.

---

## 4. NVIDIA GPU Generations: The Family Tree

NVIDIA names GPU generations after famous scientists and architects. Each
generation brings a step-change in performance:

```
  Generation Timeline (datacenter GPUs):

  2020      2022        2024         2025
  ──┼─────────┼───────────┼────────────┼──
    │         │           │            │
  Ampere    Hopper     Blackwell   Blackwell
  (A100)    (H100)     (B200)      Ultra
            (H200)     (B300)      (GB300)
                       (GB200)
```

### What the Letters Mean

| Prefix | What It Is | Example |
|--------|-----------|---------|
| **H** | Hopper generation (2022-2023) | H100, H200 |
| **B** | Blackwell generation (2024) | B200, B300 |
| **GB** | Grace-Blackwell: Blackwell GPU + Grace ARM CPU on one board | GB200, GB300 |

### The Numbers

The number after the letter generally indicates the tier:

| Number | Tier | Typical Use |
|--------|------|------------|
| x00 | Top tier | Flagship datacenter GPU |
| x00 + higher = newer | Same tier, more memory or newer revision | H200 = H100 + more memory |

### Specific GPUs in InferenceX

| GPU | Generation | Memory | Key Feature |
|-----|-----------|--------|------------|
| **H100** | Hopper | 80 GB HBM3 | Workhorse of 2023-2024 AI. Most widely deployed. |
| **H200** | Hopper | 141 GB HBM3e | Same chip as H100 but with 76% more memory and faster memory bandwidth. Models that barely fit on H100 run comfortably on H200. |
| **B200** | Blackwell | 192 GB HBM3e | Current generation (2024). Roughly 2-3x faster than H100 for inference. Has dedicated FP4 hardware. |
| **B300** | Blackwell | 288 GB HBM3e | Next revision of Blackwell with more memory. |
| **GB200** | Grace-Blackwell | 192 GB HBM3e per GPU | Two B200 GPUs paired with a Grace ARM CPU. Connected via ultra-fast NVLink. Deployed in NVL72 racks (36 GB200 modules = 72 GPUs). |
| **GB300** | Grace-Blackwell | 288 GB HBM3e per GPU | Next revision with more memory per GPU. |

**Why it matters for InferenceX:** Different GPUs have wildly different
performance characteristics. The B200 might be 2x faster than H200 for the same
model. InferenceX measures exactly how much faster, on real hardware, every night.

---

## 5. GPU Anatomy: What's Inside the Chip

A datacenter GPU is not like a gaming GPU. Here are the key components:

### Streaming Multiprocessors (SMs)

The basic compute units. A B200 has 160 SMs, each containing:
- CUDA cores (general-purpose math)
- Tensor Cores (specialized matrix multiplication hardware)
- Shared memory and registers

### Tensor Cores

The secret sauce for AI. Normal CUDA cores do one multiplication at a time.
Tensor Cores do a 4x4 matrix multiply in a single clock cycle. This is why
GPUs are so fast at AI - the most common AI operation (matrix multiply) has
dedicated hardware.

Each generation adds support for lower precisions:
- Hopper Tensor Cores: FP8, FP16, BF16, FP32
- Blackwell Tensor Cores: **FP4**, FP8, FP16, BF16, FP32

FP4 support is a big deal - it means Blackwell can process twice as many
operations per second compared to FP8 on the same hardware.

### Memory Controller + HBM

The GPU needs to feed data to all those Tensor Cores fast enough that they
don't sit idle. This is handled by High Bandwidth Memory (HBM) - stacks of
memory chips physically mounted on top of the GPU package.

---

## 6. Number Precision: FP32, FP16, FP8, FP4

Every parameter in an AI model is stored as a floating-point number. The
"precision" is how many bits are used to represent each number:

```
  FP32 (32 bits per number) - Full precision
  ┌────────────────────────────────┐
  │ sign │ exponent │  mantissa    │  Can represent: 3.14159265358979
  │  1   │    8     │     23       │
  └────────────────────────────────┘

  FP16 (16 bits per number) - Half precision
  ┌────────────────┐
  │ s │ exp │ mant │  Can represent: 3.14159
  │ 1 │  5  │  10  │
  └────────────────┘

  FP8 (8 bits per number) - Quarter precision
  ┌────────┐
  │s│ex│mnt│  Can represent: 3.14
  │1│ 4│ 3 │
  └────────┘

  FP4 (4 bits per number) - Eighth precision
  ┌────┐
  │s│em│  Can represent: 3.0
  │1│ 3│
  └────┘
```

### The Tradeoff

| Precision | Size per Parameter | Speed | Accuracy |
|-----------|-------------------|-------|----------|
| FP32 | 4 bytes | Baseline | Perfect |
| FP16/BF16 | 2 bytes | ~2x faster | Negligible loss |
| FP8 | 1 byte | ~4x faster | Small loss |
| FP4 | 0.5 bytes | ~8x faster | Noticeable loss |

**Why lower precision is faster:**
1. **Less memory:** FP4 DeepSeek-R1 = 335 GB. FP8 = 670 GB. FP4 fits on fewer GPUs.
2. **More operations per second:** Tensor Cores can do 2x more FP4 ops than FP8 ops per clock.
3. **Less memory bandwidth:** The GPU spends less time loading numbers from memory.

**Why InferenceX tests both FP4 and FP8:** The accuracy loss from quantization
(reducing precision) can make the model give wrong answers. InferenceX runs
accuracy evaluations (GSM8K math problems, GPQA reasoning) to check if the
model is still correct after quantization.

---

## 7. GPU Memory: HBM and Why It Matters

### What Is HBM?

HBM (High Bandwidth Memory) is special memory designed for GPUs. Unlike the
DDR5 RAM in your laptop (which has one or two channels), HBM has thousands of
data lanes stacked vertically.

| Memory Type | Bandwidth | Typical Use |
|-------------|----------|------------|
| DDR5 (laptop) | ~50 GB/s | Regular computing |
| GDDR6X (gaming GPU) | ~1,000 GB/s | Gaming |
| HBM3 (H100) | ~3,350 GB/s | AI datacenter |
| HBM3e (B200) | ~8,000 GB/s | AI datacenter |

### Why Bandwidth Matters for Inference

During inference, the GPU's bottleneck is usually **memory bandwidth**, not
computation. Here's why:

When the model generates one token, it needs to read nearly ALL the model's
parameters from memory, but only does a small amount of math with each one.
The Tensor Cores can compute faster than the memory can deliver data. This
makes inference "memory-bound."

This is why the B200 (8 TB/s bandwidth) can be much faster than H100
(3.35 TB/s) for inference - it can feed data to the Tensor Cores faster.

### Memory Capacity

| GPU | Memory | Can Hold (FP8) |
|-----|--------|---------------|
| H100 | 80 GB | Small models only (up to ~70B parameters) |
| H200 | 141 GB | Medium models |
| B200 | 192 GB | Larger models |
| B300 | 288 GB | Even larger models |

DeepSeek-R1 at FP8 (~670 GB) doesn't fit on a single GPU of any kind. You
need at least 4 B200s (768 GB total) or 8 H100s (640 GB total).

---

## 8. Connecting GPUs: NVLink, NVSwitch, and PCIe

When a model is split across multiple GPUs, those GPUs need to talk to each
other constantly. The speed of this communication is critical.

### PCIe (Slow)

The standard connection on any server motherboard. Like a regular highway.
- PCIe Gen5: ~64 GB/s per direction
- Used for: CPU-to-GPU communication, connecting GPUs on budget servers

### NVLink (Fast)

NVIDIA's proprietary GPU-to-GPU interconnect. Like a dedicated express lane.
- NVLink 4.0 (Hopper): 900 GB/s total
- NVLink 5.0 (Blackwell): 1,800 GB/s total
- Used for: Connecting GPUs within a server

### NVSwitch (Fastest, for Many GPUs)

A dedicated network switch chip that connects ALL GPUs to ALL other GPUs at
full NVLink speed. Without NVSwitch, you'd need point-to-point NVLink cables
between every pair of GPUs (impractical for 8+ GPUs).

```
  Without NVSwitch (mesh):          With NVSwitch (star):
  ┌─────┐    ┌─────┐               ┌─────┐  ┌─────┐
  │GPU 0│────│GPU 1│               │GPU 0│  │GPU 1│
  │     │╲  ╱│     │               │     │╲╱│     │
  └─────┘ ╲╱ └─────┘               └─────┘ │ └─────┘
           ╱╲                          ┌────┴────┐
  ┌─────┐╱  ╲┌─────┐               │ NVSwitch │
  │GPU 2│────│GPU 3│               └────┬────┘
  └─────┘    └─────┘               ┌─────┐ │ ┌─────┐
  (N*(N-1)/2 links needed)         │GPU 2│╱╲│GPU 3│
                                   └─────┘  └─────┘
                                   (every GPU talks to every
                                    GPU at full speed)
```

### GB200 NVL72

The GB200 NVL72 is a full server rack containing:
- 36 Grace-Blackwell modules (each: 1 Grace CPU + 2 B200 GPUs)
- = 72 GPUs total in one rack
- All connected via NVLink + NVSwitch at full bandwidth
- Every GPU can talk to every other GPU at 1.8 TB/s

This is the hardware behind multi-node InferenceX benchmarks.

---

## 9. How Inference Actually Works: Prefill and Decode

When you send a message to an AI chatbot, two distinct phases happen:

### Phase 1: Prefill (Processing Your Input)

The model reads your entire input prompt at once. This is highly parallel -
all input tokens are processed simultaneously in one big matrix multiplication.

- **Compute-bound:** Lots of math, GPU Tensor Cores are busy
- **Fast per token:** Processing 1000 input tokens takes roughly the same time
  as processing 100 (because they're processed in parallel)

### Phase 2: Decode (Generating the Response)

The model generates output one token at a time. Each new token depends on all
previous tokens, so this is sequential - you can't parallelize it.

- **Memory-bound:** For each token, the GPU reads the entire model from memory
  but does relatively little math
- **Slow per token:** Each token takes about the same amount of time regardless
  of how many GPUs you have

```
  Your prompt: "Explain quantum computing in simple terms"

  PREFILL (fast, parallel)                DECODE (slow, sequential)
  ┌─────────────────────────┐            ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐
  │ Process all input       │            │Q │→│ua│→│nt│→│um│→│  │→ ...
  │ tokens at once          │            │  │ │  │ │  │ │  │ │co│
  │ (one big matrix mult)   │            └──┘ └──┘ └──┘ └──┘ └──┘
  └─────────────────────────┘            one token at a time
  Time: ~100ms                           Time: ~50ms per token
```

### Why This Matters for InferenceX

InferenceX measures:
- **TTFT (Time To First Token):** Primarily measures prefill speed. How long
  until the first word of the response appears.
- **TPOT (Time Per Output Token):** Measures decode speed. How fast subsequent
  words appear.
- **Disaggregated inference:** Some configs in InferenceX split prefill and
  decode onto DIFFERENT machines, optimizing each independently.

---

## 10. Parallelism: Why One GPU Is Never Enough

DeepSeek-R1 doesn't fit on one GPU. How do you split it across multiple GPUs?
There are several strategies, and InferenceX tests different combinations.

### Tensor Parallelism (TP)

Split each layer of the model across GPUs. Every GPU holds a slice of every
layer and they must communicate after each layer's computation.

```
  TP=4: Each GPU holds 1/4 of every layer

  Layer 1:  [GPU0: 1/4] [GPU1: 1/4] [GPU2: 1/4] [GPU3: 1/4]
              ↕            ↕            ↕            ↕
  Layer 2:  [GPU0: 1/4] [GPU1: 1/4] [GPU2: 1/4] [GPU3: 1/4]
              ↕            ↕            ↕            ↕
  Layer 3:  [GPU0: 1/4] [GPU1: 1/4] [GPU2: 1/4] [GPU3: 1/4]
```

- **Pros:** Reduces memory per GPU. Every GPU participates in every request.
- **Cons:** Requires fast GPU-to-GPU communication (NVLink). Communication
  overhead increases with more GPUs.
- **In InferenceX configs:** `tp: 8` means the model is split across 8 GPUs.

### Expert Parallelism (EP)

Specific to Mixture-of-Experts models like DeepSeek-R1 (explained in next
section). The "expert" sub-networks are distributed across GPUs.

- **In InferenceX configs:** `ep: 4` means experts are spread across 4 GPUs.

### Data Parallelism (DP)

Run multiple copies of the model (or parts of it), each handling different
requests. Like having multiple checkout lanes at a grocery store.

- **In InferenceX configs:** `dp-attn: true` means the attention computation
  is replicated across GPUs rather than split.

### How They Combine

InferenceX tests various combinations:
- `tp: 8, ep: 1` = Pure tensor parallelism, 8 GPUs
- `tp: 4, ep: 4` = 4-way tensor parallel with 4-way expert parallel
- `tp: 8, ep: 8, dp-attn: true` = All three strategies combined

The optimal combination depends on the model, GPU, and concurrency level.
That's why InferenceX benchmarks so many configurations.

---

## 11. Mixture of Experts: The DeepSeek-R1 Architecture

DeepSeek-R1 has 671 billion parameters but doesn't use all of them for every
token. It uses a **Mixture of Experts (MoE)** architecture.

### How MoE Works

The model has many "expert" sub-networks (like specialist consultants). For
each input token, a "router" decides which experts are most relevant and only
activates those few.

```
  Standard Model (Dense):           MoE Model:
  ┌───────────────────┐             ┌───────────────────┐
  │                   │             │     Router        │
  │  One big network  │             │  "Who's best for  │
  │  ALL parameters   │             │   this token?"    │
  │  used every time  │             └──┬──┬──┬──┬──┬──┘
  │                   │                │  │  │  │  │
  └───────────────────┘             ┌──▼┐│  │  │┌─▼──┐
  Total: 70B params                 │Ex1││  │  ││Ex8 │  ← only 2 of 8
  Active: 70B params                │ ✓ ││  │  ││ ✓  │    experts are
                                    └───┘│  │  │└────┘    activated
                                    ┌───┐│  │  │┌────┐
                                    │Ex2││  │  ││Ex7 │
                                    │   ││  │  ││    │  ← the rest
                                    └───┘│  │  │└────┘    sit idle
                                    ...  ...   ...
                                    Total: 671B params
                                    Active: ~37B params per token
```

### Why MoE Is Clever

- **671B total parameters** = huge knowledge capacity
- **~37B active parameters per token** = inference cost similar to a 37B model
- You get the intelligence of a 671B model at the cost of a much smaller one

### The Catch

All 671B parameters must be stored in GPU memory, even though only ~37B are
used at a time. This is why DeepSeek-R1 needs so many GPUs despite being
relatively fast to run.

### Expert Parallelism (EP) for MoE

With EP, different experts live on different GPUs. When a token needs Expert 3,
the GPU holding Expert 3 processes it. This distributes the memory load but
requires routing tokens between GPUs.

---

## 12. Speculative Decoding: Predicting the Future

The decode phase generates one token at a time, which is slow. Speculative
decoding is an optimization to generate multiple tokens per step.

### The Idea

1. A small, fast "draft" model guesses the next N tokens quickly
2. The big model verifies all N guesses in one parallel forward pass
3. If the guesses were correct (they often are), you generated N tokens in
   the time it normally takes to generate ~1

```
  Normal decoding:
  Step 1: Generate "The"     (full model forward pass)
  Step 2: Generate "quick"   (full model forward pass)
  Step 3: Generate "brown"   (full model forward pass)
  Step 4: Generate "fox"     (full model forward pass)
  = 4 forward passes for 4 tokens

  Speculative decoding:
  Step 1: Draft model guesses "The quick brown fox"  (fast, tiny model)
  Step 2: Big model verifies all 4 at once           (one forward pass)
  Step 3: All correct! Accept all 4 tokens.
  = 1 big forward pass for 4 tokens (+ 1 tiny draft pass)
```

### MTP (Multi-Token Prediction)

DeepSeek-R1 has a built-in draft capability called MTP. Instead of a separate
draft model, the main model has extra "prediction heads" that can guess
multiple future tokens.

**In InferenceX configs:**
- `spec-decoding: "none"` = normal one-token-at-a-time decoding
- `spec-decoding: "mtp"` = Multi-Token Prediction enabled

When you see MTP configs testing up to concurrency 512 while non-MTP maxes out
at 64, that's because MTP makes each request faster, freeing up GPU resources
to handle more simultaneous requests.

---

## 13. The KV Cache: The Model's Working Memory

When the model generates tokens, it needs to "remember" all previous tokens
in the conversation. This memory is called the **KV Cache** (Key-Value Cache).

### What It Stores

For each token in the conversation (input + generated output), the model
computes and stores two vectors called "key" and "value" for each attention
layer. These are needed to generate every subsequent token.

### Why It Matters

The KV cache grows with conversation length:
- 1,024 tokens (short chat): ~1 GB per request on DeepSeek-R1 (FP8, TP=8)
- 8,192 tokens (long context): ~8 GB per request

With 64 simultaneous users, that's **64-512 GB just for KV caches** - a
significant fraction of total GPU memory.

### max-model-len in InferenceX

The `max-model-len` field in benchmark configs tells the inference engine how
much memory to pre-allocate for KV caches:

```
max-model-len = isl + osl + 200
              = 1024 + 1024 + 200
              = 2248 tokens
```

The 200-token buffer is a safety margin. Larger max-model-len = more memory
reserved for KV cache = less room for concurrent users.

---

## 14. Inference Engines: SGLang, vLLM, TensorRT-LLM

An inference engine is the software that sits between the AI model and the
users. It loads the model onto GPUs and serves requests. Think of it as the
"operating system" for AI inference.

### SGLang

- Built by researchers at UC Berkeley (LMSYS group - same people who made
  Chatbot Arena)
- Known for fast scheduling and memory management
- Open source, rapidly improving
- Image tags in configs: `lmsysorg/sglang:v0.5.x`

### vLLM

- Also from UC Berkeley, one of the first open-source inference engines
- Pioneered "PagedAttention" - a technique that manages KV cache memory like
  an operating system manages virtual memory (pages)
- Very widely used, large community
- Image tags in configs: varies by deployment

### TensorRT-LLM (TRT)

- Built by NVIDIA themselves
- Optimized specifically for NVIDIA hardware using TensorRT (NVIDIA's deep
  learning compiler that fuses operations and optimizes for specific GPU
  architectures)
- Typically fastest on NVIDIA hardware but NVIDIA-only
- Image tags in configs: `nvcr.io/nvidia/tensorrt-llm/...`

### ATOM

- AMD's optimized inference engine
- Built specifically for AMD GPUs (MI300X, MI325X, MI355X)
- Uses ROCm (AMD's equivalent of CUDA)

### Dynamo

- NVIDIA's distributed inference framework
- Handles multi-node orchestration (splitting work across multiple servers)
- Built on top of TensorRT-LLM
- In configs: `framework: dynamo-trt`

### Why InferenceX Tests All of Them

Each engine has different optimizations. SGLang might be fastest for high
concurrency while TRT wins at low concurrency. vLLM might have better memory
efficiency. The only way to know is to measure, which is exactly what
InferenceX does.

---

## 15. Benchmarking: What InferenceX Actually Measures

### Throughput Metrics

| Metric | Unit | Plain English |
|--------|------|--------------|
| `tput_per_gpu` | tokens/sec/GPU | Total tokens (input + output) processed per GPU per second. **The headline number.** Higher = better. |
| `output_tput_per_gpu` | tokens/sec/GPU | Only counting generated output tokens. More meaningful for real workloads since input processing is "free" from the user's perspective. |

### Latency Metrics

| Metric | Unit | Plain English |
|--------|------|--------------|
| `mean_ttft` | seconds | **Time To First Token.** How long you stare at a blank screen before the first word appears. Measures prefill speed. |
| `p99_ttft` | seconds | The worst-case TTFT that 99% of users experience. The 1% unluckiest users wait longer than this. Important for SLAs (Service Level Agreements). |
| `mean_tpot` | seconds | **Time Per Output Token.** Time between each word appearing. Determines how fast the text "streams" to you. |
| `mean_e2el` | seconds | **End-to-End Latency.** Total time from sending request to receiving the complete response. |

### The Throughput-Latency Tradeoff

This is the fundamental tension in inference:

```
  Low concurrency (4 users):
  ├─ Low throughput (GPU is underutilized)
  └─ Low latency (each user gets fast response)

  High concurrency (512 users):
  ├─ High throughput (GPU is fully utilized)
  └─ High latency (each user waits longer)
```

InferenceX sweeps across concurrency levels (4, 8, 16, 32, 64, 128, 256, 512)
to map out this tradeoff curve. This is why a single model+GPU+framework
combination generates many benchmark jobs at different concurrency levels.

---

## 16. Infrastructure: Docker, Slurm, Enroot

### Docker

A way to package software into a "container" - a self-contained bundle that
includes the application, all its libraries, and configuration. Like a
shipping container for software.

Why InferenceX uses it: Each inference engine (SGLang v0.5.6, TRT 1.2.0, etc.)
has specific library dependencies. Docker ensures the exact right versions are
used every time. The `image` field in configs points to a Docker container.

### Slurm

A job scheduler for computer clusters. When you have 100+ GPU machines, you
need software to decide which job runs on which machine. Slurm handles this.

Like a restaurant host who seats customers at available tables. You submit a
job ("I need 8 GPUs for 2 hours"), Slurm finds an available machine, allocates
the GPUs, and runs your job.

InferenceX uses Slurm for multi-node benchmarks where the model spans multiple
physical servers.

### Enroot

NVIDIA's container runtime for HPC (High Performance Computing) environments.
Similar to Docker but designed for GPU clusters managed by Slurm. It can run
Docker container images but integrates better with Slurm's GPU allocation.

### GitHub Actions

GitHub's built-in CI/CD (Continuous Integration/Continuous Deployment) system.
Whenever code changes, it automatically runs predefined workflows.

InferenceX uses GitHub Actions to:
1. Detect changes to `perf-changelog.yaml`
2. Generate the benchmark job matrix
3. Dispatch each job to the right GPU machine (self-hosted runners)
4. Collect results and publish to the dashboard

---

## 17. Putting It All Together: Reading a Config Entry

Now you can read any InferenceX config with understanding. Here's one:

```json
{
    "image": "lmsysorg/sglang:v0.5.8-cu130-amd64",
    "model": "deepseek-ai/DeepSeek-R1-0528",
    "model-prefix": "dsr1",
    "precision": "fp8",
    "framework": "sglang",
    "runner": "b200",
    "isl": 1024,
    "osl": 1024,
    "tp": 8,
    "conc": 64,
    "max-model-len": 2248,
    "ep": 1,
    "dp-attn": false,
    "spec-decoding": "mtp",
    "exp-name": "dsr1_1k1k",
    "disagg": false,
    "run-eval": false
}
```

Reading this, you now know:

> "Run the **SGLang v0.5.8** inference engine (in a Docker container built for
> **CUDA 13.0** on **x86-64** CPUs) serving the **DeepSeek-R1** model (671B
> parameter MoE) at **FP8 precision** (~670 GB) on **8 NVIDIA B200 GPUs**
> (192 GB each = 1,536 GB total, plenty of room). The model is split across
> all 8 GPUs using **tensor parallelism** (each holds 1/8 of each layer).
> **Expert parallelism is not used** (ep=1). **Multi-Token Prediction** is
> enabled for faster decoding. Simulate **64 concurrent users**, each sending
> **1,024 tokens of input** and expecting **1,024 tokens of output**. Reserve
> **2,248 tokens** of KV cache per request. This is a **single-machine**
> benchmark (not disaggregated). Don't run accuracy evals for this particular
> concurrency level."

---

## 18. NVIDIA Product Name Decoder Ring

NVIDIA uses a lot of product names. Here's a decoder:

### Software

| Name | What It Is |
|------|-----------|
| **CUDA** | NVIDIA's GPU programming language/toolkit. Like a specialized compiler that lets code run on GPU cores. Version numbers (e.g., CUDA 12.9, 13.0) indicate which GPU features are supported. |
| **cuDNN** | CUDA Deep Neural Network library. Pre-built, optimized implementations of common neural network operations. |
| **TensorRT** | NVIDIA's deep learning inference optimizer. Takes a trained model and compiles it into a faster version optimized for specific NVIDIA hardware. |
| **Triton** | (NVIDIA's, not OpenAI's) An inference serving platform that manages multiple models and handles request routing. |
| **NeMo** | NVIDIA's framework for training and deploying LLMs. |
| **NCCL** | "Nickel" - NVIDIA Collective Communications Library. Handles GPU-to-GPU communication (e.g., all-reduce operations during tensor parallelism). |
| **ROCm** | AMD's equivalent of CUDA. AMD's GPU programming toolkit. |

### Hardware Jargon

| Term | What It Means |
|------|--------------|
| **SXM** | The physical form factor (socket) for datacenter GPUs. Like how CPUs have LGA or AM5 sockets. SXM GPUs get more power and cooling than PCIe GPUs. |
| **NVL** | NVLink. When you see "NVL72", it means 72 GPUs connected via NVLink. |
| **DGX** | NVIDIA's pre-built AI server brand. A DGX system is a complete server with 8 GPUs, NVLink, networking, etc. |
| **HGX** | NVIDIA's AI server baseboard design. OEMs (Dell, HPE, Supermicro) build their own servers using HGX as the GPU module. |
| **OCI** | Oracle Cloud Infrastructure. One of the cloud providers where InferenceX runs benchmarks. |

### Container Registry Prefixes

In InferenceX config files:
- `lmsysorg/sglang:...` = Docker Hub, LMSYS organization, SGLang container
- `nvcr.io/nvidia/...` = NVIDIA Container Registry (NGC), NVIDIA's private container hosting
- `nvcr.io#nvidia/...` = Same as above (the `#` is an alternate notation)

---

## 19. AMD Equivalent Concepts

InferenceX also benchmarks AMD GPUs. Here's the translation:

| NVIDIA | AMD | Notes |
|--------|-----|-------|
| CUDA | ROCm | GPU programming toolkit |
| Tensor Cores | Matrix Cores | Dedicated matrix multiplication hardware |
| H100 | MI300X | Previous-gen flagship datacenter GPU |
| H200 | MI325X | More memory variant |
| B200 | MI355X | Current-gen flagship |
| NVLink | Infinity Fabric | GPU interconnect |
| cuDNN | MIOpen | Neural network library |
| TensorRT | (no direct equivalent) | AMD uses different optimization paths |
| DGX | Instinct Platform | Pre-built AI server |
| HBM3e | HBM3e | Same memory technology (AMD and NVIDIA both use it) |
| CDNA3 | - | AMD's GPU architecture generation for MI300X/MI325X |
| CDNA4 | - | AMD's GPU architecture generation for MI355X |

---

## 20. Further Reading

If you want to go deeper:

**On GPU Architecture:**
- NVIDIA's Blackwell Architecture Whitepaper (search: "NVIDIA Blackwell whitepaper")
- "GPU Programming" by Hwu, Kirk, El Hajj (textbook)

**On LLM Inference:**
- "Efficiently Scaling Transformer Inference" (Google, 2022)
- "PagedAttention" paper (vLLM team, 2023)
- "Speculative Decoding" papers (Leviathan et al., 2022; Chen et al., 2023)

**On InferenceX Specifically:**
- [HANDS_ON_WALKTHROUGH.md](HANDS_ON_WALKTHROUGH.md) - Run the config generator yourself
- [ARCHITECTURE.md](ARCHITECTURE.md) - How InferenceX's pipeline works internally
- [Session Transcript](session_transcript.md) - See actual output from generating 25 benchmark jobs
- [InferenceX Dashboard](https://inferencemax.ai/) - Live benchmark results

---

*This primer was written to help readers of InferenceX documentation who don't
have a background in GPU computing or AI inference. It covers the concepts
needed to understand what InferenceX benchmarks and why.*
