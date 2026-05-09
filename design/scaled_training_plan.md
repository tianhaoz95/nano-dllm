# MiCA-BD3LM Scaled Training Plan (GSM8K Focus)

**Date:** 2026-05-06
**Status:** Approved / Initiated
**Objective:** Improve mathematical reasoning (GSM8K) of the Qwen3-0.6B model by scaling up training tokens, context length, and adapter capacity.

---

## 1. Hardware & Environment
- **Platform:** NVIDIA GB10 (Blackwell)
- **Memory:** 128 GB Unified LPDDR5X
- **Optimization:** BF16, Flash Attention (SDPA), MiCA PEFT.
- **Task Management:** `task-spooler` (tsp) with `systemd-run` memory limits.

## 2. Model Configuration
- **Base Model:** `Qwen/Qwen3-0.6B` (Frozen)
- **Adapter:** MiCA (Minor Component Adaptation)
- **Target Modules:** `q_proj`, `v_proj`
- **Rank ($r$):** 32
- **Alpha ($\alpha$):** 32.0
- **Trainable Parameters:** ~1.8M (0.3% of model)

## 3. Training Hyperparameters
- **Total Steps:** 20,000
- **Global Batch Size:** 32 (8 per device x 4 grad accumulation)
- **Max Sequence Length:** 1024 tokens
- **Learning Rate:** 1e-4 (Cosine decay, 5% warmup)
- **Weight Decay:** 0.01
- **Precision:** BF16

## 4. Dataset Mixture
- **DCLM Baseline (1.0):** 1,000,000 samples (General knowledge, language priors)
- **OpenCoder FineWeb-Math:** 1,000,000 samples (Specialized reasoning, problem-solving)
- **Total Estimated Tokens:** ~650M tokens.

## 5. WSD Curriculum Strategy
| Phase | Block Size ($L_B$) | Steps | Description |
| :--- | :--- | :--- | :--- |
| **Warmup AR** | 1 | 1,000 | Transition from causal to bidirectional. |
| **Warmup Transition** | 4, 32, 128, 512 | 2,000 | Gradual increase in diffusion block size. |
| **Stable (BD3LM)** | **1024** | **15,000** | Full bidirectional denoising at scale. |
| **Decay (Refinement)**| 256, 64, 32 | 2,000 | Refine for final inference block size. |

---

## 6. Resource Estimates
- **Peak Memory:** ~12–15 GB
- **Throughput:** ~30,000 tokens/sec
- **Estimated Duration:** ~7.2 hours.
- **Output Directory:** `./outputs/mica-bd3lm-gsm8k-scaled`

---

## 7. Execution Command
```bash
tsp systemd-run --user --scope -p MemoryMax=100G \
    python scripts/train.py \
    --dataset_args "mlfoundations/dclm-baseline-1.0[train:1000000] + OpenCoder-LLM/opc-fineweb-math-corpus[train:1000000]" \
    --max_steps 20000 \
    --max_length 1024 \
    --mica_rank 32 \
    --mica_alpha 32 \
    --output_dir ./outputs/mica-bd3lm-gsm8k-scaled
```
