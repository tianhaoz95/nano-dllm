# MiCA-BD3LM: Project Context & Instructions

This project implements **MiCA-BD3LM**, a training recipe for converting pretrained autoregressive (AR) language models into Block Diffusion Language Models (BD3LM). It leverages **Minor Component Adaptation (MiCA)** for parameter-efficient fine-tuning and the **Warmup–Stable–Decay (WSD)** block-size curriculum for stable convergence.

## 1. Project Overview

*   **Objective:** Efficiently adapt AR models (specifically Qwen3-0.6B) to the block diffusion objective.
*   **Key Techniques:**
    *   **MiCA:** Updates only the minor singular directions of weight matrices (q_proj, v_proj), preserving AR priors while injecting bidirectional patterns.
    *   **BD3LM:** A block-wise masked diffusion objective.
    *   **WSD Curriculum:** A progressive block-size schedule (from AR-causal to full bidirectional and back).
*   **Base Model:** `Qwen/Qwen3-0.6B` (stored in `./models/Qwen3-0.6B`).
*   **Framework:** `dLLM` (cloned in `./dllm_repo` and installed in editable mode).

## 2. Directory Structure

*   `./design/`: Research plans and technical specifications (see `research_plan.md`).
*   `./models/`: Local model checkpoints (e.g., `Qwen3-0.6B`).
*   `./dllm_repo/`: Fork of the `dLLM` framework containing the core training and inference logic.
*   `./refs/`: Reference papers (PDFs).
*   `.venv/`: Python virtual environment with all necessary dependencies (`torch`, `transformers`, `accelerate`, `deepspeed`, `peft`, `dllm`, etc.).

## 3. Core Framework: dLLM

The project relies on the `dLLM` library. Key components include:
*   `BD3LMTrainer`: Located in `dllm_repo/dllm/core/trainers/bd3lm.py`. Handles the block diffusion loss and masking logic.
*   `MDLMSampler`: Used for evaluation and inference.

## 4. Key Commands (WIP)

### Environment Setup
```bash
source .venv/bin/activate
```

### Training (Expected)
Training will likely involve a custom script utilizing `BD3LMTrainer` and a new `MiCAConfig`.
*   **TODO:** Implement `MiCAConfig` and `MiCALinear` as PEFT adapters.
*   **TODO:** Implement the WSD scheduler for block sizes.

### Evaluation
Use the `lm-evaluation-harness` integrated within `dllm_repo`.

## 5. Development Conventions

*   **GPU Task Management:** For any GPU-related work (training, benchmarking, evaluation, etc.), **MUST** use `task-spooler` to ensure tasks share the GPU and don't collide.
    *   **Prefix:** Prepend `tsp` to your command: `tsp python train.py ...`
    *   **Monitor Queue:** Run `tsp` without arguments to see the job list and statuses.
    *   **View Output (Live):** Run `tsp -t [job_id]` to tail the output of a job (like `tail -f`).
    *   **View Full Output:** Run `tsp -c [job_id]` to cat the entire output.
    *   **Kill Job:** Run `tsp -k [job_id]` to stop a running job.
*   **PEFT Strategy:** Favor MiCA over LoRA. MiCA rank $r=16$, alpha $\alpha=16$.
*   **Weight Targets:** Focus on `q_proj` and `v_proj` in attention layers.
*   **Precision:** Use **BF16** for training.
*   **Hardware:** Optimized for **NVIDIA GB10 (Blackwell)** with unified memory (128GB).
*   **SVD Precomputation:** Perform SVD on target weights once before training starts to identify minor singular vectors.

## 6. Implementation Checklist (Priority)

1.  [ ] Implement `MiCAConfig` and `MiCALinear` module.
2.  [ ] Write SVD precomputation script for Qwen3-0.6B.
3.  [ ] Implement the WSD block-size scheduler.
4.  [ ] Set up the training script using `BD3LMTrainer`.
