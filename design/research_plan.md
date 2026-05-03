# MiCA-BD3LM: Parameter-Efficient AR-to-Block-Diffusion Conversion via Minor Component Adaptation and Warmup–Stable–Decay Training

**Date:** 2026-05-03  
**Status:** Draft Research Plan

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Motivation & Core Hypothesis](#2-research-motivation--core-hypothesis)
3. [Feasibility Study](#3-feasibility-study)
4. [Related Work & Positioning](#4-related-work--positioning)
5. [Hardware Analysis & Model Selection](#5-hardware-analysis--model-selection)
6. [Methodology](#6-methodology)
7. [Experiment Design](#7-experiment-design)
8. [Evaluation Protocol](#8-evaluation-protocol)
9. [Expected Results & Risk Analysis](#9-expected-results--risk-analysis)
10. [Paper Structure](#10-paper-structure)
11. [Implementation Checklist](#11-implementation-checklist)

---

## 1. Executive Summary

We propose **MiCA-BD3LM**: a training recipe that converts a pretrained autoregressive (AR) language model into a Block Diffusion Language Model (BD3LM) using (a) Minor Component Adaptation (MiCA) as the parameter-efficient fine-tuning (PEFT) strategy, and (b) the Warmup–Stable–Decay (WSD) block-size curriculum from LLaDA2.0 for fast convergence. The central claim is that adapting only the *minor singular directions* of AR weight matrices is uniquely well-suited to the AR→diffusion paradigm shift, because dominant directions are already saturated by causal next-token prediction and the new bidirectional denoising signal requires learning in underutilized weight subspaces. This recipe produces functional BDLMs with **6–10% of the trainable parameters of LoRA** while converging faster or comparably under equivalent token budgets.

The full experiment is designed to complete on a single **NVIDIA GB10 DGX Spark** (128 GB unified memory) within **48 hours**, using **Qwen3-0.6B** as the base model.

---

## 2. Research Motivation & Core Hypothesis

### 2.1 The AR-to-Diffusion Gap

AR models generate tokens strictly left-to-right under causal attention. DLMs require bidirectional attention and learn to denoise corrupted sequences. Directly switching the training objective causes unstable optimization and catastrophic forgetting of AR knowledge. This motivates *continual pre-training* (CPT) rather than training from scratch, and the progressive WSD curriculum to bridge the distributional gap gradually.

### 2.2 Why MiCA Fits the AR-to-Diffusion Setting

MiCA (Minor Component Adaptation) constrains weight updates to the subspace spanned by the *least significant singular vectors* of each weight matrix — directions that contribute minimally to the AR model's variance but may encode high plasticity for new task objectives.

**The key alignment argument:**

| Property | AR Pre-Training | Diffusion Adaptation |
|---|---|---|
| Dominant singular directions | Saturated with LM priors, causal patterns | Interference risk if overwritten |
| Minor singular directions | Underutilized by causal pretraining | High plasticity; ideal for new bidirectional patterns |
| LoRA | Updates unconstrained rank-r subspace — may overwrite dominant AR directions | Higher forgetting risk |
| MiCA | Constrains to minor SVD directions | Preserves AR priors; injects diffusion patterns into plastic subspace |

This alignment suggests: **MiCA's inductive bias is geometrically matched to the AR-to-DLM conversion task**. LoRA may interfere with carefully learned AR representations because its updates are not spectral-aware. MiCA avoids this by construction.

### 2.3 Core Hypotheses

**H1 (Convergence):** Under the WSD curriculum, MiCA reaches the same masked language modeling (MLM) loss as LoRA with fewer gradient steps.

**H2 (Parameter Efficiency):** MiCA achieves competitive final BD3LM performance (LAMBADA, HellaSwag) using 6–10x fewer trainable parameters than LoRA.

**H3 (Knowledge Retention):** MiCA-converted BDLMs retain more of the original AR model's factual knowledge (measured by perplexity on held-out causal LM text) compared to LoRA-converted variants.

**H4 (Subspace Specificity):** Minor-direction adaptation outperforms equivalent-rank major-direction and random-direction adaptations in the AR→diffusion conversion setting, mirroring MiCA's ablation results from the original paper.

---

## 3. Feasibility Study

### 3.1 Technical Feasibility

All three component techniques have demonstrated prior success:

| Component | Prior Evidence | Risk Level |
|---|---|---|
| AR → BD3LM conversion | dLLM Tiny-A2D: Qwen3-0.6B → BD3LM with SFT only, ~18K steps | Low |
| MiCA PEFT | 5.9x knowledge gain vs LoRA on LLaMA-2-7B, 6M vs 67M params | Low |
| WSD curriculum | LLaDA2.0: smooth AR→MDLM→BDLM transition at 100B scale | Low |
| MiCA + diffusion (novel combination) | Untested — this is the research contribution | Medium |

### 3.2 Key Unknowns

1. **SVD initialization cost:** Computing SVD for all attention weight matrices in Qwen3-0.6B is a one-time O(d³) operation. For d=1024 (hidden dim), this is fast (<2 minutes total).

2. **Compatibility with BD3LM objective:** The BD3LM loss involves masking contiguous blocks, not individual tokens. MiCA was validated on a causal LM objective. We need to verify that minor SVD directions are still the most plastic under the masked block reconstruction objective. This is theoretically motivated but empirically unconfirmed — one of our ablations directly tests this.

3. **Interaction with WSD block-size changes:** During WSD warmup, the attention mask structure changes per phase (block-causal → full bidirectional → block-causal again). It is unclear whether minor singular directions shift between phases. We will track singular value evolution during training as an analysis experiment.

### 3.3 Compute Feasibility

On NVIDIA GB10 DGX Spark (128 GB unified memory, ~1 PFLOP FP4, ~273 GB/s bandwidth):

- **Qwen3-0.6B** in BF16: ~1.2 GB model weights; fits trivially
- Batch size 8, sequence length 512, gradient accumulation 4: ~4 GB activation memory
- Estimated training throughput: ~2,000–3,000 tokens/s for SFT-scale training
- Training budget: ~1,000–2,000 steps per experiment phase (~30 min per run)
- Total experimental conditions (see §7): 6 runs × ~2h = **~12h total**, well within 48h

---

## 4. Related Work & Positioning

### 4.1 Parameter-Efficient Fine-Tuning for Diffusion Models

- **LoRA (Hu et al., 2021):** Standard PEFT baseline; unconstrained rank-r updates. Used in dLLM's Tiny-A2D (r=128, α=256).
- **PiSSA (Meng et al., 2024):** Uses principal SVD components for initialization — the *opposite* of MiCA's minor-direction focus.
- **MiCA (Rüdiger & Raschka, 2026):** Constrains to minor SVD directions; 6-60% LoRA parameter footprint; 5.9x knowledge gain. **This work extends MiCA to the diffusion domain.**

### 4.2 AR-to-Diffusion Conversion

- **DiffuLLaMA/DiffuGPT (Gong et al., 2025):** Converts GPT-2/LLaMA (127M–7B) to MDLM via causal→bidirectional attention mask annealing + LoRA.
- **RND1 (Chandrasegaran et al., 2025):** Scales to 30B via straight AR-to-bidirectional conversion; finds that constraining dense layer updates prevents catastrophic forgetting — a finding that motivates MiCA's approach.
- **SDAR (Cheng et al., 2025):** Synergistic diffusion-autoregression paradigm using Qwen-3 series; achieves AR-comparable performance.
- **LLaDA2.0 (Bie et al., 2025):** WSD curriculum at 100B scale; strongest evidence that progressive block-size scheduling is critical for stable CPT.
- **dLLM (Zhou et al., 2026):** Unified framework; Qwen3-0.6B→BD3LM with only SFT (no CPT), demonstrating minimal-compute conversion is feasible.

### 4.3 Novelty Statement

No existing work combines spectrally-grounded PEFT (specifically minor-component adaptation) with the AR-to-block-diffusion conversion pipeline. RND1's finding that constraining dense-layer updates helps knowledge retention is circumstantial evidence for our hypothesis, but uses frozen (not spectral) constraints. **Our work is the first to ask: which subspace in the AR weight matrix is the most natural landing zone for diffusion-style adaptation?**

---

## 5. Hardware Analysis & Model Selection

### 5.1 Hardware Profile

```
GPU:     NVIDIA GB10 (Blackwell)
Memory:  128 GB unified LPDDR5X (shared CPU/GPU)
CPU:     20-core ARM64
Bandwidth: 273 GB/s
AI TOPS: ~1000 TOPS (FP4), ~200 TOPS (BF16 estimate)
CUDA:    13.0
Driver:  580.142
```

The GB10 is a unified memory architecture — GPU and CPU share the same 128 GB pool, which is beneficial for loading large model checkpoints and optimizer states without explicit host-device transfers.

### 5.2 Model Selection Rationale

**Primary model: `Qwen3-0.6B`**

| Criterion | Justification |
|---|---|
| Proven baseline | dLLM paper converts Qwen3-0.6B to BD3LM (SFT only, 18K steps) — we have a direct reference point |
| Memory fit | 0.6B BF16 ≈ 1.2 GB weights; easily fits with optimizer state (~5–6 GB total) |
| Speed | Enables multiple ablation runs within 48 hours |
| SVD cost | SVD of attention matrices (dim 1024) takes <5 minutes total |
| Architecture | GQA attention; q_proj and v_proj are primary MiCA targets |

**Secondary model: `Qwen3-1.7B`** (time-permitting)  
If primary experiments finish early (~<36h), run a single MiCA-WSD vs LoRA-WSD comparison on 1.7B to probe scale sensitivity.

**Why not larger?**  
A 7B model in BF16 requires ~14 GB weights + optimizer state of ~56 GB (AdamW) = ~70 GB. Combined with activation memory and the GB10's 273 GB/s bandwidth (slower than HBM), a 7B training run would take ~20–30h for a single experiment run, leaving no budget for ablations.

---

## 6. Methodology

### 6.1 MiCA Adapter Implementation for BD3LM

Following the MiCA paper, for each target weight matrix `W ∈ ℝ^(d_out × d_in)`:

1. **Compute SVD:** `W = UΣVᵀ`
2. **Select minor subspace:** `B = U[:, -r:]`  (last r left singular vectors)
3. **Initialize:** `A = 0`  (zero initialization → identity at start)
4. **Trainable:** only `A ∈ ℝ^(r × d_in)`; `B` and `W` are frozen
5. **Update:** `W_final = W + (α/r) · B · A`

**Target modules:** `q_proj`, `v_proj` (following both MiCA and dLLM papers)  
**Rank:** r = 16 (primary); r = 32 (ablation)  
**Alpha:** α = 16

This setup produces ~4M trainable parameters for Qwen3-0.6B, compared to ~67M for LoRA at equivalent rank.

### 6.2 WSD Training Curriculum (adapted for small-scale)

LLaDA2.0's WSD uses 100B+ tokens. We adapt it proportionally for our compute budget:

| Phase | Block Size L_B | Steps | Attention Mask | Purpose |
|---|---|---|---|---|
| Warmup-1 | 1 (AR) | 500 | Causal | Stabilize; minimal change from AR |
| Warmup-2 | 4 | 500 | Block-causal | Introduce short-range diffusion |
| Warmup-3 | 32 | 500 | Block-causal | Medium-range |
| Warmup-4 | 128 | 500 | Block-causal | Approaching full seq |
| Stable | 512 (=seq_len) | 2000 | Full bidirectional (MDLM) | Deep diffusion learning |
| Decay-1 | 64 | 300 | Block-causal | Transition back |
| Decay-2 | 32 | 200 | Block-causal | Final efficient BDLM block size |

Total: ~4,500 steps. At 512-token sequences, batch size 8, grad_accum 4 → ~9M tokens per run.

**Document-level attention mask:** Applied throughout; ensures attention does not bleed across packed document boundaries (crucial for bidirectional phases, following LLaDA2.0).

**Top-k checkpoint merge:** After training, average the top-3 checkpoints by validation loss.

### 6.3 Baseline Conditions

| Condition | PEFT Method | WSD | Notes |
|---|---|---|---|
| C0 | None (frozen AR) | No | Reference upper bound on AR capability |
| C1 | Full FT | No (SFT only) | dLLM Tiny-A2D baseline |
| C2 | LoRA (r=16) | No | PEFT baseline, no curriculum |
| C3 | MiCA (r=16) | No | MiCA without curriculum |
| C4 | LoRA (r=16) | Yes | WSD baseline |
| **C5** | **MiCA (r=16)** | **Yes** | **Primary contribution** |
| C6 | MiCA-Major (r=16) | Yes | Ablation: major SVD directions |
| C7 | MiCA-Random (r=16) | Yes | Ablation: random SVD directions |

C4 vs C5 is the primary comparison. C6 and C7 are the ablation studies matching MiCA paper's Table 3.

### 6.4 Training Objective

For each block `B_k`, the BD3LM loss averaged over the WSD schedule:

```
L_BD3LM(θ) = Σ_k E_{t~U(0,1),x_0} [ (1/t) Σ_{i∈mask(B_k,t)} -log p_θ(x_0^i | x_t^{B_k}, x^{<B_k}) ]
```

During the Stable (MDLM) phase, this simplifies to the standard masked diffusion loss with full sequence context.

**Mask ratio bandwidth:** Clip `α_t ∈ [0.1, 0.9]` following LLaDA2.0 to avoid trivial masking regimes.

### 6.5 Framework

Use the `dLLM` framework (github.com/ZHZisZZ/dllm):
- `BD3LMTrainer` for block diffusion training
- Custom `MiCAConfig` implementing the SVD initialization for PEFT
- `MDLMSampler` for evaluation inference

---

## 7. Experiment Design

### 7.1 Timeline (48-hour budget)

```
Hours 0–4:   Environment setup, MiCA adapter implementation, SVD precomputation
Hours 4–8:   Run C1 (Full FT, SFT only) — reproduce dLLM baseline
Hours 8–12:  Run C2 (LoRA, no WSD)
Hours 12–16: Run C3 (MiCA, no WSD)
Hours 16–24: Run C4 (LoRA + WSD)        ← critical baseline
Hours 24–32: Run C5 (MiCA + WSD)        ← primary contribution
Hours 32–36: Run C6 (MiCA-Major + WSD)  ← ablation
Hours 36–40: Run C7 (MiCA-Random + WSD) ← ablation
Hours 40–48: Evaluation, result collection, figure generation
```

### 7.2 Dataset

**Training:** OpenWebText subset (1M samples, English) or a preprocessed slice of the Tulu-3 SFT dataset (matching dLLM paper), tokenized to 512 tokens per chunk with document-boundary padding markers.

**Validation:** Held-out 10K samples from the same distribution.

**Evaluation benchmarks** (see §8): HellaSwag, LAMBADA, and a 500-sample GSM8K subset.

### 7.3 Hyperparameters

| Hyperparameter | Value |
|---|---|
| Model | Qwen/Qwen3-0.6B |
| Precision | BF16 |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning rate | 1e-4 (cosine decay, 10% warmup) |
| Batch size | 8 |
| Gradient accumulation | 4 (effective batch 32) |
| Max sequence length | 512 |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| MiCA rank r | 16 |
| MiCA alpha α | 16 |
| MiCA targets | q_proj, v_proj |
| LoRA rank r | 16 (same for fair comparison) |
| Block size (final BDLM) | 32 |
| Mask ratio bandwidth | [0.1, 0.9] |

### 7.4 Metrics Tracked Per Step

- Training MLM loss
- Validation MLM loss (every 100 steps)
- Gradient norm per parameter group
- Singular value distribution of MiCA's `A` matrix (to track subspace utilization)
- GPU/CPU memory usage

---

## 8. Evaluation Protocol

### 8.1 Primary Metrics

| Metric | Task | Why |
|---|---|---|
| MLM perplexity | Held-out text | Direct measure of diffusion quality |
| HellaSwag accuracy | Sentence completion | Tests commonsense reasoning retention |
| LAMBADA accuracy | Long-range word prediction | Tests bidirectional context modeling |
| GSM8K (500-sample) | Math reasoning | Tests higher-order reasoning |
| Trainable param count | — | Parameter efficiency |
| Steps to convergence | — | Training efficiency |

### 8.2 AR Knowledge Retention

Reload the converted DLM weights as a causal LM (mask out diffusion objective; use causal attention), compute perplexity on WikiText-2. A lower perplexity degradation from the original AR baseline indicates better knowledge retention.

### 8.3 Inference Configuration

Following dLLM and LLaDA2.0:
- Block size: 32
- Denoising threshold: 0.95
- MDLMSampler for LAMBADA/HellaSwag scoring
- Temperature: 0.0 (greedy)

### 8.4 Convergence Analysis

Plot validation loss curves for all 8 conditions on the same axes. Identify:
1. **Area under the curve (AUC)**: total loss integrated over training — lower is better convergence efficiency
2. **Steps to reach target loss** (e.g., 0.9 nats): convergence speed
3. **Final loss plateau**: asymptotic performance

---

## 9. Expected Results & Risk Analysis

### 9.1 Expected Findings

Based on the theoretical argument and related evidence:

| Comparison | Expected Outcome | Confidence |
|---|---|---|
| C5 (MiCA+WSD) vs C4 (LoRA+WSD) | MiCA reaches equivalent loss 10–30% faster; fewer params | Medium-High |
| C5 vs C3 (MiCA, no WSD) | WSD provides 0.1–0.2 nats improvement in final loss | High (supported by LLaDA2.0) |
| C5 vs C6 (Major SVD+WSD) | Minor outperforms major (matches MiCA paper's ablation) | Medium |
| C5 vs C7 (Random SVD+WSD) | Minor outperforms random | Medium-High |
| AR knowledge retention | MiCA > LoRA | Medium (based on MiCA paper) |

### 9.2 Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| MiCA performs similarly to LoRA (null result) | Medium | Still publishable as negative result; analyze *why* via singular value tracking |
| OOM on GB10 during WSD stable phase | Low | Sequence length 512 is conservative; reduce batch size if needed |
| SVD computation of BF16 weights unstable | Low | Cast to FP32 for SVD, then cast back |
| WSD curriculum too short to show effect | Medium | Extend stable phase if loss hasn't plateaued by step 1500 |
| dLLM framework incompatibility with custom PEFT | Medium | Implement as a standalone wrapper; MiCA's architecture is simple |

### 9.3 Significance if Hypothesis Confirmed

- Validates a spectral theory of the AR-to-diffusion representational transition
- Enables sub-7B AR model conversion on consumer hardware (GB10, Mac Studio) in <24h
- Opens direction for MiCA+RL post-training for diffusion models (noted as future work in MiCA paper)

### 9.4 Significance if Hypothesis Refuted

- Provides empirical evidence that the diffusion paradigm shift does *not* preferentially target minor singular subspaces
- Informs future PEFT design for diffusion: may suggest major or full-rank adaptation is necessary
- Still validates WSD as a standalone contribution at small scale

---

## 10. Paper Structure

### Proposed Title
**"Spectral Adaptation for Efficient AR-to-Diffusion Conversion: Minor Component Adaptation Under Warmup–Stable–Decay Training"**

### Venue Target
- Primary: ICLR 2027 or NeurIPS 2026
- Backup: ACL 2027 (if framed as efficient DLM recipe)
- ArXiv preprint: immediately after experiments complete

### Paper Outline

**Abstract** (250 words)  
One-paragraph summary: problem, method, key finding, hardware accessibility claim.

**1. Introduction**  
- Motivation: AR→DLM conversion is expensive; PEFT reduces cost but *which* subspace?  
- Gap: No prior work asks whether spectral structure of AR weights matters for diffusion adaptation  
- Contribution bullets (4): MiCA-BD3LM recipe, spectral analysis, WSD scaling, GB10 reproducibility

**2. Background**  
- 2.1 Block Diffusion Language Models (BD3LM objective, eq. 1-3)  
- 2.2 Parameter-Efficient Fine-Tuning: LoRA and MiCA  
- 2.3 AR-to-DLM Conversion (DiffuLLaMA, LLaDA2.0, dLLM)  
- 2.4 Warmup–Stable–Decay Curriculum

**3. MiCA-BD3LM**  
- 3.1 Theoretical Motivation (spectral geometry argument)  
- 3.2 MiCA Adapter Design for Diffusion Models  
- 3.3 WSD Integration  
- 3.4 Document-level Attention Mask  
- 3.5 Training Algorithm (pseudocode)

**4. Experimental Setup**  
- 4.1 Model and Dataset  
- 4.2 Baselines (C0–C7)  
- 4.3 Hyperparameters  
- 4.4 Evaluation Protocol  
- 4.5 Hardware (NVIDIA GB10 DGX Spark, 128 GB)

**5. Results**  
- 5.1 Main Comparison: MiCA-WSD vs LoRA-WSD (Table 1: benchmarks; Figure 1: convergence curves)  
- 5.2 Ablation: WSD Curriculum Effect (C3 vs C5, C2 vs C4)  
- 5.3 Ablation: Subspace Specificity (C5 vs C6 vs C7 — mirroring MiCA paper Table 3)  
- 5.4 Knowledge Retention Analysis (AR perplexity preservation)  
- 5.5 Parameter Efficiency Analysis (Figure 2: param count vs performance Pareto)

**6. Analysis**  
- 6.1 Singular Value Evolution During WSD Phases  
- 6.2 Effect of MiCA Rank  
- 6.3 WSD Phase Budget Sensitivity  
- 6.4 Qualitative Generation Examples

**7. Related Work**  
- PEFT methods; AR-to-diffusion conversion; DLM training recipes

**8. Conclusion**  
- Summary of findings  
- Limitations: single model family (Qwen3), limited token budget, single GPU  
- Future work: larger scale, MiCA+RL for diffusion, multi-modal DLMs

**Appendix**  
- A: Full hyperparameter tables  
- B: Extended benchmark results  
- C: MiCA implementation details (SVD initialization code)  
- D: Training curves for all conditions  
- E: Reproducibility checklist

### Target Length
8 pages + unlimited appendix (ICLR format)

---

## 11. Implementation Checklist

### Phase 0: Setup (Hours 0–4)

- [ ] `pip install dllm` and verify GPU detection
- [ ] Download `Qwen/Qwen3-0.6B` checkpoint (~1.2 GB)
- [ ] Implement `MiCAConfig` and `MiCALinear` module (SVD init, freeze B, train A)
- [ ] Write unit test: verify `W + (α/r)·B·A` matches `W` at init (A=0)
- [ ] Write SVD precomputation script for all q_proj/v_proj matrices
- [ ] Implement WSD block-size scheduler (step-based phase transitions)
- [ ] Implement document-level attention mask compatible with BD3LMTrainer
- [ ] Write evaluation script: MLM perplexity, HellaSwag, LAMBADA

### Phase 1: Baseline Runs (Hours 4–16)

- [ ] Run C1: reproduce dLLM Tiny-A2D (Full FT, SFT only, BD3LM, block=32)
- [ ] Compare eval loss against dLLM paper Table 3 (Qwen3-0.6B-diffusion-bd3lm-v0.1)
- [ ] Run C2: LoRA (r=16), SFT only
- [ ] Run C3: MiCA (r=16), SFT only

### Phase 2: WSD Runs (Hours 16–40)

- [ ] Run C4: LoRA + WSD curriculum
- [ ] Run C5: MiCA + WSD curriculum  ← primary
- [ ] Run C6: MiCA-Major + WSD
- [ ] Run C7: MiCA-Random + WSD

### Phase 3: Analysis (Hours 40–48)

- [ ] Collect all checkpoints; apply top-3 checkpoint merge per run
- [ ] Evaluate all 8 conditions on HellaSwag, LAMBADA, GSM8K-500
- [ ] Compute AR knowledge retention (WikiText-2 perplexity)
- [ ] Generate convergence curves (matplotlib)
- [ ] Generate parameter efficiency Pareto plot
- [ ] Compute singular value statistics for MiCA A-matrix evolution

---

## References

- Rüdiger & Raschka (2026). *MiCA Learns More Knowledge Than LoRA and Full Fine-Tuning.* arXiv:2604.01694.
- Zhou et al. (2026). *dLLM: Simple Diffusion Language Modeling.* arXiv:2602.22661.
- Bie et al. (2025). *LLaDA2.0: Scaling Up Diffusion Language Models to 100B.* arXiv:2512.15745.
- Arriola et al. (2025). *Block Diffusion: Interpolating between Autoregressive and Diffusion Language Models.* ICLR 2025.
- Gong et al. (2025). *Scaling Diffusion Language Models via Adaptation from Autoregressive Models.* ICLR 2025. (DiffuLLaMA)
- Chandrasegaran et al. (2025). *RND1: Simple, Scalable AR-to-Diffusion Conversion.* Radical Numerics.
- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.
- Sahoo et al. (2024). *Simple and Effective Masked Diffusion Language Models.* NeurIPS 2024.
- Nie et al. (2025). *Scaling Up Masked Diffusion Models on Text.* ICLR 2025. (LLaDA)
- Yang et al. (2025). *Qwen3 Technical Report.* arXiv:2505.09388.
