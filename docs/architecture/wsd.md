# WSD Block-Size Curriculum

Training a model to handle bidirectional context is challenging. To stabilize convergence, we use the **Warmup–Stable–Decay (WSD)** scheduler for the block size.

## The Schedule

The block size $K$ follows a progressive schedule across the 20,000 training steps:

| Phase | Steps | Block Size | Effective Objective |
|-------|-------|------------|---------------------|
| `warmup_ar` | 0 – 1000 | 1 | Pure Autoregressive |
| `warmup_4` | 1001 – 1500 | 4 | Initial Bidirectional |
| `warmup_32` | 1501 – 2000 | 32 | Medium Block |
| `warmup_128` | 2001 – 2500 | 128 | Large Block |
| `warmup_512` | 2501 – 3000 | 512 | Full Sequence (Diffusion) |
| **`stable`** | 3001 – 18000 | **1024** | Steady-state Diffusion |
| `decay_256` | 18001 – 19000 | 256 | Refinement |
| `decay_64` | 19001 – 19500 | 64 | Fine-grained |
| `decay_32` | 19501 – 20000 | 32 | High-precision |

## Implementation

The scheduler is implemented as a custom `TrainerCallback`. At each step, the callback:
1.  Checks the current step count.
2.  Determines the active phase.
3.  Updates the `trainer.block_size` attribute.
4.  Logs phase transitions to the console and W&B.
