# Training Guide

This guide explains how to execute and monitor a MiCA-BD3LM training run.

## Environment Setup

Ensure you have the virtual environment activated:
```bash
source .venv/bin/activate
```

## Starting a Run

We use `task-spooler` (`tsp`) and `systemd-run` to manage jobs and enforce memory limits on the NVIDIA GB10 system.

```bash
tsp systemd-run --user --scope \
    -p MemoryMax=100G \
    -p MemorySwapMax=0 \
    -p MemoryHigh=90G \
    env WANDB_API_KEY=your_key_here \
    WANDB_ENTITY=your_entity_here \
    WANDB_PROJECT=nano-dllm \
    WANDB_MODE=online \
    python scripts/train.py \
    --output_dir ./outputs/your-run-name \
    --max_steps 20000 \
    --learning_rate 1e-4
```

### Key Parameters
*   `--mica_rank`: Rank of the adapters (default: 32).
*   `--resume_from_checkpoint`: Path to a directory (e.g., `./outputs/run1/checkpoint-1000`) to continue training.
*   `--report_to`: Set to `"wandb"` for live tracking.

## Monitoring

### Terminal
Monitor the live logs using `tsp`:
```bash
tsp           # List all jobs
tsp -t <ID>   # Tail output of job <ID>
```

### Weights & Biases
If `--report_to wandb` is active, you can view live charts for:
*   `train/loss`: Training loss.
*   `train/accuracy`: Token-level prediction accuracy at masked positions.
*   `gsm8k/accuracy`: Post-checkpoint benchmark results.
*   `system/gpu_utilization`: Hardware health.
