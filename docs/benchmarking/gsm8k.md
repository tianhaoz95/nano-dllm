# GSM8K Benchmarking Protocol

We use the GSM8K (Grade School Math) dataset to track the reasoning and mathematical capabilities of the model throughout its adaptation.

## Automated Benchmark Callback

Every time the model saves a checkpoint (default: every 1000 steps), the `GSMBenchmarkCallback` is triggered.

### The Workflow
1.  **Pause**: Training is paused.
2.  **Clear**: The main model is moved to the **CPU**, and `torch.cuda.empty_cache()` is called.
3.  **Evaluate**: 32 random samples from the GSM8K test set are evaluated across four configurations:
    *   **Base AR**: Original model baseline.
    *   **Zero-MiCA**: BD3LM logic but with adapters zeroed out (verifies logic integrity).
    *   **BS 1 Trained**: Current adapters with block size 1 (AR mode).
    *   **BS 8 Trained**: Current adapters with block size 8 (Diffusion mode).
4.  **Log**: Results are logged to W&B under the `gsm8k/` prefix.
5.  **Resume**: Model is moved back to the GPU and training resumes.

## Why 32 Samples?
32 samples provide a statistically significant "health check" while keeping the evaluation fast enough (~2-3 minutes) to avoid stalling the training process for too long.

## Manual Execution
You can run the full benchmark suite manually using:
```bash
python scripts/benchmark_gsm8k.py --checkpoint ./outputs/your-run/checkpoint-X
```
