# Memory Optimizations

Block Diffusion training, especially with larger block sizes, can be extremely memory-intensive due to the quadratic scaling of attention and the storage of activations for backpropagation.

## Implemented Safegaurds

To ensure stable training on systems with 128GB Unified Memory (like the NVIDIA GB10), we have implemented several optimizations:

### 1. Gradient Checkpointing
Enabled by default in `scripts/train.py`. It discards intermediate activations during the forward pass and re-computes them during the backward pass.
*   **Impact**: Significantly reduces peak VRAM usage.
*   **Cost**: Approximately 20-30% increase in wall-clock training time.

### 2. Safer Micro-Batching
We use a combination of small per-device batch sizes and gradient accumulation to maintain a stable effective batch size.
*   **Per-Device Batch Size**: Reduced from 8 to **4**.
*   **Gradient Accumulation**: Increased from 4 to **8**.
*   **Effective Batch Size**: Remains **32**.

### 3. Logit Filtering
In `BD3LMTrainer`, we optimize the loss calculation by only computing the LM head logits for the **masked tokens** in the current block, rather than the entire sequence. This drastically reduces the memory footprint of the final layer activations.

## Monitoring Memory
Use the provided `scripts/mem_monitor.py` or the `systemd-run` memory high-water mark logs to check if the 100GB limit is being approached.
