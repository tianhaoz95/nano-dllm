# Resource Management

The `nano-dllm` project is optimized for high-memory unified memory systems, specifically the **NVIDIA GB10 (Blackwell)**.

## Hardware Specs
*   **Architecture**: NVIDIA GB10
*   **VRAM/RAM**: 128GB Unified Memory

## Kernel-Level Protections

To prevent OOM (Out Of Memory) events from crashing the entire system or causing massive swap-thrashing, we use `systemd-run` to enforce strict resource slices.

### The Standard Limit
We enforce a **100GB** limit on training processes.

```bash
systemd-run --user --scope \
    -p MemoryMax=100G \
    -p MemorySwapMax=0 \
    -p MemoryHigh=90G \
    ...
```

*   **`MemoryMax=100G`**: Kills the process immediately if it exceeds 100GB.
*   **`MemorySwapMax=0`**: Prevents the process from ever using disk swap (which would cause training to crawl).
*   **`MemoryHigh=90G`**: Triggers proactive kernel memory reclamation when usage hits 90GB.

## Job Spooling

We use **`task-spooler` (`tsp`)** to manage the GPU queue. This ensures that only one heavy training or benchmarking job runs at a time, preventing VRAM contention.

*   **`tsp <command>`**: Queue a job.
*   **`tsp -k <ID>`**: Kill a job.
*   **`tsp -C`**: Clear the finished job list.
