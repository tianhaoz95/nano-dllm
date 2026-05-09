# nano-dllm Developer Documentation

Welcome to the technical documentation for **nano-dllm**. This project implements a high-efficiency training recipe for converting pretrained autoregressive (AR) language models into **Block Diffusion Language Models (BD3LM)**.

## Project Vision

The goal of `nano-dllm` is to bridge the gap between fast, causal generation and the high-quality, non-autoregressive refinement capabilities of diffusion models. By using **Minor Component Adaptation (MiCA)**, we adapt large pretrained models (specifically Qwen3-0.6B) with minimal parameter overhead while preserving their massive causal knowledge.

## Core Components

*   **[MiCA PEFT](architecture/mica.md)**: Parameter-efficient fine-tuning that targets the minor singular directions of weight matrices.
*   **[BD3LM](architecture/bd3lm.md)**: A block-wise masked diffusion objective that enables bidirectional context within a sequence.
*   **[WSD Curriculum](architecture/wsd.md)**: A Warmup–Stable–Decay scheduler that progressively increases block sizes to stabilize training.

## Recent Enhancements

*   **Memory Optimizations**: Integrated gradient checkpointing and dynamic micro-batching to handle quadratic attention memory scaling on NVIDIA GB10 hardware.
*   **Automated Benchmarking**: A post-checkpoint callback that automatically evaluates model performance on GSM8K across multiple configurations (AR baseline, Zero-MiCA, and trained adapters).
*   **Weights & Biases Integration**: Live tracking of training loss, token-level accuracy, and benchmark results.

## Quick Links

*   **[Training Guide](training/guide.md)**: How to start and monitor a training run.
*   **[GSM8K Protocol](benchmarking/gsm8k.md)**: Understanding our evaluation metrics.
*   **[Resource Management](operations/resources.md)**: Optimizing for the Blackwell (GB10) architecture.
