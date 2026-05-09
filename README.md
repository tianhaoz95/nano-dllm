# nano-dllm

[**Developer Documentation**](https://tianhaozhou95.github.io/nano-dllm/) | [**W&B Dashboard**](https://wandb.ai/tianhaozhou95-heji-technology-llc/nano-dllm)

A high-efficiency training recipe for converting pretrained autoregressive (AR) language models into Block Diffusion Language Models (BD3LM).

## Features

- **MiCA PEFT**: Targets minor singular directions to preserve AR priors.
- **BD3LM**: Block-wise diffusion objective for bidirectional sequence modeling.
- **WSD Curriculum**: Progressive block-size scheduling (1 → 1024).
- **Automated GSM8K Benchmarking**: Integrated post-checkpoint evaluation.
- **Hardware Optimized**: Tailored for NVIDIA GB10 (Blackwell) with unified memory safeguards.

## Documentation

Comprehensive documentation is available at [tianhaozhou95.github.io/nano-dllm/](https://tianhaozhou95.github.io/nano-dllm/).

- [Architecture Overview](https://tianhaozhou95.github.io/nano-dllm/architecture/mica/)
- [Training Guide](https://tianhaozhou95.github.io/nano-dllm/training/guide/)
- [Memory Optimizations](https://tianhaozhou95.github.io/nano-dllm/training/optimizations/)
- [GSM8K Protocol](https://tianhaozhou95.github.io/nano-dllm/benchmarking/gsm8k/)
