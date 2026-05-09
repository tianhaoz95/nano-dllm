# GSM8K Benchmark — MiCA-BD3LM

**Date**: 2026-05-06 21:46:17  
**Checkpoint**: `/home/tianhaoz/github/nano-dllm/outputs/run-mica-wsd-001/checkpoint-final`  
**Base model**: `/home/tianhaoz/github/nano-dllm/models/Qwen3-0.6B`  
**Few-shot**: 5  
**Test examples**: 32  
**Decoding**: greedy (temperature=0)  

## Results

| Model | Strict-Match ↑ | Flexible-Extract ↑ | Correct (strict) | Wall time |
|-------|---------------|-------------------|-----------------|-----------|
| MiCA-BD3LM (checkpoint-final) | 0.2812 (28.12%) | 0.0625 (6.25%) | 9/32 | 1.2 min |

## Metric Definitions

- **Strict-match**: generation contains `#### <number>` and the number matches gold (commas/$ stripped).
- **Flexible-extract**: last number found anywhere in the generation matches gold.

**Total optimizer steps**: 4500 · MiCA rank=16 α=16 targets=q_proj,v_proj · BF16
