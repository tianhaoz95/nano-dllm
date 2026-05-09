# GSM8K Benchmark — MiCA-BD3LM

**Date**: 2026-05-08 18:40:09  
**Checkpoint**: `outputs/mica-bd3lm-gsm8k-scaled/checkpoint-1000`  
**Base model**: `/home/tianhaoz/github/nano-dllm/models/Qwen3-0.6B`  
**Few-shot**: 5  
**Test examples**: 50  
**Decoding**: greedy (temperature=0)  

## Results

| Model | Strict-Match ↑ | Flexible-Extract ↑ | Correct (strict) | Wall time |
|-------|---------------|-------------------|-----------------|-----------|
| MiCA-BD3LM (checkpoint-final) | 0.4200 (42.00%) | 0.0400 (4.00%) | 21/50 | 4.3 min |

## Metric Definitions

- **Strict-match**: generation contains `#### <number>` and the number matches gold (commas/$ stripped).
- **Flexible-extract**: last number found anywhere in the generation matches gold.

**Total optimizer steps**: 4500 · MiCA rank=16 α=16 targets=q_proj,v_proj · BF16
