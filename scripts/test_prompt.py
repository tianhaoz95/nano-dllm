#!/usr/bin/env python3
"""
Simple sanity check for MiCA-BD3LM model inference.
"""

import argparse
import os
import sys

import torch
import transformers
from safetensors.torch import load_file as safetensors_load

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dllm_repo"))
sys.path.insert(0, REPO_ROOT)

from dllm.pipelines.a2d import A2DQwen3Config, A2DQwen3LMHeadModel
from dllm.core.samplers import BD3LMSampler, BD3LMSamplerConfig
from mica import apply_mica

DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, "outputs", "run-mica-wsd-001", "checkpoint-final")
MICA_RANK = 16
MICA_ALPHA = 16.0
MICA_TARGETS = ["q_proj", "v_proj"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="What is the capital of China?")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=1, help="Block size (default: 1 for accuracy; 32+ may currently produce low quality)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading checkpoint: {args.checkpoint}")
    cfg = A2DQwen3Config.from_pretrained(args.checkpoint)
    model = A2DQwen3LMHeadModel(cfg).to(dtype=torch.bfloat16)
    apply_mica(model, target_modules=MICA_TARGETS, rank=MICA_RANK, alpha=MICA_ALPHA)
    
    state = safetensors_load(os.path.join(args.checkpoint, "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=False)
    model.tie_weights()
    model.config._attn_implementation = "sdpa"
    model = model.to(device).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Prompt: {args.prompt}")
    print(f"Sampling with steps={args.steps}, block_size={args.block_size} ...")

    sampler_config = BD3LMSamplerConfig(
        steps=args.steps,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        block_size=args.block_size,
        return_dict=True
    )
    sampler = BD3LMSampler(model=model, tokenizer=tokenizer)
    
    inputs_list = tokenizer([args.prompt])["input_ids"]
    prompt_len = len(inputs_list[0])
    
    # BD3LMSampler pads the prompt to a multiple of block_size
    padded_prompt_len = ((prompt_len + args.block_size - 1) // args.block_size) * args.block_size
    
    print(f"Prompt length: {prompt_len}, Padded length: {padded_prompt_len}")
    
    with torch.no_grad():
        outputs = sampler.sample(inputs_list, sampler_config)
    
    sequence = outputs.sequences[0]
    # The sampler output contains: [padding] + [prompt] + [generation]
    # Total length before generation is padded_prompt_len
    gen_ids = sequence[padded_prompt_len:]
    
    # Filter out mask tokens if any remain
    mask_id = tokenizer.mask_token_id
    gen_ids = gen_ids[gen_ids != mask_id]
    
    full_text = tokenizer.decode(sequence, skip_special_tokens=False)
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Stop at common repetition/boundary strings
    for s in ["\n\n\n", "Question:", "</s>", "<|im_end|>"]:
        if s in gen_text:
            gen_text = gen_text.split(s)[0]

    print("\n--- FULL OUTPUT (with special tokens) ---")
    print(full_text)
    print("\n--- GENERATED TEXT ---")
    print(gen_text.strip())

if __name__ == "__main__":
    main()
