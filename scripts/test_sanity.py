import torch
import sys
import os
import argparse
from safetensors.torch import load_file
import transformers

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dllm_repo"))
sys.path.insert(0, REPO_ROOT)

from dllm.pipelines.a2d import A2DQwen3Config, A2DQwen3LMHeadModel
from dllm.core.samplers import BD3LMSampler, BD3LMSamplerConfig
from mica import apply_mica

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero_init", action="store_true", help="Use zero-initialized MiCA")
    parser.add_argument("--checkpoint", default="./outputs/run-mica-wsd-001/checkpoint-final")
    parser.add_argument("--question", default="What is the capital of China?")
    args = parser.parse_args()

    device = "cuda"
    checkpoint = args.checkpoint
    
    print(f"Loading model from {checkpoint}...")
    cfg = A2DQwen3Config.from_pretrained(checkpoint)
    model = A2DQwen3LMHeadModel(cfg).to(dtype=torch.bfloat16)
    
    # Apply MiCA
    apply_mica(model, target_modules=["q_proj", "v_proj"], rank=16, alpha=16.0)
    
    if not args.zero_init:
        print("Loading trained weights...")
        state = load_file(os.path.join(checkpoint, "model.safetensors"), device="cpu")
        model.load_state_dict(state, strict=False)
    else:
        print("Using ZERO-INITIALIZED MiCA (should match base model)...")
        # Ensure A is zero
        for n, p in model.named_parameters():
            if ".A" in n:
                p.data.zero_()

    model.tie_weights()
    model = model.to(device).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    sampler_config = BD3LMSamplerConfig(
        steps=128,
        max_new_tokens=64,
        temperature=0.0,
        block_size=1,
        return_dict=False
    )
    sampler = BD3LMSampler(model=model, tokenizer=tokenizer)
    
    prompt = f"Question: {args.question}\nAnswer:"
    inputs_list = tokenizer([prompt])["input_ids"]
    
    print(f"\nQuestion: {args.question}")
    with torch.no_grad():
        outputs_seq = sampler.sample(inputs_list, sampler_config)
    
    seq = outputs_seq[0]
    p_len = len(inputs_list[0])
    # BD3LMSampler pads prefix to block_size. For block_size=1, padded_p_len = p_len
    gen_ids = seq[p_len:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    
    print(f"Generated Answer: {gen_text.strip()}")

if __name__ == "__main__":
    main()
