import torch
import sys
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "dllm_repo"))

from dllm.pipelines.a2d import A2DQwen3Config, A2DQwen3LMHeadModel
from mica import apply_mica
import transformers
from safetensors.torch import load_file

def main():
    checkpoint = "./outputs/run-mica-wsd-001/checkpoint-final"
    device = "cuda"
    
    cfg = A2DQwen3Config.from_pretrained(checkpoint)
    model = A2DQwen3LMHeadModel(cfg).to(dtype=torch.bfloat16)
    apply_mica(model, target_modules=["q_proj", "v_proj"], rank=16, alpha=16.0)
    state = load_file(os.path.join(checkpoint, "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=False)
    model.tie_weights()
    model.config._attn_implementation = "sdpa"
    model = model.to(device).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        
    prompt = "Question: What is 2+2?\nAnswer:"
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=50, do_sample=False)
        
    print("OUTPUT:", tokenizer.decode(out[0]))

if __name__ == "__main__":
    main()
