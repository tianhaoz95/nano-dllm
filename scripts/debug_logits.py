import torch
import sys
import os
from safetensors.torch import load_file
import transformers

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dllm_repo"))
sys.path.insert(0, REPO_ROOT)

from dllm.pipelines.a2d import A2DQwen3Config, A2DQwen3LMHeadModel
from mica import apply_mica

def main():
    checkpoint = "./outputs/run-mica-wsd-001/checkpoint-final"
    device = "cuda"
    
    cfg = A2DQwen3Config.from_pretrained(checkpoint)
    model = A2DQwen3LMHeadModel(cfg).to(dtype=torch.bfloat16)
    apply_mica(model, target_modules=["q_proj", "v_proj"], rank=16, alpha=16.0)
    state = load_file(os.path.join(checkpoint, "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=False)
    model.tie_weights()
    model = model.to(device).eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint)
    
    # Test 1: AR Continuation (predict next token for "Beijing is the capital of China")
    prompt = "Beijing is the capital of China"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(inputs["input_ids"]).logits
    
    last_logits = logits[0, -1]
    top_vals, top_idx = torch.topk(last_logits, k=10)
    print(f"\n--- AR Continuation for: '{prompt}' ---")
    for i in range(10):
        token = tokenizer.decode([top_idx[i]])
        print(f"  {i+1}. {token!r} (id={top_idx[i]}, logit={top_vals[i]:.2f})")

    # Test 2: Mask Prediction (predict mask for "Beijing is the capital of [MASK]")
    mask_token = tokenizer.mask_token
    prompt_masked = f"Beijing is the capital of {mask_token}"
    inputs_masked = tokenizer(prompt_masked, return_tensors="pt").to(device)
    mask_pos = (inputs_masked["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    
    with torch.no_grad():
        logits_m = model(inputs_masked["input_ids"]).logits
    
    mask_logits = logits_m[0, mask_pos[0]]
    top_vals_m, top_idx_m = torch.topk(mask_logits, k=10)
    print(f"\n--- Mask Prediction for: '{prompt_masked}' ---")
    for i in range(10):
        token = tokenizer.decode([top_idx_m[i]])
        print(f"  {i+1}. {token!r} (id={top_idx_m[i]}, logit={top_vals_m[i]:.2f})")

if __name__ == "__main__":
    main()
