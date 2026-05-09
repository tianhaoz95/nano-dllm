
import time
import torch
import transformers
from dllm.pipelines.a2d import A2DQwen3Config, A2DQwen3LMHeadModel
from dllm.core.trainers.bd3lm import BD3LMTrainer, BD3LMConfig
from dllm.core.schedulers import LinearAlphaScheduler

def profile():
    model_path = "./models/Qwen3-0.6B"
    config = A2DQwen3Config.from_pretrained(model_path)
    config._attn_implementation = "sdpa"
    model = A2DQwen3LMHeadModel(config).to(dtype=torch.bfloat16, device="cuda")
    
    # Mock input
    x = torch.randn(8, 2048, 1024, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    layer = model.model.layers[0]
    
    print("Starting single layer profiling...")
    
    # Warmup
    for _ in range(2):
        out = layer(x, attention_mask=None, position_ids=None)
        out[0].sum().backward()
    
    torch.cuda.synchronize()
    start = time.time()
    for i in range(10):
        iter_start = time.time()
        out = layer(x, attention_mask=None, position_ids=None)
        torch.cuda.synchronize()
        fwd_time = time.time() - iter_start
        
        loss = out[0].sum()
        loss.backward()
        torch.cuda.synchronize()
        bwd_time = time.time() - iter_start - fwd_time
        
        print(f"Layer Step {i}: Fwd {fwd_time:.4f}s | Bwd {bwd_time:.4f}s")
    
    end = time.time()
    print(f"Average single layer: {(end - start) / 10:.4f}s")

if __name__ == "__main__":
    profile()
