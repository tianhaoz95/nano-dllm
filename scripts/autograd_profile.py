
import torch
import transformers
from dllm.pipelines.a2d import A2DQwen3Config, A2DQwen3LMHeadModel
from dllm.core.trainers.bd3lm import BD3LMTrainer, BD3LMConfig
from dllm.core.schedulers import LinearAlphaScheduler
from torch.profiler import profile, record_function, ProfilerActivity

def profile_autograd():
    model_path = "./models/Qwen3-0.6B"
    config = A2DQwen3Config.from_pretrained(model_path)
    config._attn_implementation = "sdpa"
    model = A2DQwen3LMHeadModel(config).to(dtype=torch.bfloat16, device="cuda")
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
    
    args = BD3LMConfig(
        output_dir="./tmp",
        per_device_train_batch_size=8,
        max_steps=10,
        bf16=True,
        block_size=1024,
        eval_strategy="no"
    )
    
    trainer = BD3LMTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        scheduler=LinearAlphaScheduler()
    )
    
    # Mock input
    input_ids = torch.randint(0, 151936, (8, 1024), device="cuda")
    labels = input_ids.clone()
    inputs = {"input_ids": input_ids, "labels": labels}
    
    print("Starting autograd profiling...")
    
    # Warmup
    for _ in range(2):
        trainer.compute_loss(model, inputs)
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_step"):
            loss = trainer.compute_loss(model, inputs)
            loss.backward()
    
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

if __name__ == "__main__":
    profile_autograd()
