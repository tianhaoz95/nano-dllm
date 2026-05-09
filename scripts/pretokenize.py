
import os
import sys
import torch
import transformers
from datasets import load_dataset, concatenate_datasets, DatasetDict
from functools import partial

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dllm_repo"))
sys.path.insert(0, REPO_ROOT)

import dllm

def pretokenize():
    model_path = "./models/Qwen3-0.6B"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
    
    print("Loading datasets...")
    # Math corpus
    math_ds = load_dataset("OpenCoder-LLM/opc-fineweb-math-corpus", split="train", streaming=True).take(1000000)
    
    # Alpaca
    alpaca_ds = load_dataset("tatsu-lab/alpaca", split="train")
    
    # Process math (convert to same format as alpaca if needed, but here we just need 'text')
    # Actually opc-fineweb-math-corpus has 'text' field.
    
    def gen_from_streaming(ds):
        for ex in ds:
            yield {"text": ex["text"]}
            
    from datasets import Dataset
    print("Materializing math dataset...")
    math_ds = Dataset.from_generator(partial(gen_from_streaming, math_ds))
    
    print("Combining datasets...")
    # Only keep 'text' column
    alpaca_ds = alpaca_ds.remove_columns([c for c in alpaca_ds.column_names if c != 'text'])
    math_ds = math_ds.remove_columns([c for c in math_ds.column_names if c != 'text'])
    
    combined_ds = concatenate_datasets([alpaca_ds, math_ds])
    
    print(f"Total samples: {len(combined_ds)}")
    
    print("Tokenizing and grouping...")
    tok_fn = partial(
        dllm.utils.tokenize_and_group,
        tokenizer=tokenizer,
        text_field="text",
        seq_length=1024,
        insert_eos=True,
        drop_tail=True,
    )
    
    tokenized_ds = combined_ds.map(
        tok_fn,
        batched=True,
        batch_size=1000,
        num_proc=16,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    output_path = "./datasets/tokenized_math_alpaca_1024"
    os.makedirs(output_path, exist_ok=True)
    tokenized_ds.save_to_disk(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    pretokenize()
