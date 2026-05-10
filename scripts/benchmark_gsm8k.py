#!/usr/bin/env python3
"""
GSM8K accuracy benchmark for MiCA-BD3LM.
Supports multiple configurations and automated logging during training.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import transformers
from datasets import load_dataset
from safetensors.torch import load_file as safetensors_load

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dllm_repo"))
sys.path.insert(0, REPO_ROOT)

from dllm.pipelines.a2d import A2DQwen3Config, A2DQwen3LMHeadModel
from dllm.core.samplers import BD3LMSampler, BD3LMSamplerConfig
from mica import apply_mica

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL  = os.path.join(REPO_ROOT, "models", "Qwen3-0.6B")

MICA_TARGETS = ["q_proj", "v_proj"]
NUM_FEWSHOT  = 5
MAX_NEW_TOKENS = 512
STOP_STRINGS = ["Question:", "</s>", "<|im_end|>"]

# ── Answer extraction ──────────────────────────────────────────────────────────

_STRICT_RE   = re.compile(r"####\s*(-?[0-9][0-9,]*\.?[0-9]*)")
_FLEXIBLE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

def _normalise(s: str) -> str:
    return s.replace(",", "").replace("$", "").strip().rstrip(".")

def extract_strict(text: str) -> str | None:
    m = _STRICT_RE.search(text)
    return _normalise(m.group(1)) if m else None

def extract_flexible(text: str) -> str | None:
    all_matches = _FLEXIBLE_RE.findall(text)
    if not all_matches:
        return None
    last = all_matches[-1]
    raw = last[0] if last[0] else last[1]
    return _normalise(raw)

def answers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    return _normalise(pred) == _normalise(gold)

# ── Few-shot prompt builder ────────────────────────────────────────────────────

def build_fewshot_prompt(train_examples: list[dict], question: str) -> str:
    lines = []
    for ex in train_examples[:NUM_FEWSHOT]:
        lines.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
    lines.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(lines)

# ── Model loaders ──────────────────────────────────────────────────────────────

def load_mica_model(checkpoint_path: str, rank: int, alpha: float, device: str, zero_mica: bool = False):
    """Load A2DQwen3 + MiCA from a training checkpoint directory."""
    cfg = A2DQwen3Config.from_pretrained(checkpoint_path)
    cfg.model_type = "qwen3"
    model = A2DQwen3LMHeadModel(cfg).to(dtype=torch.bfloat16)
    apply_mica(model, target_modules=MICA_TARGETS, rank=rank, alpha=alpha)
    
    if not zero_mica:
        state = safetensors_load(
            os.path.join(checkpoint_path, "model.safetensors"), device="cpu"
        )
        model.load_state_dict(state, strict=False)
    else:
        # Zero out MiCA adapters explicitly
        for m in model.modules():
            if hasattr(m, "A") and hasattr(m, "B"):
                torch.nn.init.zeros_(m.A)
                torch.nn.init.zeros_(m.B)
    
    model.tie_weights()
    model.config._attn_implementation = "sdpa"
    return model.to(device).eval()

def load_base_model(model_path: str, device: str):
    """Load the base Qwen3-0.6B AR model for comparison."""
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        attn_implementation="sdpa",
    )
    return model.eval()

def load_tokenizer(path: str) -> transformers.PreTrainedTokenizer:
    tok = transformers.AutoTokenizer.from_pretrained(path, padding_side="left")
    if not tok.pad_token:
        tok.pad_token = tok.eos_token
    return tok

# ── Evaluation loop ────────────────────────────────────────────────────────────

def evaluate(
    model,
    tokenizer: transformers.PreTrainedTokenizer,
    test_ds,
    fewshot_examples: list[dict],
    batch_size: int,
    device: str,
    label: str,
    block_size: int = 1,
    steps_per_block: int = None,
) -> dict:

    strict_correct = 0
    total          = len(test_ds)

    eos_token_ids = [
        tokenizer.eos_token_id,
        *[tokenizer.convert_tokens_to_ids(s) for s in STOP_STRINGS if s in tokenizer.get_vocab()],
    ]
    eos_token_ids = list({x for x in eos_token_ids if x is not None and x >= 0})

    t0 = time.time()
    for batch_start in range(0, total, batch_size):
        batch = test_ds[batch_start : batch_start + batch_size]
        questions = batch["question"] if isinstance(batch["question"], list) else [batch["question"]]
        gold_answers_raw = batch["answer"] if isinstance(batch["answer"], list) else [batch["answer"]]

        prompts = [build_fewshot_prompt(fewshot_examples, q) for q in questions]
        gold_nums = [extract_strict(a) for a in gold_answers_raw]

        if getattr(model.config, "model_type", "") == "a2d-qwen3":
            sampler_config = BD3LMSamplerConfig(
                steps=128,
                steps_per_block=steps_per_block,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                block_size=block_size,
                return_dict=False
            )
            sampler = BD3LMSampler(model=model, tokenizer=tokenizer)
            inputs_list = tokenizer(prompts)["input_ids"]
            outputs_seq = sampler.sample(inputs_list, sampler_config, right_shift_logits=True)
            
            generations = []
            for i, seq in enumerate(outputs_seq):
                p_len = len(inputs_list[i])
                padded_p_len = ((p_len + block_size - 1) // block_size) * block_size
                gen_ids = seq[padded_p_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                for stop_str in STOP_STRINGS:
                    if stop_str in gen_text:
                        gen_text = gen_text.split(stop_str)[0]
                generations.append(gen_text)
            del sampler
        else:
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(device)
            with torch.no_grad():
                out_ids = model.generate(
                    **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    temperature=None, top_p=None, eos_token_id=eos_token_ids,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_len = enc["input_ids"].shape[1]
            gen_ids    = out_ids[:, prompt_len:]
            generations = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for gold_num, gen in zip(gold_nums, generations):
            if answers_match(extract_strict(gen), gold_num or ""):
                strict_correct += 1

        torch.cuda.empty_cache()

    return {
        "label": label,
        "strict_acc": strict_correct / total,
        "elapsed_s": time.time() - t0,
    }

# ── Main Benchmarking Function ────────────────────────────────────────────────

def run_automated_benchmark(
    checkpoint_path: str,
    base_model_path: str = DEFAULT_BASE_MODEL,
    limit: int = 32,
    rank: int = 32,
    alpha: float = 32.0,
    batch_size: int = 16,
    device: str = "cuda",
):
    """
    Runs the 5-part benchmark requested:
    1. Base AR
    2. Block Size 8 + Zero MiCA (Untrained)
    3. Block Size 1 + Zero MiCA (Untrained)
    4. Block Size 1 + Trained MiCA
    5. Block Size 8 + Trained MiCA
    """
    print(f"\n[GSMBench] Starting automated benchmark for {checkpoint_path} (limit={limit})")
    
    train_ds = load_dataset("gsm8k", "main", split="train")
    test_ds  = load_dataset("gsm8k", "main", split="test").select(range(limit))
    fewshot_examples = list(train_ds.select(range(NUM_FEWSHOT)))
    
    tokenizer = load_tokenizer(checkpoint_path)
    results = {}

    # 1. Base AR
    print("  Evaluating Configuration 1: Base AR ...")
    model = load_base_model(base_model_path, device)
    r = evaluate(model, tokenizer, test_ds, fewshot_examples, batch_size, device, "Base AR")
    results["gsm8k/base_ar_acc"] = r["strict_acc"]
    del model
    torch.cuda.empty_cache()

    # 2. Block Size 8 + Zero MiCA (Untrained)
    print("  Evaluating Configuration 2: BS=8 + Zero MiCA ...")
    model = load_mica_model(checkpoint_path, rank, alpha, device, zero_mica=True)
    r = evaluate(model, tokenizer, test_ds, fewshot_examples, batch_size, device, "BS8-ZeroMiCA", block_size=8)
    results["gsm8k/bs8_zero_mica_acc"] = r["strict_acc"]
    del model
    torch.cuda.empty_cache()

    # 3. Block Size 1 + Zero MiCA (Untrained)
    print("  Evaluating Configuration 3: BS=1 + Zero MiCA ...")
    model = load_mica_model(checkpoint_path, rank, alpha, device, zero_mica=True)
    r = evaluate(model, tokenizer, test_ds, fewshot_examples, batch_size, device, "BS1-ZeroMiCA", block_size=1)
    results["gsm8k/bs1_zero_mica_acc"] = r["strict_acc"]
    del model
    torch.cuda.empty_cache()

    # 4. Block Size 1 + Trained MiCA
    print("  Evaluating Configuration 4: BS=1 + Trained MiCA ...")
    model = load_mica_model(checkpoint_path, rank, alpha, device, zero_mica=False)
    r = evaluate(model, tokenizer, test_ds, fewshot_examples, batch_size, device, "BS1-TrainedMiCA", block_size=1)
    results["gsm8k/bs1_trained_mica_acc"] = r["strict_acc"]
    
    # 5. Block Size 8 + Trained MiCA
    print("  Evaluating Configuration 5: BS=8 + Trained MiCA ...")
    r = evaluate(model, tokenizer, test_ds, fewshot_examples, batch_size, device, "BS8-TrainedMiCA", block_size=8)
    results["gsm8k/bs8_trained_mica_acc"] = r["strict_acc"]
    
    del model
    torch.cuda.empty_cache()
    
    print(f"[GSMBench] Results: {results}")
    return results

if __name__ == "__main__":
    # Command line usage support
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=32.0)
    args = parser.parse_args()
    
    res = run_automated_benchmark(args.checkpoint, args.base_model, args.limit, args.rank, args.alpha)
    print(json.dumps(res, indent=2))
