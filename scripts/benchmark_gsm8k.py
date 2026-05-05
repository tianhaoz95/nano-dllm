#!/usr/bin/env python3
"""
GSM8K accuracy benchmark for MiCA-BD3LM trained checkpoint.

Evaluates the checkpoint-final model against the full GSM8K test set (1319
examples) using 5-shot chain-of-thought prompting and greedy decoding.
Also evaluates the base Qwen3-0.6B for comparison.

Two metrics (matching lm-evaluation-harness gsm8k.yaml):
  - strict-match : answer after "#### " matches exactly (ignoring commas/$)
  - flexible-extract : last number found in generation matches

Usage (tsp required per project conventions):
    source .venv/bin/activate
    tsp python scripts/benchmark_gsm8k.py

    # Custom checkpoint:
    tsp python scripts/benchmark_gsm8k.py \\
        --checkpoint ./outputs/run-mica-wsd-001/checkpoint-final \\
        --output    ./results/gsm8k_benchmark.md \\
        --batch_size 16
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
from mica import apply_mica

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = os.path.join(REPO_ROOT, "outputs", "run-mica-wsd-001", "checkpoint-final")
DEFAULT_BASE_MODEL  = os.path.join(REPO_ROOT, "models", "Qwen3-0.6B")
DEFAULT_OUTPUT      = os.path.join(REPO_ROOT, "results", "gsm8k_benchmark.md")

MICA_RANK    = 16
MICA_ALPHA   = 16.0
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

def load_mica_model(checkpoint_path: str, device: str):
    """Load A2DQwen3 + MiCA from a training checkpoint directory."""
    cfg = A2DQwen3Config.from_pretrained(checkpoint_path)
    model = A2DQwen3LMHeadModel(cfg).to(dtype=torch.bfloat16)
    apply_mica(model, target_modules=MICA_TARGETS, rank=MICA_RANK, alpha=MICA_ALPHA)
    # Load checkpoint weights (MiCALinear keys: weight, B buffers + A parameter)
    state = safetensors_load(
        os.path.join(checkpoint_path, "model.safetensors"), device="cpu"
    )
    # lm_head.weight is tied to embed_tokens.weight in Qwen3 — load non-strictly
    # then re-tie so generation uses the correct embeddings.
    model.load_state_dict(state, strict=False)
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
) -> dict:
    strict_correct = 0
    flex_correct   = 0
    total          = len(test_ds)
    results        = []

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

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                eos_token_id=eos_token_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens
        prompt_len = enc["input_ids"].shape[1]
        gen_ids    = out_ids[:, prompt_len:]
        generations = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for q, gold_raw, gold_num, gen in zip(questions, gold_answers_raw, gold_nums, generations):
            strict_pred = extract_strict(gen)
            flex_pred   = extract_flexible(gen)
            s_ok = answers_match(strict_pred, gold_num or "")
            f_ok = answers_match(flex_pred,   gold_num or "")
            if s_ok:
                strict_correct += 1
            if f_ok:
                flex_correct += 1
            results.append({
                "question": q,
                "gold": gold_raw,
                "gold_num": gold_num,
                "generation": gen,
                "strict_pred": strict_pred,
                "flex_pred": flex_pred,
                "strict_match": s_ok,
                "flex_match": f_ok,
            })

        done = min(batch_start + batch_size, total)
        elapsed = time.time() - t0
        eta = elapsed / done * (total - done) if done else 0
        print(
            f"  [{label}] {done}/{total} "
            f"strict={strict_correct/done:.3f}  flex={flex_correct/done:.3f}  "
            f"ETA {eta/60:.1f}m",
            end="\r",
            flush=True,
        )

    print()
    elapsed = time.time() - t0
    return {
        "label": label,
        "n_examples": total,
        "strict_acc": strict_correct / total,
        "flex_acc": flex_correct / total,
        "strict_correct": strict_correct,
        "flex_correct": flex_correct,
        "elapsed_s": elapsed,
        "results": results,
    }

# ── Markdown report ────────────────────────────────────────────────────────────

def write_markdown(output_path: str, eval_results: list[dict], meta: dict):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# GSM8K Benchmark — MiCA-BD3LM",
        "",
        f"**Date**: {ts}  ",
        f"**Checkpoint**: `{meta['checkpoint']}`  ",
        f"**Base model**: `{meta['base_model']}`  ",
        f"**Few-shot**: {NUM_FEWSHOT}  ",
        f"**Test examples**: {eval_results[0]['n_examples']}  ",
        f"**Decoding**: greedy (temperature=0)  ",
        "",
        "## Results",
        "",
        "| Model | Strict-Match ↑ | Flexible-Extract ↑ | Correct (strict) | Wall time |",
        "|-------|---------------|-------------------|-----------------|-----------|",
    ]

    for r in eval_results:
        lines.append(
            f"| {r['label']} "
            f"| {r['strict_acc']:.4f} ({r['strict_acc']*100:.2f}%) "
            f"| {r['flex_acc']:.4f} ({r['flex_acc']*100:.2f}%) "
            f"| {r['strict_correct']}/{r['n_examples']} "
            f"| {r['elapsed_s']/60:.1f} min |"
        )

    lines += [
        "",
        "## Metric Definitions",
        "",
        "- **Strict-match**: generation contains `#### <number>` and the number matches gold (commas/$ stripped).",
        "- **Flexible-extract**: last number found anywhere in the generation matches gold.",
        "",
        "## WSD Training Curriculum (reference)",
        "",
        "| Phase | Block size | Steps |",
        "|-------|-----------|-------|",
        "| warmup_ar  | 1   | 500  |",
        "| warmup_4   | 4   | 500  |",
        "| warmup_32  | 32  | 500  |",
        "| warmup_128 | 128 | 500  |",
        "| stable     | 512 | 2000 |",
        "| decay_64   | 64  | 300  |",
        "| decay_32   | 32  | 200  |",
        "",
        "**Total**: 4500 optimizer steps · MiCA rank=16 α=16 targets=q_proj,v_proj · BF16",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nReport written to {output_path}")

# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  default=DEFAULT_CHECKPOINT)
    p.add_argument("--base_model",  default=DEFAULT_BASE_MODEL)
    p.add_argument("--output",      default=DEFAULT_OUTPUT)
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--skip_base",   action="store_true",
                   help="Skip evaluating the base Qwen3-0.6B baseline")
    p.add_argument("--limit",       type=int, default=None,
                   help="Evaluate on first N examples only (for quick tests)")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — a GPU is required.")
    device = "cuda"
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU  : {gpu.name}")
    print(f"VRAM : {gpu.total_memory / 1e9:.1f} GB")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\nLoading GSM8K ...")
    train_ds = load_dataset("gsm8k", "main", split="train")
    test_ds  = load_dataset("gsm8k", "main", split="test")
    fewshot_examples = list(train_ds.select(range(NUM_FEWSHOT)))
    if args.limit:
        test_ds = test_ds.select(range(args.limit))
    print(f"  train (few-shot pool) : {len(train_ds)}")
    print(f"  test                  : {len(test_ds)}")

    all_results = []

    # ── MiCA-BD3LM checkpoint ─────────────────────────────────────────────────
    print(f"\nLoading MiCA-BD3LM checkpoint: {args.checkpoint}")
    tokenizer = load_tokenizer(args.checkpoint)
    mica_model = load_mica_model(args.checkpoint, device)
    total = sum(p.numel() for p in mica_model.parameters())
    trainable = sum(p.numel() for p in mica_model.parameters() if p.requires_grad)
    print(f"  params: {total:,}  trainable: {trainable:,} ({100*trainable/total:.3f}%)")
    print("\nEvaluating MiCA-BD3LM ...")
    r = evaluate(mica_model, tokenizer, test_ds, fewshot_examples,
                 args.batch_size, device, "MiCA-BD3LM (checkpoint-final)")
    all_results.append(r)
    del mica_model
    torch.cuda.empty_cache()

    # ── Base model baseline ───────────────────────────────────────────────────
    if not args.skip_base:
        print(f"\nLoading base model: {args.base_model}")
        base_tok = load_tokenizer(args.base_model)
        base_model = load_base_model(args.base_model, device)
        print("\nEvaluating base Qwen3-0.6B ...")
        r_base = evaluate(base_model, base_tok, test_ds, fewshot_examples,
                          args.batch_size, device, "Qwen3-0.6B (base AR)")
        all_results.append(r_base)
        del base_model
        torch.cuda.empty_cache()

    # ── Save JSON + markdown ──────────────────────────────────────────────────
    json_path = args.output.replace(".md", ".json")
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        # Omit per-sample generations from JSON to keep it small
        json.dump(
            [{k: v for k, v in r.items() if k != "results"} for r in all_results],
            f, indent=2,
        )
    print(f"Summary JSON written to {json_path}")

    meta = {"checkpoint": args.checkpoint, "base_model": args.base_model}
    write_markdown(args.output, all_results, meta)

    print("\n=== Final Results ===")
    for r in all_results:
        print(f"  {r['label']}")
        print(f"    strict-match    : {r['strict_acc']*100:.2f}%")
        print(f"    flexible-extract: {r['flex_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
