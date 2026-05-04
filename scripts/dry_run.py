#!/usr/bin/env python3
"""
Dry-run: verify the BD3LM + MiCA training pipeline works end-to-end on GPU.

Usage (tsp required per project conventions):
    source .venv/bin/activate
    tsp python scripts/dry_run.py

Monitor:
    tsp          # check queue / status
    tsp -t <id>  # live output
"""

import os
import sys
import time

import torch
import transformers
from datasets import Dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dllm_repo"))
sys.path.insert(0, REPO_ROOT)

from dllm.pipelines.a2d import A2DQwen3Config, A2DQwen3LMHeadModel  # noqa: E402
from dllm.core.trainers.bd3lm import BD3LMConfig, BD3LMTrainer       # noqa: E402
from mica import apply_mica                                            # noqa: E402

MODEL_PATH   = os.path.join(REPO_ROOT, "models", "Qwen3-0.6B")
OUTPUT_DIR   = os.path.join(REPO_ROOT, "outputs", "dry_run")
BLOCK_SIZE   = 32
SEQ_LEN      = 64   # multiple of BLOCK_SIZE
NUM_TRAIN    = 16
NUM_EVAL     = 4
MAX_STEPS    = 5
MICA_RANK    = 16
MICA_ALPHA   = 16.0
MICA_TARGETS = ["q_proj", "v_proj"]


def make_dummy_dataset(vocab_size: int, num_samples: int, seq_len: int) -> Dataset:
    """Random in-memory dataset; labels == input_ids (required by BD3LMTrainer)."""
    rng = torch.Generator()
    rng.manual_seed(42)
    id_hi = min(vocab_size - 1, 5000)
    records = []
    for _ in range(num_samples):
        ids = torch.randint(100, id_hi, (seq_len,), generator=rng).tolist()
        records.append({"input_ids": ids, "labels": ids.copy()})
    return Dataset.from_list(records)


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    t0 = time.time()
    print("=" * 60)
    print("  BD3LM + MiCA Dry-Run Verification")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — a GPU is required.")
    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU  : {gpu.name}")
    print(f"VRAM : {gpu.total_memory / 1e9:.1f} GB")

    transformers.set_seed(42)

    # ── 1. Tokenizer ─────────────────────────────────────────────────
    print("\n[1/5] Loading tokenizer ...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        MODEL_PATH, padding_side="right"
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"mask_token": "<|mask|>"})
    print(f"      vocab size    : {len(tokenizer)}")
    print(f"      mask_token_id : {tokenizer.mask_token_id}")

    # ── 2. Model (Qwen3-0.6B → A2DQwen3) ─────────────────────────────
    print("\n[2/5] Building A2DQwen3 model ...")
    src_cfg  = transformers.AutoConfig.from_pretrained(MODEL_PATH)
    cfg_dict = {k: v for k, v in src_cfg.to_dict().items()
                if k not in ("model_type", "architectures")}
    a2d_cfg  = A2DQwen3Config(**cfg_dict)

    print("      copying weights from Qwen3-0.6B (CPU) ...")
    src_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = A2DQwen3LMHeadModel(a2d_cfg).to(dtype=torch.bfloat16)
    model.load_state_dict(src_model.state_dict(), strict=False)
    del src_model

    model.resize_token_embeddings(len(tokenizer))
    model.config._attn_implementation = "sdpa"

    # ── 3. Apply MiCA ─────────────────────────────────────────────────
    print(f"\n[3/5] Applying MiCA (rank={MICA_RANK}, alpha={MICA_ALPHA}) ...")
    print(f"      targets       : {MICA_TARGETS}")

    t_svd = time.time()
    apply_mica(model, target_modules=MICA_TARGETS, rank=MICA_RANK, alpha=MICA_ALPHA)
    print(f"      SVD time      : {time.time() - t_svd:.1f}s")

    total, trainable = count_params(model)
    print(f"      total params  : {total:,}")
    print(f"      trainable     : {trainable:,}  ({100 * trainable / total:.3f}%)")

    # Sanity check: ΔW must be exactly 0 at init because A is zero-initialised
    for name, m in model.named_modules():
        if hasattr(m, "A") and hasattr(m, "B"):
            delta_max = (m.scaling * m.B @ m.A).abs().max().item()
            assert delta_max == 0.0, f"Non-zero delta at init in {name}: {delta_max}"
    print("      init check    : ΔW = 0 for all MiCA layers ✓")

    model = model.cuda()
    print(f"      device        : {next(model.parameters()).device}")

    # ── 4. Datasets ───────────────────────────────────────────────────
    print("\n[4/5] Building synthetic datasets ...")
    train_ds = make_dummy_dataset(len(tokenizer), NUM_TRAIN, SEQ_LEN)
    eval_ds  = make_dummy_dataset(len(tokenizer), NUM_EVAL,  SEQ_LEN)
    print(f"      train : {len(train_ds)} × {SEQ_LEN} tokens")
    print(f"      eval  : {len(eval_ds)} × {SEQ_LEN} tokens")

    # ── 5. Train ──────────────────────────────────────────────────────
    print(f"\n[5/5] Running BD3LMTrainer for {MAX_STEPS} steps ...")
    training_args = BD3LMConfig(
        output_dir                  = OUTPUT_DIR,
        overwrite_output_dir        = True,
        max_steps                   = MAX_STEPS,
        per_device_train_batch_size = 2,
        per_device_eval_batch_size  = 2,
        gradient_accumulation_steps = 1,
        bf16                        = True,
        learning_rate               = 1e-4,
        logging_steps               = 1,
        eval_strategy               = "steps",
        eval_steps                  = MAX_STEPS,
        save_steps                  = MAX_STEPS,
        eval_on_start               = False,
        report_to                   = "none",
        block_size                  = BLOCK_SIZE,
        dataloader_num_workers      = 0,
    )
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True
    )
    trainer = BD3LMTrainer(
        model          = model,
        tokenizer      = tokenizer,
        train_dataset  = train_ds,
        eval_dataset   = eval_ds,
        args           = training_args,
        data_collator  = collator,
    )
    trainer.train()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  PASSED — BD3LM + MiCA dry run finished in {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
