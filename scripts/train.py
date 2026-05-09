#!/usr/bin/env python3
"""
MiCA-BD3LM full training script.

Converts Qwen3-0.6B → A2DQwen3, applies MiCA PEFT (0.18% trainable params),
and trains with BD3LM objective under the WSD block-size curriculum.

Usage (tsp required per project conventions):
    source .venv/bin/activate
    tsp python scripts/train.py

    # Override defaults:
    tsp python scripts/train.py \\
        --model_name_or_path ./models/Qwen3-0.6B \\
        --dataset_args "openwebtext" \\
        --output_dir ./outputs/mica-bd3lm-run1 \\
        --max_steps 4500

Monitor:
    tsp          # queue / status
    tsp -t <id>  # live output
"""

import functools
import os
import sys
from dataclasses import dataclass, field

import torch
import transformers

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "dllm_repo"))
sys.path.insert(0, REPO_ROOT)

import dllm
from dllm.pipelines.a2d import A2DQwen3Config, A2DQwen3LMHeadModel
from dllm.core.trainers.bd3lm import BD3LMConfig, BD3LMTrainer
from mica import apply_mica, WSDBlockSizeScheduler, WSDBlockSizeCallback
from scripts.benchmark_gsm8k import run_automated_benchmark

logger = dllm.utils.get_default_logger(__name__)


# ── Callbacks ─────────────────────────────────────────────────────────────────

class GSMBenchmarkCallback(transformers.TrainerCallback):
    """
    Automated benchmark triggered after each model save.
    Pauses training by moving model to CPU and clearing VRAM to run evaluations.
    """
    def __init__(self, mica_args):
        self.mica_args = mica_args

    def on_save(self, args, state, control, **kwargs):
        # We only run benchmark if this is a save step and not the very start
        if state.is_world_process_zero and state.global_step > 0:
            checkpoint_dir = f"checkpoint-{state.global_step}"
            checkpoint_path = os.path.join(args.output_dir, checkpoint_dir)
            
            logger.info(f"\n[GSMBench] Pausing training for benchmark at step {state.global_step}...")
            
            # 1. Clear VRAM: Move main model to CPU
            model = kwargs["model"]
            original_device = next(model.parameters()).device
            model.to("cpu")
            torch.cuda.empty_cache()
            
            try:
                # 2. Run the 4-part benchmark (32 samples)
                results = run_automated_benchmark(
                    checkpoint_path=checkpoint_path,
                    limit=32,
                    rank=self.mica_args.mica_rank,
                    alpha=self.mica_args.mica_alpha,
                    device="cuda"
                )
                
                # 3. Log to W&B
                if "wandb" in args.report_to:
                    import wandb
                    if wandb.run is not None:
                        wandb.log(results, step=state.global_step)
                        logger.info(f"[GSMBench] Logged to W&B: {results}")
            
            except Exception as e:
                logger.error(f"[GSMBench] Benchmark failed: {e}")
            
            finally:
                # 4. Resume: Move model back to GPU
                logger.info("[GSMBench] Resuming training...")
                model.to(original_device)
                torch.cuda.empty_cache()


# ── Argument dataclasses ───────────────────────────────────────────────────────

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="./models/Qwen3-0.6B",
        metadata={"help": "Path to the base AR model (Qwen3-0.6B)."},
    )


@dataclass
class DataArguments:
    dataset_args: str = field(
        default="openwebtext",
        metadata={
            "help": (
                "Dataset spec for dllm.data.load_pt_dataset.  "
                "Examples: 'openwebtext', "
                "'wikitext[name:wikitext-103-v1,train:50000,test:2000]', "
                "'mlfoundations/dclm-baseline-1.0[train:1000000,test:5000]'"
            )
        },
    )
    text_field: str = field(default="text")
    max_length: int = field(default=1024, metadata={"help": "Tokens per training sequence."})
    streaming: bool = field(default=True)
    load_preprocessed_data: bool = field(default=False)
    num_proc: int = field(default=8, metadata={"help": "Workers for non-streaming map."})
    insert_eos: bool = field(default=True)
    drop_tail: bool = field(default=True)


@dataclass
class MiCAArguments:
    mica_rank: int = field(default=32)
    mica_alpha: float = field(default=32.0)
    mica_targets: str = field(
        default="q_proj,v_proj",
        metadata={"help": "Comma-separated list of linear layer names to adapt."},
    )


@dataclass
class TrainingArguments(BD3LMConfig):
    output_dir: str = field(default="./outputs/mica-bd3lm-scaled")

    # Training budget
    max_steps: int = field(default=20000)   # Scale up for GSM8K
    learning_rate: float = field(default=1e-4)
    per_device_train_batch_size: int = field(default=4)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=8)  # effective batch = 32
    max_grad_norm: float = field(default=1.0)
    weight_decay: float = field(default=0.01)

    # LR schedule
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.05)

    # Precision & logging
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=1000)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=500)
    save_only_model: bool = field(default=True)
    report_to: str = field(default="wandb")
    overwrite_output_dir: bool = field(default=True)
    dataloader_num_workers: int = field(default=4)
    resume_from_checkpoint: str = field(default=None)

    # BD3LM — initial block_size; WSD callback overrides this per-step
    block_size: int = field(default=1)

    # WSD curriculum
    use_wsd: bool = field(default=True)
    wsd_warmup_ar_steps: int = field(default=1000)
    wsd_warmup_4_steps: int = field(default=500)
    wsd_warmup_32_steps: int = field(default=500)
    wsd_warmup_128_steps: int = field(default=500)
    wsd_warmup_512_steps: int = field(default=500)
    wsd_stable_steps: int = field(default=15000)
    wsd_decay_256_steps: int = field(default=1000)
    wsd_decay_64_steps: int = field(default=500)
    wsd_decay_32_steps: int = field(default=500)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def _build_model_and_tokenizer(model_path: str):
    logger.info("Loading tokenizer ...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, padding_side="right"
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"mask_token": "<|mask|>"})

    logger.info("Converting Qwen3 → A2DQwen3 (copying weights) ...")
    src_cfg = transformers.AutoConfig.from_pretrained(model_path)
    cfg_dict = {k: v for k, v in src_cfg.to_dict().items()
                if k not in ("model_type", "architectures")}
    a2d_cfg = A2DQwen3Config(**cfg_dict)

    src_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = A2DQwen3LMHeadModel(a2d_cfg).to(dtype=torch.bfloat16)
    model.load_state_dict(src_model.state_dict(), strict=False)
    del src_model

    model.resize_token_embeddings(len(tokenizer))
    model.config._attn_implementation = "sdpa"

    return model, tokenizer


def _build_wsd_scheduler(args: TrainingArguments) -> WSDBlockSizeScheduler:
    from mica.wsd_scheduler import WSDPhase
    return WSDBlockSizeScheduler([
        WSDPhase("warmup_ar",  block_size=1,              steps=args.wsd_warmup_ar_steps),
        WSDPhase("warmup_4",   block_size=4,              steps=args.wsd_warmup_4_steps),
        WSDPhase("warmup_32",  block_size=32,             steps=args.wsd_warmup_32_steps),
        WSDPhase("warmup_128", block_size=128,            steps=args.wsd_warmup_128_steps),
        WSDPhase("warmup_512", block_size=512,            steps=args.wsd_warmup_512_steps),
        WSDPhase("stable",     block_size=args.max_length if hasattr(args, "max_length") else 1024,
                               steps=args.wsd_stable_steps),
        WSDPhase("decay_256",  block_size=256,            steps=args.wsd_decay_256_steps),
        WSDPhase("decay_64",   block_size=64,             steps=args.wsd_decay_64_steps),
        WSDPhase("decay_32",   block_size=32,             steps=args.wsd_decay_32_steps),
    ])


# ── Main ───────────────────────────────────────────────────────────────────────

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, MiCAArguments, TrainingArguments)
    )
    model_args, data_args, mica_args, training_args = parser.parse_args_into_dataclasses()

    transformers.set_seed(training_args.seed)
    logger.info("=== MiCA-BD3LM Training ===")
    logger.info(f"Model : {model_args.model_name_or_path}")
    logger.info(f"Data  : {data_args.dataset_args}")
    logger.info(f"MiCA  : rank={mica_args.mica_rank}  alpha={mica_args.mica_alpha}  targets={mica_args.mica_targets}")
    logger.info(f"WSD   : {training_args.use_wsd}")
    logger.info(f"Steps : {training_args.max_steps}")

    model_path = _resolve_path(model_args.model_name_or_path)

    # ── Model & tokenizer ─────────────────────────────────────────────────
    model, tokenizer = _build_model_and_tokenizer(model_path)

    # ── MiCA ──────────────────────────────────────────────────────────────
    targets = [t.strip() for t in mica_args.mica_targets.split(",")]
    logger.info(f"Applying MiCA to {targets} ...")
    apply_mica(model, target_modules=targets, rank=mica_args.mica_rank, alpha=mica_args.mica_alpha)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")

    # ── Dataset ───────────────────────────────────────────────────────────
    logger.info(f"Loading dataset '{data_args.dataset_args}' (streaming={data_args.streaming}, preprocessed={data_args.load_preprocessed_data}) ...")
    dataset = dllm.data.load_pt_dataset(
        data_args.dataset_args,
        streaming=data_args.streaming,
        load_preprocessed_data=data_args.load_preprocessed_data,
    )

    if not data_args.load_preprocessed_data:
        tok_fn = functools.partial(
            dllm.utils.tokenize_and_group,
            tokenizer=tokenizer,
            text_field=data_args.text_field,
            seq_length=data_args.max_length,
            insert_eos=data_args.insert_eos,
            drop_tail=data_args.drop_tail,
        )
        map_kwargs = ({} if data_args.streaming else
                      {"num_proc": data_args.num_proc, "desc": "Tokenizing"})

        train_cols = dataset["train"].column_names
        dataset = dataset.map(
            tok_fn,
            batched=True,
            remove_columns=train_cols,
            **map_kwargs,
        )

    if data_args.streaming:
        dataset["train"] = dataset["train"].shuffle(
            seed=training_args.seed, buffer_size=10_000
        )

    train_ds = dataset["train"]
    eval_ds  = dataset.get("test", None)
    if eval_ds is None:
        logger.info("No test split found. Disabling evaluation.")
        training_args.eval_strategy = "no"
        training_args.eval_steps = None
    
    logger.info(f"Dataset ready.  eval_split={'yes' if eval_ds is not None else 'none'}")

    # ── WSD curriculum ────────────────────────────────────────────────────
    wsd_scheduler = None
    if training_args.use_wsd:
        wsd_scheduler = _build_wsd_scheduler(training_args)
        logger.info(f"WSD: {wsd_scheduler}")

    # ── Trainer ───────────────────────────────────────────────────────────
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, return_tensors="pt", padding=True
    )
    trainer = BD3LMTrainer(
        model          = model,
        processing_class = tokenizer,
        train_dataset  = train_ds,
        eval_dataset   = eval_ds,
        args           = training_args,
        data_collator  = collator,
    )

    if wsd_scheduler is not None:
        trainer.add_callback(
            WSDBlockSizeCallback(trainer=trainer, scheduler=wsd_scheduler)
        )
        trainer.add_callback(GSMBenchmarkCallback(mica_args=mica_args))

    logger.info("Starting training ...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    final_dir = os.path.join(training_args.output_dir, "checkpoint-final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info(f"Saved to {final_dir}")


if __name__ == "__main__":
    train()
