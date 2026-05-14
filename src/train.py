"""
End-to-end fine-tuning pipeline for COBOL-Archaeologist.

Stages (run in order, or individually via --stage):
  1  generate   — run base Qwen2.5-Coder-1.5B-Instruct over all logic blocks
                  to produce quality training cards (skips echo placeholder cards)
  2  build      — convert inference envelopes into chat-format SFT JSONL
  3  train      — LoRA fine-tune on A100 (BF16, no quantisation needed)
  4  merge      — merge LoRA adapter into base weights and save in HF format

After stage 4, convert to GGUF and load into Ollama (see README for commands).

Usage
-----
  # Full pipeline (recommended for first run on supercomputer):
  python src/train.py

  # Individual stages:
  python src/train.py --stage generate
  python src/train.py --stage build
  python src/train.py --stage train
  python src/train.py --stage merge

Requirements (install once on the supercomputer):
  pip install transformers peft trl accelerate bitsandbytes datasets

Paths (all relative to repo root, override with env vars or flags):
  LOGIC_BLOCKS   data/processed/logic_blocks.jsonl
  INDEX_DIR      data/index
  ENVELOPES      outputs/train_envelopes.jsonl   (stage 1 output / stage 2 input)
  TRAIN_FILE     outputs/finetune_train.jsonl    (stage 2 output / stage 3 input)
  LORA_DIR       outputs/lora_adapter            (stage 3 output / stage 4 input)
  MERGED_DIR     outputs/merged_model            (stage 4 output)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ── constants ────────────────────────────────────────────────────────────────

MODEL_ID     = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
LOGIC_BLOCKS = ROOT / "data/processed/logic_blocks.jsonl"
INDEX_DIR    = ROOT / "data/index"
ENVELOPES    = ROOT / "outputs/train_envelopes.jsonl"
TRAIN_FILE   = ROOT / "outputs/finetune_train.jsonl"
LORA_DIR     = ROOT / "outputs/lora_adapter"
MERGED_DIR   = ROOT / "outputs/merged_model"

# A100 training config (BF16, no quantisation)
TRAIN_CFG = dict(
    num_train_epochs        = 5,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 2,
    learning_rate           = 2e-4,
    lr_scheduler_type       = "cosine",
    warmup_ratio            = 0.05,
    logging_steps           = 5,
    save_strategy           = "epoch",
    bf16                    = True,
    fp16                    = False,
    max_seq_length          = 2048,
    packing                 = True,
    dataset_text_field      = "text",
    report_to               = "none",
)

LORA_CFG = dict(
    r               = 16,
    lora_alpha      = 32,
    lora_dropout    = 0.05,
    bias            = "none",
    task_type       = "CAUSAL_LM",
    target_modules  = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
)

# ── stage 1 — generate ───────────────────────────────────────────────────────

def stage_generate(args):
    """Run base HF model over all logic blocks to produce training envelopes."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from cobol_archaeologist.schemas import LogicBlock
    from cobol_archaeologist.model.prompt import render_prompt
    from cobol_archaeologist.model.parse_output import parse_card, CardParseError
    from cobol_archaeologist.rag.index import RegulationIndex

    print(f"[stage 1] Loading model {MODEL_ID} in BF16 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Load RAG index if available
    reg_index = None
    try:
        reg_index = RegulationIndex.load(INDEX_DIR)
        print(f"[stage 1] Loaded regulation index from {INDEX_DIR}")
    except Exception:
        print("[stage 1] No regulation index found — skipping RAG context")

    blocks = []
    with LOGIC_BLOCKS.open() as f:
        for line in f:
            if line.strip():
                blocks.append(LogicBlock.model_validate(json.loads(line)))
    print(f"[stage 1] {len(blocks)} logic blocks to process")

    ENVELOPES.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped = 0

    with ENVELOPES.open("w", encoding="utf-8") as out:
        for i, block in enumerate(blocks):
            retrieved = []
            chunk_ids = []
            if reg_index is not None:
                try:
                    hits = reg_index.search(block.code, top_k=3)
                    retrieved = [h.chunk for h in hits]
                    chunk_ids = [h.chunk.id for h in hits]
                except Exception:
                    pass

            prompt = render_prompt(block, retrieved=retrieved,
                                   include_static=True, include_rag=bool(retrieved))
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )

            try:
                card = parse_card(response)
                envelope = {
                    "logic_block_id":    block.id,
                    "model":             MODEL_ID,
                    "mode":              "code+static" + ("+rag" if retrieved else ""),
                    "retrieved_chunk_ids": chunk_ids,
                    "card":              json.loads(card.model_dump_json()),
                }
                out.write(json.dumps(envelope, ensure_ascii=False) + "\n")
                kept += 1
            except CardParseError:
                skipped += 1

            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(blocks)}  kept={kept}  skipped={skipped}")

    print(f"[stage 1] Done — kept {kept}, skipped {skipped} → {ENVELOPES}")


# ── stage 2 — build corpus ────────────────────────────────────────────────────

def stage_build(args):
    """Convert envelopes → chat-format SFT JSONL."""
    from cobol_archaeologist.finetune.dataset import build_corpus

    chunks_path = INDEX_DIR / "chunks.jsonl"
    summary = build_corpus(
        envelopes    = ENVELOPES,
        logic_blocks = LOGIC_BLOCKS,
        out_path     = TRAIN_FILE,
        chunks_path  = chunks_path if chunks_path.exists() else None,
        include_static = True,
        include_rag    = True,
        quality_filter = True,
    )
    print(f"[stage 2] {json.dumps(summary, indent=2)}")


# ── stage 3 — train ──────────────────────────────────────────────────────────

def stage_train(args):
    """LoRA fine-tune on A100 (BF16, no quantisation)."""
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    if not TRAIN_FILE.exists():
        sys.exit(f"[stage 3] Training file not found: {TRAIN_FILE}\n"
                 "  Run  python src/train.py --stage build  first.")

    n = sum(1 for _ in TRAIN_FILE.open())
    print(f"[stage 3] {n} training examples in {TRAIN_FILE}")
    if n < 10:
        print("  WARNING: Very few examples. Consider running stage 1 first to generate more.")

    print(f"[stage 3] Loading {MODEL_ID} in BF16 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(**LORA_CFG)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=str(TRAIN_FILE), split="train")

    def to_text(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )}

    ds_text = ds.map(to_text, remove_columns=ds.column_names)

    LORA_DIR.mkdir(parents=True, exist_ok=True)
    cfg = SFTConfig(output_dir=str(LORA_DIR), **TRAIN_CFG)

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds_text, tokenizer=tokenizer)
    trainer.train()
    trainer.model.save_pretrained(str(LORA_DIR))
    tokenizer.save_pretrained(str(LORA_DIR))
    print(f"[stage 3] Adapter saved → {LORA_DIR}")


# ── stage 4 — merge ──────────────────────────────────────────────────────────

def stage_merge(args):
    """Merge LoRA adapter into base weights and save as HF model."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    if not LORA_DIR.exists():
        sys.exit(f"[stage 4] Adapter not found: {LORA_DIR}\n"
                 "  Run  python src/train.py --stage train  first.")

    print(f"[stage 4] Loading base model {MODEL_ID} in BF16 ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )

    print(f"[stage 4] Merging adapter from {LORA_DIR} ...")
    merged = PeftModel.from_pretrained(base, str(LORA_DIR))
    merged = merged.merge_and_unload()

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"[stage 4] Merged model saved → {MERGED_DIR}")
    print()
    print("Next steps — convert to GGUF and load into Ollama:")
    print("  git clone https://github.com/ggerganov/llama.cpp")
    print("  cd llama.cpp && pip install -r requirements.txt")
    print(f"  python convert_hf_to_gguf.py {MERGED_DIR} \\")
    print(f"      --outfile outputs/cobol-archaeologist.f16.gguf --outtype f16")
    print(f"  ./build/bin/llama-quantize outputs/cobol-archaeologist.f16.gguf \\")
    print(f"      outputs/cobol-archaeologist.q4_k_m.gguf q4_k_m")
    print(f"  ollama create cobol-archaeologist:v1 -f scripts/Modelfile.cobol-archaeologist")


# ── CLI ──────────────────────────────────────────────────────────────────────

STAGES = {"generate": stage_generate, "build": stage_build,
          "train": stage_train, "merge": stage_merge}

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        choices=[*STAGES, "all"],
        default="all",
        help="Which stage to run (default: all stages in order)",
    )
    args = parser.parse_args()

    stages = list(STAGES.values()) if args.stage == "all" else [STAGES[args.stage]]
    for fn in stages:
        fn(args)


if __name__ == "__main__":
    main()
