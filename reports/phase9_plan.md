# Phase 9 — Fine-tuning Qwen2.5-Coder-1.5B with LoRA

This phase adapts the base Ollama model so it produces faithful, JSON-valid
Business Intent Cards specifically for COBOL banking code.

## Pipeline

```
infer envelopes ──┐
logic_blocks      ├─► finetune-build ─► finetune_train.jsonl
chunks (RAG)      ┘                            │
                                               ▼
                                    Colab LoRA SFT (qwen-coder-1.5b)
                                               │
                                               ▼
                                  cobol-archaeologist-merged/ (HF)
                                               │
                          llama.cpp convert+quantize → q4_k_m.gguf
                                               │
                              ollama create cobol-archaeologist:v1
                                               │
                                               ▼
                          compare-baselines vs Phase-8 table
```

## 1. Build the SFT corpus

We reuse the 35 records already produced in Phase 8 and (optionally) extend
them with more inferences:

```powershell
python -m cobol_archaeologist.cli finetune-build `
    --envelopes outputs/baseline_predictions.jsonl `
    --logic-blocks data/processed/logic_blocks.jsonl `
    --chunks data/index/chunks.jsonl `
    --out outputs/finetune_train.jsonl
```

Output is chat-format JSONL (`messages: [system,user,assistant]`) with a
quality filter that drops echo-style placeholders and short cards.

To grow the corpus, run more inferences (overnight) and re-build:

```powershell
python -m cobol_archaeologist.cli infer `
    --logic-blocks data/processed/logic_blocks.jsonl `
    --backend ollama --index-dir data/index --offline `
    --limit 500 --out outputs/baseline_500.jsonl --tag code+static+rag

python -m cobol_archaeologist.cli finetune-build `
    --envelopes outputs/baseline_500.jsonl `
    --logic-blocks data/processed/logic_blocks.jsonl `
    --chunks data/index/chunks.jsonl `
    --out outputs/finetune_train_500.jsonl
```

## 2. Train (Google Colab T4)

Open `scripts/finetune_lora.ipynb` in Colab, upload `finetune_train.jsonl`,
run all cells. The notebook produces a merged HF model in
`cobol-archaeologist-merged/`.

Hyper-parameters (already set in the notebook):
- LoRA r=16, α=32, dropout 0.05, all attention + MLP projections
- 3 epochs, lr 2e-4 cosine, eff. batch 8, fp16, max_seq_len 2048

## 3. Quantize and load into Ollama

After downloading `cobol-archaeologist-merged/` locally:

```powershell
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
python convert_hf_to_gguf.py ..\cobol-archaeologist-merged `
    --outfile ..\cobol-archaeologist.f16.gguf --outtype f16
.\build\bin\llama-quantize ..\cobol-archaeologist.f16.gguf `
    ..\cobol-archaeologist.q4_k_m.gguf q4_k_m

cd ..\cobol-archaeologist
ollama create cobol-archaeologist:v1 -f scripts\Modelfile.cobol-archaeologist
```

## 4. Re-run the ablation table

```powershell
python -m cobol_archaeologist.cli compare-baselines `
    --golden data/generated/golden_eval.jsonl `
    --backend ollama `
    --backend-args '{"model":"cobol-archaeologist:v1"}' `
    --index-dir data/index --offline `
    --out-dir reports/phase9
```

Append the resulting row to `reports/comparison_table.md` and the Phase 9
section is done.
