# COBOL-Archaeologist

Recover business intent from undocumented COBOL programs in banking/financial systems and emit
structured **Business Intent Cards** (What / Why / Code Evidence / Regulation Link / Confidence).

## Pipeline

```
Raw COBOL files
   v
COBOL cleaner (column 1-6 / col-7 indicator handling, EBCDIC fallback)
   v
Paragraph extractor (regex, with optional Lark grammar)
   v
Static analysis (vars read/written, conditions, PERFORM, file refs, copybooks)
   v
Logic-block segmenter -> data/processed/logic_blocks.jsonl
   v
Weak labeler (rules + optional small-LLM refiner)
   v
Regulation RAG (PDF -> chunks -> embeddings -> FAISS)
   v
Inference (pluggable LLM backend) -> Business Intent Cards
   v
Eval harness (faithfulness, JSON validity, ROUGE-L, regulation precision)
```

## Install

```powershell
pip install uv          # once, if you don't have it
uv sync                 # creates .venv and installs all base + dev + api deps
```

Optional extras (RAG embeddings, HuggingFace models):
```powershell
uv sync --extra rag --extra model
```

## Data

```powershell
.\data\download.ps1                  # downloads the public assets
.\data\download.ps1 -GenerateSynthetic # writes the golden seed datasets
```

See [data/README.md](data/README.md) and [data/manifest.json](data/manifest.json).

## Quickstart

```powershell
# 1. Discover COBOL files in an extracted dataset
uv run python -m cobol_archaeologist.cli ingest `
    --root data/raw/ibm-cics-banking-sample-cbsa `
    --out data/processed/sources.csv

# 2. Segment + weakly label into logic blocks
uv run python -m cobol_archaeologist.cli segment `
    --root data/raw/ibm-cics-banking-sample-cbsa `
    --out data/processed/logic_blocks.jsonl --label

# 3. Build the regulation index (use --offline for the hashing embedder)
uv run python -m cobol_archaeologist.cli index-regulations `
    --pdf data/raw/rbi-kyc-master-direction/RBI-Master-Direction-KYC.pdf `
          data/raw/basel-iii-framework/basel-iii-bcbs189.pdf `
    --out-dir data/index --offline

# 4. Run inference (default: deterministic EchoBackend; switch with --backend)
uv run python -m cobol_archaeologist.cli infer `
    --logic-blocks data/processed/logic_blocks.jsonl `
    --index-dir data/index `
    --out results/cards.jsonl --backend echo

# 5. Evaluate against the golden set
uv run python -m cobol_archaeologist.cli eval `
    --golden data/generated/golden_eval.jsonl `
    --out-dir results
```

## LLM backends

Switch via `--backend {echo|hf|openai|ollama}`. Pass JSON kwargs with `--backend-args`:

```powershell
--backend hf --backend-args '{"model_name":"Qwen/Qwen2.5-Coder-1.5B-Instruct"}'
--backend openai --backend-args '{"model":"gpt-4o-mini"}'
--backend ollama --backend-args '{"model":"qwen2.5-coder:1.5b"}'
```

`OPENAI_API_KEY` / `OPENAI_BASE_URL` and `OLLAMA_HOST` are honored.

## API server

```powershell
uv run uvicorn cobol_archaeologist.api.main:app --reload --port 8000
```

## Tests

```powershell
uv run pytest
```

## Layout

- [src/cobol_archaeologist/ingest](src/cobol_archaeologist/ingest) - cleaner + discovery
- [src/cobol_archaeologist/parser](src/cobol_archaeologist/parser) - paragraphs, copybooks, optional Lark grammar
- [src/cobol_archaeologist/static_analysis](src/cobol_archaeologist/static_analysis) - fact extraction
- [src/cobol_archaeologist/segmenter](src/cobol_archaeologist/segmenter) - logic-block builder
- [src/cobol_archaeologist/labels](src/cobol_archaeologist/labels) - weak labeler (rules + optional LLM refiner)
- [src/cobol_archaeologist/rag](src/cobol_archaeologist/rag) - PDF loader, chunker, embeddings, FAISS index
- [src/cobol_archaeologist/model](src/cobol_archaeologist/model) - prompt, backends, runner, parser
- [src/cobol_archaeologist/eval](src/cobol_archaeologist/eval) - metrics + report runner
- [examples/run_end_to_end.py](examples/run_end_to_end.py) - end-to-end demo against the test fixture
