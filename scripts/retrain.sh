#!/bin/bash
set -e

PYTHON=/dgxa_home/se23ucse073/.miniconda/envs/cobol/bin/python
ROOT=/dgxa_home/se23ucse073/cobol-archaeologist
export CUDA_VISIBLE_DEVICES=0

cd "$ROOT"

echo "=== Re-installing package ==="
$PYTHON -m pip install -e . -q

echo "=== Clearing stale outputs ==="
rm -rf "$ROOT/outputs/lora_adapter" \
       "$ROOT/outputs/merged_model" \
       "$ROOT/outputs/cobol-archaeologist.f16.gguf" \
       "$ROOT/outputs/cobol-archaeologist.q4_k_m.gguf"

echo "=== Stage 2: Build dataset ==="
$PYTHON src/train.py --stage build

echo "=== Stage 3: LoRA fine-tune ==="
$PYTHON src/train.py --stage train

echo "=== Stage 4: Merge adapter ==="
$PYTHON src/train.py --stage merge

echo "=== Convert to GGUF ==="
cd "$ROOT/llama.cpp"
$PYTHON convert_hf_to_gguf.py "$ROOT/outputs/merged_model" \
    --outfile "$ROOT/outputs/cobol-archaeologist.f16.gguf" --outtype f16

echo "=== Quantize ==="
cd "$ROOT"
llama.cpp/build/bin/llama-quantize \
    outputs/cobol-archaeologist.f16.gguf \
    outputs/cobol-archaeologist.q4_k_m.gguf q4_k_m

echo "=== Load into Ollama ==="
~/.local/bin/ollama create cobol-archaeologist:v1 -f scripts/Modelfile.cobol-archaeologist

echo "=== DONE ==="
