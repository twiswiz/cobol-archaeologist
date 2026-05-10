# Baseline Comparison

Golden set: `data/generated/golden_eval.jsonl`  ·  N = 5

| Mode | JSON valid | Faithfulness | ROUGE-L (what) | ROUGE-L (why) |
|---|---:|---:|---:|---:|
| echo (heuristic) | 1.00 | 0.000 | 0.040 | 0.060 |
| ollama code-only | 1.00 | 0.893 | 0.247 | 0.126 |
| ollama + static | 0.80 | 0.917 | 0.066 | 0.064 |
| ollama + static + rag | 1.00 | 0.933 | 0.156 | 0.093 |
