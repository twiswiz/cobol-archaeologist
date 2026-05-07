"""Run the eval harness over a golden set."""
from __future__ import annotations

import json
from pathlib import Path

from ..model.backend import LLMBackend, get_backend
from ..model.parse_output import CardParseError
from ..model.runner import generate_card
from ..schemas import LogicBlock
from .metrics import CardEvalResult, aggregate, evaluate_card


def run_eval(golden_path: Path, backend: LLMBackend | None = None, out_dir: Path | None = None) -> dict:
    backend = backend or get_backend("echo")
    out_dir = out_dir or Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[CardEvalResult] = []
    rows: list[dict] = []
    with golden_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            sample = json.loads(line)
            block = LogicBlock.model_validate(sample["block"])
            reference = sample.get("reference", {})
            try:
                card = generate_card(block, backend)
                res = evaluate_card(card, block, reference)
                rows.append({"id": block.id, "card": card.model_dump(), "eval": res.__dict__})
            except CardParseError as exc:
                res = CardEvalResult(json_valid=False, faithfulness=0.0, rouge_what=0.0, rouge_why=0.0, reg_precision=None)
                rows.append({"id": block.id, "error": str(exc), "eval": res.__dict__})
            results.append(res)

    summary = aggregate(results)
    (out_dir / "eval_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = ["# COBOL-Archaeologist Eval Report", ""]
    for k, v in summary.items():
        md.append(f"- **{k}**: {v}")
    (out_dir / "eval_report.md").write_text("\n".join(md), encoding="utf-8")
    return summary
