"""Run the eval harness over a golden set."""
from __future__ import annotations

import json
from pathlib import Path

from ..model.backend import LLMBackend, get_backend
from ..model.parse_output import CardParseError
from ..model.runner import generate_card
from ..schemas import LogicBlock
from .metrics import CardEvalResult, aggregate, evaluate_card


def _load_index(index_dir: Path | None, offline: bool):
    if not index_dir or not Path(index_dir).exists():
        return None, None
    from ..rag.embed import get_embedder
    from ..rag.index import RegulationIndex

    return RegulationIndex.load(Path(index_dir)), get_embedder(prefer_st=not offline)


def run_eval(
    golden_path: Path,
    backend: LLMBackend | None = None,
    out_dir: Path | None = None,
    include_static: bool = True,
    include_rag: bool = True,
    index_dir: Path | None = None,
    offline: bool = True,
    k: int = 3,
) -> dict:
    backend = backend or get_backend("echo")
    out_dir = out_dir or Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    index, embedder = (None, None)
    if include_rag:
        index, embedder = _load_index(index_dir, offline)

    results: list[CardEvalResult] = []
    rows: list[dict] = []
    with golden_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            sample = json.loads(line)
            block = LogicBlock.model_validate(sample["block"])
            reference = sample.get("reference", {})
            retrieved = []
            if include_rag and index is not None and embedder is not None:
                qtext = " ".join([block.paragraph] + block.vars_read + block.vars_written + block.conditions)
                qv = embedder.encode([qtext])[0]
                retrieved = [h.chunk for h in index.search(qv, k=k)]
            try:
                card = generate_card(
                    block,
                    backend,
                    retrieved=retrieved,
                    include_static=include_static,
                    include_rag=include_rag,
                )
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
    for k_, v in summary.items():
        md.append(f"- **{k_}**: {v}")
    (out_dir / "eval_report.md").write_text("\n".join(md), encoding="utf-8")
    return summary
