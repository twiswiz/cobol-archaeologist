"""Command-line interface for the COBOL-Archaeologist pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .ingest.discover import discover
from .labels.weak import label_blocks
from .model.backend import get_backend
from .model.runner import generate_card
from .rag.chunker import chunk_pages
from .rag.embed import embed_chunks, get_embedder
from .rag.index import RegulationIndex
from .rag.pdf_loader import load_pdf
from .schemas import LogicBlock, RegulationChunk
from .segmenter.segment import segment_file


def _write_jsonl(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.model_dump_json() + "\n")


def cmd_ingest(args: argparse.Namespace) -> None:
    n = discover(Path(args.root), Path(args.out))
    print(f"Discovered {n} COBOL files -> {args.out}")


def cmd_segment(args: argparse.Namespace) -> None:
    root = Path(args.root)
    blocks: list[LogicBlock] = []
    paths = [root] if root.is_file() else list(root.rglob("*"))
    for p in paths:
        if p.is_file() and p.suffix.lower() in {".cbl", ".cob", ".cobol"}:
            try:
                blocks.extend(segment_file(p))
            except Exception as exc:  # pragma: no cover
                print(f"skip {p}: {exc}")
    if args.label:
        blocks = label_blocks(blocks)
    _write_jsonl(Path(args.out), blocks)
    print(f"Wrote {len(blocks)} logic blocks -> {args.out}")


def cmd_index_regulations(args: argparse.Namespace) -> None:
    chunks: list[RegulationChunk] = []
    for pdf in args.pdf:
        pages = load_pdf(Path(pdf))
        chunks.extend(chunk_pages(pages))
    embedder = get_embedder(prefer_st=not args.offline)
    vectors = embed_chunks(chunks, embedder=embedder)
    index = RegulationIndex(vectors, chunks)
    index.save(Path(args.out_dir), vectors)
    print(f"Indexed {len(chunks)} chunks -> {args.out_dir}")


def cmd_search_regulations(args: argparse.Namespace) -> None:
    index = RegulationIndex.load(Path(args.index_dir))
    embedder = get_embedder(prefer_st=not args.offline)
    qv = embedder.encode([args.query])[0]
    hits = index.search(qv, k=args.k)
    for h in hits:
        print(f"[{h.score:.3f}] {h.chunk.id} {h.chunk.source} p{h.chunk.page}")
        print(h.chunk.text[:240].replace("\n", " "))
        print("-" * 60)


def cmd_infer(args: argparse.Namespace) -> None:
    backend = get_backend(args.backend, **(json.loads(args.backend_args) if args.backend_args else {}))
    index = None
    embedder = None
    if args.index_dir:
        index = RegulationIndex.load(Path(args.index_dir))
        embedder = get_embedder(prefer_st=not args.offline)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with Path(args.logic_blocks).open("r", encoding="utf-8") as fh, out_path.open("w", encoding="utf-8") as out:
        for line in fh:
            if not line.strip():
                continue
            if args.limit is not None and n >= args.limit:
                break
            block = LogicBlock.model_validate(json.loads(line))
            retrieved = []
            if index is not None and embedder is not None:
                qtext = " ".join([block.paragraph] + block.vars_read + block.vars_written + block.conditions)
                qv = embedder.encode([qtext])[0]
                retrieved = [h.chunk for h in index.search(qv, k=args.k)]
            try:
                card = generate_card(block, backend, retrieved=retrieved)
                out.write(card.model_dump_json() + "\n")
                n += 1
            except Exception as exc:
                out.write(json.dumps({"logic_block_id": block.id, "error": str(exc)}) + "\n")
    print(f"Wrote {n} cards -> {args.out}")


def cmd_eval(args: argparse.Namespace) -> None:
    from .eval.run import run_eval

    backend = get_backend(args.backend, **(json.loads(args.backend_args) if args.backend_args else {}))
    summary = run_eval(Path(args.golden), backend=backend, out_dir=Path(args.out_dir))
    print(json.dumps(summary, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("cobol-archaeologist")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="Discover COBOL sources under a root.")
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("segment", help="Parse and segment files into logic blocks.")
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--label", action="store_true")
    s.set_defaults(func=cmd_segment)

    s = sub.add_parser("index-regulations", help="Build a FAISS regulation index.")
    s.add_argument("--pdf", nargs="+", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--offline", action="store_true", help="Use the hashing embedder.")
    s.set_defaults(func=cmd_index_regulations)

    s = sub.add_parser("search-regulations", help="Query a regulation index.")
    s.add_argument("--index-dir", required=True)
    s.add_argument("--query", required=True)
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--offline", action="store_true")
    s.set_defaults(func=cmd_search_regulations)

    s = sub.add_parser("infer", help="Generate Business Intent Cards.")
    s.add_argument("--logic-blocks", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--backend", default="echo", choices=["echo", "hf", "openai", "ollama"])
    s.add_argument("--backend-args", default=None, help="JSON kwargs for the backend.")
    s.add_argument("--index-dir", default=None)
    s.add_argument("-k", type=int, default=3)
    s.add_argument("--offline", action="store_true")
    s.add_argument("--limit", type=int, default=None, help="Process only the first N blocks.")
    s.set_defaults(func=cmd_infer)

    s = sub.add_parser("eval", help="Evaluate against a golden JSONL file.")
    s.add_argument("--golden", required=True)
    s.add_argument("--out-dir", default="reports")
    s.add_argument("--backend", default="echo", choices=["echo", "hf", "openai", "ollama"])
    s.add_argument("--backend-args", default=None)
    s.set_defaults(func=cmd_eval)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
