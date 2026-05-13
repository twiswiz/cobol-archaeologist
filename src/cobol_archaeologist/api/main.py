"""FastAPI application — COBOL Archaeologist REST API."""
from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cobol_archaeologist.api.loader import (
    DATA_DIR,
    INDEX_DIR,
    get_block_by_id,
    get_blocks,
    get_cards,
    get_cards_by_block_id,
)
from cobol_archaeologist.schemas import BusinessIntentCard, LogicBlock


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm caches on startup
    get_blocks()
    get_cards()
    get_cards_by_block_id()
    yield


app = FastAPI(
    title="COBOL Archaeologist API",
    version="1.0.0",
    description="Explore logic blocks and business intent cards extracted from COBOL programs.",
    lifespan=lifespan,
)

# Allow any origin so the Vercel frontend can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PagedBlocks(BaseModel):
    total: int
    page: int
    size: int
    items: list[LogicBlock]


class PagedCards(BaseModel):
    total: int
    page: int
    size: int
    items: list[BusinessIntentCard]


class StatsResponse(BaseModel):
    total_blocks: int
    total_cards: int
    label_distribution: dict[str, int]
    top_files: list[dict]


class InferRequest(BaseModel):
    backend: str = "ollama"
    model: Optional[str] = None


class AnalyseRequest(BaseModel):
    code: str
    paragraph: str = "FREEFORM"
    backend: str = "ollama"
    model: Optional[str] = None


class RegSearchHit(BaseModel):
    chunk_id: str
    source: str
    section: Optional[str]
    page: Optional[int]
    text: str
    score: float


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"build": "phase9-faiss-fix", "commit": os.getenv("GIT_COMMIT", "unknown")}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@app.get("/stats", response_model=StatsResponse)
def stats():
    blocks = get_blocks()
    cards = get_cards()

    label_dist: dict[str, int] = {}
    file_counts: dict[str, int] = {}

    for b in blocks:
        label = b.weak_label or "unlabeled"
        label_dist[label] = label_dist.get(label, 0) + 1
        fname = b.source_file.split("/")[-1].split("\\")[-1]
        file_counts[fname] = file_counts.get(fname, 0) + 1

    top_files = sorted(
        [{"file": k, "count": v} for k, v in file_counts.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:10]

    return StatsResponse(
        total_blocks=len(blocks),
        total_cards=len(cards),
        label_distribution=label_dist,
        top_files=top_files,
    )


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

@app.get("/blocks", response_model=PagedBlocks)
def list_blocks(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    label: Optional[str] = Query(None),
    q: Optional[str] = Query(None, description="Search paragraph name or code"),
):
    blocks = get_blocks()

    if label:
        blocks = [b for b in blocks if b.weak_label == label]
    if q:
        q_lower = q.lower()
        blocks = [
            b for b in blocks
            if q_lower in b.paragraph.lower() or q_lower in b.code.lower()
        ]

    total = len(blocks)
    start = (page - 1) * size
    items = blocks[start : start + size]

    return PagedBlocks(total=total, page=page, size=size, items=items)


@app.get("/blocks/{block_id}", response_model=LogicBlock)
def get_block(block_id: str):
    block = get_block_by_id(block_id)
    if not block:
        raise HTTPException(status_code=404, detail="Block not found")
    return block


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

@app.get("/cards", response_model=PagedCards)
def list_cards(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
):
    cards = get_cards()
    total = len(cards)
    start = (page - 1) * size
    items = cards[start : start + size]
    return PagedCards(total=total, page=page, size=size, items=items)


@app.get("/cards/{block_id}", response_model=BusinessIntentCard)
def get_card(block_id: str):
    mapping = get_cards_by_block_id()
    card = mapping.get(block_id)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found for this block")
    return card


# ---------------------------------------------------------------------------
# On-demand inference
# ---------------------------------------------------------------------------

@app.post("/infer/{block_id}", response_model=BusinessIntentCard)
def infer_block(block_id: str, req: InferRequest = InferRequest()):
    """Generate a Business Intent Card on demand via the configured backend."""
    block = get_block_by_id(block_id)
    if not block:
        raise HTTPException(status_code=404, detail="Block not found")

    from cobol_archaeologist.model.backend import get_backend
    from cobol_archaeologist.model.runner import generate_card
    from cobol_archaeologist.rag.embed import get_embedder
    from cobol_archaeologist.rag.index import RegulationIndex

    kwargs = {}
    if req.model:
        kwargs["model"] = req.model

    backend = get_backend(req.backend, **kwargs)

    # Try to load the regulation index for RAG context
    retrieved = []
    if INDEX_DIR.exists():
        try:
            embedder = get_embedder(prefer_st=False)
            idx = RegulationIndex.load(INDEX_DIR)
            query_vec = embedder.encode([block.paragraph])[0]
            hits = idx.search(query_vec, k=3)
            retrieved = [h.chunk for h in hits]
        except Exception:
            retrieved = []

    try:
        card = generate_card(block, backend, retrieved)
    except Exception as e:
        import traceback
        raise HTTPException(
            status_code=502,
            detail=f"Backend '{req.backend}' failed: {type(e).__name__}: {e}",
        )
    if card is None:
        raise HTTPException(status_code=500, detail="Model failed to produce a valid card")

    # Persist so the next GET /cards/{block_id} returns the fresh result
    _inferred = DATA_DIR / "reports" / "inferred_cards.jsonl"
    _inferred.parent.mkdir(parents=True, exist_ok=True)
    with _inferred.open("a", encoding="utf-8") as _fh:
        _fh.write(card.model_dump_json() + "\n")

    # Invalidate cards cache so next /cards request reflects this
    get_cards.cache_clear()
    get_cards_by_block_id.cache_clear()

    return card


# ---------------------------------------------------------------------------
# Freeform / ad-hoc analysis
# ---------------------------------------------------------------------------

@app.post("/analyse", response_model=BusinessIntentCard)
def analyse_freeform(req: AnalyseRequest):
    """Generate a Business Intent Card from pasted COBOL code (no block needed)."""
    block = LogicBlock(
        id=f"freeform-{uuid.uuid4().hex[:8]}",
        source_file="freeform",
        paragraph=req.paragraph or "FREEFORM",
        code=req.code,
    )

    from cobol_archaeologist.model.backend import get_backend
    from cobol_archaeologist.model.runner import generate_card
    from cobol_archaeologist.rag.embed import get_embedder
    from cobol_archaeologist.rag.index import RegulationIndex

    kwargs = {}
    if req.model:
        kwargs["model"] = req.model

    backend = get_backend(req.backend, **kwargs)

    retrieved = []
    if INDEX_DIR.exists():
        try:
            embedder = get_embedder(prefer_st=False)
            idx = RegulationIndex.load(INDEX_DIR)
            query_vec = embedder.encode([req.paragraph])[0]
            hits = idx.search(query_vec, k=3)
            retrieved = [h.chunk for h in hits]
        except Exception:
            retrieved = []

    try:
        card = generate_card(block, backend, retrieved)
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Backend '{req.backend}' failed: {type(e).__name__}: {e}",
        )
    if card is None:
        raise HTTPException(status_code=500, detail="Model failed to produce a valid card")

    return card


# ---------------------------------------------------------------------------
# Regulation search
# ---------------------------------------------------------------------------

@app.get("/regulations/search", response_model=list[RegSearchHit])
def search_regulations(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=20),
):
    if not INDEX_DIR.exists():
        raise HTTPException(status_code=503, detail="Regulation index not built yet")

    try:
        from cobol_archaeologist.rag.embed import get_embedder
        from cobol_archaeologist.rag.index import RegulationIndex

        embedder = get_embedder(prefer_st=False)
        idx = RegulationIndex.load(INDEX_DIR)
        query_vec = embedder.encode([q])[0]
        hits = idx.search(query_vec, k=k)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}\n{tb}",
        )

    return [
        RegSearchHit(
            chunk_id=h.chunk.id,
            source=h.chunk.source,
            section=h.chunk.section,
            page=h.chunk.page,
            text=h.chunk.text,
            score=float(h.score),
        )
        for h in hits
    ]
