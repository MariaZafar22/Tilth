"""Small RAG layer: chunk markdown guides, embed with Gemini, retrieve by cosine
similarity. Falls back to keyword scoring if embeddings are unavailable."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

KB_DIR = Path(__file__).parent / "knowledge_base"
EMBED_MODEL = "gemini-embedding-001"

_client = None
_chunks: list[dict] | None = None
_vectors: np.ndarray | None = None


def configure(client) -> None:
    """Give the retriever a Gemini client (call once at start-up)."""
    global _client, _vectors
    _client = client
    _vectors = None


def load_chunks() -> list[dict]:
    """Split every guide into chunks, one per '## ' section."""
    global _chunks
    if _chunks is not None:
        return _chunks
    chunks = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = path.stem.replace("_", " ").title()
        for block in re.split(r"\n(?=## )", text):
            block = block.strip()
            if len(block) < 40:
                continue
            heading = block.splitlines()[0].lstrip("# ").strip()
            chunks.append({"source": path.name, "section": f"{title} - {heading}", "text": block})
    _chunks = chunks
    return chunks


def _embed(texts: list[str]) -> np.ndarray:
    result = _client.models.embed_content(model=EMBED_MODEL, contents=texts)
    return np.array([e.values for e in result.embeddings], dtype=np.float32)


def _keyword_scores(query: str, chunks: list[dict]) -> np.ndarray:
    words = set(re.findall(r"[a-z]{3,}", query.lower()))
    return np.array([
        sum(w in c["text"].lower() for w in words) for c in chunks
    ], dtype=np.float32)


def search(query: str, top_k: int = 3) -> list[dict]:
    global _vectors
    chunks = load_chunks()
    scores = None
    if _client is not None:
        try:
            if _vectors is None:
                _vectors = _embed([c["text"] for c in chunks])
            q = _embed([query])[0]
            norms = np.linalg.norm(_vectors, axis=1) * np.linalg.norm(q) + 1e-9
            scores = (_vectors @ q) / norms
        except Exception:
            scores = None
    if scores is None:
        scores = _keyword_scores(query, chunks)
    best = np.argsort(scores)[::-1][:top_k]
    return [
        {"source": chunks[i]["source"], "section": chunks[i]["section"],
         "score": round(float(scores[i]), 3), "text": chunks[i]["text"]}
        for i in best
    ]
