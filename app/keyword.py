import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from app.store import get_all_chunks


def _tokenize(text: str) -> list[str]:
    """Lowercase, then split on anything that isn't a letter or digit."""
    return re.findall(r"[a-z0-9]+", text.lower())


@lru_cache(maxsize=1)
def _index() -> tuple[BM25Okapi, list[dict]]:
    """Build a BM25 index over every stored chunk.

    Cached, because rebuilding per query would be wasteful. Call
    reset_index() after indexing new documents.
    """
    chunks = get_all_chunks()
    if not chunks:
        raise RuntimeError("No chunks in the store. Index a document first.")
    corpus = [_tokenize(c["text"]) for c in chunks]
    return BM25Okapi(corpus), chunks


def reset_index() -> None:
    """Drop the cached index so the next search rebuilds it."""
    _index.cache_clear()


def keyword_search(query: str, n_results: int = 20) -> list[dict]:
    """Return the highest-scoring chunks by BM25."""
    bm25, chunks = _index()
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(
        zip(scores, chunks), key=lambda pair: pair[0], reverse=True
    )

    hits = []
    for score, chunk in ranked[:n_results]:
        if score <= 0:
            break          # no term overlap at all
        hits.append(
            {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "strategy": chunk["strategy"],
                "bm25_score": round(float(score), 4),
                "citation": f"{chunk['source']}, p.{chunk['page']}",
            }
        )
    return hits