from app.keyword import keyword_search
from app.rerank import rerank
from app.store import search as vector_search

RRF_K = 60          # damping constant; higher flattens rank differences


def _chunk_key(hit: dict) -> tuple:
    """Identify a chunk across retrievers, which return different fields."""
    return (hit["source"], hit["page"], hit["text"][:80])


def hybrid_search(
    query: str,
    n_results: int = 5,
    candidates: int = 20,
    use_rerank: bool = True,
) -> list[dict]:
    """Fuse vector and keyword results, then optionally rerank.

    Reciprocal Rank Fusion scores by rank position rather than raw score,
    because the two scales are not comparable (cosine similarity ~0-1 vs
    unbounded BM25). A chunk both retrievers rank highly beats one that
    only a single retriever favours.

    The cross-encoder then rescores the fused candidates by reading each
    (query, chunk) pair together — accurate, but too slow to run over the
    whole store, hence retrieve-then-rerank.
    """
    vector_hits = vector_search(query, candidates)
    keyword_hits = keyword_search(query, candidates)

    fused: dict[tuple, dict] = {}

    def add(hits: list[dict], retriever: str) -> None:
        """Add one retriever's ranked hits, contributing at most once per chunk."""
        seen: set[tuple] = set()
        for rank, hit in enumerate(hits, start=1):
            key = _chunk_key(hit)
            if key in seen:          # duplicate content from one retriever
                continue
            seen.add(key)
            entry = fused.setdefault(
                key, {**hit, "rrf_score": 0.0, "found_by": []}
            )
            entry["rrf_score"] += 1 / (RRF_K + rank)
            entry["found_by"].append(retriever)
            entry[f"{retriever}_rank"] = rank

    add(vector_hits, "vector")
    add(keyword_hits, "keyword")

    ranked = sorted(fused.values(), key=lambda h: h["rrf_score"], reverse=True)

    for hit in ranked:
        hit["rrf_score"] = round(hit["rrf_score"], 5)

    if use_rerank:
        return rerank(query, ranked[:candidates], n_results)
    return ranked[:n_results]