from functools import lru_cache

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model():
    """Load the cross-encoder once. Downloads ~80 MB on first use."""
    from sentence_transformers import CrossEncoder

    return CrossEncoder(CROSS_ENCODER_MODEL)


def rerank(query: str, hits: list[dict], top_n: int = 5) -> list[dict]:
    """Rescore candidates by reading each (query, chunk) pair together.

    Embedding similarity compares two independently computed vectors, so
    the model never sees the query and chunk at the same time. A
    cross-encoder does, and can judge whether the chunk actually answers
    the question rather than merely sitting in the same topic area.
    """
    if not hits:
        return []

    pairs = [(query, hit["text"]) for hit in hits]
    scores = _model().predict(pairs)

    for hit, score in zip(hits, scores):
        hit["rerank_score"] = round(float(score), 4)

    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_n]