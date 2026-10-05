"""Precision@5 evaluation runner for the Week 9 RAG retriever.

Loads eval/labelled.json, runs hybrid_search for each labelled question,
computes precision@5 against the ground-truth pages, and writes
eval/results.json.

Run from the project root:
    python -m eval.run_eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.hybrid import hybrid_search  # noqa: E402

K = 5
EVAL_DIR = Path(__file__).resolve().parent
LABELLED_PATH = EVAL_DIR / "labelled.json"
RESULTS_PATH = EVAL_DIR / "results.json"


def precision_at_k(hits: list[dict], relevant_pages: list[int], k: int = K) -> float:
    """Precision@k = (# relevant in top-k) / k.

    Denominator is always k, matching README.md — even if retrieval returns
    fewer than k results.
    """
    relevant_set = {int(p) for p in relevant_pages}
    top_k = hits[:k]
    relevant = sum(1 for h in top_k if int(h.get("page", -1)) in relevant_set)
    return relevant / k


def main() -> None:
    items = json.loads(LABELLED_PATH.read_text(encoding="utf-8-sig"))

    per_question: list[dict] = []
    for item in items:
        q = item["q"]
        relevant_pages = item["relevant_pages"]

        hits = hybrid_search(q, n_results=K)
        score = precision_at_k(hits, relevant_pages, K)

        per_question.append({
            "q": q,
            "relevant_pages": relevant_pages,
            "retrieved_pages": [h.get("page") for h in hits[:K]],
            "retrieved_citations": [h.get("citation") for h in hits[:K]],
            "precision@5": round(score, 4),
        })

    mean_precision = (
        sum(r["precision@5"] for r in per_question) / len(per_question)
        if per_question else 0.0
    )

    output = {
        "k": K,
        "n_questions": len(per_question),
        "mean_precision@5": round(mean_precision, 4),
        "per_question": per_question,
    }

    RESULTS_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()