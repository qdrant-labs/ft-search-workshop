"""Quality metrics for the workshop eval module.

ESCI graded relevance:
    E (Exact)       -> rel 3
    S (Substitute)  -> rel 2
    C (Complement)  -> rel 1
    I (Irrelevant)  -> rel 0

Gain is computed as ``2 ** rel - 1`` for nDCG (standard graded-gain form).
Recall@k uses the binary "relevant if grade in {E, S}" convention common in
ESCI baselines, which gives stable, interpretable numbers for the slide.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

ESCI_REL_MAP: Dict[str, int] = {"E": 3, "S": 2, "C": 1, "I": 0}
# Grades considered "relevant" for recall calculations.
_RECALL_RELEVANT_GRADES = {"E", "S"}


def _grade_to_rel(grade) -> int:
    """ESCI label -> integer relevance, robust to None / float / numeric inputs.

    Production qrels are string letters ("E"/"S"/"C"/"I"); we still tolerate
    None (treat as Irrelevant) and numeric scores (e.g. a future mirror that
    ships floats) so the eval cells never crash mid-loop on a single bad row.
    """
    if grade is None:
        return 0
    if isinstance(grade, str):
        return ESCI_REL_MAP.get(grade.upper(), 0)
    try:
        return int(grade)
    except (TypeError, ValueError):
        return 0


def _gain(rel: int) -> float:
    return float(2 ** rel - 1)


def _dcg(rels: Sequence[int]) -> float:
    return float(sum(_gain(r) / np.log2(i + 2) for i, r in enumerate(rels)))


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    qrels: Mapping[str, str],
    k: int = 10,
) -> float:
    """Graded nDCG@k using the ESCI E/S/C/I gain mapping.

    Args:
        retrieved_ids: Ranked list of product ids from the retriever.
        qrels: Mapping product_id -> ESCI grade ("E"/"S"/"C"/"I"). Products
            absent from the mapping are treated as grade "I" (gain 0).
        k: Cutoff.

    Returns:
        nDCG@k in [0, 1]. Returns 0.0 when there are no relevant items.
    """
    if k <= 0 or not retrieved_ids:
        return 0.0

    top = list(retrieved_ids)[:k]
    rels = [_grade_to_rel(qrels.get(pid, "I")) for pid in top]
    dcg = _dcg(rels)

    ideal_rels = sorted((_grade_to_rel(g) for g in qrels.values()), reverse=True)[:k]
    idcg = _dcg(ideal_rels)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def recall_at_k(
    retrieved_ids: Sequence[str],
    qrels: Mapping[str, str],
    k: int = 10,
) -> float:
    """Recall@k using ESCI {E, S} as the relevant set.

    Returns 0.0 if there are no relevant items in the qrels for this query.
    """
    if k <= 0 or not retrieved_ids:
        return 0.0

    relevant = {pid for pid, g in qrels.items() if str(g).upper() in _RECALL_RELEVANT_GRADES}
    if not relevant:
        return 0.0
    top = set(list(retrieved_ids)[:k])
    return len(top & relevant) / len(relevant)


def bootstrap_ci(
    per_query_scores: Sequence[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 13,
) -> Tuple[float, float, float]:
    """Bootstrap mean and (lo, hi) confidence interval over per-query scores.

    Returns:
        (mean, lo, hi). For an empty input returns ``(0.0, 0.0, 0.0)``.
    """
    scores = np.asarray(list(per_query_scores), dtype=float)
    if scores.size == 0:
        return 0.0, 0.0, 0.0

    rng = np.random.default_rng(seed)
    n = scores.size
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    means = scores[idx].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return float(scores.mean()), lo, hi


__all__ = [
    "ndcg_at_k",
    "recall_at_k",
    "bootstrap_ci",
    "ESCI_REL_MAP",
]
