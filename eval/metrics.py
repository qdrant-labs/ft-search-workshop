"""Quality metrics for the workshop eval module.

ESCI graded relevance:
    E (Exact)       -> rel 3
    S (Substitute)  -> rel 2
    C (Complement)  -> rel 1
    I (Irrelevant)  -> rel 0

Gain is computed as ``2 ** rel - 1`` for nDCG (standard graded-gain form).
MRR@k, Recall@k, and Precision@k use the binary "relevant if grade in {E, S}"
convention common in ESCI baselines, which gives stable, interpretable numbers
for the slide.

This module also exposes ``explain_metric`` -- a tiny one-liner translator
for notebooks or docs that need a plain-English metric description.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Union

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


def precision_at_k(
    retrieved_ids: Sequence[str],
    qrels: Mapping[str, str],
    k: int = 10,
) -> float:
    """Precision@k using ESCI {E, S} as the relevant set."""
    if k <= 0 or not retrieved_ids:
        return 0.0

    relevant = {pid for pid, g in qrels.items() if str(g).upper() in _RECALL_RELEVANT_GRADES}
    if not relevant:
        return 0.0
    top = list(retrieved_ids)[:k]
    return sum(1 for pid in top if pid in relevant) / min(k, len(top))


def mrr_at_k(
    retrieved_ids: Sequence[str],
    qrels: Mapping[str, str],
    k: int = 10,
) -> float:
    """MRR@k using ESCI {E, S} as the relevant set."""
    if k <= 0 or not retrieved_ids:
        return 0.0

    relevant = {pid for pid, g in qrels.items() if str(g).upper() in _RECALL_RELEVANT_GRADES}
    if not relevant:
        return 0.0
    for rank, pid in enumerate(list(retrieved_ids)[:k], start=1):
        if pid in relevant:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Plain-language explanations for headline metrics
#
# Calibration is workshop-tuned, not derived from a user study. The point is
# to give a reader without an IR background a mental model next to the
# precise number, not to replace it.
# ---------------------------------------------------------------------------
Number = Union[int, float]


def explain_metric(metric_name: str, value: Number) -> str:
    """Return a one-line plain-language explanation for a metric value.

    Supported metrics (case-insensitive, separators ignored):
        - ``ndcg@10`` / ``ndcg_at_10`` / ``ndcg10``
        - ``mrr@10`` / ``mrr_at_10`` / ``mrr10``
        - ``recall@10`` / ``recall_at_10`` / ``recall10``
        - ``precision@10`` / ``precision_at_10`` / ``precision10``

    Unknown metrics return a generic ``"<metric> = <value>"`` string so the
    notebook never crashes on a typo during the live demo.
    """
    key = "".join(ch for ch in metric_name.lower() if ch.isalnum())
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{metric_name} = {value}"

    if key in {"ndcg10", "ndcgat10"}:
        # nDCG@10 is rank-weighted graded relevance on a 0-1 scale: partial
        # credit for Substitute/Complement, not just Exact, with higher ranks
        # weighted more heavily (log-discounted DCG / ideal DCG). It is NOT
        # "share of top-10 that are Exact" -- earlier translations said that
        # and it didn't survive a sharp PM question.
        return (
            f"ranking quality {v:.2f} / 1.0 "
            "(higher = better products closer to rank 1; rank-weighted graded relevance E/S/C/I)"
        )
    if key in {"recall10", "recallat10"}:
        # Recall@10 = average per query of (E/S products in top-10) / (total
        # E/S products in qrels). NOT "fraction of queries with any hit"
        # (that's Success@10).
        pct = max(0.0, min(100.0, v * 100.0))
        return f"top-10 contains {pct:.0f}% of known-relevant products per query, on average"
    if key in {"mrr10", "mrrat10"}:
        return f"first known-relevant product appears around rank {1 / v:.1f}, on average" if v else "no known-relevant product found in top 10"
    if key in {"precision10", "precisionat10"}:
        pct = max(0.0, min(100.0, v * 100.0))
        return f"{pct:.0f}% of top-10 results are known-relevant, on average"

    return f"{metric_name} = {value}"


__all__ = [
    "ndcg_at_k",
    "mrr_at_k",
    "recall_at_k",
    "precision_at_k",
    "ESCI_REL_MAP",
    "explain_metric",
]
