"""Plain-language explanations for headline metrics.

These are deliberately approximate -- they exist to give a reader without an IR background a
mental model next to the precise number, not to replace it. Calibration
is workshop-tuned, not derived from a user study.
"""

from __future__ import annotations

from typing import Union

Number = Union[int, float]


def explain_metric(metric_name: str, value: Number) -> str:
    """Return a one-line plain-language explanation for a metric value.

    Supported metrics (case-insensitive, separators ignored):
        - ``ndcg@10`` / ``ndcg_at_10`` / ``ndcg10``
        - ``recall@10`` / ``recall_at_10`` / ``recall10``
        - ``ms_per_query`` / ``latency_ms`` / ``p95_ms``

    Unknown metrics return a generic ``"<metric> = <value>"`` string so the
    notebook never crashes on a typo during the live demo.
    """
    key = _normalize(metric_name)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return f"{metric_name} = {value}"

    if key in {"ndcg10", "ndcgat10"}:
        return _ndcg10(v)
    if key in {"recall10", "recallat10"}:
        return _recall10(v)
    if key in {"msperquery", "latencyms", "p95ms", "p50ms", "p99ms"}:
        return _latency(v)
    return f"{metric_name} = {value}"


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _ndcg10(value: float) -> str:
    """nDCG@10 -> mathematically honest one-liner.

    nDCG@10 is **rank-weighted graded relevance on a 0–1 scale**. It gives
    partial credit for Substitute and Complement (graded gain), not just
    Exact, and weights higher ranks more heavily (log-discounted DCG /
    ideal DCG). It is NOT "share of top-10 that are Exact" — earlier
    versions of this translation said that and it didn't survive a sharp
    PM question. We render the number on its native scale with a one-line
    description of what it actually measures.
    """
    return (
        f"ranking quality {value:.2f} / 1.0 "
        "(higher = better products closer to rank 1; rank-weighted graded relevance E/S/C/I)"
    )


def _recall10(value: float) -> str:
    """Recall@10 -> mathematically honest one-liner.

    Recall@10 = average per query of (E/S products in top-10) / (total E/S
    products in qrels). NOT "fraction of queries with any hit" (that would
    be Success@10).
    """
    pct = max(0.0, min(100.0, value * 100.0))
    return (
        f"top-10 contains {pct:.0f}% of known-relevant products per query, on average"
    )


def _latency(value: float) -> str:
    """ms/query -> "X queries per second per worker at p95"."""
    if value <= 0:
        return "Latency unavailable"
    qps = 1000.0 / value
    if qps >= 100:
        qps_str = f"{qps:.0f}"
    elif qps >= 10:
        qps_str = f"{qps:.1f}"
    else:
        qps_str = f"{qps:.2f}"
    return f"{qps_str} queries per second per worker at p95"


__all__ = ["explain_metric"]
