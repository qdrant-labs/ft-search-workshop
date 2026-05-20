"""Eval helpers for the Qdrant e-commerce search workshop.

Public API re-exported here for convenient use from the lab notebook:

    from eval import (
        ndcg_at_k, recall_at_k, bootstrap_ci,
        compare_results, inspect_sparse_vector,
        explain_metric,
        SpladeEncoder,
    )
"""

from eval.metrics import (
    ndcg_at_k,
    recall_at_k,
    bootstrap_ci,
    ESCI_REL_MAP,
)
from eval.viewer import compare_results, inspect_sparse_vector
from eval.explain import explain_metric
from eval.encoders import SpladeEncoder

__all__ = [
    "ndcg_at_k",
    "recall_at_k",
    "bootstrap_ci",
    "ESCI_REL_MAP",
    "compare_results",
    "inspect_sparse_vector",
    "explain_metric",
    "SpladeEncoder",
]
