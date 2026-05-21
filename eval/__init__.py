"""Eval helpers for the Qdrant e-commerce search workshop.

Public API re-exported here for convenient use from the lab notebook:

    from eval import (
        ndcg_at_k, mrr_at_k, recall_at_k, precision_at_k, bootstrap_ci,
        compare_results, inspect_sparse_vector,
        explain_metric,
    )
"""

from eval.metrics import (
    ndcg_at_k,
    mrr_at_k,
    recall_at_k,
    precision_at_k,
    bootstrap_ci,
    ESCI_REL_MAP,
    explain_metric,
)
from eval.viewer import compare_results, inspect_sparse_vector, render_query_block

__all__ = [
    "ndcg_at_k",
    "mrr_at_k",
    "recall_at_k",
    "precision_at_k",
    "bootstrap_ci",
    "ESCI_REL_MAP",
    "compare_results",
    "inspect_sparse_vector",
    "render_query_block",
    "explain_metric",
]
