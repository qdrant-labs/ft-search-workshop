"""Side-by-side result viewers for the lab notebook.

Two helpers:

* ``compare_results`` renders a per-rank, per-model HTML table with the ESCI
  grade tag (E/S/C/I/--) styled by color so attendees feel the failure modes
  at a glance.
* ``inspect_sparse_vector`` renders the top active terms of a SPLADE-style
  sparse vector against a vocab, so attendees see that sparse is interpretable.

Both ``display()`` HTML inline in Jupyter and return the HTML object for
chaining or saving.
"""

from __future__ import annotations

import html
import string
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from IPython.display import HTML, display

# Stopwords + punctuation filter for the sparse-vector inspection cell.
# SPLADE's MLM head emits high weights on these tokens for almost every
# query (artifacts of the pretraining objective on natural text), and they
# crowd out the meaningful product terms in the top-K display. Filtering
# them is a presentation choice for the workshop -- the underlying sparse
# vector is unchanged.
_PUNCT_CHARS = set(string.punctuation)
_STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "for", "with", "on", "at",
    "in", "is", "was", "be", "by", "this", "that", "it", "as", "from",
    "are", "but", "not", "all", "any", "no",
})


def _is_uninformative_token(token: str) -> bool:
    """Whether to hide a token from sparse-vector inspection display."""
    if not token:
        return True
    if all(ch in _PUNCT_CHARS for ch in token):
        return True
    if token.lower() in _STOPWORDS:
        return True
    return False

# Color palette for ESCI grades. Kept colorblind-aware (green / amber / blue /
# red / neutral) so the rows read at a glance from the back row of a room.
_GRADE_STYLE: Dict[str, Tuple[str, str]] = {
    "E": ("#1b7f3a", "#e6f4ec"),  # exact -- green
    "S": ("#9a6b00", "#fdf3d8"),  # substitute -- amber
    "C": ("#1f5fa8", "#e6eff9"),  # complement -- blue
    "I": ("#a3251f", "#fbe6e4"),  # irrelevant -- red
    "--": ("#555", "#f0f0f0"),    # ungraded
}


def _grade_for(product_id: str, qrels: Optional[Mapping[str, str]]) -> str:
    if not qrels:
        return "--"
    grade = qrels.get(product_id)
    if grade is None:
        return "--"
    grade = str(grade).upper()
    return grade if grade in _GRADE_STYLE else "--"


def _grade_tag(grade: str) -> str:
    fg, bg = _GRADE_STYLE.get(grade, _GRADE_STYLE["--"])
    return (
        f'<span style="display:inline-block;min-width:1.6em;text-align:center;'
        f'padding:1px 6px;border-radius:8px;font-weight:600;font-size:0.85em;'
        f'color:{fg};background:{bg};">{html.escape(grade)}</span>'
    )


def _coerce_result_item(item: Any) -> Tuple[str, str]:
    """Normalize a single result entry to ``(product_id, title)``.

    Accepts either a 2-tuple/2-list ``(pid, title)`` or a Qdrant
    ``ScoredPoint``-like object exposing ``.id`` and ``.payload``.
    """
    if isinstance(item, (tuple, list)) and len(item) == 2:
        pid, title = item
        return str(pid), str(title)
    if hasattr(item, "id") and hasattr(item, "payload"):
        payload = item.payload or {}
        pid = str(payload.get("product_id", item.id))
        # `scripts/setup_collections._payload` stores the title under
        # ``product_title`` (matches the ESCI column name). Prefer that; fall
        # back to ``title`` for forward-compat; finally degrade to the raw
        # point id so the row still renders if both keys are absent.
        title = str(payload.get("product_title", payload.get("title", item.id)))
        return pid, title
    raise TypeError(
        "compare_results expected a (product_id, title) tuple or a ScoredPoint "
        f"with .id/.payload, got {type(item).__name__}"
    )


def compare_results(
    query: str,
    models_results: Mapping[str, Sequence[Tuple[str, str]]],
    esci_qrels: Optional[Mapping[str, str]] = None,
    max_rows: int = 10,
) -> HTML:
    """Render a side-by-side comparison table of retrieval results.

    Args:
        query: The query string (rendered as header).
        models_results: dict ``{model_name: [(product_id, product_title), ...]}``.
            Order of the dict drives column order.
        esci_qrels: Optional mapping ``{product_id: "E"/"S"/"C"/"I"}`` for the
            current query. Products absent from the mapping render as ``--``.
        max_rows: How many ranks to display (default 10).

    Returns:
        An ``IPython.display.HTML`` object. Also ``display()``s it inline.
    """
    model_names = list(models_results.keys())
    n_rows = min(
        max_rows,
        max((len(v) for v in models_results.values()), default=0),
    )

    header_cells = "".join(
        f'<th style="text-align:left;padding:6px 10px;border-bottom:1px solid #ddd;'
        f'font-size:0.9em;background:#fafafa;">{html.escape(name)}</th>'
        for name in model_names
    )

    rows_html: List[str] = []
    for i in range(n_rows):
        cells: List[str] = [
            f'<td style="padding:4px 8px;color:#888;font-variant-numeric:tabular-nums;">'
            f'{i + 1}</td>'
        ]
        for name in model_names:
            results = models_results[name]
            if i < len(results):
                pid, title = _coerce_result_item(results[i])
                grade = _grade_for(pid, esci_qrels)
                cells.append(
                    f'<td style="padding:4px 8px;border-bottom:1px solid #f0f0f0;'
                    f'vertical-align:top;max-width:340px;">'
                    f'{_grade_tag(grade)} '
                    f'<span style="font-size:0.92em;">{html.escape(str(title))}</span>'
                    f'<div style="color:#999;font-size:0.75em;">{html.escape(str(pid))}</div>'
                    f'</td>'
                )
            else:
                cells.append(
                    '<td style="padding:4px 8px;border-bottom:1px solid #f0f0f0;'
                    'color:#bbb;">--</td>'
                )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    html_doc = f"""
    <div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:8px 0 16px;">
      <div style="font-size:0.95em;margin-bottom:6px;">
        <span style="color:#666;">query:</span>
        <span style="font-weight:600;">{html.escape(query)}</span>
      </div>
      <table style="border-collapse:collapse;width:100%;font-size:0.92em;">
        <thead>
          <tr>
            <th style="text-align:left;padding:6px 8px;border-bottom:1px solid #ddd;
                background:#fafafa;width:2.4em;">#</th>
            {header_cells}
          </tr>
        </thead>
        <tbody>
          {''.join(rows_html)}
        </tbody>
      </table>
      <div style="margin-top:6px;font-size:0.78em;color:#666;">
        ESCI grades:
        {_grade_tag('E')} exact &nbsp;
        {_grade_tag('S')} substitute &nbsp;
        {_grade_tag('C')} complement &nbsp;
        {_grade_tag('I')} irrelevant &nbsp;
        {_grade_tag('--')} ungraded
      </div>
    </div>
    """
    obj = HTML(html_doc)
    display(obj)
    return obj


def inspect_sparse_vector(
    sparse_vec: Any,
    vocab: Optional[Mapping[int, str]] = None,
    top_k: int = 15,
    skip_uninformative: bool = True,
) -> HTML:
    """Render the most active terms of a sparse (SPLADE-style) vector.

    Args:
        sparse_vec: Either a ``qdrant_client`` SparseVector-like object with
            ``.indices`` / ``.values``, a dict ``{index: weight}``, or a tuple
            ``(indices, values)``.
        vocab: Mapping ``token_id -> token`` for human-readable display.
        top_k: Number of top-weighted terms to show.
        skip_uninformative: If True (default), hide pure-punctuation tokens
            and common English stopwords from the top-K display. SPLADE's
            MLM head emits high weights on these for almost any query
            (pretraining artifact), and they crowd out the meaningful product
            terms. Set False to see the raw top-K.
    """
    vocab = vocab or {}
    indices, values = _coerce_sparse(sparse_vec)
    if len(indices) == 0:
        obj = HTML("<em>(empty sparse vector)</em>")
        display(obj)
        return obj

    all_pairs = sorted(zip(indices, values), key=lambda x: float(x[1]), reverse=True)
    if skip_uninformative:
        all_pairs = [
            (idx, w) for idx, w in all_pairs
            if not _is_uninformative_token(str(vocab.get(int(idx), "")))
        ]
    pairs = all_pairs[:top_k]
    if not pairs:
        obj = HTML("<em>(all top-weighted tokens were filtered as uninformative)</em>")
        display(obj)
        return obj
    max_w = max(float(v) for _, v in pairs) or 1.0

    rows: List[str] = []
    for idx, weight in pairs:
        token = vocab.get(int(idx), f"#{int(idx)}")
        pct = max(2.0, 100.0 * float(weight) / max_w)
        rows.append(
            "<tr>"
            f'<td style="padding:3px 8px;font-family:ui-monospace,monospace;'
            f'font-size:0.88em;">{html.escape(str(token))}</td>'
            f'<td style="padding:3px 8px;color:#666;font-variant-numeric:tabular-nums;'
            f'font-size:0.85em;">{float(weight):.3f}</td>'
            f'<td style="padding:3px 8px;width:60%;">'
            f'<div style="background:#e6eff9;height:8px;width:{pct:.1f}%;'
            f'border-radius:4px;"></div></td>'
            "</tr>"
        )

    html_doc = f"""
    <div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:8px 0 16px;">
      <div style="font-size:0.85em;color:#666;margin-bottom:4px;">
        sparse vector -- top {len(pairs)} active terms of {len(indices)} total
      </div>
      <table style="border-collapse:collapse;font-size:0.9em;">
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """
    obj = HTML(html_doc)
    display(obj)
    return obj


def _coerce_sparse(sparse_vec: Any) -> Tuple[List[int], List[float]]:
    """Accept several shapes for the sparse vector input."""
    if hasattr(sparse_vec, "indices") and hasattr(sparse_vec, "values"):
        return list(sparse_vec.indices), [float(v) for v in sparse_vec.values]
    if isinstance(sparse_vec, dict):
        return list(sparse_vec.keys()), [float(v) for v in sparse_vec.values()]
    if isinstance(sparse_vec, tuple) and len(sparse_vec) == 2:
        idx, vals = sparse_vec
        return list(idx), [float(v) for v in vals]
    raise TypeError(
        "inspect_sparse_vector expects a SparseVector, dict, or (indices, values) tuple"
    )


def render_query_block(
    query: str,
    results: Sequence[Any],
    expanded: bool = False,
) -> HTML:
    """Render one query's results, optionally inside a ``<details><summary>`` block.

    Args:
        query: The query string (used as the header / summary).
        results: Iterable of Qdrant ``ScoredPoint``-like objects or ``(pid, title)``
            tuples. Items are passed through ``_coerce_result_item``.
        expanded: When ``True`` the result list renders inline. When ``False``
            it's wrapped in a collapsed ``<details>`` block so the caller can
            stack many queries in a long list without overflowing the screen.

    Returns:
        An ``IPython.display.HTML`` object. Also ``display()``s it inline.
    """
    rows: List[str] = []
    for i, item in enumerate(results[:10], start=1):
        pid, title = _coerce_result_item(item)
        rows.append(
            "<tr>"
            f'<td style="padding:2px 8px;color:#888;">{i}</td>'
            f'<td style="padding:2px 8px;">{html.escape(str(title))}</td>'
            "</tr>"
        )
    table = (
        "<table style='border-collapse:collapse;font-size:0.9em;margin:6px 0 14px;'>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )

    if expanded:
        html_doc = (
            "<div style='font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:8px 0 16px;'>"
            f"<div style='font-size:0.95em;margin-bottom:6px;'>"
            f"<span style='color:#666;'>query:</span> "
            f"<span style='font-weight:600;'>{html.escape(query)}</span>"
            "</div>"
            f"{table}"
            "</div>"
        )
    else:
        html_doc = (
            f"<details><summary><b>query:</b> {html.escape(query)}</summary>"
            f"{table}</details>"
        )

    obj = HTML(html_doc)
    display(obj)
    return obj


__all__ = ["compare_results", "inspect_sparse_vector", "render_query_block"]
