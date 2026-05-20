"""Build the workshop's evaluation data files from the Amazon ESCI dataset.

This script is the canonical recipe for producing the three eval artifacts the
workshop ships in ``data/``:

* ``qrels_hero.json`` — graded relevance for the 10 hero queries in
  ``data/hero_queries.json``. Schema:
  ``{hero_id: {product_id: grade}}``  where ``grade`` is one of the ESCI
  string labels ``"E"`` (Exact), ``"S"`` (Substitute), ``"C"`` (Complement),
  ``"I"`` (Irrelevant). Downstream consumers (``eval/metrics.py``,
  ``eval/viewer.py``, the notebook's sanity-check cell) all expect strings.
* ``eval_subset_200.json`` — a 200-query stratified sample of the ESCI test
  split, with each query's qrels inlined for self-contained evaluation.
  Schema: ``[{"query_id", "query", "qrels": {product_id: grade}}, ...]``.
  Same string-grade convention as ``qrels_hero.json``.
* ``eval_full_2k.json`` — the full ESCI test set (~2K queries), same schema
  as the subset.

Corpus coverage filtering
-------------------------
All qrels are filtered to product_ids that exist in ``data/products_10k.jsonl``
— the curated 10K-product subset that ``scripts/setup_collections.py`` actually
indexes. Products outside that subset are physically unreachable, so counting
them would corrupt Recall@10 (inflated denominator) and nDCG@10 (ideal DCG
over unreachable relevances). The script emits a per-split coverage report
and **hard-fails** if any hero query has no Exact-grade product in the
indexed corpus — CP1's failure-mode hook depends on each hero having at
least one reachable Exact.

Stratification (200-subset)
---------------------------
Queries are bucketed by token-count tertile (short / medium / long) before
sampling so the subset is not biased toward short or long queries. With
``--n-subset 200`` and 3 tertiles, each tertile contributes ~66 queries.

Idempotency
-----------
The script overwrites all three output files on each run. There is no
"append" mode — re-running with a different ``--seed`` will produce a
different stratified subset, which is by design (for sensitivity checks).

This script does **not** ship the ESCI data itself. You need to install
``datasets`` and pull ``tasksource/esci`` (or your preferred ESCI mirror)
on the machine you run it on. The script will tell you if the import fails.

CLI
---
::

    python scripts/build_eval_data.py --out-dir data/ --n-subset 200 --seed 13

Author: workshop eval-data team. See ``WORKSHOP.md`` for the consuming
notebook path.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

# ESCI label normalization. We emit the string grade ("E"/"S"/"C"/"I")
# directly so that eval/metrics.py (which does ``ESCI_REL_MAP[grade.upper()]``)
# and eval/viewer.py (which colors rows by grade letter) both work without
# any type-juggling. Unknown labels fall back to "I" (irrelevant).
ESCI_GRADES: set[str] = {"E", "S", "C", "I"}


def _normalize_grade(raw_label: str) -> str:
    """Normalize an ESCI label to one of ``"E"``, ``"S"``, ``"C"``, ``"I"``.

    Note: the first-character slice works for both single-letter labels
    (``"E"``) AND full-word labels (``"Exact"``, ``"Substitute"``,
    ``"Complement"``, ``"Irrelevant"``) because each of the four canonical
    ESCI words happens to start with a unique letter. If a future ESCI
    fork adds a new tier (e.g. ``"Excellent"`` collides with ``"Exact"``),
    revisit this — the coincidence will silently miscategorize.
    """
    label = str(raw_label).upper()[:1]
    return label if label in ESCI_GRADES else "I"

# Default ESCI mirror on HuggingFace. Swap this if your team uses a different
# fork — e.g. ``amazon-science/esci-data`` (raw parquet) or an internal mirror.
DEFAULT_ESCI_DATASET = "tasksource/esci"


def _load_esci(dataset_name: str) -> Any:
    """Load ESCI via the ``datasets`` library. Fail loudly if missing."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "ERROR: `datasets` is not installed.\n"
            "    pip install datasets\n"
            f"and ensure you can pull `{dataset_name}` from HuggingFace "
            "(or pass --dataset to point at a different ESCI mirror)."
        ) from exc

    return load_dataset(dataset_name)


def _load_indexed_product_ids(products_path: Path) -> set[str]:
    """Load product_ids from the indexed 10K-product subset.

    Used to constrain qrels: products outside this subset are physically
    unreachable from a search against the ``products`` collection, so they
    would corrupt Recall@10 (inflated denominator) and nDCG@10 (ideal DCG
    over unreachable relevances). All qrels produced by this script are
    filtered against this set.
    """
    if not products_path.exists():
        raise SystemExit(
            f"ERROR: {products_path} not found.\n"
            "This file is the curated 10K-product subset and must exist BEFORE\n"
            "building qrels (see README.md). Otherwise qrels would include\n"
            "products not in the indexed corpus and the headline metrics\n"
            "(Recall@10, nDCG@10) would silently undercount.\n"
            "Build it first, or pass --products to point at your subset."
        )
    ids: set[str] = set()
    with products_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ids.add(str(json.loads(line)["product_id"]))
    if not ids:
        raise SystemExit(f"ERROR: {products_path} is empty.")
    return ids


def _tertile_bucket(token_count: int, t1: int, t2: int) -> int:
    """Return 0 / 1 / 2 for short / medium / long, given the two tertile cuts."""
    if token_count <= t1:
        return 0
    if token_count <= t2:
        return 1
    return 2


def _stratified_sample(
    test_rows: list[dict[str, Any]],
    n_subset: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample ``n_subset`` queries, balanced across query-length tertiles."""
    # One row per (query_id), since stratification is on the query, not pairs.
    unique_queries: dict[str, str] = {}
    for row in test_rows:
        unique_queries.setdefault(row["query_id"], row["query"])

    qids = list(unique_queries.keys())
    token_counts = sorted(len(unique_queries[q].split()) for q in qids)
    if len(token_counts) < 3:
        # Degenerate case — not enough data to stratify; fall back to flat sample.
        rng.shuffle(qids)
        return [{"query_id": q, "query": unique_queries[q]} for q in qids[:n_subset]]

    t1 = token_counts[len(token_counts) // 3]
    t2 = token_counts[(2 * len(token_counts)) // 3]

    buckets: list[list[str]] = [[], [], []]
    for q in qids:
        buckets[_tertile_bucket(len(unique_queries[q].split()), t1, t2)].append(q)

    per_bucket = n_subset // 3
    remainder = n_subset - per_bucket * 3
    chosen: list[str] = []
    for i, bucket in enumerate(buckets):
        take = per_bucket + (1 if i < remainder else 0)
        rng.shuffle(bucket)
        chosen.extend(bucket[:take])

    return [{"query_id": q, "query": unique_queries[q]} for q in chosen]


def _qrels_for_queries(
    rows: list[dict[str, Any]],
    query_ids: set[str],
    indexed_product_ids: set[str],
) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    """Build ``{query_id: {product_id: grade}}`` from raw ESCI rows, filtered
    to the indexed corpus.

    ``grade`` is the ESCI string label ("E"/"S"/"C"/"I"), normalized.
    Unknown labels are coerced to "I" with a single summary warning at the
    end so a silent label-mirror drift doesn't quietly collapse all metrics.

    Returns ``(qrels, stats)`` where ``stats`` reports the number of rows
    skipped as ``unreachable`` (product not in the indexed corpus) and
    ``unknown_label`` (esci_label outside E/S/C/I).
    """
    qrels: dict[str, dict[str, str]] = {}
    unknown_count = 0
    unreachable_count = 0
    for row in rows:
        qid = row["query_id"]
        if qid not in query_ids:
            continue
        pid = row["product_id"]
        if pid not in indexed_product_ids:
            unreachable_count += 1
            continue
        raw = row["esci_label"]
        norm = _normalize_grade(raw)
        if norm == "I" and str(raw).upper()[:1] not in ESCI_GRADES:
            unknown_count += 1
        qrels.setdefault(qid, {})[pid] = norm
    if unknown_count:
        print(
            f"  WARN: {unknown_count} rows had unrecognized esci_label values; "
            f"defaulted to 'I'. Check the dataset's label vocabulary."
        )
    return qrels, {"unreachable": unreachable_count, "unknown_label": unknown_count}


def _attach_qrels(
    sampled: list[dict[str, Any]],
    qrels: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Inline qrels onto each sampled query record."""
    return [{**s, "qrels": qrels.get(s["query_id"], {})} for s in sampled]


def _build_hero_qrels(
    hero_queries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    indexed_product_ids: set[str],
    rows_by_qid: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, dict[str, str]]:
    """Map each hero query to its ESCI qrels, filtered to the indexed corpus.

    Resolution order per hero:

    1. **Explicit override** — if the hero record has a non-null
       ``esci_qid`` field, use it directly. This is the recommended path:
       hard-curating the mapping eliminates the risk of a generic-vs-specific
       text collision (e.g. "red running shoes" vs "women's red running
       shoes size 8"). Set this in ``data/hero_queries.json``.
    2. **Exact text match** — case-insensitive, whitespace-stripped.
    3. **Substring containment fallback** — bidirectional. **EMITS A
       WARNING** because this is precisely the case where a specific hero
       can inherit qrels from a less-specific ESCI query and silently
       corrupt the demo. Curate ``esci_qid`` to bypass.

    After resolution, each hero's qrels are filtered to product_ids in
    ``indexed_product_ids``. The function then asserts every hero has at
    least one Exact-grade product in the filtered set — CP1's failure-mode
    hook depends on rendering at least one Exact row per hero. Fails loudly
    if any hero is empty so the VM build never publishes corrupt qrels.

    ``rows_by_qid`` is an optional pre-built index ``query_id -> [rows]`` to
    avoid an O(heroes × pairs) scan over ``rows``.
    """
    by_query_text: dict[str, str] = {}  # query_text -> query_id (first seen)
    for row in rows:
        by_query_text.setdefault(row["query"].lower().strip(), row["query_id"])

    if rows_by_qid is None:
        rows_by_qid = {}
        for row in rows:
            rows_by_qid.setdefault(row["query_id"], []).append(row)

    hero_qrels: dict[str, dict[str, str]] = {}
    fallback_count = 0
    for hero in hero_queries:
        hero_id = hero["id"]
        text = hero["text"].lower().strip()

        # 1. Explicit override.
        explicit_qid = hero.get("esci_qid")
        if explicit_qid is not None:
            qid = explicit_qid
            # ESCI's query_id is int in tasksource/esci; JSON preserves the
            # type. If the hero_queries.json was written with a string qid
            # but ESCI gave us ints, coerce to int so the lookup matches.
            if qid not in rows_by_qid and isinstance(qid, str) and qid.isdigit():
                qid = int(qid)
            print(f"  ok:   hero {hero_id!r} -> esci_qid {qid!r} (explicit)")
        else:
            # 2. Exact match.
            qid = by_query_text.get(text)
            if qid is None:
                # 3. Substring fallback — loud.
                for candidate_text, candidate_qid in by_query_text.items():
                    if text in candidate_text or candidate_text in text:
                        qid = candidate_qid
                        fallback_count += 1
                        print(
                            f"  WARN: hero {hero_id!r} matched via substring "
                            f"fallback to ESCI qid {qid!r} (text={candidate_text!r}). "
                            f"Hard-curate `esci_qid` in hero_queries.json to bypass."
                        )
                        break

        if qid is None:
            print(f"  WARN: no ESCI match for hero {hero_id!r} ({hero['text']!r})")
            hero_qrels[hero_id] = {}
            continue
        hero_qrels[hero_id] = {
            row["product_id"]: _normalize_grade(row["esci_label"])
            for row in rows_by_qid.get(qid, [])
            if row["product_id"] in indexed_product_ids
        }
    if fallback_count:
        print(
            f"  WARN: {fallback_count} hero(es) used the substring fallback. "
            f"Verify each before pilot."
        )

    # Coverage assertion: every hero must have >=1 Exact-grade product in
    # the indexed corpus, otherwise CP1's failure-mode hook collapses (no
    # Exacts to surface, grade tags all read S/C/I or '--'). Fail loudly so
    # the VM build never publishes empty hero qrels.
    missing_exact: list[str] = []
    for hero_id, hq in hero_qrels.items():
        if not any(grade == "E" for grade in hq.values()):
            missing_exact.append(hero_id)
    if missing_exact:
        raise SystemExit(
            "ERROR: the following hero queries have no Exact-grade products "
            "in the indexed corpus after coverage filtering:\n"
            f"  {missing_exact}\n"
            "CP1's failure-mode hook depends on each hero having >=1 Exact "
            "in products_10k.jsonl. Either:\n"
            "  (a) Set explicit `esci_qid` in hero_queries.json to an ESCI "
            "query whose Exact-grade products ARE in products_10k.jsonl, or\n"
            "  (b) Expand products_10k.jsonl to include the missing products."
        )
    return hero_qrels


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"  wrote {path} ({path.stat().st_size:,} bytes)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-subset", type=int, default=200)
    parser.add_argument("--n-full", type=int, default=2000,
        help="Cap the 'full' eval set to this many queries (random sample "
             "from the reachable pool). Default 2000 matches the workshop's "
             "advertised '2K test queries' framing and keeps the live wrap "
             "eval under ~6 min on the VM. Set to 0 to disable the cap.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dataset", type=str, default=DEFAULT_ESCI_DATASET)
    parser.add_argument(
        "--hero-queries",
        type=Path,
        default=Path("data/hero_queries.json"),
        help="Path to the workshop's hero queries JSON.",
    )
    parser.add_argument(
        "--products",
        type=Path,
        default=Path("data/products_10k.jsonl"),
        help="Indexed 10K-product subset (JSONL). All qrels are filtered to "
             "product_ids in this file so unreachable products don't corrupt "
             "Recall@10 / nDCG@10.",
    )
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    indexed_ids = _load_indexed_product_ids(args.products)
    print(f"Loaded {len(indexed_ids):,} indexed product_ids from {args.products}")

    print(f"Loading {args.dataset} ...")
    ds = _load_esci(args.dataset)
    # ESCI mirrors vary in split naming; try common conventions.
    # `load_dataset` may return a DatasetDict (typical) or a bare Dataset
    # (some forks / custom mirrors that already pin a split). Handle both.
    if hasattr(ds, "get"):
        test_split = ds.get("test") or ds.get("validation") or next(iter(ds.values()))
    else:
        test_split = ds
    rows: list[dict[str, Any]] = [dict(r) for r in test_split]
    print(f"  loaded {len(rows):,} (query, product) pairs from test split")

    # Build a single ``query_id -> [rows]`` index once and reuse everywhere.
    # The previous implementation rebuilt the qid->query map via a
    # ``next(... for r in rows ...)`` per qid, which was O(|qids| × |rows|)
    # — at 2K queries × ~200K pairs that's ~4e8 scans and looks hung to a
    # VM-build operator.
    rows_by_qid: dict[str, list[dict[str, Any]]] = {}
    qid_to_query: dict[str, str] = {}
    for row in rows:
        qid = row["query_id"]
        rows_by_qid.setdefault(qid, []).append(row)
        qid_to_query.setdefault(qid, row["query"])

    # Pre-filter the pool to queries that have >=1 reachable Exact in the
    # indexed corpus. Otherwise the stratified subset is mostly queries
    # with empty qrels (since ESCI's test split has hundreds of thousands
    # of products and we index only ~10K), and the live metric reveal in
    # CP2/CP3 becomes noisy on an effective sample much smaller than the
    # advertised size.
    reachable_qids: set[Any] = set()
    for qid, rows_for_q in rows_by_qid.items():
        if any(
            row["product_id"] in indexed_ids and _normalize_grade(row["esci_label"]) == "E"
            for row in rows_for_q
        ):
            reachable_qids.add(qid)
    print(
        f"  [coverage] {len(reachable_qids):,}/{len(qid_to_query):,} test queries have "
        f">=1 Exact-grade product in the indexed corpus; pool restricted to those."
    )

    if not reachable_qids:
        raise SystemExit(
            "ERROR: no test queries have any Exact-grade product in the indexed "
            "corpus. products_10k.jsonl is probably too small or doesn't overlap "
            "ESCI's test products. Expand products_10k.jsonl."
        )

    # Cap to --n-full random queries (default 2K) so the live wrap eval
    # finishes within the workshop's time budget.
    if args.n_full and len(reachable_qids) > args.n_full:
        reachable_qids_list = sorted(reachable_qids, key=lambda x: str(x))
        rng.shuffle(reachable_qids_list)
        reachable_qids = set(reachable_qids_list[:args.n_full])
        print(f"  capped full eval pool to {len(reachable_qids):,} (random sample, seed={args.seed})")

    reachable_rows = [r for r in rows if r["query_id"] in reachable_qids]

    # 1. Full test set (queries with reachable qrels) with inlined qrels.
    print("Building eval_full_2k.json ...")
    full_queries = [
        {"query_id": qid, "query": qid_to_query[qid]}
        for qid in sorted(reachable_qids, key=lambda x: (str(x)))
    ]
    full_qrels, full_stats = _qrels_for_queries(reachable_rows, reachable_qids, indexed_ids)
    print(
        f"  [coverage] full set: {len(full_qrels):,} queries with qrels "
        f"(skipped {full_stats['unreachable']:,} unreachable pairs from filtering)"
    )
    _write_json(args.out_dir / "eval_full_2k.json", _attach_qrels(full_queries, full_qrels))

    # 2. Stratified subset.
    print(f"Building eval_subset_{args.n_subset}.json ...")
    subset = _stratified_sample(reachable_rows, args.n_subset, rng)
    subset_qids = {s["query_id"] for s in subset}
    subset_qrels, subset_stats = _qrels_for_queries(reachable_rows, subset_qids, indexed_ids)
    n_subset_reachable = sum(1 for v in subset_qrels.values() if v)
    print(
        f"  [coverage] subset {args.n_subset}: {n_subset_reachable}/{len(subset)} queries "
        f"have >=1 reachable product (should be 100% after the pre-filter)"
    )
    _write_json(
        args.out_dir / f"eval_subset_{args.n_subset}.json",
        _attach_qrels(subset, subset_qrels),
    )

    # 3. Hero qrels (with hard-fail coverage assertion).
    print("Building qrels_hero.json ...")
    hero_queries = json.loads(args.hero_queries.read_text())
    hero_qrels = _build_hero_qrels(
        hero_queries, rows, indexed_ids, rows_by_qid=rows_by_qid
    )
    for hero_id, hq in hero_qrels.items():
        n_exact = sum(1 for g in hq.values() if g == "E")
        n_total = len(hq)
        print(f"  [coverage] hero {hero_id!r}: {n_total} qrels ({n_exact} Exact)")
    _write_json(args.out_dir / "qrels_hero.json", hero_qrels)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
