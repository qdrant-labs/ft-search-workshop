"""Curate hero queries from Amazon ESCI by finding test queries where the
baseline dense (MiniLM) model demonstrably buries the Exact result.

This script inverts the historical workflow (hand-craft heroes -> hope they
map to ESCI). Instead it picks heroes *from* ESCI's real test queries,
filtered to queries where the baseline buries an Exact-graded product
deeper than rank ``--min-rank``. Result: every hero is provably broken on
baseline, and ``esci_qid`` is set deterministically by construction.

Outputs:

* ``data/hero_queries.json`` -- ``--n-heroes`` records with ``id``, ``text``
  (verbatim ESCI), ``esci_qid``, ``category``, ``failure_mode``,
  ``why_baseline_fails`` (the rank-of-first-Exact evidence).
* ``data/products_10k.jsonl`` -- unique products from the ESCI test split,
  trimmed to ``--products-cap`` most-frequent if larger. Schema:
  ``{"product_id", "product_title", "product_brand", "product_color"}``.

Pipeline:

1. Load ESCI test split (filtered to US locale by default).
2. Build product universe (unique products from the split).
3. Embed queries + products with MiniLM via FastEmbed.
4. Brute-force cosine top-K retrieval per query.
5. Compute rank-of-first-Exact per query.
6. Filter qualifying queries (rank >= ``--min-rank`` or no Exact in top-K).
7. Heuristic categorize by Exact product's metadata (electronics / apparel
   / home / beauty / other).
8. Diversify-pick ``--n-heroes`` across categories.
9. Write outputs.

CLI::

    python scripts/curate_heroes.py --out-dir data/ --n-heroes 10 --top-k 20
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from fastembed import TextEmbedding
from tqdm import tqdm


DEFAULT_ESCI_DATASET = "tasksource/esci"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _normalize_grade(raw_label: str) -> str:
    label = str(raw_label).upper()[:1]
    return label if label in {"E", "S", "C", "I"} else "I"


# Category heuristics. Order matters; first match wins. Coarse buckets
# matching the workshop's four target categories. Operators can hand-edit
# the output JSON if a hero lands in the wrong bucket.
CATEGORY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("electronics", re.compile(
        r"\b(iphone|ipad|samsung|sony|bose|nintendo|playstation|xbox|"
        r"headphone|earbud|laptop|tablet|monitor|computer|tv|television|"
        r"speaker|smartwatch|watch|router|drone|gaming|console|"
        r"keyboard|mouse|webcam|charger|cable|battery|usb|hdmi|"
        r"wireless|bluetooth|wifi|camera|lens|microphone|amplifier|"
        r"adapter|stereo|earphone|airpod|ssd|hdd|gpu|cpu|psu|ram|"
        r"motherboard|drive|memory|storage|power supply|powerstrip|"
        r"hp|dell|lenovo|asus|acer|lg|tcl|rca|evga|nvidia|amd|intel|"
        r"seagate|sandisk|kingston|adata|crucial|hard\s*drive)\b",
        re.IGNORECASE)),
    ("home", re.compile(
        r"\b(pillow|blanket|sheet|comforter|duvet|towel|curtain|rug|"
        r"lamp|chair|table|sofa|couch|desk|shelf|shelves|cabinet|"
        r"drawer|mattress|bedframe|frame|vase|candle|bowl|plate|"
        r"mug|cup|pot|pan|knife|fork|spoon|spatula|cutting board|"
        r"cookware|bedding|throw|quilt|kitchen|bathroom|garage|"
        r"planter|gallon|quart|liter|water heater|appliance|"
        r"hose|broom|vacuum|cleaning|detergent|paper towel)\b",
        re.IGNORECASE)),
    ("beauty", re.compile(
        r"\b(lipstick|mascara|foundation|eyeshadow|blush|concealer|"
        r"primer|moisturizer|serum|cleanser|toner|shampoo|conditioner|"
        r"perfume|cologne|fragrance|polish|tint|matte|gloss|spf|"
        r"sunscreen|exfoliant|skincare|makeup|cosmetic|nail|lipgloss|"
        r"lip balm|hair color|hair dye|eyeliner|essential oil|"
        r"diffuser|difusor|aroma|salon|spa)\b",
        re.IGNORECASE)),
    ("apparel", re.compile(
        # Apparel-only -- toys/games/gifts/books deliberately excluded; those
        # are recommendation-intent queries, not specificity failures, and
        # don't fit the workshop's failure-mode taxonomy.
        r"\b(shirt|t-shirt|tshirt|dress|jeans|pants|trousers|skirt|"
        r"jacket|coat|sweater|hoodie|shoes|shoe|sneakers|boots|sandals|"
        r"hat|cap|gloves|scarf|socks|underwear|bra|swimsuit|leggings|"
        r"blouse|tie|belt|wallet|purse|handbag|tote|backpack|"
        r"clothing|garment|footwear|outerwear|sportswear|fleece)\b",
        re.IGNORECASE)),
]


def _categorize(*texts: str) -> str:
    """Best-effort category. Matches against any of the provided text fields
    (query, product title, brand). First-match-wins. Falls back to 'other'.
    """
    haystack = " ".join(t for t in texts if t)
    for category, pattern in CATEGORY_PATTERNS:
        if pattern.search(haystack):
            return category
    return "other"


_BRAND_PAT = re.compile(
    r"\b(apple|samsung|sony|bose|nike|adidas|lego|nintendo|microsoft|google|"
    r"kitchenaid|hp|dell|lenovo|asus|acer|lg|panasonic|canon|nikon|gopro|"
    r"fitbit|garmin|amazon|kindle|fire|echo|alexa|roku|tcl|rca|jbl|"
    r"logitech|razer|corsair|evga|nvidia|amd|intel|seagate|sandisk|"
    r"crucial|kingston|adata|wd|toshiba|seiko|casio|fossil|levi'?s)\b",
    re.IGNORECASE,
)
_NUMERIC_UNIT_PAT = re.compile(
    r"\b\d+\.?\d*\s*"
    r"(gb|tb|mb|ml|oz|lb|kg|liter|gallon|gal|inch|in|cm|mm|ft|foot|feet|"
    r"w|watt|watts|v|volt|amp|hz|mhz|ghz|mp|mph|year|years|month|months|"
    r"day|days|piece|pieces|pack|count|ct|qt|cup|cups|tsp|tbsp|hp|"
    r"pound|pounds|pcs|ml|cc|btu|cfm|rpm|psi|degree|degrees)\b",
    re.IGNORECASE,
)
_BARE_NUMBER_PAT = re.compile(r"\b\d{1,5}\b")
_SIZE_PAT = re.compile(
    r"\b(size\s+\d+|size\s+(small|medium|large|xs|xl|xxl)|"
    r"\b(xs|sm|md|lg|xl|xxl|xxs)\b)",
    re.IGNORECASE,
)
_COLOR_PAT = re.compile(
    r"\b(red|blue|green|black|white|yellow|orange|purple|pink|gray|grey|"
    r"brown|silver|gold|beige|tan|navy|teal|maroon|cream|ivory)\b",
    re.IGNORECASE,
)
_GENDER_PAT = re.compile(
    r"\b(men'?s|women'?s|girls'?|boys'?|kids|baby|toddler|infant|unisex|mens|womens)\b",
    re.IGNORECASE,
)
_DIMENSION_PAT = re.compile(r"\b\d+\s*(x|by|×)\s*\d+\b", re.IGNORECASE)


def _specificity_score(query: str) -> int:
    """Score how concretely-specific a query is. Higher = better workshop hero.

    The hero narrative needs queries with crisp semantic details that
    baseline dense provably *should* capture but doesn't (a specific
    storage size, a brand+model identifier, a numeric dimension). This
    score prefers those queries over generic ones.
    """
    score = 0
    if _NUMERIC_UNIT_PAT.search(query):
        score += 4  # numeric attribute with unit -- strongest signal
    elif _BARE_NUMBER_PAT.search(query):
        score += 2  # bare number (e.g. 'iphone 11')
    if _DIMENSION_PAT.search(query):
        score += 3
    if _BRAND_PAT.search(query):
        score += 3
    if _SIZE_PAT.search(query):
        score += 2
    if _COLOR_PAT.search(query):
        score += 1
    if _GENDER_PAT.search(query):
        score += 1
    if len(query.split()) >= 4:
        score += 1
    return score


def _has_obvious_typo(query: str) -> bool:
    """Coarse heuristic for typo-ridden queries (3+ consecutive same char,
    or words with very unusual letter distributions). Filters out garbage
    queries from real Amazon search logs without doing a real spell-check.
    """
    if re.search(r"(.)\1{2,}", query):  # 'matttress', 'inpjone' won't catch but 'aaa' will
        return True
    # Words that don't appear to contain vowels and are >= 4 chars are suspect.
    for word in query.split():
        if len(word) >= 4 and not re.search(r"[aeiouAEIOU]", word):
            return True
    return False


def _describe_failure_mode(query: str) -> str:
    """Best-effort failure-mode label from query text. Heuristic only --
    operators should hand-correct on review.
    """
    ql = query.lower()
    if _NUMERIC_UNIT_PAT.search(ql):
        return "specificity (numeric attribute)"
    if _DIMENSION_PAT.search(ql):
        return "specificity (dimension)"
    if _SIZE_PAT.search(ql):
        return "specificity (size)"
    qualifiers = sum(bool(pat.search(ql)) for pat in (_COLOR_PAT, _GENDER_PAT, _SIZE_PAT))
    has_brand = bool(_BRAND_PAT.search(ql))
    has_number = bool(_BARE_NUMBER_PAT.search(ql))
    if qualifiers >= 2:
        return "multi-attribute"
    if has_brand and has_number:
        return "identity (brand + model)"
    if has_brand:
        return "identity (brand)"
    if has_number:
        return "specificity (numeric)"
    return "general"


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:max_len]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _embed_batch(embedder: TextEmbedding, texts: list[str], label: str) -> np.ndarray:
    """Embed ``texts`` and return an (N, D) float32 matrix, with progress."""
    print(f"  embedding {len(texts):,} {label} ...")
    out = []
    for v in tqdm(embedder.embed(texts), total=len(texts)):
        out.append(np.asarray(v, dtype=np.float32))
    return np.stack(out)


def _select_and_write_heroes(
    args: argparse.Namespace,
    candidates: list[dict[str, Any]],
    products_path: Path | None,
) -> int:
    """Filter, diversify-pick, and write hero_queries.json from a candidate
    pool. ``products_path`` is logged for context; reuse-candidates mode
    passes ``None`` and assumes products_10k.jsonl was already written.

    Recomputes ``category``, ``failure_mode``, ``specificity_score``, and
    ``word_count`` on each candidate so the selection is robust to dumps
    written before logic changes.
    """
    for c in candidates:
        c["category"] = _categorize(
            c.get("query", ""),
            c.get("rep_product_title", ""),
            c.get("rep_product_brand", ""),
        )
        c["failure_mode"] = _describe_failure_mode(c.get("query", ""))
        c["specificity_score"] = _specificity_score(c.get("query", ""))
        c["word_count"] = len(c.get("query", "").split())

    before = len(candidates)
    filtered = [
        c for c in candidates
        if c["word_count"] >= args.min_words
        and c["specificity_score"] >= args.min_specificity
        and c["rank_first_exact"] >= args.min_rank
        and c["rank_first_exact"] <= args.max_rank
        and c["n_exact"] >= args.min_exacts
        and not _has_obvious_typo(c["query"])
    ]
    print(f"  filtered candidates: {before:,} -> {len(filtered):,} "
          f"(min_words={args.min_words}, min_specificity={args.min_specificity}, "
          f"rank in [{args.min_rank}, {args.max_rank}], min_exacts={args.min_exacts}, no obvious typos)")

    # --include-qids: operator override. Skip the filter + auto-pick stages
    # entirely and pull the requested qids straight from the full candidate
    # pool (NOT the filtered subset, so the operator can choose anything).
    if args.include_qids:
        requested = [q.strip() for q in args.include_qids.split(",") if q.strip()]
        by_qid = {str(c["query_id"]): c for c in candidates}
        chosen = [by_qid[q] for q in requested if q in by_qid]
        missing = [q for q in requested if q not in by_qid]
        print(f"  --include-qids: {len(chosen)}/{len(requested)} qids found in candidates")
        if missing:
            print(f"  WARN: missing qids (not in candidates dump): {missing}")
        return _write_hero_records(args, chosen, products_path)

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in filtered:
        by_category[c["category"]].append(c)
    for cat_list in by_category.values():
        cat_list.sort(key=lambda c: (
            -c["specificity_score"],
            -c["rank_first_exact"],
            -c["n_exact"],
        ))

    # Greedy diversification: cycle through target categories, taking one
    # at a time, until we hit n_heroes. This gives a more even mix than
    # taking N from each category up-front.
    target_categories = ["electronics", "home", "apparel", "beauty"]
    cat_cursors: dict[str, int] = {cat: 0 for cat in target_categories}
    chosen: list[dict[str, Any]] = []
    chosen_qids: set[str] = set()
    while len(chosen) < args.n_heroes:
        made_progress = False
        for cat in target_categories:
            if len(chosen) >= args.n_heroes:
                break
            cat_list = by_category.get(cat, [])
            cursor = cat_cursors[cat]
            if cursor < len(cat_list):
                pick = cat_list[cursor]
                cat_cursors[cat] = cursor + 1
                if pick["query_id"] in chosen_qids:
                    continue
                chosen.append(pick)
                chosen_qids.add(pick["query_id"])
                made_progress = True
        if not made_progress:
            break

    # Fill any remainder from "other" or remaining filtered candidates.
    if len(chosen) < args.n_heroes:
        remaining = sorted(
            (c for c in filtered if c["query_id"] not in chosen_qids),
            key=lambda c: (
                -c["specificity_score"],
                -c["rank_first_exact"],
                -c["n_exact"],
            ),
        )
        chosen.extend(remaining[:args.n_heroes - len(chosen)])

    return _write_hero_records(args, chosen, products_path)


def _write_hero_records(
    args: argparse.Namespace,
    chosen: list[dict[str, Any]],
    products_path: Path | None,
) -> int:
    """Write hero_queries.json from the chosen candidates + print the summary."""
    args.out_dir.mkdir(parents=True, exist_ok=True)

    hero_records = []
    for c in chosen:
        rank_human = (c["rank_first_exact"] + 1) if c["rank_first_exact"] >= 0 else None
        slug = _slug(c["query"])
        hero_records.append({
            "id": f"{c['category'][:3]}_{slug}"[:60],
            "text": c["query"],
            "esci_qid": c["query_id"],
            "category": c["category"],
            "failure_mode": c["failure_mode"],
            "why_baseline_fails": (
                f"Baseline (MiniLM) ranks the first Exact-grade product at "
                f"position {rank_human} of {c['n_exact']} Exact(s) available"
                if rank_human is not None else
                f"Baseline (MiniLM) returns NO Exact-grade product in top-{args.top_k}; "
                f"{c['n_exact']} Exact(s) exist in the indexed corpus"
            ),
        })

    hero_path = args.out_dir / "hero_queries.json"
    hero_path.write_text(json.dumps(hero_records, indent=2, ensure_ascii=False) + "\n")
    print(f"\n  wrote {hero_path} ({hero_path.stat().st_size:,} bytes)")

    if products_path is None:
        print(f"  (skipped products_10k.jsonl -- reuse-candidates mode; existing file preserved)")

    print("\n=== Selected heroes ===")
    cat_counts = Counter(h["category"] for h in hero_records)
    print(f"category mix: {dict(cat_counts)}")
    fm_counts = Counter(h["failure_mode"] for h in hero_records)
    print(f"failure modes: {dict(fm_counts)}")
    for h in hero_records:
        print(f"  [{h['category']:11s}] [{h['failure_mode']:34s}] "
              f"{h['text']!r}  (qid={h['esci_qid']})")
    print("\nDone.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-heroes", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=20,
        help="Top-K results to consider when computing rank-of-first-Exact.")
    parser.add_argument("--min-rank", type=int, default=3,
        help="Heroes need first-Exact at rank >= this in baseline retrieval "
             "(0-indexed). Use 3 to filter for instructive failures "
             "(Exact pushed below position 3).")
    parser.add_argument("--max-rank", type=int, default=15,
        help="Cap first-Exact rank at this (0-indexed). Queries where the "
             "Exact is never in top-K are excluded — those are likely "
             "impossible queries, not buried-but-recoverable ones.")
    parser.add_argument("--min-words", type=int, default=3,
        help="Require this many tokens minimum in the query text. Filters "
             "out single-word generic queries ('cellphone').")
    parser.add_argument("--min-specificity", type=int, default=2,
        help="Minimum specificity score (see _specificity_score). Filters "
             "out generic queries with no concrete semantic detail.")
    parser.add_argument("--min-exacts", type=int, default=2,
        help="Require this many Exact-grade products available (more makes "
             "the failure-mode story more reliable).")
    parser.add_argument("--dataset", default=DEFAULT_ESCI_DATASET)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--locale", default="us",
        help="Filter ESCI to this product_locale. Empty string disables.")
    parser.add_argument("--products-cap", type=int, default=10_000,
        help="Trim products_10k.jsonl to this many most-frequent products if "
             "the test split has more.")
    parser.add_argument("--candidates-debug", type=Path, default=None,
        help="If set, dump the full candidate list (pre-selection) here for "
             "inspection.")
    parser.add_argument("--reuse-candidates", type=Path, default=None,
        help="If set, skip ESCI load + embedding + retrieval and re-select "
             "heroes from this candidates dump (produced by an earlier run "
             "with --candidates-debug). Use to iterate on selection logic "
             "without paying the embedding cost again.")
    parser.add_argument("--include-qids", default="",
        help="Comma-separated ESCI query_ids to force-include as heroes, "
             "overriding auto-selection. Useful after inspecting the "
             "candidates dump and hand-picking the strongest stories.")
    args = parser.parse_args(argv)

    _ = random.Random(args.seed)  # currently unused; reserved for future tie-breaking

    if args.reuse_candidates:
        candidates_raw = json.loads(args.reuse_candidates.read_text())
        print(f"Loaded {len(candidates_raw):,} candidates from {args.reuse_candidates}")
        # Skip to selection.
        return _select_and_write_heroes(args, candidates_raw, products_path=None)

    # 1. Load ESCI.
    print(f"Loading {args.dataset} ...")
    ds = load_dataset(args.dataset)
    test_split = ds.get("test") if hasattr(ds, "get") else ds
    if test_split is None:
        test_split = next(iter(ds.values()))
    rows = [dict(r) for r in test_split]
    print(f"  loaded {len(rows):,} (query, product) pairs from test split")

    # Locale filter (ESCI is multilingual; default to us).
    if args.locale:
        locale_field = None
        for cand in ("product_locale", "locale", "query_locale"):
            if rows and cand in rows[0]:
                locale_field = cand
                break
        if locale_field:
            before = len(rows)
            rows = [r for r in rows if str(r.get(locale_field, "")).lower() == args.locale.lower()]
            print(f"  filtered to locale={args.locale!r} via field {locale_field!r}: {before:,} -> {len(rows):,}")
        else:
            print(f"  WARN: no locale field found in rows; skipping locale filter")

    if not rows:
        raise SystemExit("ERROR: 0 rows after filtering. Check --locale or dataset structure.")

    # 2. Universes.
    queries_by_qid: dict[str, str] = {}
    for row in rows:
        queries_by_qid.setdefault(row["query_id"], row["query"])

    products_by_pid: dict[str, dict[str, Any]] = {}
    product_freq: Counter[str] = Counter()
    for row in rows:
        pid = row["product_id"]
        product_freq[pid] += 1
        if pid not in products_by_pid:
            products_by_pid[pid] = {
                "product_id": pid,
                "product_title": str(row.get("product_title") or ""),
                "product_brand": str(row.get("product_brand") or ""),
                "product_color": str(row.get("product_color") or ""),
            }
    print(f"  {len(queries_by_qid):,} unique queries; {len(products_by_pid):,} unique products")

    if len(products_by_pid) > args.products_cap:
        top_pids = {pid for pid, _ in product_freq.most_common(args.products_cap)}
        products_by_pid = {pid: products_by_pid[pid] for pid in top_pids}
        print(f"  capped product universe to {len(products_by_pid):,} most-frequent")
    else:
        top_pids = set(products_by_pid)

    # Qrels filtered to indexed corpus.
    qrels: dict[str, dict[str, str]] = {}
    for row in rows:
        pid = row["product_id"]
        if pid not in top_pids:
            continue
        qrels.setdefault(row["query_id"], {})[pid] = _normalize_grade(row["esci_label"])

    # 3. Embed.
    print(f"Embedding with {args.model} ...")
    embedder = TextEmbedding(args.model)

    qids_ordered = list(queries_by_qid.keys())
    query_texts = [queries_by_qid[q] for q in qids_ordered]
    query_vecs = _l2_normalize(_embed_batch(embedder, query_texts, "queries"))

    pids_ordered = list(products_by_pid.keys())
    product_texts = []
    for pid in pids_ordered:
        p = products_by_pid[pid]
        product_texts.append(" ".join(filter(None, [p["product_title"], p["product_brand"], p["product_color"]])))
    product_vecs = _l2_normalize(_embed_batch(embedder, product_texts, "products"))

    # 4. Top-K retrieval.
    print(f"Running top-{args.top_k} retrieval ...")
    batch_size = 128
    all_top_pids: list[list[str]] = []
    for start in tqdm(range(0, len(query_vecs), batch_size)):
        batch = query_vecs[start:start + batch_size]
        scores = batch @ product_vecs.T  # (B, P)
        top_idx = np.argpartition(-scores, min(args.top_k, scores.shape[1] - 1), axis=1)[:, :args.top_k]
        for i, row_idx in enumerate(top_idx):
            row_scores = scores[i, row_idx]
            order = np.argsort(-row_scores)
            ranked_idx = row_idx[order]
            all_top_pids.append([pids_ordered[j] for j in ranked_idx])

    # 5 + 6. Failure filter + qualifying queries.
    candidates: list[dict[str, Any]] = []
    for qid, top_pids_list in zip(qids_ordered, all_top_pids):
        q_qrels = qrels.get(qid, {})
        exact_pids = [pid for pid, g in q_qrels.items() if g == "E"]
        if not exact_pids:
            continue
        rank_first_exact = None
        for rank, pid in enumerate(top_pids_list):
            if q_qrels.get(pid) == "E":
                rank_first_exact = rank
                break
        if rank_first_exact is None or rank_first_exact >= args.min_rank:
            rep = products_by_pid[exact_pids[0]]
            rep_text = " ".join(filter(None, [rep["product_title"], rep["product_brand"]]))
            category = _categorize(rep_text)
            query_text = queries_by_qid[qid]
            failure_mode = _describe_failure_mode(query_text)
            candidates.append({
                "query_id": qid,
                "query": query_text,
                "rank_first_exact": rank_first_exact if rank_first_exact is not None else -1,
                "n_exact": len(exact_pids),
                "category": category,
                "failure_mode": failure_mode,
                "rep_product_title": rep["product_title"],
                "rep_product_brand": rep["product_brand"],
            })

    print(f"  {len(candidates):,} queries qualify (baseline buries Exact at rank >= {args.min_rank} or absent)")

    if args.candidates_debug:
        args.candidates_debug.write_text(json.dumps(candidates, indent=2, ensure_ascii=False))
        print(f"  wrote candidate dump to {args.candidates_debug}")

    # Write products_10k.jsonl now (independent of hero selection).
    args.out_dir.mkdir(parents=True, exist_ok=True)
    products_path = args.out_dir / "products_10k.jsonl"
    with products_path.open("w") as fh:
        for pid in pids_ordered:
            fh.write(json.dumps(products_by_pid[pid], ensure_ascii=False) + "\n")
    print(f"  wrote {products_path} ({len(products_by_pid):,} products, "
          f"{products_path.stat().st_size:,} bytes)")

    return _select_and_write_heroes(args, candidates, products_path=products_path)


if __name__ == "__main__":
    sys.exit(main())
