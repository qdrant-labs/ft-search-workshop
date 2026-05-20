"""Verify fine-tuned SPLADE actually wins on the curated hero set.

The workshop's central claim is that fine-tuned SPLADE (``thierrydamiba/
splade-ecommerce-esci``) beats baseline dense (MiniLM) on the hero queries
that baseline demonstrably fails on. ``scripts/curate_heroes.py`` picks
heroes where baseline buries the Exact -- it does NOT verify that SPLADE
*recovers* the Exact.

This script closes that loop:

1. Load ``data/hero_queries.json`` and ``data/qrels_hero.json``.
2. Load ``data/products_10k.jsonl`` (the indexed corpus).
3. Encode all 10K product texts + 10 hero query texts with the
   fine-tuned SPLADE checkpoint (via ``eval.encoders.SpladeEncoder``).
4. For each hero, compute sparse-dot scores against all products, take
   top-K, and report:
   - rank_first_exact (SPLADE)
   - whether SPLADE recovers the Exact into the top-3 (the visual demo
     threshold)
5. Print a verdict table: heroes where SPLADE WINS (top-3) vs where
   SPLADE still loses (Exact still buried or absent).

The encoding step is the slow part: ~10K DistilBERT forward passes on
CPU. Expect 10-30 min on an M-series Mac. Run once during pilot prep;
the result lets you swap any "SPLADE-also-fails" hero before the room
sees it.

CLI::

    python scripts/verify_splade_wins.py [--top-k 20] [--win-rank 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


def _load_products(path: Path) -> tuple[list[str], list[str]]:
    """Return ``(product_ids, texts)``. Text format must match what was used
    by ``curate_heroes.py`` so SPLADE scores are comparable: title + brand
    + color."""
    ids: list[str] = []
    texts: list[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids.append(str(row["product_id"]))
            texts.append(" ".join(filter(None, [
                row.get("product_title", ""),
                row.get("product_brand", ""),
                row.get("product_color", ""),
            ])))
    return ids, texts


def _encode_batches(encoder: Any, texts: list[str], batch_size: int, label: str) -> list[dict[int, float]]:
    """Encode ``texts`` in batches via SpladeEncoder, return list of
    ``{token_id: weight}`` dicts (sparse vector per row)."""
    out: list[dict[int, float]] = []
    pbar = tqdm(range(0, len(texts), batch_size), desc=f"encoding {label}")
    for start in pbar:
        batch = texts[start:start + batch_size]
        pairs = encoder.encode(batch)  # List[Tuple[List[int], List[float]]]
        for token_ids, weights in pairs:
            out.append(dict(zip(token_ids, weights)))
    return out


def _sparse_dot(q: dict[int, float], p: dict[int, float]) -> float:
    """Dot product of two sparse vectors stored as ``{token: weight}`` dicts."""
    # Iterate the smaller dict for speed.
    if len(q) < len(p):
        return sum(w * p.get(t, 0.0) for t, w in q.items())
    return sum(w * q.get(t, 0.0) for t, w in p.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--win-rank", type=int, default=3,
        help="Heroes are 'won' if SPLADE places the first Exact at rank <= "
             "this (0-indexed: 3 means top-3 positions). Default 3 matches "
             "the workshop's visual demo threshold.")
    parser.add_argument("--min-wins", type=int, default=7,
        help="Exit nonzero if fewer than this many heroes are SPLADE wins. "
             "Default 7/10 matches the current pilot-accepted contract (the "
             "remaining 3 are intentional 'honest losses' kept in the set for "
             "aggregate-metric realism). Raise this only after re-curating "
             "the hero set; lowering it silently masks regression.")
    parser.add_argument("--batch-size", type=int, default=16,
        help="Encoding batch size. Larger = faster on M-series but more RAM.")
    parser.add_argument("--model", default="thierrydamiba/splade-ecommerce-esci")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--report-out", type=Path, default=None,
        help="If set, also dump a JSON report of per-hero SPLADE results here.")
    args = parser.parse_args(argv)

    heroes_path = args.data_dir / "hero_queries.json"
    qrels_path = args.data_dir / "qrels_hero.json"
    products_path = args.data_dir / "products_10k.jsonl"

    heroes = json.loads(heroes_path.read_text())
    qrels = json.loads(qrels_path.read_text())
    product_ids, product_texts = _load_products(products_path)

    print(f"Loaded {len(heroes)} heroes, qrels for {len(qrels)} heroes, "
          f"{len(product_ids):,} products.")

    # Make eval.encoders importable when run from repo root.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from eval.encoders import SpladeEncoder  # noqa: E402

    print(f"Loading SPLADE encoder ({args.model}) on {args.device} ...")
    t0 = time.time()
    encoder = SpladeEncoder(args.model, device=args.device)
    print(f"  loaded in {time.time() - t0:.1f}s")

    # Encode products (slow).
    t0 = time.time()
    product_vecs = _encode_batches(encoder, product_texts, args.batch_size, "products")
    print(f"  encoded {len(product_vecs):,} products in {time.time() - t0:.1f}s")

    # Encode hero queries.
    t0 = time.time()
    hero_texts = [h["text"] for h in heroes]
    query_vecs = _encode_batches(encoder, hero_texts, args.batch_size, "queries")
    print(f"  encoded {len(query_vecs)} queries in {time.time() - t0:.1f}s")

    print(f"\nScoring heroes against {len(product_vecs):,} products ...")
    report: list[dict[str, Any]] = []
    for hero, q_vec in zip(heroes, query_vecs):
        hero_id = hero["id"]
        hero_qrels = qrels.get(hero_id, {})

        scores = np.array([_sparse_dot(q_vec, p_vec) for p_vec in product_vecs], dtype=np.float64)
        top_idx = np.argpartition(-scores, args.top_k)[:args.top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        splade_rank = None
        for r, idx in enumerate(top_idx):
            if hero_qrels.get(product_ids[idx]) == "E":
                splade_rank = r
                break

        wins = splade_rank is not None and splade_rank <= args.win_rank - 1
        report.append({
            "hero_id": hero_id,
            "text": hero["text"],
            "esci_qid": hero["esci_qid"],
            "splade_rank_first_exact": splade_rank,
            "splade_wins": wins,
            "baseline_evidence": hero.get("why_baseline_fails", ""),
        })

    print("\n=== Verdict ===")
    print(f"win threshold: SPLADE Exact at rank <= {args.win_rank} (0-indexed: top {args.win_rank} positions)")
    wins = [r for r in report if r["splade_wins"]]
    losses = [r for r in report if not r["splade_wins"]]
    print(f"WINS: {len(wins)}/{len(report)}")
    print(f"LOSSES: {len(losses)}/{len(report)}\n")
    for r in report:
        verdict = "WIN " if r["splade_wins"] else "LOSE"
        rank_display = (r["splade_rank_first_exact"] + 1) if r["splade_rank_first_exact"] is not None else "NOT IN TOP-K"
        print(f"  [{verdict}] {r['hero_id']:42s}  SPLADE rank: {rank_display}  ({r['baseline_evidence']})")

    if args.report_out:
        args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"\n  wrote per-hero report to {args.report_out}")

    n_wins = len(wins)
    if n_wins < args.min_wins:
        print(f"\nFAIL: {n_wins}/{len(report)} wins, below --min-wins={args.min_wins}.")
        return 1
    print(f"\nPASS: {n_wins}/{len(report)} wins meets --min-wins={args.min_wins}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
