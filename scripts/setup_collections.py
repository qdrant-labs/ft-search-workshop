"""Provision the workshop's Qdrant ``products`` collection from ESCI directly.

One collection, three named vectors per point:

* ``dense``              -- ``sentence-transformers/all-MiniLM-L6-v2``
                            (FastEmbed; 384-dim cosine)
* ``bm25``               -- ``Qdrant/bm25`` (FastEmbed sparse, IDF modifier)
* ``splade_finetuned``   -- ``thierrydamiba/splade-ecommerce-esci``
                            (HuggingFace ``transformers``; override via
                            ``--finetuned-model``)

Hybrid (dense + fine-tuned SPLADE, RRF) is queried via Qdrant's Query API at
runtime -- a single ``query_points`` call with two ``Prefetch`` blocks and a
``FusionQuery(fusion=Fusion.RRF)``.

Eval contract
-------------
The headline lab eval is 2,000 ESCI US test queries, deterministically sampled
while force-including the demo queries' ``esci_qid`` values. The corpus is
every product appearing in any of those queries' qrels rows -- typically
~35-40K unique products -- so every selected test query's Exact-grade
products are reachable.

The selection is materialized into ``data/corpus_manifest.json`` at the end
of provisioning. The lab notebook reads that manifest to filter the live ESCI
test split down to exactly the 2K eval set used at provision time. The
manifest is build metadata -- never edit it by hand.

Usage::

    python scripts/setup_collections.py --qdrant-url http://localhost:6333 --recreate

Provisioning cost (CPU only, single VM):
* ~35-40K product corpus
* SPLADE-encode pass dominates (~1-2h on 4 vCPU)
* Use ``--device cuda`` if a GPU is available -- drops to a few minutes
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import torch
from datasets import load_dataset
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval import SpladeEncoder  # noqa: E402

LOG = logging.getLogger("setup_collections")

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_ESCI_DATASET = "tasksource/esci"
DEFAULT_EVAL_SAMPLE_SIZE = 2000
DEFAULT_EVAL_SAMPLE_SEED = 13

# Single collection name -- kept in sync with the lab notebook.
COLLECTION = "products"

# Vector / model identifiers.
BM25_MODEL = "Qdrant/bm25"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_DIM = 384
SPLADE_FINETUNED_MODEL_DEFAULT = "thierrydamiba/splade-ecommerce-esci"

# Named-vector identifiers on every point. The notebook references these by
# name in ``using=...`` / ``Prefetch(using=...)``.
DENSE_VECTOR_NAME = "dense"
BM25_VECTOR_NAME = "bm25"
SPLADE_VECTOR_NAME = "splade_finetuned"

# Path the notebook reads to learn which 2K queries to evaluate.
# Written at the end of every successful provisioning.
MANIFEST_PATH = Path("data/corpus_manifest.json")
MANIFEST_SAMPLE_SIZE = 50  # used by the optional pilot verification notebook


def assert_supported_runtime() -> None:
    """Fail before FastEmbed reaches a native segfault on unsupported runtimes."""
    if sys.version_info >= (3, 14):
        py = platform.python_version()
        raise RuntimeError(
            f"Python {py} is not supported for this workshop. FastEmbed/ONNX "
            f"Runtime currently segfaults during local indexing on Python 3.14. "
            f"Recreate your virtualenv with Python 3.12, then rerun this script."
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    p.add_argument("--qdrant-api-key", default=None)
    p.add_argument("--esci-dataset", default=DEFAULT_ESCI_DATASET,
                   help="HuggingFace dataset id for ESCI")
    p.add_argument("--eval-sample-size", type=int, default=DEFAULT_EVAL_SAMPLE_SIZE,
                   help="Number of test queries to sample for the headline eval")
    p.add_argument("--eval-sample-seed", type=int, default=DEFAULT_EVAL_SAMPLE_SEED,
                   help="Seed for the deterministic eval-query sample")
    p.add_argument(
        "--demo-queries",
        dest="demo_queries",
        default="data/demo_queries.json",
        help="Path to demo_queries.json; their esci_qid values are forced into the manifest",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--finetuned-model", default=SPLADE_FINETUNED_MODEL_DEFAULT)
    p.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and rebuild the products collection if it already exists",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for the SPLADE encoder",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional product cap (for smoke tests). Bypasses the manifest contract.",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


# ---------------------------------------------------------------------------
# ESCI loader + deterministic 2K sample.
#
# We use all US rows from the ESCI test split. Some curated demo queries are
# outside ESCI's ``small_version == 1`` subset, and they are still valid US
# query/product judgments for this workshop.
# ---------------------------------------------------------------------------
def _row_is_eligible(row: dict) -> bool:
    return str(row.get("product_locale", "us")).lower() == "us"


def load_esci_filtered(dataset_id: str) -> List[dict]:
    """Pull every US row in the ESCI test split.

    Returns the raw rows (we group them later). Materializing this once
    avoids a second pass over the HF cache.
    """
    LOG.info("loading ESCI from HF: %s (split=test, locale=us)", dataset_id)
    ds = load_dataset(dataset_id, split="test")
    rows = [r for r in ds if _row_is_eligible(r)]
    if not rows:
        raise RuntimeError(
            f"No eligible rows in {dataset_id} after filtering on "
            f"product_locale='us'. Check the schema -- this loader expects "
            f"a product_locale column."
        )
    LOG.info("ESCI: %d eligible rows after US filter", len(rows))
    return rows


def select_eval_query_ids(
    rows: List[dict],
    sample_size: int,
    seed: int,
    demo_qids: Set[int],
) -> List[int]:
    """Deterministically pick the eval query-id set, force-including demos.

    The headline lab eval is a fixed sample of ``sample_size`` queries
    (random, seeded) that always includes every demo query's ``esci_qid``.
    Demo queries are forced in so the CP2/CP3 narrative survives any reshuffle
    of the sample. Returns a sorted list of integer query_ids.
    """
    all_qids = sorted({int(r["query_id"]) for r in rows if "query_id" in r})
    missing_demos = demo_qids - set(all_qids)
    if missing_demos:
        raise RuntimeError(
            f"Demo esci_qids missing from ESCI test split: {sorted(missing_demos)}. "
            f"Either fix demo_queries.json or pick a different dataset mirror."
        )
    if len(demo_qids) > sample_size:
        raise RuntimeError(
            f"Demo query count ({len(demo_qids)}) exceeds requested eval sample "
            f"size ({sample_size})."
        )
    if len(all_qids) < sample_size:
        LOG.warning(
            "ESCI eligible queries (%d) < requested sample (%d); using all.",
            len(all_qids), sample_size,
        )
        final = set(all_qids)
    else:
        rng = random.Random(seed)
        sample_pool = [qid for qid in all_qids if qid not in demo_qids]
        sampled = set(rng.sample(sample_pool, k=sample_size - len(demo_qids)))
        final = sampled | demo_qids
    result = sorted(final)
    included_demos = len(demo_qids & set(result))
    LOG.info(
        "eval manifest: %d queries (%d random + %d demos)",
        len(result),
        len(result) - included_demos,
        included_demos,
    )
    return result


def collect_corpus_from_eval(
    rows: List[dict],
    eval_qids: Set[int],
    limit: int | None,
) -> pd.DataFrame:
    """Collect every product appearing in any qrels row of the eval queries.

    This is what guarantees corpus reachability for the headline metric:
    every test query's graded products are physically present.
    """
    seen: Dict[str, Dict] = {}
    for row in rows:
        try:
            qid = int(row["query_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if qid not in eval_qids:
            continue
        pid = row.get("product_id")
        if not pid or pid in seen:
            continue
        seen[pid] = {
            "product_id": pid,
            "product_title": row.get("product_title", "") or "",
            "product_brand": row.get("product_brand", "") or "",
            "product_color": row.get("product_color", "") or "",
            "product_description": row.get("product_description", "") or "",
        }
        if limit is not None and len(seen) >= limit:
            LOG.info("hit --limit=%d, stopping corpus collection early", limit)
            break

    df = pd.DataFrame(list(seen.values()))
    if df.empty:
        raise RuntimeError(
            "No products collected from eval qrels. Check whether the "
            "eval_qids actually intersect the ESCI rows."
        )
    df["_text"] = df.apply(_compose_text, axis=1)
    LOG.info("corpus: %d unique products from %d eval queries' qrels", len(df), len(eval_qids))
    return df


def write_corpus_manifest(
    manifest_path: Path,
    *,
    dataset_id: str,
    eval_query_ids: List[int],
    corpus_df: pd.DataFrame,
    sample_seed: int,
    sample_size: int,
) -> None:
    """Write build metadata the lab notebook reads at startup."""
    rng = random.Random(sample_seed)
    sorted_pids = sorted(corpus_df["product_id"].tolist())
    sample_pids = (
        sorted_pids if len(sorted_pids) <= MANIFEST_SAMPLE_SIZE
        else rng.sample(sorted_pids, k=MANIFEST_SAMPLE_SIZE)
    )
    manifest = {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": dataset_id,
        "split": "test",
        "locale": "us",
        "small_version": None,
        "eval_sample_size": sample_size,
        "eval_sample_seed": sample_seed,
        "eval_query_ids": eval_query_ids,
        "corpus_product_count": int(len(corpus_df)),
        "corpus_sample_product_ids": sorted(sample_pids),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    LOG.info(
        "wrote %s (eval_queries=%d, corpus=%d, sample=%d)",
        manifest_path, len(eval_query_ids), len(corpus_df), len(sample_pids),
    )


def _compose_text(row: pd.Series) -> str:
    parts = [str(row.get("product_title", "") or "")]
    for col in ("product_brand", "product_color", "product_description"):
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " ".join(parts).strip()


def _stable_point_id(product_id: str) -> str:
    """Convert an ESCI product_id (ASIN) to a UUID Qdrant accepts as a point id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"esci:{product_id}"))


def _payload(row: Dict) -> Dict:
    keep = ("product_id", "product_title", "product_brand", "product_color")
    return {k: row[k] for k in keep if k in row and pd.notna(row[k]) and row[k] != ""}


# ---------------------------------------------------------------------------
# Collection lifecycle
# ---------------------------------------------------------------------------
def _ensure_collection(client: QdrantClient, recreate: bool) -> bool:
    """Create the ``products`` collection (or recreate if --recreate)."""
    exists = client.collection_exists(COLLECTION)
    if exists and not recreate:
        LOG.info(
            "collection %s already exists -- skipping create "
            "(use --recreate to rebuild)",
            COLLECTION,
        )
        return False
    if exists and recreate:
        LOG.info("dropping existing collection %s", COLLECTION)
        client.delete_collection(COLLECTION)

    client.create_collection(
        COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            BM25_VECTOR_NAME: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
            SPLADE_VECTOR_NAME: models.SparseVectorParams(),
        },
    )
    LOG.info(
        "created collection %s (named vectors: %s, %s, %s)",
        COLLECTION,
        DENSE_VECTOR_NAME,
        BM25_VECTOR_NAME,
        SPLADE_VECTOR_NAME,
    )
    return True


# ---------------------------------------------------------------------------
# Population: encode all three vectors for every point, batch-upsert
# ---------------------------------------------------------------------------
def populate(
    client: QdrantClient,
    df: pd.DataFrame,
    finetuned_model: str,
    batch_size: int,
    device: str,
) -> None:
    LOG.info("loading FastEmbed encoders (dense=%s, bm25=%s)", DENSE_MODEL, BM25_MODEL)
    dense_encoder = TextEmbedding(model_name=DENSE_MODEL)
    bm25_encoder = SparseTextEmbedding(model_name=BM25_MODEL)

    LOG.info("loading fine-tuned SPLADE encoder (%s) on %s", finetuned_model, device)
    splade_encoder = SpladeEncoder(finetuned_model, device=device)

    rows = df.to_dict(orient="records")
    total = len(rows)
    t0 = time.time()

    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        texts = [r["_text"] for r in batch]

        dense_vecs = list(dense_encoder.embed(texts))
        bm25_vecs = list(bm25_encoder.embed(texts))
        splade_vecs = splade_encoder.encode(texts)

        points = []
        for r, dv, bv, (sidx, svals) in zip(batch, dense_vecs, bm25_vecs, splade_vecs):
            points.append(
                models.PointStruct(
                    id=_stable_point_id(r["product_id"]),
                    vector={
                        DENSE_VECTOR_NAME: list(map(float, dv)),
                        BM25_VECTOR_NAME: models.SparseVector(
                            indices=list(map(int, bv.indices)),
                            values=list(map(float, bv.values)),
                        ),
                        SPLADE_VECTOR_NAME: models.SparseVector(
                            indices=list(map(int, sidx)),
                            values=list(map(float, svals)),
                        ),
                    },
                    payload=_payload(r),
                )
            )
        client.upsert(COLLECTION, points=points, wait=False)
        if start % (batch_size * 4) == 0:
            LOG.info("  %s: %d / %d uploaded", COLLECTION, start + len(batch), total)

    LOG.info("  %s: populated %d points in %.1fs", COLLECTION, total, time.time() - t0)


# ---------------------------------------------------------------------------
# SPLADE vocab dump (for the notebook's sparse-vector inspection cell)
# ---------------------------------------------------------------------------
def write_splade_vocab(model_name: str, out_path: str = "data/splade_vocab.json") -> None:
    """Write a ``{int_id: token}`` mapping for the SPLADE model's tokenizer.

    The lab notebook's sparse-vector inspection cell uses this to render
    human-readable terms (``iphone``) instead of raw token indices
    (``#1045``).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = {int(v): k for k, v in tokenizer.vocab.items()}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocab, ensure_ascii=False))
    LOG.info("wrote %s (%d tokens)", out, len(vocab))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    assert_supported_runtime()

    demo_qids: Set[int] = set()
    demo_path = Path(args.demo_queries)
    if demo_path.exists():
        for h in json.loads(demo_path.read_text()):
            if "esci_qid" in h:
                demo_qids.add(int(h["esci_qid"]))
        LOG.info("loaded %d demo qids from %s", len(demo_qids), demo_path)
    else:
        LOG.warning("demo_queries file %s not found; manifest will exclude demo forcing", demo_path)

    rows = load_esci_filtered(args.esci_dataset)
    eval_qids = select_eval_query_ids(
        rows,
        sample_size=args.eval_sample_size,
        seed=args.eval_sample_seed,
        demo_qids=demo_qids,
    )
    df = collect_corpus_from_eval(rows, set(eval_qids), args.limit)

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)

    created = _ensure_collection(client, args.recreate)
    if created:
        populate(
            client,
            df,
            finetuned_model=args.finetuned_model,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        LOG.info("collection exists; skipping population (use --recreate to rebuild)")

    write_splade_vocab(args.finetuned_model, "data/splade_vocab.json")
    _verify_collection_ready(client, expected_count=len(df))
    write_corpus_manifest(
        MANIFEST_PATH,
        dataset_id=args.esci_dataset,
        eval_query_ids=eval_qids,
        corpus_df=df,
        sample_seed=args.eval_sample_seed,
        sample_size=args.eval_sample_size,
    )

    LOG.info("done")
    return 0


def _verify_collection_ready(
    client: QdrantClient,
    expected_count: int,
    max_wait_seconds: float = 120.0,
    poll_interval: float = 2.0,
) -> None:
    """Poll the ``products`` collection until points_count and status agree."""
    deadline = time.time() + max_wait_seconds
    last_count = -1
    last_status = "unknown"
    while time.time() < deadline:
        info = client.get_collection(COLLECTION)
        last_count = getattr(info, "points_count", 0) or 0
        last_status = str(getattr(info, "status", "unknown"))
        if last_count >= expected_count and last_status.lower() in {"green", "status.green"}:
            LOG.info(
                "  %s ready: count=%d, status=%s",
                COLLECTION, last_count, last_status,
            )
            return
        LOG.info(
            "  waiting for %s (count=%d/%d, status=%s)",
            COLLECTION, last_count, expected_count, last_status,
        )
        time.sleep(poll_interval)
    raise RuntimeError(
        f"{COLLECTION} did not reach ready state within {max_wait_seconds}s "
        f"(final count={last_count}/{expected_count}, status={last_status}). "
        f"Re-run with --recreate or investigate the upsert log."
    )


if __name__ == "__main__":
    sys.exit(main())
