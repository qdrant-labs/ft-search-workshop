"""Build the single Qdrant ``products`` collection used in the workshop.

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

Usage::

    python scripts/setup_collections.py \\
        --products data/products_10k.jsonl \\
        --qdrant-url http://localhost:6333

The products file is jsonl by default; ``.parquet`` and ``.csv`` are also
auto-detected by suffix. Required columns: ``product_id``, ``product_title``.
Optional indexed/payload columns: ``product_brand``, ``product_color``,
``product_description``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.encoders import SpladeEncoder  # noqa: E402

LOG = logging.getLogger("setup_collections")

DEFAULT_PRODUCTS = "data/products_10k.jsonl"
DEFAULT_QDRANT_URL = "http://localhost:6333"

# Single collection name -- kept in sync with notebooks and other scripts.
COLLECTION = "products"

# Vector / model identifiers.
BM25_MODEL = "Qdrant/bm25"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DENSE_DIM = 384
SPLADE_FINETUNED_MODEL_DEFAULT = "thierrydamiba/splade-ecommerce-esci"

# Named-vector identifiers on every point. The notebook + benchmark script
# both reference these by name in ``using=...`` / ``Prefetch(using=...)``.
DENSE_VECTOR_NAME = "dense"
BM25_VECTOR_NAME = "bm25"
SPLADE_VECTOR_NAME = "splade_finetuned"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--products",
        default=DEFAULT_PRODUCTS,
        help="Path to ESCI products jsonl/parquet/csv",
    )
    p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    p.add_argument("--qdrant-api-key", default=None)
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
        help="Optional product cap (for smoke tests)",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Product loader (jsonl-first, parquet/csv auto-detected)
# ---------------------------------------------------------------------------
def load_products(path: str, limit: int | None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"products file not found: {p}")
    suffix = p.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        rows: List[Dict] = []
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
        df = pd.DataFrame(rows)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
        if limit is not None:
            df = df.head(limit).copy()
    else:
        df = pd.read_csv(p)
        if limit is not None:
            df = df.head(limit).copy()
    required = {"product_id", "product_title"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"products file missing columns: {missing}")
    df["_text"] = df.apply(_compose_text, axis=1)
    LOG.info("loaded %d products from %s", len(df), p)
    return df


def _compose_text(row: pd.Series) -> str:
    parts = [str(row.get("product_title", "") or "")]
    for col in ("product_brand", "product_color", "product_description"):
        v = row.get(col)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return " ".join(parts).strip()


def _stable_point_id(product_id: str) -> str:
    """Convert an arbitrary ESCI product_id to a UUID Qdrant accepts as a point id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"esci:{product_id}"))


def _payload(row: Dict) -> Dict:
    keep = ("product_id", "product_title", "product_brand", "product_color")
    return {k: row[k] for k in keep if k in row and pd.notna(row[k])}


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
    (``#1045``). One-shot side effect of the build -- call after the
    ``products`` collection is populated.
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

    df = load_products(args.products, args.limit)
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

    # Final assertion: vm/SETUP.md promises this script asserts collection
    # count + status before exiting 0. Earlier upserts used wait=False, so
    # poll get_collection() until indexing settles or we time out.
    _verify_collection_ready(client, expected_count=len(df))

    LOG.info("done")
    return 0


def _verify_collection_ready(
    client: QdrantClient,
    expected_count: int,
    max_wait_seconds: float = 120.0,
    poll_interval: float = 2.0,
) -> None:
    """Poll the ``products`` collection until both points_count and status agree.

    Exits the process non-zero (via ``RuntimeError``) if the collection
    doesn't reach green status with the expected count inside the timeout —
    this is the VM-build pipeline's load-bearing assertion.
    """
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
