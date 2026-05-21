"""Provision the workshop's Qdrant ``products`` collection from ESCI directly.

One collection, three named vectors per point:

* ``dense``              -- ``sentence-transformers/all-MiniLM-L6-v2``
                            (FastEmbed; 384-dim cosine)
* ``bm25``               -- ``Qdrant/bm25`` (FastEmbed sparse, IDF modifier)
* ``splade_finetuned``   -- ``thierrydamiba/splade-ecommerce-esci``
                            (HuggingFace ``transformers``)

The lab also demonstrates the production hybrid pattern: dense + fine-tuned
SPLADE via Qdrant's Query API, using two ``Prefetch`` blocks and a
``FusionQuery(fusion=Fusion.DBSF)`` at runtime.

Eval contract
-------------
The headline lab eval is 2,000 ESCI US test queries, deterministically sampled
while force-including the demo queries' ``esci_qid`` values. The corpus always
includes every product that appears in any of those queries' qrels rows --
typically ~35-40K unique products -- so every selected test query's
Exact-grade products are reachable.

``--corpus-distractors N`` optionally appends N additional ESCI US products
that do NOT appear in any eval query's qrels. Distractors raise the realism
of the catalog (more noise for lexical retrieval to compete against) without
touching the recall ceiling. Default 0 preserves the original workshop corpus.

The selection is materialized into ``data/corpus_manifest.json`` at the end
of provisioning. The lab notebook reads that manifest to filter the live ESCI
test split down to exactly the 2K eval set used at provision time.

Usage::

    python scripts/setup_collections.py --recreate
    python scripts/setup_collections.py --recreate --corpus-distractors 80000

Provisioning cost (single VM):
* qrels-only corpus (~37K)              ~1-2h on 4 vCPU, minutes on GPU
* qrels + 80K distractors (~120K)       ~3-4h on 4 vCPU
* Use ``--device cuda`` or ``--device mps`` to drop encode time substantially.
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
DEFAULT_EVAL_SAMPLE_SIZE = 2000
DEFAULT_EVAL_SAMPLE_SEED = 13
DEFAULT_CORPUS_DISTRACTORS = 0

# Fixed dataset + locale -- the workshop is built around this ESCI mirror.
ESCI_DATASET = "tasksource/esci"
ESCI_SPLIT = "test"
ESCI_LOCALE = "us"

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

# SPLADE on MPS OOMs at batch=256 because the MLM logits tensor is
# (batch, seq_len, vocab) ~= (256, 256, 30522) per intermediate. Sub-batch
# the SPLADE pass while keeping the outer indexing batch at --batch-size.
SPLADE_MPS_SUB_BATCH = 32

# Path the notebook reads to learn which 2K queries to evaluate.
# Written at the end of every successful provisioning.
MANIFEST_PATH = Path("data/corpus_manifest.json")
DEMO_QUERIES_PATH = Path("data/demo_queries.json")


def assert_supported_runtime() -> None:
    """Fail before FastEmbed reaches a native segfault on unsupported runtimes."""
    if sys.version_info >= (3, 14):
        py = platform.python_version()
        raise RuntimeError(
            f"Python {py} is not supported for this workshop. FastEmbed/ONNX "
            f"Runtime currently segfaults during local indexing on Python 3.14. "
            f"Recreate your virtualenv with Python 3.12, then rerun this script."
        )


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    p.add_argument("--eval-sample-size", type=int, default=DEFAULT_EVAL_SAMPLE_SIZE,
                   help="Number of test queries to sample for the headline eval")
    p.add_argument("--eval-sample-seed", type=int, default=DEFAULT_EVAL_SAMPLE_SEED,
                   help="Seed for the deterministic eval-query sample")
    p.add_argument("--corpus-distractors", type=int, default=DEFAULT_CORPUS_DISTRACTORS,
                   help="Extra non-qrels products to append to the corpus (default 0)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--recreate", action="store_true",
                   help="Drop and rebuild the products collection if it already exists")
    p.add_argument("--device", default=_default_device(),
                   help="Torch device for the SPLADE encoder (cpu/mps/cuda)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# ESCI loader + deterministic 2K sample.
#
# We use all US rows from the ESCI test split. Some curated demo queries are
# outside ESCI's ``small_version == 1`` subset, and they are still valid US
# query/product judgments for this workshop.
# ---------------------------------------------------------------------------
def load_esci_us_test() -> List[dict]:
    """Pull every US row in the ESCI test split."""
    LOG.info("loading ESCI from HF: %s (split=%s, locale=%s)",
             ESCI_DATASET, ESCI_SPLIT, ESCI_LOCALE)
    ds = load_dataset(ESCI_DATASET, split=ESCI_SPLIT)
    rows = [r for r in ds if str(r.get("product_locale", "us")).lower() == ESCI_LOCALE]
    if not rows:
        raise RuntimeError(
            f"No US rows in {ESCI_DATASET} split={ESCI_SPLIT}. "
            f"Check the schema -- this loader expects a product_locale column."
        )
    LOG.info("ESCI: %d eligible rows after US filter", len(rows))
    return rows


def load_demo_qids() -> Set[int]:
    if not DEMO_QUERIES_PATH.exists():
        LOG.warning("demo queries file %s not found; manifest will exclude demo forcing",
                    DEMO_QUERIES_PATH)
        return set()
    qids = {int(h["esci_qid"]) for h in json.loads(DEMO_QUERIES_PATH.read_text())
            if "esci_qid" in h}
    LOG.info("loaded %d demo qids from %s", len(qids), DEMO_QUERIES_PATH)
    return qids


def select_eval_query_ids(
    rows: List[dict],
    sample_size: int,
    seed: int,
    demo_qids: Set[int],
) -> List[int]:
    """Deterministically pick the eval query-id set, force-including demos.

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
        LOG.warning("ESCI eligible queries (%d) < requested sample (%d); using all.",
                    len(all_qids), sample_size)
        final = set(all_qids)
    else:
        rng = random.Random(seed)
        sample_pool = [qid for qid in all_qids if qid not in demo_qids]
        sampled = set(rng.sample(sample_pool, k=sample_size - len(demo_qids)))
        final = sampled | demo_qids
    result = sorted(final)
    included_demos = len(demo_qids & set(result))
    LOG.info("eval manifest: %d queries (%d random + %d demos)",
             len(result), len(result) - included_demos, included_demos)
    return result


def _product_row(row: dict) -> Dict:
    return {
        "product_id": row["product_id"],
        "product_title": row.get("product_title", "") or "",
        "product_brand": row.get("product_brand", "") or "",
        "product_color": row.get("product_color", "") or "",
        "product_description": row.get("product_description", "") or "",
    }


def collect_corpus_from_eval(rows: List[dict], eval_qids: Set[int]) -> pd.DataFrame:
    """Collect every product appearing in any qrels row of the eval queries."""
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
        seen[pid] = _product_row(row)

    if not seen:
        raise RuntimeError(
            "No products collected from eval qrels. Check whether the "
            "eval_qids actually intersect the ESCI rows."
        )
    df = pd.DataFrame(list(seen.values()))
    df["_text"] = df.apply(_compose_text, axis=1)
    LOG.info("corpus (qrels coverage): %d unique products from %d eval queries",
             len(df), len(eval_qids))
    return df


def collect_distractors(
    rows: List[dict],
    eval_qids: Set[int],
    existing_pids: Set[str],
    target: int,
    seed: int,
) -> pd.DataFrame:
    """Sample N distractor products from rows whose query is NOT in the eval set.

    Distractors are unique products that do not appear in any eval query's
    qrels. They add lexical noise so BM25 has false positives to compete with,
    bringing the workshop's task closer to a real catalog search.
    """
    if target <= 0:
        return pd.DataFrame()

    candidates: Dict[str, Dict] = {}
    for row in rows:
        try:
            qid = int(row["query_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if qid in eval_qids:
            continue
        pid = row.get("product_id")
        if not pid or pid in existing_pids or pid in candidates:
            continue
        candidates[pid] = _product_row(row)

    pool = list(candidates)
    if not pool:
        LOG.warning("no distractor candidates available; skipping")
        return pd.DataFrame()

    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen = pool[:target]
    if len(chosen) < target:
        LOG.warning("requested %d distractors but only %d available; using all",
                    target, len(chosen))

    df = pd.DataFrame([candidates[pid] for pid in chosen])
    df["_text"] = df.apply(_compose_text, axis=1)
    LOG.info("corpus (distractors): %d products sampled from non-eval queries",
             len(df))
    return df


def write_corpus_manifest(
    manifest_path: Path,
    *,
    eval_query_ids: List[int],
    qrels_count: int,
    distractor_count: int,
    sample_seed: int,
    sample_size: int,
) -> None:
    manifest = {
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_id": ESCI_DATASET,
        "split": ESCI_SPLIT,
        "locale": ESCI_LOCALE,
        "eval_sample_size": sample_size,
        "eval_sample_seed": sample_seed,
        "eval_query_ids": eval_query_ids,
        "corpus_qrels_count": qrels_count,
        "corpus_distractor_count": distractor_count,
        "corpus_product_count": qrels_count + distractor_count,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    LOG.info("wrote %s (eval_queries=%d, corpus=%d qrels + %d distractors)",
             manifest_path, len(eval_query_ids), qrels_count, distractor_count)


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
    exists = client.collection_exists(COLLECTION)
    if exists and not recreate:
        LOG.info("collection %s already exists -- skipping create "
                 "(use --recreate to rebuild)", COLLECTION)
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
            BM25_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF),
            SPLADE_VECTOR_NAME: models.SparseVectorParams(),
        },
    )
    LOG.info("created collection %s (named vectors: %s, %s, %s)",
             COLLECTION, DENSE_VECTOR_NAME, BM25_VECTOR_NAME, SPLADE_VECTOR_NAME)
    return True


# ---------------------------------------------------------------------------
# Population: encode all three vectors for every point, batch-upsert
# ---------------------------------------------------------------------------
def populate(
    client: QdrantClient,
    df: pd.DataFrame,
    batch_size: int,
    device: str,
) -> None:
    LOG.info("loading FastEmbed encoders (dense=%s, bm25=%s)", DENSE_MODEL, BM25_MODEL)
    dense_encoder = TextEmbedding(model_name=DENSE_MODEL)
    bm25_encoder = SparseTextEmbedding(model_name=BM25_MODEL)

    LOG.info("loading fine-tuned SPLADE encoder (%s) on %s",
             SPLADE_FINETUNED_MODEL_DEFAULT, device)
    splade_encoder = SpladeEncoder(SPLADE_FINETUNED_MODEL_DEFAULT, device=device)

    rows = df.to_dict(orient="records")
    total = len(rows)
    t0 = time.time()

    splade_chunk = SPLADE_MPS_SUB_BATCH if device == "mps" else batch_size

    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        texts = [r["_text"] for r in batch]

        dense_vecs = list(dense_encoder.embed(texts))
        bm25_vecs = list(bm25_encoder.embed(texts))
        splade_vecs = []
        for i in range(0, len(texts), splade_chunk):
            splade_vecs.extend(splade_encoder.encode(texts[i : i + splade_chunk]))

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


def write_splade_vocab(model_name: str, out_path: str = "data/splade_vocab.json") -> None:
    """Write a ``{int_id: token}`` mapping for the SPLADE model's tokenizer.

    The lab notebook's sparse-vector inspection cell uses this to render
    human-readable terms (``iphone``) instead of raw token indices (``#1045``).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = {int(v): k for k, v in tokenizer.vocab.items()}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocab, ensure_ascii=False))
    LOG.info("wrote %s (%d tokens)", out, len(vocab))


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
            LOG.info("  %s ready: count=%d, status=%s",
                     COLLECTION, last_count, last_status)
            return
        LOG.info("  waiting for %s (count=%d/%d, status=%s)",
                 COLLECTION, last_count, expected_count, last_status)
        time.sleep(poll_interval)
    raise RuntimeError(
        f"{COLLECTION} did not reach ready state within {max_wait_seconds}s "
        f"(final count={last_count}/{expected_count}, status={last_status}). "
        f"Re-run with --recreate or investigate the upsert log."
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    assert_supported_runtime()

    demo_qids = load_demo_qids()
    rows = load_esci_us_test()
    eval_qids = select_eval_query_ids(
        rows,
        sample_size=args.eval_sample_size,
        seed=args.eval_sample_seed,
        demo_qids=demo_qids,
    )

    qrels_df = collect_corpus_from_eval(rows, set(eval_qids))
    distractor_df = collect_distractors(
        rows,
        eval_qids=set(eval_qids),
        existing_pids=set(qrels_df["product_id"]),
        target=args.corpus_distractors,
        seed=args.eval_sample_seed,
    )
    corpus_df = pd.concat([qrels_df, distractor_df], ignore_index=True) \
        if not distractor_df.empty else qrels_df
    LOG.info("corpus total: %d products (%d qrels + %d distractors)",
             len(corpus_df), len(qrels_df), len(distractor_df))

    client = QdrantClient(url=args.qdrant_url)

    created = _ensure_collection(client, args.recreate)
    if created:
        populate(client, corpus_df, batch_size=args.batch_size, device=args.device)
    else:
        LOG.info("collection exists; skipping population (use --recreate to rebuild)")

    write_splade_vocab(SPLADE_FINETUNED_MODEL_DEFAULT, "data/splade_vocab.json")
    _verify_collection_ready(client, expected_count=len(corpus_df))
    write_corpus_manifest(
        MANIFEST_PATH,
        eval_query_ids=eval_qids,
        qrels_count=len(qrels_df),
        distractor_count=len(distractor_df),
        sample_seed=args.eval_sample_seed,
        sample_size=args.eval_sample_size,
    )

    LOG.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
