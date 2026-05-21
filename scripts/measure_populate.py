"""Measure where time goes in the populate loop, on this machine.

Profiles dense / BM25 / SPLADE encoders + upsert HTTP roundtrip on real
ESCI product texts (title + brand + color + description), at the same
batch size we use in production indexing. Reports per-step ms-per-batch
and projected total indexing time at 120K so we can decide whether the
parallelism refactor is worth doing.

Not part of the workshop flow -- diagnostic only.
"""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from datasets import load_dataset
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from retrieval import SpladeEncoder
from scripts.setup_collections import (
    BM25_MODEL,
    DENSE_MODEL,
    SPLADE_FINETUNED_MODEL_DEFAULT,
    _compose_text,
)

BATCH = 256
N_BATCHES = 3       # warmup + 2 measured -- short enough to finish in <2 min
TOTAL_TARGET = 120_000
MPS_SUB_BATCH = 32  # SPLADE on MPS OOMs at batch 256 with full-length texts


def splade_encode_chunked(encoder: SpladeEncoder, texts: list[str], chunk: int) -> list:
    out = []
    for i in range(0, len(texts), chunk):
        out.extend(encoder.encode(texts[i:i+chunk]))
    return out


def sample_real_texts(n: int) -> list[str]:
    """Pull n real ESCI product texts (title + brand + color + description)."""
    print(f"loading {n} real ESCI product rows...")
    ds = load_dataset("tasksource/esci", split="test")
    rows = []
    seen = set()
    for r in ds:
        if str(r.get("product_locale", "us")).lower() != "us":
            continue
        pid = r.get("product_id")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        rows.append({
            "product_id": pid,
            "product_title": r.get("product_title", "") or "",
            "product_brand": r.get("product_brand", "") or "",
            "product_color": r.get("product_color", "") or "",
            "product_description": r.get("product_description", "") or "",
        })
        if len(rows) >= n:
            break
    df = pd.DataFrame(rows)
    return [_compose_text(s) for _, s in df.iterrows()]


def timed(label: str, fn) -> float:
    t = time.time()
    fn()
    dt = time.time() - t
    return dt


def main() -> int:
    texts = sample_real_texts(BATCH * N_BATCHES)
    lengths = [len(t.split()) for t in texts]
    print(f"text token-ish lengths: min={min(lengths)} p50={sorted(lengths)[len(lengths)//2]} "
          f"p95={sorted(lengths)[int(len(lengths)*0.95)]} max={max(lengths)}")
    print()

    print("loading encoders...")
    dense = TextEmbedding(model_name=DENSE_MODEL)
    bm25 = SparseTextEmbedding(model_name=BM25_MODEL)
    splade_cpu = SpladeEncoder(SPLADE_FINETUNED_MODEL_DEFAULT, device="cpu")
    splade_mps = SpladeEncoder(SPLADE_FINETUNED_MODEL_DEFAULT, device="mps")

    client = QdrantClient(host="localhost", port=6333)
    if not client.collection_exists("measure_smoke"):
        client.create_collection(
            "measure_smoke",
            vectors_config={"d": models.VectorParams(size=384, distance=models.Distance.COSINE)},
        )

    print()
    print("warmup (1 batch each)...")
    timed("warmup dense", lambda: list(dense.embed(texts[:BATCH])))
    timed("warmup bm25", lambda: list(bm25.embed(texts[:BATCH])))
    timed("warmup splade-cpu", lambda: splade_cpu.encode(texts[:BATCH]))
    timed("warmup splade-mps",
          lambda: splade_encode_chunked(splade_mps, texts[:BATCH], MPS_SUB_BATCH))

    print()
    print(f"timing {N_BATCHES-1} batches of {BATCH} real product texts...")
    print()

    dense_times, bm25_times, splade_cpu_times, splade_mps_times, upsert_times = [], [], [], [], []

    for i in range(1, N_BATCHES):
        batch = texts[i*BATCH:(i+1)*BATCH]

        dense_times.append(timed(f"dense b{i}", lambda: list(dense.embed(batch))))
        bm25_times.append(timed(f"bm25 b{i}", lambda: list(bm25.embed(batch))))
        splade_cpu_times.append(timed(f"splade-cpu b{i}", lambda: splade_cpu.encode(batch)))
        splade_mps_times.append(timed(
            f"splade-mps b{i}",
            lambda: splade_encode_chunked(splade_mps, batch, MPS_SUB_BATCH),
        ))

        # Synthetic upsert: throw 256 random dense vectors at Qdrant to time
        # the HTTP roundtrip only. Real upsert also serializes sparse vectors;
        # this lower-bounds the actual cost.
        import random
        dummy_points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector={"d": [random.random() for _ in range(384)]},
                payload={"i": j},
            )
            for j in range(BATCH)
        ]
        upsert_times.append(timed(
            f"upsert b{i}",
            lambda: client.upsert("measure_smoke", points=dummy_points, wait=False),
        ))

    def avg(lst): return sum(lst) / len(lst)

    d, b, sc, sm, u = (avg(x) for x in (dense_times, bm25_times,
                                         splade_cpu_times, splade_mps_times, upsert_times))

    serial_cpu = d + b + sc + u
    serial_mps = d + b + sm + u
    parallel_mps = max(d + b + u, sm)   # SPLADE on MPS overlaps with all CPU work

    print()
    print(f"  dense (CPU/ONNX)   : {d*1000:7.0f} ms/batch  ({BATCH/d:8.1f} docs/s)")
    print(f"  bm25 (CPU/ONNX)    : {b*1000:7.0f} ms/batch  ({BATCH/b:8.1f} docs/s)")
    print(f"  splade (CPU, b={BATCH})         : {sc*1000:7.0f} ms/batch  ({BATCH/sc:8.1f} docs/s)")
    print(f"  splade (MPS, sub-b={MPS_SUB_BATCH})       : {sm*1000:7.0f} ms/batch  ({BATCH/sm:8.1f} docs/s)")
    print(f"  upsert (dense-only): {u*1000:7.0f} ms/batch  (lower bound)")
    print()
    print("projected total time at 120K (n_batches = 469):")
    n_b = TOTAL_TARGET / BATCH
    print(f"  current (serial, CPU SPLADE):   {serial_cpu*n_b/60:5.1f} min")
    print(f"  serial, MPS SPLADE:             {serial_mps*n_b/60:5.1f} min")
    print(f"  CPU-encoders || MPS-SPLADE:     {parallel_mps*n_b/60:5.1f} min")

    client.delete_collection("measure_smoke")
    return 0


if __name__ == "__main__":
    sys.exit(main())
