# HackerSquad VM — Build Spec

Target deliverable: a single VM image that an 80-minute workshop attendee can boot, open in their browser, and start running notebook cells against in under 60 seconds. No network is required during the lab (all models cached, all data local). This document is the source of truth for the VM team.

---

## Goal & constraints

Build a reproducible VM image that runs Qdrant locally with one pre-populated `products` collection (dense + two sparse named vectors per point), JupyterLab pre-opened on the main lab notebook, and all embedding models cached on disk. The full 2K eval is **computed live** in the wrap cell (~2–6 min on the VM; exact range CPU-dependent, verified at pilot). **Nothing is precomputed.** The image must boot to a ready state — no installs, no downloads, no warmup — and survive a Qdrant restart without data loss.

---

## Software inventory

Pin exact versions in the image. The notebook will assert these at startup.

| Component | Version | Source | Notes |
|---|---|---|---|
| OS | Ubuntu 22.04 LTS | base image | x86_64 |
| Docker Engine | 24.x | apt | for Qdrant container |
| Qdrant | `qdrant/qdrant:v1.12.4` | Docker Hub | pinned; do not use `:latest` |
| Python | 3.11.x | deadsnakes PPA or pyenv | 3.11 required for fastembed wheels |
| JupyterLab | 4.2.x | pip | server config below |
| `qdrant-client` | 1.12.x | pip | matches Qdrant minor |
| `fastembed` | 0.4.x | pip | provides BM25 + MiniLM |
| `transformers` | 4.44.x | pip | for SPLADE inference |
| `torch` | 2.4.x (CPU build) | pip `--extra-index-url` cpu | no GPU expected on VM |
| `pandas` | 2.2.x | pip | eval tables |
| `numpy` | 1.26.x | pip | bootstrap CI + percentiles |
| `ipython` | 8.x | pip | |
| `tqdm` | 4.x | pip | progress in eval cells |

Freeze the final environment with `pip freeze > /opt/workshop/requirements.lock.txt` and bake that into the image.

---

## Model cache

Pre-download to `/home/workshop/.cache/huggingface/` (and symlink to `/root/.cache/huggingface/` for safety). All three models must resolve offline with `HF_HUB_OFFLINE=1`.

| Model | HF repo | Used by | Approx size |
|---|---|---|---|
| MiniLM dense baseline | `sentence-transformers/all-MiniLM-L6-v2` | FastEmbed (via its own wrapper) for the `dense` named vector | ~90 MB |
| BM25 (FastEmbed) | `Qdrant/bm25` | FastEmbed for the `bm25` named sparse vector | ~5 MB (tokenizer + idf) |
| Fine-tuned SPLADE | `thierrydamiba/splade-ecommerce-esci` | `transformers` for the `splade_finetuned` named sparse vector | ~440 MB |


FastEmbed caches separately under `/home/workshop/.cache/fastembed/`. Prime it by running each FastEmbed model class once during image build with `cache_dir=` set explicitly.

Validate offline mode at end of build:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "
from transformers import AutoTokenizer, AutoModelForMaskedLM
for repo in ['thierrydamiba/splade-ecommerce-esci']:
    AutoTokenizer.from_pretrained(repo)
    AutoModelForMaskedLM.from_pretrained(repo)
print('OK: SPLADE models load offline')
"
```

---

## Data inventory

All under `/opt/workshop/data/`. Read-only to the attendee user.

| File | Size | Purpose |
|---|---|---|
| `hero_queries.json` | ~10 KB | 10 diversified hero queries with failure-mode tags. Top-of-notebook list for CP1/CP2. |
| `qrels_hero.json` | ~5 KB | Hero qrels keyed by hero id: `{hero_id: {product_id: grade}}`. **`grade` is the ESCI string label — one of `"E"` (Exact), `"S"` (Substitute), `"C"` (Complement), `"I"` (Irrelevant)**. Derived from ESCI at VM build time by `scripts/build_eval_data.py`. Consumed by `eval/metrics.py`, `eval/viewer.py`, and the notebook's hero-query sanity check — all of which expect strings, NOT floats. Used by `compare_results` to render ESCI grade tags per row. |
| `ood_queries.json` | ~3 KB | 5 out-of-domain queries for the "where SPLADE breaks" wrap demo. |
| `products_10k.jsonl` | ~30 MB | ESCI subset, 10K products (`product_id`, `product_title`, `product_brand`, `product_color`, etc). Source: `tasksource/esci` filtered to top 10K most-frequent products. |
| `eval_subset_200.json` | ~80 KB | 200-query subset with inlined qrels: `[{"query_id": ..., "query": ..., "qrels": {product_id: grade}}, ...]`. Stratified sample of the 2K test set. Used by CP2 + CP3 live metric reveals. |
| `eval_full_2k.json` | ~700 KB | Full 2K test set, same schema as `eval_subset_200.json`. Used by the wrap-cell live 2K eval. |
| `splade_vocab.json` | ~250 KB | DistilBERT tokenizer vocab keyed by integer token id: `{"1045": "iphone", ...}`. Used by the sparse-vector inspection cell to render readable terms. Generated at image-bake time by `scripts/setup_collections.py`. |

The full ESCI dump is *not* shipped — only the 10K-product subset and the 2K-query slice with their relevance labels.

> **Note:** `qrels_hero.json`, `eval_subset_200.json`, and `eval_full_2k.json` are produced by `scripts/build_eval_data.py` — the canonical source for the workshop's qrels and eval slices. Regenerate them from the ESCI dump via that script at image-bake time; do not hand-edit.

---

## Collection state (Qdrant at minute 0)

Qdrant container running with persistent storage at `/var/lib/qdrant/storage/`. The single `products` collection is fully populated and indexed before the image is sealed.

| Collection | Vector type | Embedding source | Size |
|---|---|---|---|
| `products` | dense (384, cosine) + sparse (bm25 IDF, splade_finetuned) | FastEmbed `sentence-transformers/all-MiniLM-L6-v2` (dense), FastEmbed `Qdrant/bm25` (sparse bm25, IDF modifier), `thierrydamiba/splade-ecommerce-esci` via `transformers` (sparse splade_finetuned) | 10K points |

Each point carries all three named vectors (`dense`, `bm25`, `splade_finetuned`) and a payload of `{product_id, product_title, product_brand, product_color}`.

Population is driven by `/opt/workshop/scripts/setup_collections.py`. The script is idempotent: drops and recreates the collection when `--recreate` is set, embeds all 10K products, batch-upserts at 256 per batch, then asserts `count == 10000` and `status == "green"` before exiting 0. Run it once during image build; do not run on every boot.

The `bm25` named sparse vector uses `SparseVectorParams(modifier=Modifier.IDF)`; the `splade_finetuned` named sparse vector uses default `SparseVectorParams()`. Keep `on_disk=False` (sparse hot in RAM) for live demo latency.

---


## VM startup sequence

At boot (systemd units, in order):

1. `qdrant.service` — `docker compose up -d` in `/opt/workshop/qdrant/` (compose file mounts the persistent storage volume). Health-check loop until `GET /healthz` returns 200; max 30s.
2. `workshop-jupyter.service` — runs as user `workshop`, launches `jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --ServerApp.token='' --ServerApp.password='' --notebook-dir=/opt/workshop/notebooks` (token disabled because the VM is single-tenant and ephemeral; do NOT reuse this config outside the workshop).
3. `workshop-welcome.service` — paints the welcome banner on tty1 with the JupyterLab URL.

Pre-built at image-bake time, idle at boot:
- Qdrant collections (populated, on-disk)
- Model cache (HuggingFace + FastEmbed)
- Python venv at `/opt/workshop/venv/` with all packages installed
- Main lab notebook at `/opt/workshop/notebooks/lab.ipynb`, with `/opt/workshop/notebooks/` owned by user `workshop` so attendees can edit and save. **The notebook must live inside `/opt/workshop/` so cell 2's `Path.cwd().resolve().parent` resolves the repo root correctly and `sys.path` picks up `eval/` and `scripts/`.** The active tab when JupyterLab opens is set via the JupyterLab workspace JSON at `/home/workshop/.jupyter/lab/workspaces/default-37a8.jupyterlab-workspace`, whose `"file"` field points at `/opt/workshop/notebooks/lab.ipynb`.

Idle (no work at boot, attendee triggers): notebook cell execution.

---

## Verification checklist

Run all checks on a freshly-booted instance from the candidate image. All must pass before publishing.

```bash
# 1. Qdrant health
curl -fsS http://localhost:6333/healthz
curl -fsS http://localhost:6333/readyz

# 2. The products collection exists, is green, and has 10K points
curl -fsS http://localhost:6333/collections/products | jq -e '.result.status == "green" and .result.points_count >= 10000' \
  || { echo "FAIL: products"; exit 1; }

# 3. Python imports smoke test (offline)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/workshop/venv/bin/python -c "
import qdrant_client, fastembed, transformers, torch, pandas
from transformers import AutoTokenizer, AutoModelForMaskedLM
from fastembed import TextEmbedding, SparseTextEmbedding
print('OK: imports')
"

# 4. JupyterLab reachable
curl -fsS http://localhost:8888/lab | grep -q "JupyterLab"

# 5. Notebook opens, all cells parse (no execution)
/opt/workshop/venv/bin/jupyter nbconvert --to script /opt/workshop/notebooks/lab.ipynb --stdout > /dev/null


# 7. Hero query smoke — embed and search the products collection's dense vector end-to-end
/opt/workshop/venv/bin/python -c "
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
client = QdrantClient(host='localhost', port=6333)
emb = next(iter(TextEmbedding('sentence-transformers/all-MiniLM-L6-v2').embed(['65 lg tv'])))
res = client.query_points(collection_name='products', query=list(emb), using='dense', limit=5).points
assert len(res) == 5, 'expected 5 results'
print('OK: dense smoke query returned 5 results')
"

# 8. Data files present
for f in hero_queries.json qrels_hero.json ood_queries.json products_10k.jsonl eval_subset_200.json eval_full_2k.json splade_vocab.json; do
  test -s /opt/workshop/data/$f || { echo "FAIL: missing data/$f"; exit 1; }
done

# 9. No outbound network needed — disable interface and re-run #1, #3, #7
sudo ip link set eth0 down
sleep 2
curl -fsS http://localhost:6333/healthz && curl -fsS http://localhost:6333/collections/products | jq -e '.result.points_count == 10000'
sudo ip link set eth0 up
```

All 9 must return 0. Save the verification log into the image at `/opt/workshop/BUILD_VERIFIED.txt` with timestamp and image hash.

---

## Sizing

| Resource | Min | Recommended | Notes |
|---|---|---|---|
| vCPU | 4 | 8 | SPLADE inference is CPU-bound; 8 cores keeps live cells <10s on the 200-query slice. |
| RAM | 8 GB | 16 GB | Qdrant collections in RAM (~2 GB) + SPLADE model resident (~1.5 GB) + headroom for notebook kernel. |
| Disk | 25 GB | 40 GB | OS ~8 GB · Qdrant storage ~3 GB · model cache ~1 GB · data ~50 MB · Docker layers + venv ~8 GB. SSD strongly preferred — sparse retrieval IO scales with disk speed. |
| Network (lab time) | none | none | offline by design |
| Network (build time) | required | required | HF + PyPI |

Target single-attendee VM. Do not multi-tenant.

---

## What attendees see

On first browser hit to the VM URL:

1. **Welcome screen** (rendered as the first JupyterLab cell output, also pinned in `/etc/motd` for ssh users):

   ```
   Fine-Tuning AI Search for E-commerce — Workshop VM
   ---------------------------------------------------
   Qdrant:       http://localhost:6333  (healthy)
   JupyterLab:   http://<vm-ip>:8888/lab
   Notebook:     lab.ipynb (already open)
   Models:       MiniLM, BM25, fine-tuned SPLADE — cached
   Collection:   products (dense + bm25 + splade_finetuned named vectors) — populated (10K points)

   Run the health-check cell to confirm everything is live, then follow the instructor.
   ```

2. **JupyterLab opens directly on `lab.ipynb`** (no file-browser detour).

3. **The notebook's dedicated health-check cell** connects to Qdrant on localhost, confirms the `products` collection has 10K points, and loads the SPLADE encoder offline. Expected output (one `[INFO]` debug line + three `[OK]` lines):

   ```
   [INFO] repo root: /opt/workshop
   [OK] Qdrant reachable at http://localhost:6333
   [OK] products collection populated (10,000 points)
   [OK] SpladeEncoder loads thierrydamiba/splade-ecommerce-esci offline
   ```

If any line is missing or the cell raises, the attendee raises a hand and the instructor escalates to the VM team. The lab has no fallback safety net — a non-ready state means we troubleshoot before the attendee runs further cells.

---

## Handoff

Deliverables to workshop owner:
- VM image ID + region(s)
- `BUILD_VERIFIED.txt` (verification log)
- `requirements.lock.txt`
- One-page changelog if rebuilding (pinned versions changed, model SHAs, dataset row count)
