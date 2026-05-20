# Fine-Tuning AI Search for E-commerce — Workshop Repo

An 80-minute hands-on workshop comparing five retrieval approaches for e-commerce product search: BM25, dense (MiniLM), fine-tuned SPLADE, and two hybrid recipes (RRF over dense+BM25 and dense+SPLADE). Source material is Qdrant's `sparse-embeddings-ecommerce` 5-part article series; ground truth is Amazon ESCI.

Content authorities:
- [`agenda.md`](agenda.md) — public-facing agenda
- [`WORKSHOP.md`](WORKSHOP.md) — detailed plan (timing, checkpoints, audience handling)

## Repo layout

```
notebooks/      Lab notebook + self-study training notebook + instructor pilot-verify notebook
slides/         Intro deck outline (13 slides, 20 min)
eval/           Metrics, viewers, SPLADE encoder — imported by the lab
scripts/        VM-bake-time data + collection builders
data/           Curated query lists (committed); build outputs (gitignored, produced by scripts/)
vm/             HackerSquad VM build spec
agenda.md       Public agenda
WORKSHOP.md     Detailed workshop plan
requirements.txt
```

## Data artifacts

`data/` ships two curated files only. The rest are build outputs that must be produced before the workshop runs.

| File | Committed? | Source | Notes |
|---|---|---|---|
| `data/hero_queries.json` | yes | hand-curated | 10 hero queries with failure-mode tags |
| `data/ood_queries.json` | yes | hand-curated | 5 OOD queries for the wrap demo |
| `data/products_10k.jsonl` | no | curated 10K-product subset of ESCI (top-10K most-frequent products in the test split) | ~2 MB; produced by `scripts/curate_heroes.py` |
| `data/splade_vocab.json` | no | DistilBERT tokenizer vocab keyed by token id | produced by `scripts/setup_collections.py` |
| `data/qrels_hero.json` | no | ESCI grades for the 10 heroes, filtered to indexed corpus | produced by `scripts/build_eval_data.py` |
| `data/eval_subset_200.json` | no | 200-query stratified sample, qrels filtered to indexed corpus | produced by `scripts/build_eval_data.py` |
| `data/eval_full_2k.json` | no | full ~2K-query ESCI test set, qrels filtered to indexed corpus | produced by `scripts/build_eval_data.py` |

All qrels are filtered to product_ids that exist in `products_10k.jsonl`. Products outside the indexed subset are physically unreachable, so counting them would corrupt Recall@10 (inflated denominator) and nDCG@10 (ideal DCG over unreachable relevances).

## Build order

The four build steps must run in this order — each depends on the previous artifact.

```bash
# 1. Curate heroes + products from ESCI.
#    Loads ESCI test split, embeds queries + products with MiniLM, runs
#    baseline retrieval, picks 10 hero queries where MiniLM demonstrably
#    buries the Exact-grade product. Writes:
#      - data/hero_queries.json  (10 heroes with esci_qid set)
#      - data/products_10k.jsonl (top-10K most-frequent products in test split)
#      - data/_candidates_debug.json (full candidate pool, for iteration)
#
#    First run picks heroes by category. To hand-curate (recommended):
#    run once, inspect data/_candidates_debug.json, then re-run with
#    --reuse-candidates and --include-qids to pin specific stories.
python scripts/curate_heroes.py --candidates-debug data/_candidates_debug.json
# Optional second pass with hand-picked qids:
python scripts/curate_heroes.py \
    --reuse-candidates data/_candidates_debug.json \
    --include-qids "5021,9719,96622,9912,1199,1703,3225,111097,93904,104623"

# 2. Stand up Qdrant + index all three named vectors. Also writes data/splade_vocab.json.
python scripts/setup_collections.py --recreate

# 3. Build qrels + eval splits. Requires data/products_10k.jsonl so qrels can be
#    filtered to the indexed corpus. Hard-fails if any hero query has no
#    Exact-grade product inside the indexed subset.
python scripts/build_eval_data.py --out-dir data/
```

After step 3 you have all seven files in `data/`. The lab notebook reads them at runtime.

## Local dev

The lab notebook expects Qdrant on `localhost:6333` and reads paths relative to its parent directory (so it works the same locally and on the VM).

```bash
# Qdrant (docker)
docker run -d -p 6333:6333 -v "$PWD/qdrant_data:/qdrant/storage" qdrant/qdrant:v1.12.4

# Python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build data (see "Build order" above)

# Open the lab — launch JupyterLab from inside notebooks/ so the kernel's cwd
# is notebooks/ and Path.cwd().resolve().parent resolves to the repo root.
cd notebooks && jupyter lab lab.ipynb
```

On the VM the equivalent contract is: notebook at `/opt/workshop/notebooks/lab.ipynb`, JupyterLab launched with `--notebook-dir=/opt/workshop/notebooks`. Same `Path.cwd().resolve().parent` resolution.

## VM build

See [`vm/SETUP.md`](vm/SETUP.md). Deliverable: a HackerSquad VM that boots into JupyterLab on `lab.ipynb` with Qdrant healthy, the `products` collection populated (10K points, three named vectors), and all models cached for offline use.

## Workshop materials

- **Slides** — `slides/intro_outline.md` (13 slides, 20 min)
- **Main lab** — `notebooks/lab.ipynb` (60 min, three checkpoints)
- **Self-study takeaway** — `notebooks/splade_training.ipynb` (training pipeline; not run in the room)
- **Instructor pilot QA** — `notebooks/pilot_verify.ipynb` (pre-pilot smoke tests)
