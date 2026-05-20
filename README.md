# Fine-Tuning AI Search for E-commerce — Workshop Repo

This repo contains the hands-on lab from the workshop. It compares five retrieval approaches for e-commerce product search:

- BM25
- Dense MiniLM
- Fine-tuned SPLADE
- Hybrid dense + BM25
- Hybrid dense + SPLADE

The lab uses Amazon ESCI relevance labels and Qdrant. Product-side vectors are built once into a local Qdrant collection; retrieval results, metric tables, and bootstrap confidence intervals are computed live when you run the notebook.

## Repo Layout

```
notebooks/      Main lab, training takeaway, and pilot verification notebook
slides/         Intro deck outline
eval/           Metrics, viewers, SPLADE encoder
scripts/        Local collection builder
data/           Curated query lists plus generated local build outputs
agenda.md       Public agenda
WORKSHOP.md     Detailed workshop plan
requirements.txt
```

## Prerequisites

- Python 3.11
- Docker
- Recommended: 16 GB RAM and enough time for the SPLADE product-encoding pass

The collection build indexes roughly 20K-30K products. On CPU, SPLADE product encoding can take a while. That is normal.

## Local Setup

Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start Qdrant:

```bash
docker run -d \
  --name ft-search-qdrant \
  -p 6333:6333 \
  -v "$PWD/qdrant_data:/qdrant/storage" \
  qdrant/qdrant:v1.12.4
```

Build the local product collection:

```bash
python scripts/setup_collections.py --recreate
```

That command loads the ESCI test split, selects the deterministic 2K-query eval set, indexes every product referenced by those queries, and writes:

- `data/corpus_manifest.json`
- `data/splade_vocab.json`
- Qdrant collection `products`

## Run The Lab

Open `notebooks/lab.ipynb` in your IDE or notebook environment.

Select the Python environment you created above, then run the Setup cell first. It checks Qdrant, validates `data/corpus_manifest.json`, loads ESCI qrels, and prepares the demo queries.

## Data Files

Committed:

- `data/demo_queries.json` — 10 curated product-search demo queries
- `data/bad_queries.json` — 5 bad queries for the query-routing demo

Generated locally:

- `data/corpus_manifest.json` — selected eval query IDs, expected product count, reachability sample
- `data/splade_vocab.json` — tokenizer vocab for sparse-vector inspection
- `qdrant_data/` — local Qdrant storage

## Training Takeaway

`notebooks/splade_training.ipynb` is the self-study notebook for how the fine-tuned SPLADE model was produced. It is not required for the main lab.

The training notebook is illustrative and expects more than the basic lab environment. For a real run, use a GPU and install its extra dependencies, such as `sentence-transformers`, `matplotlib`, and `huggingface_hub`.

## Useful Commands

Rebuild the product collection from scratch:

```bash
python scripts/setup_collections.py --recreate
```

Use a smaller product cap for a quick smoke test:

```bash
python scripts/setup_collections.py --recreate --limit 500
```

Run the pilot verification notebook after building the collection:

Open `notebooks/pilot_verify.ipynb` in your IDE or notebook environment.
