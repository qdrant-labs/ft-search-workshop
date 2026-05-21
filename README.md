# Fine-Tuning AI Search for E-commerce — Workshop Repo

This repo contains the hands-on lab from the workshop. It compares four retrieval approaches for e-commerce product search:

- BM25
- Generic dense MiniLM
- Fine-tuned SPLADE
- Hybrid generic dense + SPLADE

The lab uses Amazon ESCI relevance labels and Qdrant. Product-side vectors are built once into a local Qdrant collection; retrieval results, metric tables, and bootstrap confidence intervals are computed live when you run the notebook.

## Repo Layout

```
notebooks/      Main lab, training takeaway, and pilot verification notebook
slides/         Intro deck outline
eval/           Metrics and result viewers
retrieval/      Retrieval model helpers
scripts/        Local collection builder
data/           Curated query lists plus generated local build outputs
agenda.md       Public agenda
WORKSHOP.md     Detailed workshop plan
requirements.txt
```

## Prerequisites

- Python 3.12. Confirm with `python3.12 --version`.
- Docker
- Recommended: 16 GB RAM
- Set aside one to two hours for the SPLADE product-encoding

The collection build indexes roughly 35K-40K products. On CPU, SPLADE product encoding can take a while. That is normal.

On macOS, install Python 3.12 with `brew install python@3.12` if `python3.12` is not already available.

## Local Setup

Create a Python environment and install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install -r requirements.txt
```

Avoid Python 3.14 for this repo for now: the current FastEmbed/ONNX Runtime stack can segfault during local indexing on macOS.

Start Qdrant:

```bash
docker run -d \
  --name ft-search-qdrant \
  -p 6333:6333 \
  -v "$PWD/qdrant_data:/qdrant/storage" \
  qdrant/qdrant:v1.18.0
```

Build the local product collection:

```bash
python scripts/setup_collections.py --recreate
```

That command loads the ESCI test split (labeled product-search queries with Exact/Substitute/Complement/Irrelevant grades), selects the deterministic 2K-query eval set, indexes every product referenced by those queries, and writes:

- `data/corpus_manifest.json`
- `data/splade_vocab.json`
- Qdrant collection `products`

`data/splade_vocab.json` is only for notebook inspection. SPLADE sparse vectors are stored as token IDs plus weights; this file maps those token IDs back to readable tokens so the lab can show which terms fired for a query. Qdrant does not need this file for retrieval.

This corpus construction is workshop-specific. We use a subset of ESCI and index only products with relevance labels for the selected eval queries so every graded product is reachable during the lab. In a normal production workflow, you would index your full product catalog first, then evaluate against query logs, judgments, clicks, or other relevance data.

## Run The Lab

Open `notebooks/lab.ipynb` in your IDE or notebook environment.

Select the Python environment you created above, then run the Setup cell first. It connects to Qdrant, loads ESCI qrels, and prepares the demo queries.

## Data Files

Committed:

- `data/demo_queries.json` — 10 curated product-search demo queries
- `data/bad_queries.json` — 5 bad queries for the query-routing demo

Generated locally:

- `data/corpus_manifest.json` — selected eval query IDs and build metadata
- `data/splade_vocab.json` — SPLADE token ID to token text map for sparse-vector inspection
- `qdrant_data/` — local Qdrant storage

## Training Takeaway

`notebooks/splade_training.ipynb` is the self-study notebook for how the fine-tuned SPLADE model was produced. It is not required for the main lab.

The training notebook is illustrative and expects more than the basic lab environment. For a real run, use a GPU and install these extra dependencies:

```bash
python -m pip install sentence-transformers accelerate pyarrow matplotlib huggingface_hub
```

## Useful Commands

Rebuild the product collection from scratch:

```bash
python scripts/setup_collections.py --recreate
```

Use a smaller product cap for a quick smoke test:

```bash
python scripts/setup_collections.py --recreate --limit 500
```
