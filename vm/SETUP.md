# Workshop Machine Setup

This document is the instructor/ops guide for preparing the workshop machines.
The intended flow is simple:

1. Create one normal Linux machine.
2. Install the repo, Python environment, Qdrant, models, and product index.
3. Verify the lab end to end.
4. Stop the machine and clone it once per attendee.

Each attendee should get a single-tenant clone. Do not run multiple attendees on one shared machine.

## Target State

Every cloned attendee machine should boot with:

- Qdrant running on `localhost:6333`
- Built-in IDE opens `/opt/workshop`
- `notebooks/lab.ipynb` ready to open in that IDE
- Python dependencies installed in `/opt/workshop/venv`
- Qdrant collection `products` already populated
- Models and ESCI dataset cached
- Generated files present:
  - `/opt/workshop/data/corpus_manifest.json`
  - `/opt/workshop/data/splade_vocab.json`

The product index is prebuilt. Retrieval results, metric tables, and bootstrap confidence intervals are computed live during the workshop.

## Recommended Machine

| Resource | Recommended |
|---|---|
| OS | Ubuntu 22.04 LTS x86_64 |
| vCPU | 8 |
| RAM | 16 GB |
| Disk | 40 GB SSD |
| Network | Required during setup and workshop |

The collection build is CPU-bound and can take a while because it encodes product text with SPLADE.

## Install System Dependencies

Run as an admin user on the source machine:

```bash
sudo apt-get update
sudo apt-get install -y \
  ca-certificates \
  curl \
  git \
  jq \
  python3.11 \
  python3.11-venv \
  python3.11-dev
```

Install Docker Engine using your provider's preferred method, then verify:

```bash
docker --version
```

## Create Workshop User

```bash
sudo useradd -m -s /bin/bash workshop || true
sudo usermod -aG docker workshop
sudo mkdir -p /opt/workshop
sudo chown -R workshop:workshop /opt/workshop
```

Log in as `workshop` for the remaining setup:

```bash
sudo -iu workshop
```

## Clone Repo

```bash
cd /opt
git clone <repo-url> workshop
cd /opt/workshop
```

If the repo is already copied onto the machine, make sure it is owned by `workshop`:

```bash
sudo chown -R workshop:workshop /opt/workshop
```

## Python Environment

```bash
cd /opt/workshop
python3.11 -m venv /opt/workshop/venv
source /opt/workshop/venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name ft-search-workshop --display-name "FT Search Workshop"
pip freeze > /opt/workshop/requirements.lock.txt
```

## Start Qdrant

Use a persistent local Docker volume directory:

```bash
mkdir -p /opt/workshop/qdrant_data
docker run -d \
  --name ft-search-qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -v /opt/workshop/qdrant_data:/qdrant/storage \
  qdrant/qdrant:v1.18.0
```

Verify Qdrant:

```bash
curl -fsS http://localhost:6333/healthz
curl -fsS http://localhost:6333/readyz
```

## Build Product Index

This is the slow step. Run it once on the source machine before cloning:

```bash
cd /opt/workshop
source /opt/workshop/venv/bin/activate
python scripts/setup_collections.py --recreate
```

This writes:

- Qdrant collection `products`
- `data/corpus_manifest.json`
- `data/splade_vocab.json`

It also warms the HuggingFace, FastEmbed, and dataset caches used by the lab.

## Built-In IDE

The workshop platform provides the IDE. Do not run a separate JupyterLab service unless the platform requires it. The important contract is that the IDE opens `/opt/workshop` and attendees can open `notebooks/lab.ipynb` from there.

Before cloning, open the built-in IDE once on the source machine and confirm:

- `/opt/workshop` is visible as the project folder
- `notebooks/lab.ipynb` opens
- the selected Python kernel points at `/opt/workshop/venv`

## Verification Checklist

Run these checks on the source machine before cloning.

```bash
# Qdrant health
curl -fsS http://localhost:6333/healthz
curl -fsS http://localhost:6333/readyz

# Collection matches manifest
EXPECTED=$(jq -r .corpus_product_count /opt/workshop/data/corpus_manifest.json)
curl -fsS http://localhost:6333/collections/products | jq -e --argjson expected "$EXPECTED" \
  '.result.status == "green" and .result.points_count == $expected'

# Python imports
/opt/workshop/venv/bin/python -c "
import qdrant_client, fastembed, transformers, torch, pandas, datasets
print('OK: imports')
"

# Required data files
for f in demo_queries.json bad_queries.json splade_vocab.json corpus_manifest.json; do
  test -s /opt/workshop/data/$f || { echo \"missing data/$f\"; exit 1; }
done

# Manifest reachability sample
/opt/workshop/venv/bin/python -c "
import json, uuid
from pathlib import Path
from qdrant_client import QdrantClient
m = json.loads(Path('/opt/workshop/data/corpus_manifest.json').read_text())
client = QdrantClient(host='localhost', port=6333)
ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f'esci:{pid}')) for pid in m['corpus_sample_product_ids']]
found = client.retrieve(collection_name='products', ids=ids, with_payload=False)
assert len(found) == len(ids), f'missing {len(ids)-len(found)} sample ids'
print(f'OK: reachability {len(found)}/{len(ids)}')
"

# Dense query smoke test
/opt/workshop/venv/bin/python -c "
from qdrant_client import QdrantClient, models
from scripts.setup_collections import COLLECTION, DENSE_MODEL, DENSE_VECTOR_NAME
client = QdrantClient(host='localhost', port=6333)
res = client.query_points(
    collection_name=COLLECTION,
    query=models.Document(text='65 lg tv', model=DENSE_MODEL),
    using=DENSE_VECTOR_NAME,
    limit=5,
    with_payload=True,
).points
assert len(res) == 5
print('OK: dense query returned 5 results')
"

# Notebook JSON parses
/opt/workshop/venv/bin/python -m json.tool /opt/workshop/notebooks/lab.ipynb >/dev/null
```

Then open the built-in IDE, open `notebooks/lab.ipynb`, and run the first Setup cell. Expected shape:

```text
[OK] Qdrant up at localhost:6333, products collection has ~38,000 points (matches manifest)
[OK] reachability spot-check: 50/50 sample products present
Loading ~2,000 ESCI eval queries from HuggingFace...
[OK] ESCI eval set: ~2,000 queries loaded
Ready. Corpus: ~38,000 products · 10 demo queries · 5 bad queries · ~2,000 eval queries.
```

## Pilot Run

Before cloning for attendees, run:

- `notebooks/pilot_verify.ipynb`
- the full `notebooks/lab.ipynb` flow with a stopwatch

Confirm:

- setup cell succeeds
- demo queries show the expected baseline failures
- SPLADE improves the focus demo queries
- bad queries return product results and support the query-routing discussion
- full 2K eval finishes within the workshop timing envelope

## Clone For Attendees

After verification:

1. Stop the source machine cleanly.
2. Clone/snapshot it using the hosting provider.
3. Create one clone per attendee.
4. Boot each clone and verify Qdrant plus IDE access:

```bash
curl -fsS http://localhost:6333/healthz
```

Then open the platform IDE for that clone and confirm `notebooks/lab.ipynb` opens.

Each clone should keep its own copy of `/opt/workshop/qdrant_data`, model caches, venv, and generated data files.

## Attendee Access

Give each attendee their own platform IDE link for their cloned machine. The notebook should already be available at `/opt/workshop/notebooks/lab.ipynb`.
