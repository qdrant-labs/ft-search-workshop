# Fine-Tuning AI Search for E-commerce — Workshop

## Context

An 80-minute hands-on workshop for e-commerce AI practitioners (search engineers, ML engineers, data scientists, plus some product managers). The goal is to show why generic embeddings break for real product search and how a fine-tuned retrieval model — specifically a fine-tuned SPLADE model — closes the gap, with numbers attendees can believe.

Delivered on HackerSquad VMs that come pre-loaded with Qdrant, models, and embeddings, so the lab time is spent on comparisons and measurement rather than setup.

Source material: Qdrant's `sparse-embeddings-ecommerce` 5-part article series. The `agenda.md` in this repo is the authority for content; the articles supply the dataset (Amazon ESCI), the fine-tuned model, and the failure-mode examples.

## Decisions confirmed

- **Total time:** 80 min — 20 min intro, 60 min hands-on.
- **Narrative arc:** baseline dense → fine-tuned SPLADE → hybrid fusion.
- **Headline takeaway:** quantified lift on real metrics. Attendees see fine-tuned beat baseline on nDCG@10 / Recall@10 on Amazon ESCI and believe the numbers.
- **Lab style:** three checkpoints with mini-exercises, structured but interactive.
- **Audience handling:** mixed-technical room (search engineers, ML engineers, data scientists, some product managers) following a **single path through the notebook** — no two-track engineer-leads / PM-watches split. To keep results readable for attendees without an IR background, every metric is paired with a **plain-language description** ("ranking quality 0.39 / 1.0 — higher = better products closer to rank 1") next to the raw number. Concept depth in the intro stays at one-liners; deeper concepts deferred to a future workshop.
- **Dataset:** Amazon ESCI (1.2M query-product pairs, 2K test queries, 10K products) — graded relevance labels (E/S/C/I) used both in nDCG@10 *and* surfaced visually in result rows.
- **Models compared:** **BM25** (FastEmbed `Qdrant/bm25` sparse) · baseline **Dense** (`all-MiniLM-L6-v2` via FastEmbed) · fine-tuned **SPLADE** (`thierrydamiba/splade-ecommerce-esci`) · **Hybrid (D+BM25)** — dense + BM25 via server-side RRF · **Hybrid (D+SPLADE)** — dense + fine-tuned SPLADE via server-side RRF. Comparing both hybrid recipes isolates *which sparse signal* helps when fused with dense, not just whether hybrid wins.
- **VM state at minute 0:** full pre-load — Qdrant + one `products` collection (with `dense`, `bm25`, and `splade_finetuned` named vectors per point) populated + models cached + notebook open + out-of-domain query bundle ready.
- **Training notebook:** **screenshot of the training loop** on the intro "fine-tuning" slide; the full notebook is a takeaway link (no live notebook switching during the intro).
- **Hybrid fusion:** RRF as the default (no weights to tune); weighted fusion as a stretch goal for fast finishers.
- **Notebook environment:** JupyterLab on the VM.
- **Eval rigor — full 2K is the authoritative claim, computed LIVE in the wrap.** The wrap cell runs all five approaches over the 2K ESCI test queries with a progress bar (**~2–6 min on the VM**; the exact range depends on CPU and is verified at pilot) and computes **bootstrap 95% CIs**; these are the numbers attendees watch materialize. The 200-query subset earlier in the lab is for "feel," not the authoritative claim. No precomputed eval JSON.
- **No precomputed artifacts.** The lab runs fully live against the pre-configured VM. Every number attendees see materializes during the workshop — no cached fallback data and no precomputed eval JSON. If a cell fails the instructor recovers verbally.
- **"Wrong tool for the job":** mandatory ~3 min in the wrap showing out-of-domain queries returning product results — reframed as a **corpus-mismatch / query-routing** lesson, not a "model is broken" or "catastrophic forgetting" claim. The product retrieval system returns products; non-shopping intent needs upstream routing or a separate index. Honest framing of domain specialization.
- **Curated query set:** 10 hero queries drawn from real ESCI test queries via `scripts/curate_heroes.py`, with verified baseline failure (rank-of-first-Exact ≥ 4) and a documented mix of failure modes — electronics (65" LG TV → 55"; Apple iPhone 11 case → wrong-model case), home (10 gallon fish tank → 5 gallon; 11-piece knife set → 7-piece), apparel (size 8 girls pants → size 6). No beauty hero in this set — ESCI's beauty queries didn't survive the specificity filter; acknowledged gap. `scripts/verify_splade_wins.py` confirms SPLADE recovers 7 of 10 into top-3; the 3 it doesn't recover stay in the set so aggregate metrics reflect reality (no cherry-picking). Explicitly framed in the lab as **illustrative demonstrations of failure modes**, not evidence of aggregate performance — the aggregate claim comes only from the full 2K eval.
- **CP1 metrics:** qualitative-only on the aggregate. ESCI grade (E/S/C/I) is shown on every result row so attendees feel the failure modes concretely without seeing nDCG yet. First aggregate metric lands in CP2.
- **"Your turn":** stretch slot in the wrap if time permits, not a dedicated lab section.
- **Skip in the room:** live training, hard-negative mining (linked as self-study).

## Workshop structure

### Intro — 20 min

| Time | Section | Content |
|---|---|---|
| 3 min | **Hook** | Live failure demo: "65 lg tv" query on baseline dense returns the 55-inch variant at rank 1; the 65-inch (Exact) is buried at rank 6. One slide, one bad result, no math yet. |
| 10 min | **Concepts (one-liners)** | Vector search → sparse vs dense → hybrid/fusion → fine-tuning → hard negatives. Each gets ~2 min max; full depth deferred to a future workshop. The "fine-tuning" slide shows a **screenshot** of the training loop ("this is what produces the model we'll use today") with a link to the takeaway notebook — no live notebook switching. |
| 4 min | **Lab roadmap** | What the three checkpoints will show; preview the lift numbers so attendees know what they're chasing. |
| 3 min | **VM check-in** | Log in, run cell 1, confirm Qdrant is up and collections are populated. |

### Hands-on — 60 min

| Time | Checkpoint | What attendees do |
|---|---|---|
| 12 min | **CP1 — Baseline dense (qualitative)** | Run the ~10 diversified hero queries against the `products` collection's `dense` named vector. **No aggregate metric yet** — pure "look at the results." Each result row shows its ESCI grade (E/S/C/I) so attendees see that baseline often returns a Substitute at position 1 with the Exact buried at position 7. Attendees label the failure modes they spot. Brief share-out. |
| 20 min | **CP2 — Fine-tuned SPLADE (first numbers)** | Same queries against the `products` collection's `splade_finetuned` named vector. Side-by-side two-way view (baseline dense + fine-tuned SPLADE) with ESCI grades on each row. Inspect one fine-tuned sparse vector — show its active terms (sparse is interpretable; dense is opaque). **First metric reveal:** nDCG@10 + Recall@10 across **BM25, baseline dense, and fine-tuned SPLADE** on the 200-query slice, with a plain-language description under each number. BM25 anchors the lexical baseline. |
| 13 min | **CP3 — Hybrid fusion (two recipes)** | Use Qdrant's Query API to fuse with RRF in a single `query_points` call — two `Prefetch` blocks + `FusionQuery(fusion=Fusion.RRF)`. Compare TWO hybrid recipes on the 200-query slice: **Hybrid (D+BM25)** — the classic production recipe — vs **Hybrid (D+SPLADE)** — the workshop's pitch. Re-run quality metrics **5-way** (BM25 / Dense / SPLADE / Hybrid (D+BM25) / Hybrid (D+SPLADE)). The comparison isolates *which sparse signal* benefits most from dense fusion. Stretch goal: try weighted fusion. |
| 15 min | **Wrap, "wrong tool" demo, Q&A** | Final metrics table — **BM25 / Dense / SPLADE / Hybrid (D+BM25) / Hybrid (D+SPLADE)** · nDCG@10 with 95% CI from the full 2K eval / Recall@10 — with plain-language descriptions next to each quality number. No latency or cost columns; the workshop stays focused on quality (latency is a Q&A topic, not a column). **Wrong-tool demo (~3 min):** five out-of-domain queries against fine-tuned SPLADE return product results regardless of intent — corpus-mismatch story, not a "model is broken" story. Mention query routing + cross-encoder rerank as the "next layer" in Q&A. Stretch "your turn" if time. Preview the "fine-tune your own" path with the takeaway notebook. |

## VM setup (target state at minute 0)

- Qdrant running and healthy, with one `products` collection populated from the ESCI catalog (10K products). Every point carries three named vectors:
  - `dense` — `all-MiniLM-L6-v2` via FastEmbed (384-dim cosine)
  - `bm25` — sparse BM25 via FastEmbed `Qdrant/bm25`, `Modifier.IDF` (lexical sanity check)
  - `splade_finetuned` — `thierrydamiba/splade-ecommerce-esci` via `transformers`
- All embedding models pre-downloaded and cached (no network during the lab).
- JupyterLab pre-opened with the main lab notebook as the active tab.
- Held-out query set (~200 queries with ESCI relevance labels) plus the full 2K test set available locally.
- **Out-of-domain query bundle** (~5 queries — web/general knowledge style, e.g., MS MARCO-style) for the "where it breaks" demo in the wrap.
- Eval helper module imported and warm:
  - `ndcg_at_k`, `recall_at_k` — quality metrics
  - `compare_results(query, models=[...])` — side-by-side result viewer with ESCI grade per row
  - `explain_metric(metric_name, value)` — renders a human-readable string next to each metric
- Curated diversified hero query list (~10 queries) at the top of the notebook, tagged by category (electronics, apparel, home, beauty).
- Takeaway folder: the self-study training notebook (`splade_training.ipynb`), the lab notebook itself (which contains the live 2K eval cell with bootstrap CIs), and links to the article series + the fine-tuned HuggingFace model.

## Materials produced

- Slide deck for intro (built from this document via Claude Design), including a training-loop screenshot on the fine-tuning slide.
- Main lab notebook — the three-checkpoint flow with the diversified hero query set, ESCI-grade-aware result viewer, and plain-language metric descriptions.
- Self-study notebook — SPLADE training pipeline (hard negatives, ANCE-style, multi-epoch). Shipped as a takeaway link; not opened in the room.
- Eval helper module (nDCG@10, Recall@k, side-by-side viewer with grades, plain-language metric descriptions).
- Out-of-domain query bundle for the "where it breaks" demo.

## Verification (how we know it works)

- **Dry-run on a clean VM, end-to-end, with a stopwatch.** Three checkpoints + wrap should hit 60 min for an experienced attendee; build slack for variable pace.
- **Confirm the notebook's hero-query verification cell asserts the Exact-grade SKU is below rank 3 on each hero query** (across electronics, home, apparel). The failure demo is load-bearing — if MiniLM happens to surface the Exact within the top 3 for one query in the actual Qdrant/HNSW retrieval (vs the brute-force cosine `scripts/curate_heroes.py` used), replace that query. Also run `scripts/verify_splade_wins.py` to confirm SPLADE recovers each Exact into top-3 on the focus heroes — otherwise the CP2 lift demo dies. "65 lg tv" → 55-inch is the highest-stakes verification (matches the slide hook + CP1/CP2/CP3 through-line).
- **Confirm the two-way lift is visible:** baseline → fine-tuned SPLADE should show a meaningful jump on the 200-query subset. If the gap is too small, either expand the subset or revisit query selection.
- **Confirm the "where it breaks" demo actually breaks.** Run the out-of-domain queries against fine-tuned SPLADE and confirm at least one returns visibly absurd results — otherwise pick different queries.
- **Confirm live eval time on the 200-query subset is under ~10s** for each model, and the full 5-way CP3 eval completes in well under 90s.
- **Confirm ESCI grades render correctly** for in-test-set queries and that out-of-test queries display as "ungraded" rather than crashing.
- **Pilot with 2–3 friendly attendees of mixed skill level** before the public run — at least one engineer and one PM.
- **Verify the full 2K eval shows a statistically robust win** (bootstrap 95% CI between models). The full 2K is the authoritative claim; the 200-query subset is only for live feel.
- **Verify every live cell runs cleanly end-to-end on the VM.** No fallback safety net — pilot runs are the only validity gate.
- **Verify hero queries are introduced in the lab as illustrative** — not as the basis for the quality claim.

## Risks and how they're handled

Findings from the Codex adversarial review and the resolutions:

- **Overscoping risk in the 60-min lab.** Decision: keep 3 CPs and trust the pre-configured VM (no fallback system). Mitigation: an explicit "what gets cut on the day" priority order:
  1. Stretch "your turn" in wrap
  2. Weighted-fusion stretch in CP3
  3. Sparse-vector inspection in CP2 (2 min → 30s)
  4. "Where it breaks" demo (3 min → 1-min mention)

  Pilot run is the validity gate; if timing slips by >10%, escalate to a 2-CP merge (CP2 + CP3).
- **Mixed-technical audience.** Decision: single-track lab with plain-language descriptions next to each metric. No engineer-leads / PM-watches split — every attendee follows one notebook path. The IR-jargon barrier is handled by the metric descriptions and the ESCI grade tags on every result row.
- **Lexical baseline gap.** Decision: BM25 included as a retrieval approach in the metric reveals (CP2, CP3, wrap). Cross-encoder reranker intentionally deferred to a follow-up; mentioned in Q&A as "the next layer" of the production stack.
- **Statistical rigor on 200-query subset.** Full 2K with bootstrap 95% CI is the authoritative claim; the 200-query subset is the in-room "feel" only, labeled as a fast approximation.
- **Cherry-picking perception on hero queries.** Queries labeled illustrative in the lab; aggregate claim rests on the full 2K eval.
- **Live failure brittleness.** Decision: no fallback system. The lab runs live against the pre-configured VM; the instructor handles any cell failure verbally. Cached/fake data would contradict the "real metrics" headline. The pilot run on a clean VM is the only validity gate.
