# Intro Deck — Slide-by-Slide Outline

Target length: 20 minutes · 13 slides · audience is mixed-technical, including some product managers. Map: slides 1–3 = hook (3 min), slides 4–9 = concepts one-liners (10 min), slides 10–11 = lab roadmap (4 min), slides 12–13 = VM check-in + transition (3 min). Each concept slide is hard-capped at ~90 seconds.

---

## Slide 1: Fine-Tuning AI Search for E-commerce
**Goal:** Set the room — who we are, what we'll do, and the promise.
**Bullets:**
- 80 minutes · 20 min intro · 60 min hands-on lab
- You'll measure a real lift on real metrics — not a toy demo
- Built on Amazon ESCI (1.2M query-product pairs, graded relevance)
- VM is already running; you just open the notebook
**Speaker notes:** Welcome, brief self-intro, frame the deal: this is hands-on and the numbers at the end will be real. Everyone runs the notebook; plain-language descriptions next to each metric keep results readable without prior IR background.
**Visual:** Title card with the workshop name, Qdrant logo, and a faint background screenshot of the JupyterLab notebook to signal "this is real work, not slides all the way down."

---

## Slide 2: The Hook — "65 lg tv"
**Goal:** Show the failure in one screen before any theory.
**Bullets:**
- Query: "65 lg tv"
- Baseline dense (all-MiniLM-L6-v2) — rank 1 is a 55-inch LG TV (Substitute)
- The Exact 65-inch LG TV is at rank 6
- This isn't a one-off — variant, color, size, brand model are exactly where generic dense embeddings consistently lose ranking signal. (Drawn from real ESCI test queries; baseline rank verified.)
**Speaker notes:** Pause on this slide. Let the room read it. The "Substitute" tag is the punchline — your retailer just sent the customer to the wrong-sized TV and they'll bounce. Say out loud: "this is what we're going to fix in the next 80 minutes." The query is one of 10 heroes attendees will run in CP1.
**Visual:** Mock-up of a search results page, e-commerce style: query bar at top with "65 lg tv" typed in. Rank 1 product card has a big red "Substitute · 55-inch" badge. Below it, rank 2/3/4 are also wrong (different sizes or brands). Way at the bottom, rank 6, a green "Exact · 65-inch LG" badge on the correct product. Stylistically a clean Amazon-store look — no Qdrant branding in the mock; it's the user's experience we're showing.

---

## Slide 3: Why this matters
**Goal:** Connect the failure mode to revenue and trust — the shared why.
**Bullets:**
- Search is a top conversion lever in e-commerce — wrong results at rank 1 cost real revenue
- A Substitute at rank 1 is worse than no result — it erodes trust
- Long-tail attributes (size, color, variant) are where embedding models silently fail
- Today: how to make retrieval honest about those attributes
**Speaker notes:** This slide is for everyone — revenue + trust framing matters whether you're building the search system or owning the product roadmap. Don't dwell — 45 seconds.
**Visual:** Three-icon row: a shopping cart with a downward arrow (conversion), a broken trust badge (trust), and a magnifying glass with a question mark (search quality). Minimal, infographic style.

---

## Slide 4: Vector search in one slide
**Goal:** Establish the mental model. Nothing more.
**Bullets:**
- Turn text into vectors · search by similarity, not keyword match
- Dense = one short, dense vector (semantic)
- Sparse = a long, mostly-zero vector (lexical-ish, interpretable)
- Both live in a vector database (Qdrant). Both retrievable in milliseconds.
**Speaker notes:** Concept one-liner. The point is to introduce "dense" and "sparse" as the two retrieval modalities the rest of the workshop will compare. Don't get pulled into ANN algorithms or HNSW — just the two modalities.
**Visual:** Side-by-side diagram. Left: a phrase "65 lg tv" arrow to a small dense rectangle labeled "[0.12, -0.04, 0.81, ...]" with caption "384 dims, all populated." Right: same phrase arrow to a tall sparse bar chart with five tall bars labeled "65, lg, tv, inch, smart" and the rest zero, captioned "30K dims, ~50 non-zero." Make sparse visibly interpretable.

---

## Slide 5: Sparse vs dense — what they actually capture
**Goal:** Make the trade-off concrete.
**Bullets:**
- Dense — semantic similarity ("running shoes" ≈ "sneakers"); blurs specifics
- Sparse — lexical precision ("65 inch" ≠ "55 inch"); blurs paraphrase
- Neither alone is enough for e-commerce
- Modern sparse models (SPLADE) learn term weights — not just TF-IDF
**Speaker notes:** This is the slide that justifies why we're going to fine-tune a sparse model rather than a dense one. SPLADE is sparse-but-learned, which is the sweet spot: keeps lexical precision *and* picks up synonyms. 90 seconds.
**Visual:** Two-column table. Header: "Dense" | "Sparse." Row 1 (Captures): "Meaning, paraphrase" | "Exact terms, attributes." Row 2 (Misses): "Variant numbers, sizes, colors" | "Synonyms, paraphrase." Row 3 (Example failure): "65 inch → 55 inch" | "'sneakers' ≠ 'running shoes'." Clean, no decoration.

---

## Slide 6: Hybrid — the obvious answer
**Goal:** Introduce fusion as the production-grade move.
**Bullets:**
- Run dense + sparse in parallel · fuse the two ranked lists
- Reciprocal Rank Fusion (RRF) — no weights to tune, just rank positions
- Qdrant's Query API does this in one request
- We'll measure two hybrid recipes — Dense+BM25 (the classic) and Dense+SPLADE (the workshop's pitch) — alongside the three solo approaches
**Speaker notes:** RRF is the default because it's parameter-free; weighted fusion exists but is a stretch goal. The point of this slide is that hybrid is not exotic — it's table stakes for production search. 60 seconds.
**Visual:** Funnel diagram. Two parallel lanes at the top (dense lane and sparse lane), each producing a ranked list of 10 products. Arrows converge into an "RRF" box at the bottom, producing a single fused top-10. Clean dataflow style.

---

## Slide 7: Fine-tuning — what it changes
**Goal:** Establish what "fine-tuned" means and why it matters here.
**Bullets:**
- Start from a base encoder (`distilbert-base-uncased`) and train the SPLADE sparse head from scratch on Amazon ESCI query-product pairs, using `sentence-transformers` `SparseEncoderTrainer`
- Model learns that "65 inch" matters and "smart" doesn't (when the customer typed a specific size)
- Result: a sparse retrieval model that speaks fluent e-commerce
**Speaker notes:** Use the screenshot to anchor "this is what produced the model you'll use today." Do NOT switch to the live training notebook — the screenshot is intentional. The takeaway notebook link is at the end of the deck. 90 seconds.
**Visual:** A screenshot mock-up of a Jupyter notebook cell showing a `sentence-transformers` `SparseEncoderTrainer` training loop: `for epoch in range(3):` ... `loss.backward()` ... `print(f"epoch {epoch} loss {loss.item():.4f}")` — with a TQDM progress bar at "1500/3200 [02:14<03:11]" and decreasing loss values 4.21 → 2.18 → 1.04 → 0.62 in the cell output. Right-side annotation: "Producing splade-ecommerce-esci · 3 epochs · hard negatives · ~6 minutes on a single A100 (100K-sample fine-tune from the article series)."

---

## Slide 8: Hard negatives — the trick that makes it work
**Goal:** One-line concept; honest that we won't do this live.
**Bullets:**
- Easy negatives: random products. Model learns "iPhone ≠ banana" — trivial.
- Hard negatives: products that look right but are wrong (55-inch vs 65-inch TV)
- Mined with the model's own predictions (ANCE-style, iterative)
- We won't run mining live — link in takeaways
**Speaker notes:** This is the "if you take one technique home" concept. Hard-negative mining is the single biggest lever in retrieval fine-tuning. We don't run it live because it's slow and not visually compelling — but we ship the notebook. 60 seconds.
**Visual:** Two stacked rows. Top row labeled "Easy negative" — query "65 lg tv" with a product card of a banana, big green check (trivially correct). Bottom row labeled "Hard negative" — same query with a 55-inch LG TV, big yellow exclamation mark (this is the one the model has to learn to push down). Caption: "Training on hard negatives is where the lift comes from."

---

## Slide 9: What we did to SPLADE for today
**Goal:** Concrete recipe for the model you'll be running.
**Bullets:**
- Base: `distilbert-base-uncased` with a SPLADE sparse head
- Data: Amazon ESCI, ~1.2M query-product pairs, graded labels (E/S/C/I)
- Hard negatives mined from baseline dense retrievals
- 3 epochs, ANCE-style refresh, ~6 minutes on a single A100 (100K-sample fine-tune in the article series) → `splade-ecommerce-esci`
**Speaker notes:** Quick recipe slide. Engineers will want this; PMs can tune out for 30 seconds. Emphasize that this is *not* magic — it's well-understood technique applied to the right dataset.
**Visual:** A "recipe card" infographic: ingredients (base model, dataset, hard negatives, epochs), procedure (4 steps), output (the fine-tuned model with its HF repo path as a badge at the bottom).

---

## Slide 10: The lab — three checkpoints
**Goal:** Roadmap. What they're about to do.
**Bullets:**
- **CP1 (12 min):** Baseline dense on 10 hero queries — qualitative, ESCI grades visible
- **CP2 (20 min):** Fine-tuned SPLADE — first nDCG@10 numbers (3-way: BM25 / baseline dense / fine-tuned SPLADE)
- **CP3 (13 min):** Hybrid fusion with RRF — two recipes side by side: Hybrid (D+BM25) and Hybrid (D+SPLADE)
- **Wrap (15 min):** Full 2K eval with CIs across all five approaches (BM25, Dense, SPLADE, Hybrid (D+BM25), Hybrid (D+SPLADE)) · "where SPLADE breaks" · Q&A
**Speaker notes:** Walk through each CP in ~20 seconds. Make clear the metric reveal lands in CP2 — CP1 is intentionally qualitative so attendees feel the failure modes before they see the numbers. CP3 introduces the second hybrid recipe so the wrap can compare all five approaches head-to-head. Total = 60 min hands-on.
**Visual:** A horizontal timeline with four blocks (CP1, CP2, CP3, Wrap) sized proportionally to their minute budgets. Each block has a one-line subtitle and an icon (eyeballs for CP1, gauge for CP2, fusion-symbol for CP3, trophy for Wrap).

---

## Slide 11: What you'll leave with — preview the lift
**Goal:** Show the destination so they know what they're chasing.
**Bullets:**
- nDCG@10: baseline ~0.32 → fine-tuned ~0.48 → hybrid ~0.52 (full 2K eval, 95% CI)
- Recall@10: comparable lift
- Quality comparison across all five approaches: BM25, Dense, SPLADE, Hybrid (D+BM25), Hybrid (D+SPLADE)
- Takeaway notebook: train your own on your own data
**Speaker notes:** Specific numbers create commitment. The room now has a target to verify. Note the CI — this is the authoritative claim, not the in-room 200-query subset. Mention briefly that the 200-query slice is for live feel only. The two hybrid rows are the punchline: same RRF call structure, different sparse leg.
**Visual:** A preview of the final wrap table: 5 rows in lineup order (BM25, Dense, SPLADE, Hybrid (D+BM25), Hybrid (D+SPLADE)), columns for nDCG@10 (with CI bars), Recall@10, and the plain-English translation under each metric. Numbers shown as placeholders ("0.32 ± 0.02") so the slide doesn't lock the team to an exact number before the eval is run. Caption: "You'll see this filled in for real at the end."

---

## Slide 12: VM check-in
**Goal:** Get every laptop into a working state in 3 minutes.
**Bullets:**
- Open the URL on your placard: `http://<vm-ip>:8888/lab`
- The notebook `lab.ipynb` is already open
- Run the **health-check cell** — should print one `[INFO]` debug line + three green `[OK]` lines:
  - `[OK]   Qdrant reachable at http://localhost:6333`
  - `[OK]   products collection populated (10K)`
  - `[OK]   SpladeEncoder loaded offline (thierrydamiba/splade-ecommerce-esci)`
- Raise your hand if any `[OK]` line is missing or replaced with `[FAIL]`
**Speaker notes:** Pause the slide and walk the room. The three `[OK]` lines are Qdrant health, the single `products` collection populated, and the fine-tuned SPLADE encoder loadable offline. Anything red goes to the VM team helper, not you. Move to slide 13 once ≥90% of hands are down.
**Visual:** Screenshot of the JupyterLab UI with the health-check cell expanded and its expected output visible (one `[INFO]` + three `[OK]` lines), with the URL bar of the browser highlighted in a callout. A small "raise hand" icon in the corner with text "anything red → flag a helper."

---

## Slide 13: Let's go — CP1 starts now
**Goal:** Transition out of slides into the notebook.
**Bullets:**
- Scroll to **Checkpoint 1** in `lab.ipynb`
- Run the hero queries one at a time
- Tag the failure modes you see in the comments cell
- 12 minutes — we share out at the end
**Speaker notes:** Stop talking. Start a visible 12-minute timer on the projector. Walk the room. The first 60 seconds always have a few people who need help finding CP1 — that's normal.
**Visual:** Big countdown timer placeholder ("12:00") with the CP1 section header as a screenshot at the bottom. Minimal text. The deck is done at this point — the rest of the workshop happens in JupyterLab.
