# Intro Deck — Slide-by-Slide Outline

Target length: 20 minutes · 13 slides · audience is mixed-technical, including some product managers. Map: slides 1–3 = hook (3 min), slides 4–9 = concepts one-liners (10 min), slides 10–11 = lab roadmap + dataset setup (4 min), slides 12–13 = machine check-in + transition (3 min). Each concept slide is hard-capped at ~90 seconds.

---

## Slide 1: Fine-Tuning AI Search for E-commerce
**Goal:** Set the room — who we are, what we'll do, and the promise.
**Bullets:**
- 80 minutes · 20 min intro · 60 min hands-on lab
- You'll measure a real lift on real metrics — not a toy demo
- Built on Amazon ESCI (1.2M query-product pairs, graded relevance)
- Workshop machine is already prepared; you just open the notebook in the built-in IDE
**Speaker notes:** Welcome, brief self-intro, frame the deal: this is hands-on and the numbers at the end will be real. Everyone runs the notebook; plain-language descriptions next to each metric keep results readable without prior IR background.
**Visual:** Title card with the workshop name, Qdrant logo, and a faint background screenshot of the notebook in the IDE to signal "this is real work, not slides all the way down."

---

## Slide 2: The Hook — "65 lg tv"
**Goal:** Show the failure in one screen before any theory.
**Bullets:**
- Query: "65 lg tv"
- Generic dense (all-MiniLM-L6-v2) — rank 1 is a 55-inch LG TV (Substitute)
- The Exact 65-inch LG TV is at rank 6
- This isn't a one-off — variant, color, size, brand model are exactly where generic dense embeddings consistently lose ranking signal. (Drawn from real ESCI test queries; rank verified.)
**Speaker notes:** Pause on this slide. Let the room read it. The "Substitute" tag is the punchline — your retailer just sent the customer to the wrong-sized TV and they'll bounce. Say out loud: "this is what we're going to fix in the next 80 minutes." The query is one of 10 demo queries attendees will run in CP1.
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

## Slide 4: The Search Stack In One Picture
**Goal:** Establish the moving parts before naming algorithms.
**Bullets:**
- Query text and product text are converted into vectors
- Qdrant stores the vectors plus product payloads
- At search time: encode the query → retrieve nearest products → rank the top results
- Today we compare three signals: dense, BM25 sparse, and fine-tuned SPLADE sparse
**Speaker notes:** This is the map for the rest of the intro. Keep it concrete: embedding models produce vectors; Qdrant indexes and searches them; relevance labels let us measure whether the retrieved products were good. Do not explain every term yet — this slide is the glossary anchor.
**Visual:** Pipeline diagram: "query" and "products" go through "embedding models" into "Qdrant index"; a search arrow returns ranked product cards with E/S/C/I badges.

---

## Slide 5: Dense Embeddings
**Goal:** Explain dense vectors for non-IR attendees.
**Bullets:**
- A dense embedding is a short list of numbers representing meaning
- Similar meanings land near each other: "running shoes" ≈ "sneakers"
- Great for paraphrase and broad semantic similarity
- Risk in e-commerce: it can blur exact attributes like size, model, color, and compatibility
**Speaker notes:** Use "coordinates on a map" as the mental model. Dense is powerful because it handles language variation, but it is not naturally strict about every typed attribute. The hook failure is exactly this: 65-inch and 55-inch TVs are semantically close but commercially different.
**Visual:** 2D map illustration with product cards clustered by meaning. "65-inch LG TV" and "55-inch LG TV" appear close together, with a warning badge: "close in embedding space, wrong for the shopper."

---

## Slide 6: HNSW — How Dense Vector Search Is Fast
**Goal:** Explain nearest-neighbor search without turning the workshop into an algorithms lecture.
**Bullets:**
- Brute force would compare the dense query vector to every dense product vector
- HNSW builds a navigable graph of nearby vectors
- Search walks the graph toward better neighbors instead of scanning everything
- Trade-off: very fast approximate search, with tunable recall / latency
**Speaker notes:** Keep this to 90 seconds. HNSW is the dense-vector index, not the embedding model. The model decides where points live; HNSW is the shortcut Qdrant uses to find nearby dense points quickly. Sparse vectors use a different retrieval path, which is why the next slide separates dense and sparse. Avoid graph construction details like `M` and `ef` unless asked.
**Visual:** Layered graph diagram: top layer has a few long-range links, bottom layer has many local links. A query marker walks from a far product to closer products, ending near the target cluster.

---

## Slide 7: Sparse Embeddings
**Goal:** Explain BM25 and SPLADE as sparse signals.
**Bullets:**
- Sparse vectors are mostly zeros; active dimensions correspond to vocabulary terms
- BM25 sparse = classic lexical signal from exact word overlap and rarity
- SPLADE sparse = neural term weights and learned term expansion
- Sparse is interpretable: you can inspect which terms fired
**Speaker notes:** This is where "sparse" becomes concrete. BM25 and SPLADE both produce sparse vectors in Qdrant, but BM25 is formula-based while SPLADE is model-based. A good example: "10 gallon fish tank" can activate literal terms like `10`, `gallon`, `fish`, `tank`; SPLADE may also activate learned related terms like `aquarium`.
**Visual:** Three-row comparison: query text → BM25 terms → SPLADE terms. BM25 row only shows literal tokens; SPLADE row shows literal tokens plus learned expansions with weighted bars.

---

## Slide 8: Hybrid Search
**Goal:** Introduce fusion as the production-grade move.
**Bullets:**
- Run dense + sparse retrieval in parallel
- Fuse the lists with Distribution-Based Score Fusion (DBSF)
- Dense catches semantic matches; sparse protects exact attributes
- We'll build generic dense + SPLADE in the lab
**Speaker notes:** DBSF normalizes each retriever's scores before adding them together. The important idea is that hybrid search is not a hack; it is a practical way to use multiple retrieval signals. Qdrant's Query API runs generic dense + SPLADE fusion as one request in the lab. Dense + BM25 is also common in production, but it is not part of the main lab path.
**Visual:** Two parallel lanes at the top (dense lane and sparse lane), each producing a scored list of product cards. Arrows converge into a "DBSF" box, then one fused top-10 list.

---

## Slide 9: How This SPLADE Model Was Trained
**Goal:** Concrete recipe for the model attendees will use.
**Bullets:**
- Base: `distilbert-base-uncased` with a SPLADE sparse head
- Data: Amazon ESCI query-product pairs with E/S/C/I relevance labels
- Hard negatives: wrong products that look plausible, like 55-inch TV for "65 lg tv"
- Training teaches term weights and expansions that matter for e-commerce retrieval
- Output: `thierrydamiba/splade-ecommerce-esci`
**Speaker notes:** This is not a live-training workshop. The intro needs enough detail for trust: labels tell the model what "good" means, hard negatives teach it what subtle failure looks like, and fine-tuning changes the sparse term weights. Mention that `notebooks/splade_training.ipynb` is the takeaway for people who want the training loop.
**Visual:** Recipe card: base model + ESCI labels + hard negatives + training loop → fine-tuned SPLADE model. Include one hard-negative mini-example: query "65 lg tv"; positive = 65-inch LG TV; hard negative = 55-inch LG TV.

---

## Slide 10: The Lab Dataset
**Goal:** Explain ESCI and the workshop subset before attendees see the notebook.
**Bullets:**
- ESCI = Amazon product-search queries paired with candidate products and relevance grades
- Grades: Exact / Substitute / Complement / Irrelevant
- For the lab we use 2,000 US test queries and index only products that appear in their labeled rows
- Result: ~38,000 indexed products, all graded products reachable during evaluation
- Full US test would be ~360,000 products: ~10 hours of indexing and ~16 GB RAM per attendee
- Production difference: normally you index the full catalog first, then evaluate against labels/clicks/logs
**Speaker notes:** This is the place to defuse confusion about the subset. We are not saying production teams should build catalogs from labels. We are making a workshop-sized corpus where the labels and indexed products line up, so every metric can be computed live and explained clearly. Keep "qrels" as a parenthetical only if needed: qrels are the labeled query-product rows.
**Visual:** Simple data-flow: ESCI test split → filter US → choose 2,000 eval queries → collect labeled products → Qdrant products collection (~38K). Add a small side note: "Workshop construction, not production indexing pattern."

---

## Slide 11: The lab — three checkpoints
**Goal:** Roadmap. What they're about to do.
**Bullets:**
- **CP1 (12 min):** BM25 baseline on 10 demo queries — qualitative, ESCI grades visible
- **CP2 (20 min):** BM25, generic dense, and fine-tuned SPLADE — first metrics on the demo set
- **CP3 (13 min):** Hybrid fusion with DBSF — generic dense + fine-tuned SPLADE
- **Wrap (15 min):** Full 2K eval with CIs across four approaches (BM25, Generic dense, SPLADE, Hybrid DBSF) · query-level findings · bad queries / query-routing demo · Q&A
**Speaker notes:** Walk through each CP in ~20 seconds. CP1 anchors the lab in the traditional lexical product-search baseline. CP2 adds the generic dense model and fine-tuned SPLADE, with the first metric reveal. CP3 adds one DBSF hybrid recipe so the lab stays focused on the SPLADE story. Total = 60 min hands-on.
**Visual:** A horizontal timeline with four blocks (CP1, CP2, CP3, Wrap) sized proportionally to their minute budgets. Each block has a one-line subtitle and an icon (eyeballs for CP1, gauge for CP2, fusion-symbol for CP3, trophy for Wrap).

---

## Slide 12: What you'll leave with — preview the lift
**Goal:** Show the destination so they know what they're chasing.
**Bullets:**
- Eval queries = 2,000 ESCI product-search queries
- Qrels = the answer key: query-product relevance labels (Exact / Substitute / Complement / Irrelevant)
- nDCG@10 is the primary score: it uses ESCI grades, not exact-match only
- MRR@10 / Recall@10 / Precision@10 are supporting checks from the same run
- Quality comparison across four approaches: BM25, Generic dense, SPLADE, Hybrid DBSF
- Takeaway notebook: train your own on your own data
**Speaker notes:** Define the measurement terms before showing the lift: an eval query is one ESCI query text, and qrels are the labeled query-product judgments we compare retrieved IDs against. The final notebook leads with nDCG@10 plus delta vs BM25, then shows supporting metrics and example queries where SPLADE helped most and where hybrid helped or hurt. Note the CI — this is the authoritative claim, not the illustrative demo-set metrics.
**Visual:** A preview of the final wrap table: 4 rows in lineup order (BM25, Generic dense, SPLADE, Hybrid DBSF), with nDCG@10 CI and delta vs BM25 as the headline columns. Add a small supporting-metrics strip for MRR@10 / Recall@10 / Precision@10 and a second panel titled "Where did the lift come from?" with 3 example query rows. Caption: "You'll see this filled in for real at the end."

---

## Slide 13: Machine check-in
**Goal:** Get every laptop into a working state in 3 minutes.
**Bullets:**
- Open the platform's built-in IDE for your assigned workshop machine
- Open `notebooks/lab.ipynb`
- Run the **Setup cell** — it should end with:
  - `Ready. Corpus: ~38,000 products · 10 demo queries · 5 bad queries · 2,000 eval queries.`
- Raise your hand if the cell errors or never reaches `Ready`
**Speaker notes:** Pause the slide and walk the room. The Setup cell connects to Qdrant, loads live ESCI qrels, and initializes the encoders. Anything red goes to a workshop helper, not you. Move into the notebook once ≥90% of hands are down.
**Visual:** Screenshot of the built-in IDE with the Setup cell expanded and its expected Ready line visible. A small "raise hand" icon in the corner with text "anything red → flag a helper."
