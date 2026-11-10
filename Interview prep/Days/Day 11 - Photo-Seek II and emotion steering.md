---
type: study-day
day: 11
date: 2026-11-18
tags: [interview, day, photo-seek, faces, emotion-steering]
---
# Day 11 (Wed Nov 18): Photo-Seek II (faces), then emotion steering

Part of [[Interview plan (Nov 2026)]]. About 7.25 h. Morning: `photo_seek`, branch `upgrade-2026-10` (prep pack
[[Photo-Seek - prep]], Trace 2, stories 4.1–4.3). Afternoon: `emotion_steering` (branch `main`, one commit `d725cbe`;
prep pack [[Emotion steering - prep]]).

**Code walkthrough:** [[Day 11 - code walkthrough]]: one small figure per pipeline, the exact code to read in order, and what each piece does.

**Resume bullets:**
- Photo-Seek (both resumes): "Built an active-learning face-clustering pipeline for offline photo organization, **fine-tuning ArcFace embeddings** on user correction signals to progressively refine identity clusters across **1,000+ photos**."
- Emotion steering (resume 4): "Built an activation-steering framework for **Llama 3.2 1B** spanning all **8 primary emotions** via Plutchik's 4 bipolar axes, injected as real-time hidden-state interventions and validated with ablations." / "Achieved **up to 90%** steering success entirely at inference time with no fine-tuning."

**The Photo-Seek faces bullet is not supported as written** (prep pack "Read these first", §5.1):
- **ArcFace is never fine-tuned.** A 512×512 linear adapter on frozen ArcFace vectors changed BCubed F1 by **−0.07** (CPLFW) and **+0.00** (LFW). The gain comes from the corrections, the constraints, and *which* faces the sampler asks about.
- **"1,000+ photos" matches nothing.** The eval clusters **5,606 held-out faces** (1,906 people) from 11,435 CPLFW photos. 1,000 is the number of **simulated reviews**. No real library was processed.
- **95.9 → 97.6** is BCubed F1 with the dev-chosen setting (agglomerative, 0.75) and **margin** sampling. The shipped defaults (HDBSCAN; uncertainty) give 91.75 and 97.2.
- Use prep pack §5.4's honest bullet: "…margin-based review sampling turns user corrections into hard clustering constraints. On 5,606 held-out cross-pose faces (CPLFW), 1,000 simulated reviews raised BCubed F1 from 95.9 to 97.6, a 41% error cut vs 7% for random review."

**The emotion bullet needs care too** (prep pack §5):
- **90% = the best of 240 cells** (Joy+, layer 9, strength 5: 18/20; 95% CI 70–97%). **Joy averaged 36%** over the sweep; Anger 3.3%, Disgust 5.9%, Sadness 7.6%.
- **"All 8 emotions":** vectors exist for all 8 poles, but only 6 can be scored by the classifier (Trust and Anticipation have no label).
- **"Validated with ablations":** layer × strength sweeps and a sign flip, but no random-vector control and no prompting baseline. The sweep code that made the CSVs is **not in the repo**, and the notebook can't run top to bottom.

**By tonight you can:**
- explain SCRFD, ArcFace's angular margin, and agglomerative clustering vs HDBSCAN;
- explain BCubed F1, active learning (uncertainty vs margin vs random) and the two refine bugs;
- explain difference-of-means steering vectors, the hook and the perplexity guard;
- say both projects' numbers honestly.

## Schedule

| # | Time | Kind | What |
|---|---|---|---|
| 0 | 15 min | recall | Day 10: CLIP, FAISS Flat, Trace 1 |
| 1 | 45 min | theory | Faces: detection, ArcFace, clustering, BCubed (A1) |
| 2 | 20 min | skim | `faces/detect.py`, `embed/arcface.py` |
| 3 | 60 min | code | `faces/cluster.py` (B1) |
| 4 | 25 min | theory | Active learning; metric learning (A2) |
| 5 | 60 min | code | `faces/active.py`, `faces/head.py` (B2) |
| 6 | 20 min | trace | Trace 2: one correction end to end (D1) |
| — | lunch | | |
| 7 | 40 min | theory | Activation steering (A3) |
| 8 | 50 min | code | `Project.ipynb`: calibration, the hook, the guard (B3) |
| 9 | 40 min | rebuild | A steering vector on GPT-2 (C) |
| 10 | 45 min | practice | Questions for both projects (E) |

## A. Theory

### A1. Faces (block 1, 45 min)

**Read:** Deng et al., *ArcFace* (2019), §1 and §3.1 (the additive angular margin, Figure 2). About 25 min. Skim the SCRFD abstract (Guo et al., 2021).
- **Pipeline:** SCRFD detects faces and 5 landmarks → align to 112×112 → ArcFace gives a 512-d unit vector. Same person → high cosine.
- **ArcFace loss:** softmax over identities with an additive margin m on the angle: `cos(θ_y + m)`. This forces tight clusters per identity.
- **Clustering:**
  - **Agglomerative (average linkage)** merges clusters while the average pairwise distance is under a threshold (0.75 here); you don't set k.
  - **HDBSCAN** is density-based and leaves noise unassigned.
  - **k-means** needs k (unknown here).
- **BCubed F1:** per face, precision = the share of its cluster with the same true identity; recall = the share of its identity in its cluster. Average over faces. It suits clustering because it doesn't need to match cluster IDs to identities.

### A2. Active learning (block 4, 25 min)

**Read:** Settles, *Active Learning Literature Survey*, §3.1 (uncertainty sampling, margin). 15 min.
- **Uncertainty:** ask about the faces farthest from their own centroid.
- **Margin:** ask about the faces where the nearest two identities are almost tied (`d2 − d1` smallest). Margin won: error cut **41%**, vs 32% for uncertainty and 7% for random.
- **Corrections as constraints:** a confirm *pins* a face to an identity; a reject becomes a *cannot-link*.
- **Metric learning** (the adapter head): a SupCon-style loss on pinned faces, initialised at W ≈ I. It added nothing. Propagation to unreviewed faces was **≤ +0.09 points**.

### A3. Activation steering (block 7, 40 min)

**Read:**
- Turner et al., *"Activation Addition"* (2023), §1–3. About 20 min.
- Rimsky et al., *"Steering Llama 2 via Contrastive Activation Addition"* (2024), §2. About 15 min.

**Concepts:**
- The **residual stream** carries a vector per token through the layers. Directions in it can encode concepts.
- **Difference of means:** run contrastive sentences (15 per pole), take the last token's hidden state, `v = mean(pos) − mean(neg)`, normalise. Your 4 Plutchik axes give 4 vectors per layer.
- **Inject:** a forward hook adds `α·v` to one layer's output at the **last position, every step**.
- **Middle layers work best** (layer 9 here): early layers hold token identity, late layers are committed to output.
- **Over-steering** breaks fluency before it shifts the mood. Hence the guard: perplexity of the steered text under the unsteered model, as a ratio to a same-seed baseline. Accept if ≤ 1.75×; otherwise retry at 75%, 50%, 25%, then fall back.

## B. Code

### B1. Clustering (block 3, 60 min, deep)

`photo_seek` `src/photoseek/faces/cluster.py` (402 lines):
- **76 `cluster`**: full clustering; **286 `_unsupervised`** (agglomerative or HDBSCAN);
- **174 `IdentityClusterer.refine`**: 201–204 the story 4.1 fix; **227–233** the true average-linkage distance `1 − x·mean(members)` with the mean *unnormalised* (story 4.2); 243–249 assignment rules; 261–273 leftovers;
- 314 `assign_new_faces` (incremental); **341 `persist`** (identity IDs stay stable; pinned faces never written).

Skim `faces/detect.py` (127 SCRFD `detect_array`) and `embed/arcface.py` (72 align, 85 `embed_faces`).

### B2. Active learning (block 5, 60 min, deep)

- `faces/active.py` (416 lines):
  - 69 `ActiveLearner`; **97 `next_batch`**: uncertainty 137–140, margin 141–144; 401 `_spread` (at most 2 questions per identity);
  - **239 `answer`** → 173 `record` (append to `corrections`, pin the face);
  - **290 `refine`**: 319–325 rejects become cannot-links; 333 corrections are marked consumed only after a successful retrain.
- `faces/head.py`: **60 `ProjectionHead.fit`** (≥ 20 labels across ≥ 3 identities, 27 and 73). Note the docstring's "25M parameters" is wrong (about 43M; §5.3).

### B3. Emotion steering (block 8, 50 min, deep)

`emotion_steering/Project.ipynb`. References are `cN:L` = cell N, line L (prep pack "Read this first"):
- `c3`: the calibration data, 15 sentences per pole;
- **`c4` `AppraisalSteerer`**: 15 `_get_all_layer_last_token_states`; 21 `calibrate_all_layers` (one pass, all 17 hidden states); **50–51** difference of means, normalise; **61** `set_layer` reads index `L+1` (index 0 is the embeddings); **64–69** the hook adds the vector to the last position; 82 chat template; 96 `manual_seed`; 98 `generate` (T = 0.8, 60 tokens);
- **`c6`**: 38 `continuation_perplexity`; **51 `generate_guarded`**; 65 the ladder; 71 the ratio; 77 the 1.75 threshold;
- `c2`: the j-hartmann DistilRoBERTa classifier; `c5`: the Gradio demo.

## C. Rebuild it yourself (block 9, 40 min)

GPT-2 small on CPU, about 50 lines:
1. 10 "happy" and 10 "sad" sentences. Collect the last-token hidden state at layer 6 (`output_hidden_states=True`; remember index 0 is the embeddings).
2. `v = mean(happy) − mean(sad)`, normalise.
3. Register a forward hook on `model.transformer.h[6]` that adds `α·v` to `output[0][:, -1, :]`. Generate from "Today I went to the shop and" at α = 0, 4, 8, 16, with the same seed.
4. Compute the perplexity of each continuation under the unhooked model. Where does fluency break?

## D. Traces

**D1 (block 6).** Prep pack Trace 2: the user types `y` for one face. Write what `next_batch` sorts by (margin), what `answer` writes, when `refine` runs (25 pending corrections), and why a crash mid-retrain loses nothing.

**D2 (in block 10, 5 min).** The Gradio defaults (layer 12, Joy 5, guard on): which hidden-state index is the vector from? How many generations does one request cost at minimum and at most? (*2 and 5.*)

## E. Questions

**Photo-Seek faces**
1. Walk me through the pipeline from a photo to an identity.
2. Did you fine-tune ArcFace? (*No. Say it first; the adapter added −0.07.*)
3. Why average-linkage agglomerative clustering? Why not k-means?
4. What is BCubed F1, and why not accuracy?
5. Uncertainty vs margin vs random: results, and why margin wins.
6. Corrections once made clustering *worse*. Why? (*Story 4.1.*)
7. A refine with zero answers changed the clusters. Why? (*Story 4.2: the null-control arm.*)
8. "1,000+ photos": what was actually clustered?

**Emotion steering**
9. How do you compute a steering vector? Why the last token?
10. Why layer 9? What happens at layer 0 and layer 15?
11. "Up to 90%": what is the mean, and what is the CI on 18/20?
12. Why do anger and disgust barely steer? (*Classifier and data limits; a punctuation confound: 6/15 anger sentences end in "!".*)
13. What does the perplexity guard measure, and how did you choose 1.75? (*Not calibrated; say so.*)
14. What ablation is missing? (*A random-vector control and a prompting baseline.*)
15. Hostile: "Your notebook can't run top to bottom and the sweep code is missing. How can I trust the CSVs?"

## My notes and fumbles

- 
