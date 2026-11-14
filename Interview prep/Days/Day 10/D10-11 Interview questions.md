---
type: study-block
day: 10
block: 11
tags: [interview, day10, questions]
---
# D10-11 · Interview practice (40 min)

Hub: [[Day 10 - Photo-Seek I]] · Previous: [[D10-10 Trace and failures]] · More questions: [[Photo-Seek - prep#7. Fifty likely questions]]

**How to use.** One question at a time. Answer **out loud** in under a minute, then open the answer. Mark each **S / OK / W** (strong, OK, weak) in the table at the end. In a live session, ask Claude to "run D10-11 as an interview": it asks one question, waits, and follows up.

Every answer separates **measured** (a number from the repo), **documented** (a stated decision) and **my reasoning** (an argument, not a fact).

## A. CLIP and the contrastive loss

**1. How does CLIP let text search images?**

> [!success]- Answer
> Two encoders, one for images and one for text, each ending in a projection to 512 numbers. They are trained together on image–caption pairs so that a photo and its own caption point the same way. So at search time I embed the query text and rank photos by cosine. **I use the pretrained open_clip ViT-B-32 `laion2b_s34b_b79k`; I did not train it.** → [[D10-02 CLIP theory]]

**2. Write the loss for a batch of N pairs.**

> [!success]- Answer
> Normalise all vectors. $S = I T^\top$ is N × N. Multiply by a learned scale $s$. Cross-entropy on each row with the diagonal as the target (image → text), the same on each column (text → image), and average the two. Positives: N on the diagonal. Negatives: $N^2 - N$, free from the batch.

*Follow-up:* **Why symmetric?** Each direction can hide a confusion the other shows; and I search text → image, so that direction must be trained. *Follow-up:* **Why huge batches?** More and harder negatives per step.

**3. What is the temperature and what happens if it is wrong?**

> [!success]- Answer
> Cosines sit in [−1, 1], too flat for a softmax. The scale $s = 1/\tau$ stretches them. Too small: gaps vanish, weak gradients. Too large: unstable. CLIP learns it, starting near 14 and capped at 100. It does not affect ranking at search time.

**4. R@1 is 39.4. Is that good, and is it yours?**

> [!success]- Answer
> **Measured:** COCO-5k, 5,000 images, 25,010 captions, R@1 39.4, R@5 65.4, R@10 75.6, about ±0.3. It is **zero-shot pretrained CLIP with exact search**, so it should match open_clip's published figure for this checkpoint; it is a sanity check of my pipeline, not a result I trained for. My contribution is the system around it. Through the real search path with the rule parser it is 35.9, because the parser turns words like "baby" into person filters.

**5. What can CLIP not do?**

> [!success]- Answer
> Identify specific people, know dates, count, handle spatial relations and negation, read fine-grained classes; text is cut at 77 tokens; scores are small in absolute value (my top hits were 0.27–0.37), so I rank and never threshold. The first two are why the parser and SQL exist.

## B. Cosine and FAISS

**6. Why inner product and not cosine or L2?**

> [!success]- Answer
> Every vector is L2-normalised when created, so inner product **is** cosine. For unit vectors L2 gives the same ranking ($\lVert a-b \rVert^2 = 2 - 2a\cdot b$). IP lets me use `IndexFlatIP` directly. The risk: the store trusts callers to normalise and does not check.

**7. Explain Flat, IVF, HNSW and PQ in one line each.**

> [!success]- Answer
> **Flat:** compare with everything; exact; O(N·d). **IVF:** k-means buckets, scan the nearest `nprobe`; misses near borders; needs training. **HNSW:** layered neighbour graph, greedy walk, ~log N hops; more memory, slow build, no deletes in FAISS. **PQ:** store codebook indexes instead of floats; 32× smaller at 64 bytes; approximate scores. → [[D10-04 FAISS theory]]

**8. Why Flat? Why not HNSW?**

> [!success]- Answer
> **Documented:** a personal library is thousands of vectors; flat search is sub-millisecond, so an approximate index trades recall for nothing. **My reasoning:** HNSW would add memory and build time, cannot delete (I replace vectors when a photo is re-embedded), and would mix index error into my retrieval numbers. With Flat every miss is CLIP's. **Honest gap:** I did not benchmark any other index.

**9. What happens at 100 million photos?**

> [!success]- Answer
> Flat needs about 205 GB and 51 billion multiply-adds per query: not viable. I would move to IVF-PQ (about 7 GB of codes) with exact re-ranking of the top few hundred, shard the index, and track recall against a Flat sample. The **hard part is filtering**: `IDSelectorBatch` over millions of ids does not scale, so I would partition by filter keys or use filter-aware ANN. And I would say plainly that 100 million is no longer a personal library. **None of this is built.**

**10. Why `IndexIDMap2`?**

> [!success]- Answer
> FAISS numbers vectors by position. The wrapper maps rows to my `photos.id`, so results join straight back to SQL, and I can remove by id before re-adding. (`IDMap2` also keeps a reverse map for `reconstruct`, which I do not use.)

**11. Why write the index to a temp file and rename?**

> [!success]- Answer
> A crash during a direct write leaves a truncated file under the real name. Writing `clip.faiss.tmp` and then `os.replace` is atomic on one filesystem: readers see the old file or the new one. And the index is derived; the catalog holds every vector.

## C. Parsing, Ollama and structured output

**12. What does the LLM do in your system? Is "routing" the right word?**

> [!success]- Answer
> It **parses** a sentence into a `StructuredQuery`: text, people, date range, place. It does not choose between backends, so "routing" is loose; the class is called `OllamaRouter`, which is where the word came from. I say "LLM query parsing". The shipped default parser is the rule one; the LLM is opt-in.

**13. Why not send the whole query to CLIP?**

> [!success]- Answer
> CLIP cannot know who "mom" is or when "last summer" was; those are facts in the catalog. And the extra words pull the text embedding toward things that merely look summery. So hard constraints go to SQL and only the visual residue goes to CLIP.

**14. What does schema-constrained decoding guarantee?**

> [!success]- Answer
> Shape: valid JSON with my five keys and types, because illegal tokens are masked at every step. **Not** content. My schema types dates as plain strings, so a wrong year, a non-date string, or a made-up person all pass. Recorded examples: "last summer" a year early; Christmas as 25–26 December; "children" as a person. → [[D10-06 Ollama and structured output]]

**15. Why not use the LLM for date parsing?**

> [!success]- Answer
> **Measured:** on dates the 7B model scored 82 against the rules' 87, with the season rule and an example already in the prompt. Date ranges are calendar arithmetic with one right answer; a few lines of code do it exactly and in 0.19 ms. When the rules do not recognise a phrase, the LLM's date is still used, which is why the hybrid reaches 94.

*Follow-up:* **Would a bigger model fix it?** Possibly; I did not test one. The rules would still be faster, free and the fallback.

**16. Explain the hybrid and its numbers.**

> [!success]- Answer
> Run both parsers. Keep the LLM's people, place and text. If the rules found a date phrase, their range replaces the LLM's. **Measured on 100 hand-written queries:** all four filters right at once on 84 (hybrid), 73 (LLM), 58 (rules). That is per query, not per field. 95% interval for 84 is about 76–90. Paired: hybrid beat LLM-only on 11 queries and lost on 0.

*Follow-up:* **Did you design it after seeing the test results?** Answer truthfully. The hybrid commit predates the eval commit, but no doc records what you had looked at. The gold set was also written by the prompt's author. Own both.

**17. The LLM server is down. What happens?**

> [!success]- Answer
> `urlopen` raises; the `except` in `OllamaRouter.route` catches any exception, sets `last_fallback`, and returns the rule parser's result. Search continues with lower parsing coverage. A hung server costs up to the 30-second timeout first.

**18. A user types a name the system does not know.**

> [!success]- Answer
> `resolve_people` returns `None` for it, `candidate_ids` returns an empty set, and the search returns nothing with "unknown people [...]". CLIP is never loaded. I chose that over ignoring the name, which would silently answer a different question. The cost is real: the LLM once produced `people=['kids']` and the search returned nothing.

## D. SQL prefiltering and design trade-offs

**19. How do filters combine with vector search?**

> [!success]- Answer
> SQL first: dates as a half-open range on `taken_at`, one `EXISTS` per person on the `faces` table. That gives a set of photo ids. FAISS gets it as an `IDSelectorBatch`, so only those vectors are scored. → [[D10-07b Search engine code]]

**20. Why is filtering after top-k a problem?**

> [!success]- Answer
> The top k of the whole library can be entirely photos that fail the filter, leaving nothing, even when matching photos exist further down. You would have to over-fetch by an unknown amount. Prefiltering always returns k if k qualify.

*Follow-up:* **Does prefiltering make it faster?** Not with Flat: it still visits every row to check the id. It makes it correct.

**21. Why one `EXISTS` per person instead of `IN`?**

> [!success]- Answer
> `IN (mom, dad)` is an OR: either person. A single face row can never be both. One `EXISTS` per person, AND-ed, means each must appear in the photo, and it never duplicates rows.

**22. Why `< date_to + 1 day`?**

> [!success]- Answer
> `taken_at` has a time part and is compared as text. `'2026-08-31 18:20:00' <= '2026-08-31'` is false, so `<=` loses the last day. Half-open ranges keep every moment of it.

**23. "A dog on the beach" returned nothing once. Why?**

> [!success]- Answer
> The parser made "beach" a place filter, but `photos.place` is empty because reverse geocoding is not built, so the filter removed everything. Fix: a place filters only if some photo carries a matching label; otherwise the word goes to CLIP as visual text.

**24. Why store vectors in SQLite as well as FAISS?**

> [!success]- Answer
> SQLite is the source of truth; the FAISS file is derived and rebuildable. A FAISS upgrade or index change is a rebuild, not a migration. Cost: every vector is stored twice. (The rebuild function exists and is tested but is not wired to a CLI command.)

## E. Honesty, failure and scale

**25. "On-device", but every number is from an H200 server?**

> [!success]- Answer
> Correct, and I would not use that word. It runs locally with no cloud calls, but it was only measured on a shared GPU server. I say "local" or "self-hosted". I have no laptop or phone measurements.

**26. How do you know it is offline?**

> [!success]- Answer
> A Python audit hook recorded every socket connect, send and DNS lookup during three real searches: six events, all to 127.0.0.1. Ollama's own sockets were sampled with `ss`: loopback only. That is process-level evidence, not OS-level isolation; I could not create a network namespace without root.

**27. Can you re-run these numbers now?**

> [!success]- Answer
> Not today. The conda env fails at `import numpy` with a missing `libmkl_gf_lp64.so.3`. The scripts and result JSONs are in the repo. The last recorded test run was 156 passed; the README says 157. I would report that discrepancy rather than pick one.

**28. What is the slowest part of a query, and what would you do about it?**

> [!success]- Answer
> **Measured:** the LLM, 1.52 s median (p90 2.19 s), against 0.19 ms for rules and about 2.8 ms for CLIP text encoding plus the flat search on 5k vectors. Options I did **not** build: try rules first and call the LLM only when they leave text unexplained; cache parses; a smaller model.

**29. What would you do next?**

> [!success]- Answer
> Reverse geocoding, so places become real filters. Validate the LLM's people against known identities before filtering. Check `date_from <= date_to`. A larger gold set written by someone else. An ablation over CLIP checkpoints.

**30. What is the weakest part of this pipeline?**

> [!success]- Answer
> Pick one and own it. Candidates: the 100-query gold set written by the system's author; places that only match visually; an unknown or generic person word emptying the result; no index benchmark behind the Flat choice.

## Score yourself

| # | topic | S / OK / W |
|---|---|---|
| 1–5 | CLIP, loss, temperature, R@1, limits | |
| 6–11 | cosine, index types, Flat, scale, ids, atomic save | |
| 12–18 | parsing, schema, dates, hybrid, fallback, unknown person | |
| 19–24 | prefilter, EXISTS, dates, place fallback, two stores | |
| 25–30 | wording, offline, re-run, latency, next steps, weakness | |

Copy every **W** into the weak-topics list of [[Interview plan (Nov 2026)]].
