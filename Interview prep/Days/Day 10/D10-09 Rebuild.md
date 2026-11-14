---
type: study-block
day: 10
block: 9
tags: [interview, day10, rebuild, exercise]
---
# D10-09 · Rebuild: a tiny search with a SQL prefilter (45 min)

Hub: [[Day 10 - Photo-Seek I]] · Previous: [[D10-08 Skim map]] · Next: [[D10-10 Trace and failures]]

**File:** `Interview prep/Days/exercises/day10_tiny_search.py`. Run it with `python3 day10_tiny_search.py`.

It needs **only the standard library** (math, sqlite3). That is on purpose: the project env cannot import numpy today, and the ideas do not need it. The original plan in the day note (real CLIP + real FAISS, [[Day 10 - Photo-Seek I#C. Rebuild it yourself (block 9, 45 min)|section C]]) stays valid for when the env works.

> [!important] Rules of the exercise
> There are **no solutions in the vault**. Write one step, run the file, read the message. The file checks the steps in order and stops at the first unwritten one. Hints below are folded; open one only after ten honest minutes. If you are stuck after the hint, ask in chat and paste your attempt.

## The toy data (already in the file)

Six photos as 2-D unit vectors. The query "beach" is the arrow at 0°. It is the same catalog as [[D10-07b Search engine code#0. How this note's values were produced]], so you know every right answer in advance.

| photo | angle | cosine with the query | taken_at | faces |
|---|---|---|---|---|
| 5 | 5° | 0.996 | 2026-03-02 10:00 | mom |
| 9 | 15° | 0.966 | 2025-07-14 09:30 | mom |
| 40 | 30° | 0.866 | 2026-07-04 11:12 | mom, dad |
| 77 | 50° | 0.643 | 2026-08-31 18:20 | mom |
| 12 | 70° | 0.342 | 2026-06-20 16:05 | mom |
| 81 | 120° | −0.500 | 2026-09-01 08:00 | dad |

## Step 1 · cosine(a, b) (5 min)

**Goal:** the cosine similarity of two plain lists, which may **not** be unit length.

**Mirrors:** `embed/base.py:79` and the `normalize` call in `clip.py`. Theory: [[D10-02 CLIP theory#Cosine similarity, in one place]].

**Before you type, predict:** `cosine([3, 4], [6, 8])`. Is it 50, 1.0 or 0.5?

> [!hint]- Hint 1
> Three numbers: the dot product, the length of `a`, the length of `b`. A length is the square root of the sum of squares.

> [!hint]- Hint 2
> `zip(a, b)` walks both lists together. What must you return if a length is zero, before dividing?

## Step 2 · flat_search(query, vectors, k, allowed=None) (10 min)

**Goal:** exact search. Score every vector, keep the best `k`. If `allowed` is a set, only those ids may be returned.

**Mirrors:** `IndexFlatIP` plus `IDSelectorBatch` ([[D10-05 FAISS code#2. search - the first half]]).

**Decide first:** what should `allowed=set()` return, and why is that different from `allowed=None`?

> [!hint]- Hint 1
> Build a list of `(id, score)` pairs, skipping ids that are not allowed. All vectors are unit length, so the score is just the dot product.

> [!hint]- Hint 2
> `pairs.sort(key=lambda t: t[1], reverse=True)` then slice `[:k]`. Check `allowed is not None` rather than `if allowed:`, or an empty set will behave like "no filter".

## Step 3 · candidate_ids(conn, date_from, date_to, person) (15 min)

**Goal:** the metadata filter. Return `None` when no filter was given; otherwise the set of matching ids (possibly empty).

**Mirrors:** `SearchEngine.candidate_ids`, `engine.py:109` ([[D10-07b Search engine code#4. candidate_ids: SQL before vectors]]).

**The trap is in the data:** photo 77 was taken at 18:20 on the last day of the range. The test fails if you lose it.

> [!hint]- Hint 1
> Collect two lists as you go: SQL fragments (`clauses`) and their values (`params`). Join the fragments with `" AND "` at the end. If `clauses` is empty, return `None`.

> [!hint]- Hint 2
> `'2026-08-31 18:20:00' <= '2026-08-31'` is **false** as text. Compare with `<` against the **next day**: `date.fromisoformat(date_to) + timedelta(days=1)`.

> [!hint]- Hint 3
> For the person: `EXISTS (SELECT 1 FROM faces f WHERE f.photo_id = p.id AND f.person = ?)`. Give the photos table the alias `p`.

## Step 4 · search(...) (5 min)

**Goal:** put it together. Filter first, then rank only the survivors.

**Mirrors:** `SearchEngine.run`, `engine.py:92–106`.

> [!hint]- Hint
> Two lines. The value step 3 returns is exactly what step 2's `allowed` expects. That is not a coincidence: it is why `None` and `set()` were kept distinct.

When step 4 passes, the file prints the pre-filter and post-filter results side by side.

## Afterwards: answer out loud (10 min)

1. The post-filter search returned `[]` for the same query. Which two photos filled its top 2, and why were they dropped?
2. You could "fix" post-filtering by asking for the top 6 instead of the top 2. Why is that not a real fix at 50,000 photos?
3. Your `flat_search` does one dot product per allowed photo. How many membership checks does it do? What does that say about whether prefiltering makes the search *faster*?
4. In your SQL, what would `person IN ('mom', 'dad')` return for the summer range, and what does the real code do instead?
5. Which line of your code plays the role of `IDSelectorBatch`?
6. What would you add to make this the real thing? (Name the four swaps.)

> [!success]- Answers
> 1. Photos 5 (0.996) and 9 (0.966): the most beach-like, but taken in March 2026 and July 2025, outside the range.
> 2. You never know how many to fetch. With a strict filter the allowed photos can sit thousands of ranks down; fetching "enough" means ranking nearly everything anyway.
> 3. One check per stored vector. Prefiltering makes the result **correct**; with a flat index it does not reduce the scan.
> 4. Photos with **either** person: 12, 40, 77. The real code adds one `EXISTS` per person, AND-ed, so "mom and dad" gives only 40.
> 5. The `if allowed is not None and pid not in allowed: continue` test inside the loop.
> 6. 2-D toy vectors → CLIP's 512-d vectors; the Python loop → `IndexIDMap2(IndexFlatIP)`; the `person` text column → `identities` + `resolve_people`; the hand-written arguments → a parsed `StructuredQuery`.

## Stretch (only if time remains)

- Add a second person argument and make "mom and dad" return only photo 40.
- Make `search` return the candidates **by date** when there is no query vector, as `engine.py:97–102` does.
- Count dot products with and without the filter and print both.
