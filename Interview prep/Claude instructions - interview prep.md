---
type: guide
tags: [interview, claude, handoff]
---
# Claude instructions: continuing interview prep

**Who should read this.** Any Claude session, on any account or machine, that helps Jayavardhan prepare for interviews.
Read it fully before you answer. If you have no access to the server, ask him to paste this note and
[[Interview plan (Nov 2026)]], or to share the files they name.

## Context

- Jayavardhan is an M.Tech CSE student at IIT Bombay, graduating in 2027. Interviews start around **2026-11-19**.
- He must defend **13 projects** from **resumes 4 and 5** (the only ones he sends; welfare from resume 3 is out of scope) as their sole author. Interviewers will ask "why this, why not that",
  "why this model or module", "what was the hardest problem" and "which trade-off did you make and why", and they will
  follow up in depth.
- **The thesis is excluded.** He started from zero on 2026-11-08. It is active work. Don't plan or quiz on it unless he asks.
- The plan, the day-by-day checklist and his weak topics are in [[Interview plan (Nov 2026)]]. **That note is the master
  tracker.** He started reading on 2026-11-08: Day N = 2026-11-08 plus N−1 days, and Day 15 (Nov 22) is the last. Interviews are around Nov 23. The plan's *Schedule* table maps each day to its material. Write each day's detailed note in `Days/` (`Day NN - <topic>.md`) one or two days ahead, following the format of the Day 01–04 notes: a timed schedule; theory with exact reading, videos and maths; code with line numbers; a rebuild exercise; a trace; questions. Then push.

## Where things live (on gpuserver1)

| What | Path |
|---|---|
| Resumes (5 variants) | `/data/jayavardhan/projects/resumes/25M0807-{1..5}.pdf` |
| Project repos | `/data/jayavardhan/projects/<folder>`: vLLMmini, photo_seek, sysml, speculative_decoding, raftkv, microexchange, decs, bountybrain, emotion_steering |
| CosyVoice code | `/data/jayavardhan/cosyvoice_rnd` (data and checkpoints on `/mnt/warm_storage/jayavardhan/cosy*`) |
| Seminar notes | `/data/jayavardhan/vaults/personal/Seminar/`, `/data/jayavardhan/mtp/qwen/report/Seminar Report.docx` |
| CS728, LeNet-5, welfare | not on the server; he collects them into `/data/jayavardhan/projects/interview_materials/<name>/` |
| Prep packs | `vault/Interview prep/Prep packs/<Project> - prep.md` (the copies in the repos are stale snapshots) |
| Day notes | `vault/Interview prep/Days/Day NN - <topic>.md`: write each day's lesson here before the day, and his notes go here too |
| One note per project | `vault/Projects/` (this vault) |
| This vault on GitHub | `jayavardhan-g/projects-vault` (repo root = the vault) |

Several repos have their real work on the branch **`upgrade-2026-10`**, not on `main`: photo_seek, bountybrain,
microexchange and raftkv. Read that branch. The numbers in the upgraded docs replace the older ones.

## State on 2026-11-04

- **11-10: day notes for Days 5–15 are all written** (`Days/`), from the prep packs and the code. Each note's top section lists the resume claims to fix. Open items: copy the CS728/LeNet-5 code (Day 9 has no line numbers yet); find the CosyVoice WER CSVs or the presentation (they are not on the server); write the CosyVoice, CS728 and LeNet-5 prep packs the evening before their days. Revise a note only if the plan changes.
- **11-10: every day note (1–14) has a `## Code, explained` section** between B and C: hand-drawn Mermaid pipeline figures plus exact code excerpts with explanations. The separate `Days/Code/` notes were folded into the day notes. The archify diagrams were reverted at the user's request; don't use archify for this prep. Extend the section when new code arrives (CS728, LeNet-5).

- **11-11: every day (1–14) has from-scratch deep dives** at the end of its *Code, explained* section (one sentence → tiny example → formula → code line by line → questions → quiz → five things). Day 1 also has 10 animated shape figures (`d1s-*.svg`, real model sizes) under trace steps 1.1–1.11. Glossary now 323 entries (Python, torch, Triton, CUDA, C++, numpy, SQLAlchemy…).
- **Open corrections found while writing the dives (claims in older sections / project docs; the dives already state the corrected version):**
  - Day 4 A2: the "measured above the formula because bursty" story is wrong. `acceptance_rate` is accepted ÷ drafted, not α; solving α gives 0.941/0.912/0.914 (γ = 2/4/8), and α = 0.912 predicts the measured 4.19 tokens/iteration. RESULTS.md's 0.5B per-layer cost 0.098 ms should be 0.071 (1.69 ms ÷ 24).
  - Day 3 A1: the fused kernel wins mainly by removing ~9,000 small launches per step, not by moving fewer bytes. `serving.py` says the arrival bug made TTFT look "8× better"; the README numbers (24.6 → 2,448 ms) give about 100×.
  - Day 8: the LoRA adapter file is 35 MB (fp32), not 17 MB. "Label-smoothed" is likely wrong (`lsm_weight: 0`). The speech embedding and output layer stayed frozen.
  - Day 12 A2: R@k counts any chunk of the gold **document** (`eval/retrieval.py:170`), not the gold chunk. `lru_cache` also caches a failed reranker load.
  - Day 11: the guard's NaN probably needs an empty continuation, not one "under 2 tokens"; check against the notebook.
  - Day 13: the follower reads about 800 records per 200 µs batch at 4M/s, not about 1,000.
  - Day 5 §3 mermaid says F = 4; the contiguous loader really runs as `mm_tiled_contiguous<64,32>`, so F = 2.
  - Day 4 §7.6: the 23.6 ms iteration ("0.33 ms of 23.6 ms") is the 1.5B-draft run, not the 0.5B one.
  - Day 2 eviction, said precisely: `evict()` frees as many leaf candidates as it needs from one scan before re-scanning, so a parent becomes a candidate only when the current leaves can't cover the request.
  - Day 7 A1: Whisper's stem has two convolutions but only the second has stride 2 (not "two strided convolutions").
  - Day 9 A2: the 2019 lottery-ticket paper resets weights to their original initial values; "rewinding" to an early iteration came in later work.
- **11-12: new day-note format: one interleaved study guide.** He rejected separate theory / code / deep-dive sections. A day note now has one `## Study guide` made of `### Part N`. Inside a Part: the idea on tiny numbers → the exact code → a **Variables** table (name, type or shape, what it holds, value in the example) → **Traced** (the code run step by step on a small example, state after each step) → **Where the output goes** (the next function) → questions → check → five things. Readings sit at the end as optional "Sources". **Converted: Day 5 and Day 13** (use them as models). All other days still have the old layout.
  - **How to convert a day:** reuse the note's existing text and numbers as they are. Do not re-audit results or re-read the repo; open the code only for extra lines a trace needs (a struct, the caller). Work inline, **one day at a time, only the day he names**. Do not launch parallel agents for this unless he asks (they burned the usage limit twice on 11-12 with nothing delivered).
  - **11-12: Day 13's Micro Exchange parts (7–10) have 11 step animations** (`d13m-*.svg`: pipeline, ring, ordering, false sharing, matching, unlink, id table, pool, WAL append, recover, segments) and a ten-command worked session with the conservation check. They are made by `Days/figures/_gen/mxfigs.py <output dir>` (plain python3; one frame per step, the step text in a bar at the bottom). Reuse that script's helpers for step animations on other days.
  - **11-12: he still could not read lines like `const size_t head = head_.load(std::memory_order_relaxed);`.** What he asked for, and what Day 13 Parts 8–10 now have: a **primer** on the language pieces before the code (what an atomic is, load/store, each memory order, `&` masks, references), then **for every line**: what it does, why it is written that way, what uses its result next, and **what breaks if it is changed or removed**. Visual where possible: "debugger view" animations (`d13x-push`, `d13x-pop`, `d13x-match`: code on the left with the running line highlighted, variables on the right) and "what breaks" animations (`d13x-stale`, `d13x-swap`, `d13x-load`), made by `Days/figures/_gen/mxfigs2.py`. Never assume he knows a C++ or library construct; explain each the first time it appears. Use this depth for any code he says he does not understand.
  - **11-12: every animated figure has a step-through player.** An embedded SVG can't be paused in Obsidian, so each animated `figures/X.svg` has `figures/players/X.html` (Previous / Next / Play, a scrub slider, arrow keys), and every embed is followed by `[[X.html|▶ Step through it]]`, which opens in the browser. After adding or changing any animated figure, run `python3 make_players.py` in `Days/figures/_gen/` and put the same link after the embed (inside a table cell write the pipe as `\|`).
  - Links from other notes now point at `#Study guide` for converted days and `#Code, explained` for the rest.
- **11-11: section A of every day stands alone:** each reading has a "What it says" summary (own words), and each Concepts list links to its deep dives. Days that lacked Concepts lists got them.
- **11-11: Days 2–5 also have shape animations** (`d02s-`…`d05s-*.svg`, real sizes, no values) under their code headings, like Day 1's `d1s-*`.
- **11-10: [[Function glossary]]** (`Interview prep/Function glossary.md`) is generated from scripts in the session scratch dir (glossary.py, glossary_ext.py, gen_glossary.py, gloss_figs.py → figures `g-*.svg`). Each day note's code excerpts get a *New functions in this code* box (first use per note) linking to it. Add new functions there, not inline.
- 11-05: Day 1 started. 11 prep packs (all except CS728, LeNet-5 and welfare) were launched as background agents. DECS CPU-saturation question resolved: the claim holds (see plan).
- Upgrades: Photo-Seek, BountyBrain, microexchange and raftkv are done. DECS, mini-vLLM, speculative decoding, CUDA
  and emotion steering were **not** upgraded and won't be. **Start no new feature work.**
- Photo-Seek's GitHub `main` still shows the scaffold until he pushes the branch.
- Resume claims that disagree with the measured results (details in the plan's Day-1 section): Photo-Seek "fine-tuning
  ArcFace" (the adapter head added nothing), Micro Exchange 10.8M ops/s / 50× (laptop only; 3.44M / 18.1× on the server),
  emotion steering "up to 90%" (mean 36%), and CosyVoice WER 1.99
  (above 1, which needs explaining).

## How to behave

0. **One project at a time.** Work only on the project or projects scheduled for today. To save usage, never read or prepare other projects in parallel. Write a project's prep pack (prompt in the plan) only the evening before its day.

1. **Orient first, without asking.** Read the plan note: the ticked boxes, the weak topics and the latest `Log/` entry.
   Work out today's day number from the date. Say in one line where he is, and what today covers.
2. **Ground every answer in his code and docs.** Cite `path:line` or the doc section. Quote numbers only from the repo's
   RESULTS/README or the prep pack's numbers sheet, with the setup they were measured on.
3. **Never invent history.** If the docs don't say why a choice was made, say "not documented" and give the
   reasoning the code supports, labelled as reconstructed. Never invent bugs, experiences or numbers for him to
   repeat. His real "hardest problem" stories are in the PROGRESS logs, the `Log/` notes and `Lessons/`.
4. **Flag disagreements.** When the resume and the repo disagree, say so plainly and suggest honest wording. Don't
   help him defend a number the repo doesn't support.
5. **Theory and code together.** Each day's blocks interleave theory (T), deep code (C), skim (S) and practice (P), in the order that makes sense for the topic. When he asks about a concept, point to the function in his repo that uses it; when he asks about code, give the concept behind it. Quiz him on what a function does, or what happens at a line with concrete values.
6. **Teach in layers.** Start from the prerequisite concept, then how this project uses it, then the code, then the trade-off
   against the alternatives. Keep it short and plain. Check his understanding with a question rather than lecturing.
7. **Mock interviews: only on Day 15, or when he asks.** There are no mock days; daily practice is out-loud questions. Ask one question at a time and wait for his answer. Follow up on anything vague with "why not X?",
   "what breaks at 10× scale?" and "how did you measure that?". After about 15 questions, grade each answer
   (strong / OK / weak) and name the 5 topics to revise. Add the weak topics to the plan note.
8. **If he falls behind,** cut the S blocks first, then shorten the trace exercise. Never cut practice or mocks. Don't rank projects by how many resumes they appear on.
9. **Leave the project repos alone.** Read only: no commits, no pushes, no new branches and no env rebuilds unless he
   asks. Writing `INTERVIEW_PREP.md` (untracked) and editing this vault are fine.
10. **When a study session ends:**
   - tick the boxes in the plan;
   - add weak topics to the plan;
   - append a short dated entry to `Log/<today>.md` (what was covered, what was weak, what is next).

   **Always sync the vault (standing instruction, 2026-11-12).** He reads the notes on his laptop by pulling.
   - **Before editing anything in the vault:** `git pull --rebase`.
   - **After every edit:** `git add -A`, commit with a short message, `git push`. Don't leave edits unpushed.
11. **Machine rules** (from his global CLAUDE.md on gpuserver1; follow them on any account):
    - no sudo;
    - shared server: touch only `/data/jayavardhan/` and `/mnt/warm_storage/jayavardhan/`;
    - `/data/jayavardhan` has a 100 GB quota, and it read about 91 GB on 2026-10-22, so check `du -sh` before any install;
    - ask before using a GPU;
    - mamba environments created by name.

## Kick-off prompt for a new session

```text
Read /data/jayavardhan/projects/vault/Interview prep/Claude instructions - interview prep.md and follow it.
Then tell me where I am in the plan and start today's session.
```
