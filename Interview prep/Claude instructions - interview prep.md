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

   Don't commit or push the vault unless he asks.
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
