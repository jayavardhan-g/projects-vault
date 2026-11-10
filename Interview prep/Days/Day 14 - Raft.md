---
type: study-day
day: 14
date: 2026-11-21
tags: [interview, day, raft, distributed-systems]
---
# Day 14 (Sat Nov 21): Raft (raftkv)

Part of [[Interview plan (Nov 2026)]]. Repo `raftkv`. About 7 h. Prep pack: [[Raft - prep]] (original vs upgrade §0,
key files §2, design decisions §3, stories §4, numbers §5, a code-review appendix).

**Code walkthrough:** [[Day 14 - code walkthrough]]: one small figure per pipeline, the exact code to read in order, and what each piece does.

**Two versions, and a branch trap:**
- **Original** = `origin/main` = commit **`2a19135`** ("RAFT", 768-line `raft_node.cpp`). It's what GitHub shows and what your resume bullets describe.
- **Upgrade** = branch `upgrade-2026-10` = **`bad36f1`** (1,077 lines): snapshots, pre-vote, fault injection, the linearizability checker, sharding, and both bug fixes.
- **Your local `main` has been fast-forwarded to `bad36f1`** (7 commits ahead of `origin/main`). So `git show main:…` shows the *upgrade*, not the original. To read the original, use `git show 2a19135:<path>` (or `origin/main:<path>`).
- **Authorship:** all 7 upgrade commits carry `Co-Authored-By: Claude`. The honest line (prep pack §0): "The core Raft engine is mine. I used an AI assistant for the hardening pass, and I reviewed and can explain every line."

**Resume bullets (resume 5):**
- "Engineered a from-scratch **Raft consensus** system in C++, implementing leader election, log replication, conflict resolution, and **crash-safe persistent storage** with **cross-platform** cluster support."
- "Built an automated **chaos-testing pipeline** to inject node failures and verify leader failover, **zero data loss**, and automatic follower catch-up, with a **live dashboard** tracking cluster health and replication progress."

**Check before you defend** (prep pack §5):
- **"Crash-safe":** `main` had the port-before-load bug that could wipe the log on restart (story 2). `persist()` ignores write errors. SIGKILL tests can't prove fsync works.
- **"Cross-platform":** `platform.h` and a cross-compile with zig (upgrade), but **the Windows and macOS binaries were never run**, and there's no log of a mixed-OS cluster.
- **"Pipeline":** on `main` it's one script with one scenario (SIGKILL the leader once). Say "an automated chaos test".
- **"Zero data loss"** = no *acknowledged* write lost (100 keys re-read, 11/11 runs); a single sequential client; SIGKILL is not power loss.
- Never quote "bench 500 → 446 writes/sec" from the README: no data behind it. Use ~250 writes/s (3 nodes) and ~200 (5 nodes), fsync-bound.

**By tonight you can:**
- explain election, replication, the commit rule and Figure 8, and the persistence rules;
- trace both real bugs (follower commit index; port before load) with concrete logs;
- explain pre-vote, check-quorum, linearizable reads and snapshots;
- quote failover ≈ 1.0 s and ~250 writes/s with their setups.

## Schedule

| # | Time | Kind | What |
|---|---|---|---|
| 0 | 15 min | recall | Day 13: Little's law, the ring's acquire/release |
| 1 | 75 min | theory | The Raft paper §5 (A1) |
| 2 | 30 min | code | `include/raft_node.h`, `include/timing.h` (B1) |
| 3 | 90 min | code | Election, commit and the RPC handlers (B2) |
| — | lunch | | |
| 4 | 50 min | code | Replication, apply, submit (B3) |
| 5 | 25 min | code | `src/storage.cpp`: fsync + atomic rename (B4) |
| 6 | 40 min | theory | Linearizability, CAP, Raft vs Paxos, pre-vote, snapshots (A2) |
| 7 | 30 min | skim | Upgrade-only code, transport, scripts (B5) |
| 8 | 40 min | rebuild | A Raft log-matching simulator (C) |
| 9 | 20 min | trace | Does the entry survive? (D) |
| 10 | 40 min | practice | The questions (E) |

## A. Theory

### A1. Raft (block 1, 75 min)

**Read:** Ongaro & Ousterhout, *"In Search of an Understandable Consensus Algorithm"* (2014), **§5 in full** and **Figure 2** (print it). About 60 min.
**Play:** the *Raft visualization* at thesecretlivesofdata.com/raft (10 min) or raft.github.io's interactive demo.
- **Election:** randomised timeouts; a vote per term; the **election restriction** (vote only for a log at least as up to date: last term first, then length).
- **Replication:** AppendEntries with `prevLogIndex`/`prevLogTerm`; the follower rejects on mismatch; the leader backs up (yours jumps back by conflicting term).
- **Commit rule:** a leader commits an entry from its **current term** once a majority has it; earlier entries commit indirectly. **Figure 8** shows why counting replicas of an old-term entry is unsafe. Your no-op on election makes the rule fire quickly.
- **Persistence:** `currentTerm`, `votedFor` and the log must be on disk *before* replying.
- **Follower commit index** = `min(leaderCommit, index of last new entry)`. Your story 1 is exactly this line.

### A2. Beyond §5 (block 6, 40 min)

**Read:** Raft paper §6 (membership, skim) and §7 (log compaction). Ongaro's thesis §9.6 on **pre-vote** and check-quorum (10 min). Kleppmann's blog *"Please stop calling databases CP or AP"* (15 min).
- **Linearizability:** every operation appears to take effect at one instant between its call and its return. Your reads go **through the log** (safe). `--unsafe-local-reads` is the negative control.
- **CAP:** under a partition, Raft keeps consistency and gives up availability on the minority side.
- **Raft vs Paxos:** same safety guarantees; Raft has a strong leader and explicit log matching, designed to be understandable.
- **Pre-vote:** a node first asks "would you vote for me?" without raising its term, so a returning partitioned node doesn't depose a healthy leader (story 5: 0/5 vs 5/5 deposed).
- **Snapshots:** compact the log; InstallSnapshot for followers that are too far behind.

## B. Code

Line numbers are for the **upgrade** (`upgrade-2026-10`). Original (`2a19135`) lines are in brackets where they matter.

### B1. Headers (block 2, 30 min, deep)

- `include/raft_node.h`: node state; `majority()` :151; `has_quorum()` :153.
- `include/timing.h`: defaults :17–23; the `lan`/`wifi`/`slow` profiles :39–50; `validate_timing` :56–75 (story 3: election storms on Wi-Fi).

### B2. Election, commit, handlers (block 3, 90 min, deep)

`src/raft_node.cpp`:
- **216 `become_follower`** [163], **229 `become_leader`** [176] (the no-op), **255 `maybe_advance_commit`** [202] (the current-term check);
- **333 `start_election`** [225], **409 `ticker_loop`** [300] (timeouts; check-quorum);
- **711 `handle_request_vote`** [526] (the election restriction);
- **770 `handle_append_entries`** [585]: 824–846 conflict backup by term; **859–865 the fixed follower commit bound** (on the original, `2a19135:src/raft_node.cpp:663` uses `last_log_index()`: story 1).

### B3. Replication (block 4, 50 min, deep)

- **450 `replicator_loop`** [332]: one thread per follower, batches of 64, `nextIndex`/`matchIndex`;
- **588 `apply_loop`** [427]; **619 `submit`** [455] (a client command waits for commit, with a timeout);
- 174 `persist` [147] (rewrites the whole log each time, hence story 6's throughput decay).

### B4. Storage (block 5, 25 min, deep)

`src/storage.cpp` **78 `write_durably`**: write a temp file → `fsync` → `rename` over the real file → `fsync` the directory. Why each step? What does a crash between each pair of steps leave on disk?
Then `src/main.cpp` 367–381: the fixed start-up order (load state, *then* open the port: story 2).

### B5. Skim (block 7, 30 min)

Upgrade-only: `start_pre_vote` (283), `take_snapshot` (186), `handle_install_snapshot` (915). Then `src/transport.cpp` (framing 120–138), `src/state_machine.cpp`, `src/fault.cpp`, `src/main.cpp`, `scripts/chaos_test.py` (phases 193–258), `scripts/linearizability.py` (99 `check_register`), `scripts/dashboard.py`.

## C. Rebuild it yourself (block 8, 40 min)

In Python, about 80 lines, no networking: logs are lists of `(term, cmd)`.
1. `append_entries(follower_log, prev_index, prev_term, entries, leader_commit)` returning `(success, new_commit)`. Implement the Figure 2 rules exactly, including truncate-on-conflict and `commit = min(leader_commit, index of last new entry)`.
2. Replay story 1's trace (prep pack §4): B has stale term-3 entries 75..200; C sends `prev=10` with 11..74 and `leader_commit=300`. Show that the buggy bound gives 200 and the correct one gives 74.
3. Write `leader_can_commit(index, current_term, match_indices, log)` with the current-term rule. Build Figure 8's scenario and show that the naive "majority has it" rule commits an entry that is later overwritten.

## D. Trace (block 9, 20 min)

5 nodes. The leader (term 4) appends entry 50, replicates it to 1 of 4 followers, then crashes. Does entry 50 survive?
- Who can win the next election? (*The election restriction.*)
- If the follower that has entry 50 wins, is entry 50 committed at once? (*No: only when an entry from its own term commits on a majority: the no-op.*)
- If another node wins, what happens to entry 50 on the follower that had it?
- Was entry 50 ever acknowledged to the client? So is losing it a violation?

## E. Questions

1. Explain Raft leader election. Why randomised timeouts?
2. Walk me through log replication and how a follower's log is repaired.
3. Why can't a leader commit an old-term entry by counting replicas? (*Figure 8.*)
4. What must be persisted, and when? How do you make a write crash-safe?
5. Tell me about a bug you found. (*Story 1 with the concrete trace; story 2 for durability.*)
6. How are your reads linearizable? How did you test that? What did the negative control show?
7. What is pre-vote, and what did it fix? Show the numbers.
8. How fast is failover? How did you measure it? (*≈ 1.0 s median, 3 nodes; quantised by the 0.2 s poll loop.*)
9. Why only ~250 writes/s? (*One fsync per write; a new TCP connection per request; a sequential client.*)
10. "Zero data loss": under what failures? What about power loss?
11. Did you run it across operating systems?
12. Which parts did an AI assistant help with?
13. Hostile: "Why build Raft yourself instead of using etcd?"

## My notes and fumbles

- 
