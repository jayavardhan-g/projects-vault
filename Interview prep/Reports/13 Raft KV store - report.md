# Project report: raftkv, a fault-tolerant distributed key-value store on Raft (C++20)

> **For the LLM reading this.** This report describes a self project by G Jayavardhan (M.Tech CSE, IIT Bombay), Summer
> 2026, upgraded in October 2026, listed on resume 5 as "Fault Tolerant Distributed System". It was written from
> `/data/jayavardhan/projects/raftkv`, **branch `upgrade-2026-10`** (HEAD `bad36f1`): source, README (833 lines),
> `results/` and the scripts. All 8 commits are by Jayavardhan:
> - `main` = the original August build (`2a19135`, "RAFT");
> - the upgrade branch adds snapshots and InstallSnapshot, pre-vote, fault injection, a linearizability checker, live resharding, a cross-platform build check, and a full results regeneration.
>
> Labels:
> - **[CODE]**: `file:line` or function;
> - **[MEASURED]**: from `results/` via the README §9;
> - **[DOC]**: a reason written in the repo;
> - **[RECON]**: the report writer's reasoning.
>
> **Know which features are in the original and which in the upgrade** (§2).

---

## 1. Summary

A from-scratch implementation of the **Raft consensus algorithm** (Ongaro & Ousterhout, 2014) with a replicated key-value store on top. No libraries: a C++20 compiler plus Python 3 for the scripts. Features:
- **Leader election:** randomized timeouts 600–1,200 ms, majority votes, the election restriction (vote only for an up-to-date log).
- **Log replication:** a replicator thread per follower, 100 ms heartbeats, **fast backup** (the follower returns the first index of the conflicting term).
- **Commit rule:** only entries of the current term are counted directly. A **no-op** is appended on becoming leader.
- **Crash-safe storage:** term, vote and log written to a temp file → fsync → rename → fsync(dir) before any acknowledgement.
- **Linearizable reads:** `get` goes through the log.
- **Quorum check:** a leader that hasn't heard from a majority for 1.5 s steps down, and writes are refused with a reason.
- **Upgrade features:**
  - **log compaction** (snapshots + `InstallSnapshot`);
  - **pre-vote**;
  - **fault injection** inside the transport (partitions, loss, delay; no root needed);
  - a **Wing & Gong / Lowe linearizability checker**;
  - **live resharding** across several Raft groups (freeze/install/drop through both logs).
- **Cross-platform:** Linux, macOS, Windows (Winsock). Nodes on different operating systems can join one cluster. A `zig c++` cross-build check passes for 4 targets.
- **Tooling:**
  - an interactive node shell;
  - a Python client (follows redirects to the leader);
  - a live **dashboard** (polls each node's status port, raft port + 1000, every 300 ms);
  - per-node replicated-log files (`APPEND`/`COMMIT`/`DROP`/`SNAPSHOT`/`INSTALL`);
  - debug traces naming the Raft rule behind each decision;
  - timing profiles `lan` / `wifi` / `slow`.

**Key results [MEASURED]:**
- chaos test 11/11 pass (3 and 5 nodes), failover **≈ 1.0 s** median;
- **50/50 five-node linearizability runs** pass under partitions, loss and SIGKILLs (180,462 acknowledged operations);
- pre-vote stops a partitioned node from deposing the leader (deposed 0/5 vs 5/5 without it);
- live rebalance 16/0/0 → 6/5/5 gives **1.86× throughput**.

---

## 2. Resume bullets vs evidence (resume 5)

| Bullet | Verdict | Evidence |
|---|---|---|
| "Engineered a from-scratch Raft consensus system in C++, implementing leader election, log replication, conflict resolution, and crash-safe persistent storage with cross-platform cluster support" | **Supported** (all in the original `main` and kept). | `src/raft_node.cpp`: `start_election`, `handle_request_vote`, `replicator_loop`, `handle_append_entries` (conflict detection + truncation + fast backup), `src/storage.cpp` (fsync + atomic rename), `include/platform.h` (Winsock/POSIX). |
| "Built an automated chaos-testing pipeline to inject node failures and verify leader failover, zero data loss, and automatic follower catch-up, with a live dashboard tracking cluster health and replication progress" | **Supported.** | `scripts/chaos_test.py`: kill the leader (+k−1 followers), check a new leader, re-read all keys, restart, wait for catch-up and convergence → PASS/FAIL. 11/11 in the matrix. `scripts/dashboard.py`. |
| Not on the resume, but in the upgrade | Bonus material | Snapshots, pre-vote, the linearizability checker and negative controls, resharding. He can mention these as follow-up work. |

---

## 3. Background (teach first)

### 3.1 Why consensus
Replicas of a state machine must apply **the same commands in the same order** despite crashes and lost messages. Consensus picks the order. Raft does it with a **strong leader**: the leader appends client commands to its log, replicates them, and declares an entry **committed** once a **majority** stores it. Committed entries are applied in order to the key-value map on every node.

### 3.2 Raft in five rules (Figure 2 of the paper)
- **Terms:** monotonically increasing epochs. Each has at most one leader. Any message with a higher term makes the receiver a follower in that term.
- **Election:**
  - a follower that hears nothing for a randomized timeout becomes a candidate, increments the term, votes for itself and sends RequestVote;
  - it wins with a majority;
  - **one vote per term** (persisted);
  - **election restriction:** vote only if the candidate's log is at least as up to date: (lastLogTerm, lastLogIndex) compared lexicographically.
- **Log replication:** AppendEntries(term, prevLogIndex, prevLogTerm, entries, leaderCommit).
  - The follower rejects it if its log has no entry at prevLogIndex with prevLogTerm (the **consistency check**).
  - On a conflict it truncates its own log from the first conflicting entry and appends the leader's entries.
  - Log Matching Property: if two logs share (index, term), they are identical up to that point.
- **Commit:**
  - the leader advances commitIndex to N if a majority has matchIndex ≥ N **and log[N].term == currentTerm**;
  - committing an older-term entry by counting replicas alone is unsafe (**Figure 8** scenario);
  - followers set commitIndex = min(leaderCommit, index of the last *new/verified* entry).
- **Persistence:** currentTerm, votedFor and the log must be on stable storage *before* answering RPCs.

### 3.3 Safety vs liveness; CAP
- Raft is always **safe**: it never returns conflicting committed values.
- It is **live** only when a majority can communicate. With N = 2f + 1 nodes it tolerates f failures (3 → 1, 5 → 2). A 2-node cluster tolerates 0.
- Under a partition it chooses **consistency over availability** (CP): the minority side refuses writes.

### 3.4 Linearizability
- Every operation appears to take effect atomically at some instant between its call and its return, consistent with real-time order.
- **Checking a history:** search for a valid linearization (Wing & Gong), with memoisation of (set of operations linearized, state) (Lowe). This is what Porcupine and Knossos do.
- A key-value history is linearizable iff **each key's** sub-history is (locality), so the check is per key.
- An operation whose reply was lost is **indeterminate**: it may have taken effect at any point after it was sent, or never.

### 3.5 Pre-vote and snapshots (Raft thesis §9.6, paper §7)
- **Pre-vote:** before raising its term, a would-be candidate asks "would you vote for me at term+1?". Peers say no while they still hear from a leader. This stops a partitioned node that comes back with an inflated term from deposing a healthy leader.
- **Snapshot / log compaction:** save the state machine at index i, discard log entries ≤ i. A follower that needs discarded entries gets **InstallSnapshot**.

---

## 4. Code walk

### 4.1 Structure (`include/`, `src/`; about 3,000 lines of C++ plus 2,600 lines of Python scripts)
| file | role |
|---|---|
| `raft_node.h/.cpp` (211 / 1,077 lines) | the algorithm: state, RPC handlers, threads |
| `storage.h/.cpp` | durable term, vote, log (v2 format, v1 still loads) and snapshot files |
| `state_machine.h/.cpp` | key-value map, commands (SET/GET/DELETE/NOOP/SHARD_*), state digest |
| `transport.h/.cpp` | TCP request/reply RPC; messages are typed with fields separated by ASCII unit separators (0x1f/0x1e/0x1d) |
| `fault.h/.cpp` | fault injector wrapped around the transport |
| `platform.h` | sockets, fsync and atomic rename on Winsock vs POSIX |
| `timing.h` | profiles and the safety clamps on timing combinations |
| `telemetry.cpp` | status server (port + 1000) for the dashboard |
| `main.cpp` | CLI, node shell, client mode |

### 4.2 Node state (`raft_node.h`)
- **Persistent:** `current_term_`, `voted_for_`, `log_` (entries after `snapshot_index_`), `snapshot_index_/term_`.
- **Volatile:** `role_`, `leader_id_`, `commit_index_`, `last_applied_`. Leader-only: `next_index_`, `match_index_` and `last_contact_` per peer.
- **One `std::mutex mutex_`** guards everything. Condition variables `replicate_cv_` (wake the replicators) and `apply_cv_` (wake the applier).
- **Threads:** ticker (elections and the quorum check), one replicator per peer, the applier, plus transport and telemetry threads.
- Index helpers: `last_log_index()` (= `snapshot_index_` if the log is empty), `term_at(i)`, `entry_at(i) = log_[i − snapshot_index_ − 1]`, `majority() = n/2 + 1`.

### 4.3 Elections
- **`ticker_loop`** (`:409`): if a follower or candidate has been silent past `election_timeout_` (a random 600–1,200 ms, re-drawn on every leader contact or granted vote), start a pre-vote (if enabled) or an election. If leader: check `has_quorum()` (recent contact with a majority within the quorum window, default 1.5 s), else step down ("QUORUM LOST").
- **`start_pre_vote`** (`:283`) and `handle_pre_vote` (`:883`): ask at term+1 without changing our own term. Peers refuse while they've heard from a leader recently. `prevote_round_` discards late replies. A majority of yeses → `start_election`.
- **`start_election`** (`:333`): term++, vote for self, persist, send RequestVote to all peers in parallel, count `RVR` replies. Majority → `become_leader`. A higher term in a reply → `become_follower`.
- **`handle_request_vote`** (`:711-768`), rule by rule:
  1. term > current → `become_follower(term)` (clears the vote).
  2. Refuse if term < current ("stale-term").
  3. Refuse if `voted_for_` is set to someone else ("already-voted").
  4. Refuse unless the candidate's (lastTerm, lastIndex) ≥ ours ("log-behind": the **election restriction**).
  5. Otherwise grant: record `voted_for_`, reset the election timer, **persist before replying**.
- **`become_leader`** (`:229-253`): set next = last + 1, match = 0, a grace-period last_contact for every peer; **append a NOOP entry in the new term** (so the current-term commit rule can commit inherited entries); persist; try to commit; wake the replicators.

### 4.4 Replication and commit
- **`replicator_loop(peer)`** (`:450`), one thread per follower:
  - waits on `replicate_cv_` or the 100 ms heartbeat;
  - sends AppendEntries(term, leaderId, prevIndex = next − 1, prevTerm, leaderCommit, entries from `next`, in batches) or **InstallSnapshot** if `next ≤ snapshot_index_`;
  - on success: `match = prev + n`, `next = match + 1`, `maybe_advance_commit()`;
  - on rejection: `next` jumps to the follower's returned **conflict index** (fast backup: one term per round trip, e.g. `NAK n2 mismatch, next 41->35`);
  - a reply with a higher term → step down.
- **`maybe_advance_commit`** (`:255-281`): from the last index down to `commit_index_ + 1`, stop as soon as `term_at(n) != current_term_` (**only current-term entries are counted: the Figure 8 rule**). Count self + peers with match ≥ n. At a majority, set `commit_index_ = n` and wake the applier.
- **`handle_append_entries`** (`:770-882`):
  1. Reject a stale term.
  2. term > current → follower. Recognise the leader, reset the timer.
  3. If `prev_index < snapshot_index_`, drop entries already covered by the snapshot and check consistency at the snapshot point instead.
  4. **Consistency check with fast backup:**
     - if prev is beyond our log, reply conflict = our last + 1;
     - if the term at prev differs, walk back to the **first index of that conflicting term** and reply with it.
  5. For each entry: if we already have that index with the same term, skip (idempotent). If the term differs, **truncate from there** (`DROP`) and append.
  6. **commit = min(leaderCommit, index of the last entry this message verified)**, *not* the end of our log.
  7. Persist if anything changed, then reply.
- **Bug found and fixed [DOC]:** the follower originally used `min(leaderCommit, end of own log)`. A far-behind follower holding *stale* entries beyond a 64-entry batch could mark those stale entries committed. Now it is bounded by the last verified entry.
- **`apply_loop`** (`:588`): while `last_applied < commit_index`, apply the next entry to the state machine, store the result for the waiting client (`applied_results_`), and take a snapshot when the threshold is reached.

### 4.5 Client path (`submit`, `:619`; `handle_client`, `:969`)
- A follower replies with a **redirect** naming the leader's address (the C++ shell and `client.py` follow it).
- The leader refuses up front if it has no quorum ("writes are refused before being appended", so a doomed request never enters the log). It appends the command, persists, wakes the replicators, and waits for **its index to be applied** (with a client timeout). It then returns the result, checking that the entry at that index is still its own (term), since a leader change could overwrite it.
- **`GET` goes through the log**, so reads are linearizable. A deposed leader cannot serve stale data. `--unsafe-local-reads` deliberately breaks this as a negative control.

### 4.6 Storage (`storage.cpp`)
- `persist()`: serialise term, vote and the post-snapshot log → **write a temp file → fsync → rename over the real file → fsync the directory**.
  - Nothing is acknowledged before this completes, so an acknowledged write can't be lost and the file can't be half-written.
  - Cost: each persist rewrites the whole post-snapshot log, which is why compaction speeds up writes (§5).
- Snapshot file: the same temp + fsync + rename pattern. Entries are dropped from the log **only after** the snapshot is durable. A crash in between leaves duplicate coverage, which is cleaned up on restart.
- Bug fixed [DOC]: the node used to open its port *before* loading saved state, so an early AppendEntries could be overwritten by the load.

### 4.7 Snapshots and InstallSnapshot (`take_snapshot`, `:186`; `handle_install_snapshot`, `:915`)
- Every `--snapshot-every` applied entries (default 1,000): serialise the KV store → durable `.snap` → drop the covered log entries. The latest snapshot bytes are kept to send to laggards.
- The follower installs it: replace the state machine, keep any later log entries that agree with it, else discard the log. Then set `snapshot_index/term`, commit and applied.

### 4.8 Fault injection (`fault.h/.cpp`)
- Wraps `TcpTransport` inside each node, so no root or iptables is needed. Per peer it can block (partition), drop requests or replies with probability p, or add random delay.
- A lost message costs the sender its full RPC timeout, as on a real network.
- Rules change at runtime via `FAULT` messages. Client traffic is never blocked, so a client can still reach a stranded minority leader. That is the interesting case for linearizability.

### 4.9 Sharding (README §10; `state_machine.cpp`, `scripts/shardkv.py`)
- Each key hashes to `fnv1a32(key) % N` slots. **Slot ownership is replicated state** in each group's log and snapshots, so no separate configuration service is needed. Requests for an unowned slot get `WRONG_GROUP`.
- **Moving a slot** = `SHARD_FREEZE` (source stops serving and returns the keys as of that log point) → `SHARD_INSTALL` (destination starts serving) → `SHARD_DROP` (source deletes).
  - Freeze is ordered before install, so the two groups never serve the same slot at once.
  - Each step is **idempotent per migration number**, so a crashed coordinator just re-runs the move.

---

## 5. Testing and results [MEASURED]
All numbers were regenerated by `scripts/run_all.sh` (about 75 min) on a shared Linux server, loopback, 2026-10-22. Throughput is single-client, closed-loop and **dominated by one fsync per write**, so it is relative, not a capacity claim.

| Test | Result |
|---|---|
| Chaos, 3 nodes, kill the leader (11 runs) | **11/11 pass**; failover detected in **1.01 s** median (0.81–1.21) |
| Chaos, 5 nodes, kill leader + 1 follower | **11/11**; failover 0.81 s median |
| Same on the original commit `2a19135` (interleaved) | 11/11 and 11/11; 250 vs 258 writes/s (3 nodes), 193 vs 201 (5 nodes): no regression |
| Snapshot test | 10/10; every restarted follower caught up via `InstallSnapshot`; **0 of 500 keys lost** after SIGKILLing all 3 nodes |
| Isolated leader | the majority elects a new leader, the old one refuses writes, everything converges after healing (3 and 5 nodes) |
| **Pre-vote**, follower cut off 6 s (3 nodes) | **on:** term +0, leader deposed **0/5**. **off:** term +7, deposed **5/5** |
| Pre-vote, 5 nodes | on: +0, 0/5. off: +6, 5/5 |
| **Linearizability, 5 nodes** | **50/50 runs linearizable**: 180,462 acknowledged ops, 860 indeterminate, 91 SIGKILLs, 181 partitions, ≥ 567 snapshot installs |
| Linearizability, 3 nodes | 30/30: 91,875 ops, 41 SIGKILLs, 110 partitions |
| Linearizability, 5 nodes, quorum window 4 s | 20/20 |
| **Negative control**: stale local reads, quorum window 4 s | **the checker flagged 13/20 runs** |
| Negative control, default timings | flagged **0/20** (see below) |
| Write throughput per 1,000-write block, no compaction | 235 → 189 → 168 → 149 → 133 → 125 writes/s (persist rewrites a growing log) |
| Same with `--snapshot-every 1000` | 194 → 180 → 174 → 165 → 173 → 179; last block **1.43×** faster (paired median) |
| **Live rebalance 16/0/0 → 6/5/5 under load** | 10/10 pass (ownership, data, linearizability); throughput **432 → 798 ops/s (1.86× paired)**; p99 33 → 24 ms; each slot unavailable 7.5 ms median |
| Same with 38 SIGKILLs during the moves | 10/10; 1.84× |
| Negative control: install before freeze (`--unsafe-move`) | flagged 5/5 |
| Checker self-test vs brute force | agree on 3,000 random small histories (1,880 linearizable, 1,120 not) |

**Linearizability workload:** 6 concurrent clients, 8 keys, 20 s. A **nemesis** partitions the network, isolates the leader, makes links lossy, and SIGKILLs and restarts nodes every 0.8–2.5 s. Snapshots every 40 entries. 20% of requests go to a random node.

**What the negative controls teach [DOC]:**
- With default timings the stale-read bug was never caught (0/20). A client's next write to a cut-off leader blocks until that leader steps down (the 1.5 s quorum window), so almost no client reads from it after the new leader commits.
- Widening the window to 4 s exposes the bug (13/20), while the correct build passes 20/20 under the same seeds.
- So **a pass means "no violation in the interleavings produced", not a proof.**

---

## 6. Design decisions
| Decision | Alternatives | Why | Source |
|---|---|---|---|
| Raft | Paxos / Multi-Paxos, ZAB | understandable; strong leader; well-specified | [RECON] |
| Randomized timeouts 600–1,200 ms, heartbeat 100 ms | fixed timeouts | break split votes; heartbeat ≪ election timeout ≪ failover budget | [DOC] |
| Timing profiles + safety clamps | one fixed setting | Wi-Fi RTTs caused constant re-elections; unsafe combinations are auto-raised | [DOC] |
| NOOP on election | none | lets the current-term commit rule commit inherited entries (Figure 8 safety) | [DOC] |
| Fast backup by conflicting term | decrement next by 1 | one term per round trip | [DOC] |
| Reads through the log | lease reads / ReadIndex | simplest linearizable reads (costs a write path per read) | [DOC] |
| Leader quorum check + step-down | stay leader | clients get a fast "no" instead of hanging | [DOC] |
| Refuse writes before append without quorum | append and wait | doomed requests never enter the log | [DOC] |
| fsync + atomic rename of the whole state | append-only log file | simple and crash-safe; costs a full rewrite per persist (hence compaction) | [DOC] |
| Pre-vote on by default | off | stops term inflation deposing healthy leaders (0/5 vs 5/5) | [MEASURED] |
| In-process fault injection | iptables / tc netem | no root on the shared server; per-peer control | [DOC] |
| Linearizability checker + negative controls | trust passing tests | proves the checker can fail | [DOC] |
| Slot ownership as replicated state | external config service | consistent with the log; no extra component | [DOC] |
| One mutex for the node | fine-grained locks | simplicity; correctness first | [RECON] |
| Thread per peer replicator | event loop | simple blocking RPCs | [RECON] |

---

## 7. Known limits [DOC]
- Each persist rewrites the whole post-snapshot log, and snapshots are taken and sent in one piece under the lock. Fine for thousands to tens of thousands of keys, not gigabytes. Messages are capped at 32 MB.
- **Fixed membership:** no joint consensus or single-server changes.
- **No client sessions / deduplication:** a retried write after a lost reply can apply twice. The tests therefore send each op once and record lost replies as indeterminate.
- The shard coordinator's migration counter lives in memory.
- Windows and macOS binaries were built (`zig`) but not run.

---

## 8. Likely interview questions
1. *The leader crashes after replicating an entry to 1 of 4 followers (5 nodes). Does the entry survive?*
   - Not committed: only 2 of 5 have it, and a majority is 3.
   - It *may* survive: if the follower that has it wins the election (its log is more up to date, so it can win), it will replicate it, and its NOOP commits it.
   - If another node wins, the entry is truncated (`DROP`) on that follower.
   - The client was never acknowledged, so there's no safety violation either way.
2. *Explain Figure 8 and your `maybe_advance_commit`.* → §3.2, §4.4.
3. *Why must votedFor be persisted?* → otherwise a restarted node could vote twice in one term, giving two leaders.
4. *How does fast backup work?* → the follower returns the first index of the conflicting term, and the leader jumps `next` there.
5. *How are reads linearizable?* → through the log. Alternatives: ReadIndex (confirm leadership with a heartbeat round), leader leases (clock assumptions).
6. *What does pre-vote fix? Show the numbers.* → §3.5, §5.
7. *Why 600–1,200 ms, and what changes on Wi-Fi?* → §6; profiles.
8. *How did you test linearizability? Can your checker miss bugs?* → §5 negative controls. Mention indeterminate operations.
9. *What is the commit-index bug you found?* → §4.4.
10. *Why does throughput fall as the log grows?* → the full-state rewrite per persist; compaction gives 1.43×.
11. *How does resharding avoid two owners?* → freeze before install, both ordered through Raft logs; idempotent steps.
12. *How would you add membership changes?* → joint consensus (C_old,new) or single-server changes.
13. *Raft vs Paxos?* → Raft is equivalent in fault tolerance; it is decomposed into election, replication and safety with a strong leader. Multi-Paxos is more flexible (any proposer) but harder to implement.
14. *CAP?* → CP: the minority partition is unavailable for writes and reads.
15. *What breaks at 10× scale?* → full-log rewrites, the single mutex, snapshot size, the thread-per-peer model, fsync per write (batching and group commit would help).

---

## 9. Numbers cheat sheet
| Number | Meaning |
|---|---|
| 100 ms / 600–1,200 ms | heartbeat / election timeout (lan profile) |
| 250 / 1,500–3,000; 500 / 3,000–6,000 | wifi / slow profiles |
| 1.5 s | quorum window (leader steps down) |
| ~1.0 s (3 nodes), 0.81 s (5 nodes) | median failover |
| 11/11 | chaos-test passes per configuration |
| 50/50 and 30/30 | linearizable runs (5 and 3 nodes) |
| 180,462 ops, 91 SIGKILLs, 181 partitions | 5-node linearizability totals |
| 13/20 | stale-read bug caught with a 4 s window (0/20 at default) |
| 0/5 vs 5/5 | leader deposed with vs without pre-vote |
| 235 → 125 vs ~175 writes/s | no compaction vs `--snapshot-every 1000` (1.43×) |
| 1.86× | throughput after rebalancing 16/0/0 → 6/5/5 |
| 7.5 ms | per-slot unavailability during a move |
| 1,000 | default snapshot threshold |
| port + 1000 | telemetry / dashboard port |

---

## Appendix A: Key code, annotated

Real line numbers from `src/` (branch `upgrade-2026-10`). Debug-print lines are left out. `⋮` = left out; `◀` = plain-English note, with the Raft rule it implements.

### Reading the code: C++ constructs
| Construct | Meaning |
|---|---|
| `std::lock_guard` / `std::unique_lock<std::mutex>` | lock the node's single mutex for this scope; `unique_lock` can unlock/relock mid-function |
| `cv.wait_for(lock, 100ms)` | sleep until notified OR 100 ms pass (the heartbeat) |
| `std::thread([captures]{…}).detach()` | start a background thread and don't wait for it (one per vote request) |
| `std::make_shared<std::atomic<int>>(1)` | a vote counter shared by all those threads |
| `log_` | `std::vector<LogEntry{term, index, command}>` holding entries AFTER `snapshot_index_` |
| `term_at(i)`, `entry_at(i)`, `last_log_index()` | helpers that translate a Raft index into the vector (`raft_node.h`) |
| `persist()` | write term + vote + log durably (A.7) |

### A.1 Starting an election (`start_election`)
```cpp
344 ++current_term_;                         // ◀ new term
345 voted_for_ = id_;                        // ◀ vote for myself
⋮                                          // ◀ reset the election timer, persist() BEFORE asking anyone
374 std::thread([this, peer_id, request, term, votes] {    // ◀ RequestVote to each peer in parallel
    ⋮   transport_->send(...);  lock;
    ⋮   if (reply_term > current_term_) { become_follower(reply_term, ...); persist(); return; }
    ⋮   if (role_ != Role::Candidate || current_term_ != term) return;   // ◀ stale reply from an older election
404     if (total >= majority()) become_leader();          // ◀ majority = n/2 + 1
    }).detach();
```

### A.2 Granting a vote (`handle_request_vote`)
```cpp
725 if (term > current_term_) { become_follower(term, ...); dirty = true; }   // ◀ any higher term → follower (clears the vote)
⋮
741 if (term < current_term_) {                                         // ◀ RULE: refuse a stale candidate
743 } else if (voted_for_ != -1 && voted_for_ != candidate) {            // ◀ RULE: one vote per term
747     const bool up_to_date = cand_last_term > my_last_term ||
748         (cand_last_term == my_last_term && cand_last_index >= my_last_index);   // ◀ ELECTION RESTRICTION
    ⋮   if (up_to_date) {
752         voted_for_ = candidate;                                      // ◀ grant, and reset my election timer
    ⋮
766 if (dirty) persist();                                                // ◀ the vote is on disk BEFORE the reply leaves
```

### A.3 Becoming leader: the no-op (`become_leader`)
```cpp
229 void RaftNode::become_leader() {
⋮       // 230-240: role = Leader; for each peer next = last+1, match = 0, last_contact = now
248     log_.push_back(LogEntry{current_term_, last_log_index() + 1, "NOOP -"});   // ◀ an entry of MY term, so the
                                                                                   //   current-term commit rule can also commit inherited entries
250     persist();
251     maybe_advance_commit();
252     replicate_cv_.notify_all();                                                // ◀ wake every replicator now
```

### A.4 Replicating to one follower (`replicator_loop`, one thread per peer)
```cpp
456 replicate_cv_.wait_for(lock, std::chrono::milliseconds(timing_.heartbeat_ms));   // ◀ 100 ms heartbeat, or woken early
463 uint64_t next = std::max<uint64_t>(1, next_index_[peer_id]);
466 if (next <= snapshot_index_) { /* 467-505: entries compacted away → send InstallSnapshot */ }
507 const uint64_t prev_index = next - 1;
510 for (uint64_t i = next; i <= last_log_index() && batch.size() < kMaxBatch; ++i)   // ◀ up to 64 entries
511     batch.push_back(entry_at(i));
522 const uint64_t leader_commit = commit_index_;
523 lock.unlock();                                     // ◀ never hold the lock across the network call
537 if (!transport_->send(peer_id, request, reply, timing_.rpc_timeout_ms)) { ... continue; }
554 lock.lock();
557 if (reply_term > current_term_) { become_follower(reply_term, ...); persist(); continue; }
564 if (reply.num(1) == 1) {                           // ◀ success
565     uint64_t matched = prev_index + batch.size();
566     match_index_[peer_id] = std::max(match_index_[peer_id], matched);
567     next_index_[peer_id] = matched + 1;
574     maybe_advance_commit();
576 } else {                                           // ◀ consistency check failed
577     uint64_t conflict_index = reply.unum(2);
578     next_index_[peer_id] = std::max<uint64_t>(1, conflict_index);   // ◀ FAST BACKUP: jump a whole term per round trip
583     send_now = true;
```

### A.5 The commit rule (`maybe_advance_commit`): Figure 8 safety
```cpp
257 for (uint64_t n = last_log_index(); n > commit_index_; --n) {   // ◀ try the highest index first
258     if (term_at(n) != current_term_) break;      // ◀ RULE: only count replicas for entries of MY term
259     int replicas = 1;                             // ◀ myself
260     for (const auto& [peer_id, matched] : match_index_)
261         if (matched >= n) ++replicas;
263     if (replicas >= majority()) {
270         commit_index_ = n;                        // ◀ everything ≤ n (older terms too) is now committed
    ⋮       apply_cv_.notify_all(); return;
```
Why `:258` matters: an old-term entry can be on a majority and still be overwritten by a later leader (Figure 8). It becomes safe only once an entry of the current term on top of it is committed. The no-op in A.3 guarantees such an entry exists.

### A.6 Receiving AppendEntries (`handle_append_entries`)
```cpp
788 if (term < current_term_) return Message{"AER", {current_term, "0", "0"}};   // ◀ reject a stale leader
797 if (term > current_term_) { become_follower(term, ...); dirty = true; }
⋮                                                  // ◀ recognise the leader; reset the election timer
824 if (check_index > last_log_index()) {          // ◀ I don't even have prevLogIndex
825     const uint64_t conflict = last_log_index() + 1;   // ◀ "start from my end"
⋮       return Message{"AER", {term, "0", conflict}};
834 if (check_index >= 1 && term_at(check_index) != check_term) {   // ◀ CONSISTENCY CHECK failed
    ⋮   const uint64_t bad_term = term_at(check_index);
837     while (first > snapshot_index_ + 1 && term_at(first - 1) == bad_term) --first;   // ◀ first index of that term
    ⋮   return Message{"AER", {term, "0", first}};    // ◀ the leader skips the whole bad term
⋮
848 for (const LogEntry& entry : entries) {
849     if (has_index(entry.index)) {
850         if (term_at(entry.index) == entry.term) continue;     // ◀ already have it (idempotent)
852         log_.resize(entry.index - snapshot_index_ - 1);       // ◀ CONFLICT: truncate from here (logged as DROP)
853     }
854     log_.push_back(entry);
⋮
862 const uint64_t last_new_entry = entries.empty() ? check_index : entries.back().index;
863 if (leader_commit > commit_index_ && last_new_entry > commit_index_) {
865     commit_index_ = std::min(leader_commit, last_new_entry);   // ◀ THE BUG FIX: bounded by what this message VERIFIED,
                                                               //   not by the end of my (possibly stale) log
876 if (dirty) persist();                             // ◀ durable before replying success
```

### A.7 Client writes and reads (`submit`)
```cpp
654 if (!has_quorum()) { result.error = "no quorum: ..."; return result; }   // ◀ refuse BEFORE appending (no doomed entries)
⋮
668 log_.push_back(LogEntry{term, index, command});   // ◀ GET also goes through here → linearizable reads
⋮   persist(); replicate_cv_.notify_all(); maybe_advance_commit();
678 while (last_applied_ < index) {                   // ◀ wait until MY entry is applied
    ⋮   if (role_ != Role::Leader) { error = "lost leadership … (outcome unknown)"; return; }
    ⋮   apply_cv_.wait_until(lock, deadline);       // ◀ client timeout
698 if (it->second.first != term) { result.error = "this write was replaced by a newer leader"; ... }
                                                    // ◀ the index was applied, but with ANOTHER leader's entry
```

### A.8 Crash-safe storage (`storage.cpp`, `include/platform.h`)
```cpp
78  bool write_durably(const std::string& path, const std::string& dir, const std::string& blob) {
79      const std::string tmp = path + ".tmp";
80      int fd = open_file_write(tmp);
82      if (!write_all(fd, blob.data(), blob.size()) || !file_sync(fd)) {   // ◀ fsync the temp file
    ⋮       return false; }
88      if (!replace_file(tmp, path)) return false;   // ◀ atomic rename (POSIX ::rename; Windows MoveFileExA REPLACE_EXISTING)
89      sync_directory(dir);                          // ◀ fsync the directory so the rename itself survives power loss
```
After a crash the file holds either the old state or the new state, never a mix. Nothing is acknowledged before this returns.
