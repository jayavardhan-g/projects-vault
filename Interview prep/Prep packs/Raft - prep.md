> Prep pack for **Raft**, written 2026-11-05 from `raftkv/INTERVIEW_PREP.md`. The vault copy is the one to study and edit.

# raftkv — interview prep

Prepared 2026-11-05 from the repo at `/data/jayavardhan/projects/raftkv`
(branch `upgrade-2026-10`, HEAD `bad36f1`), the original spec
`/data/jayavardhan/projects/RaftKV/Project.md`, the git log and the source.
No code was changed. All `path:line` references are to the current branch
unless marked `main:`.

---

## 0. Read this first: what is original and what is the upgrade

### Two bodies of work

| | **Original (`main`)** | **Upgrade (`upgrade-2026-10`)** |
|---|---|---|
| Commits | 1 commit, `2a19135 "RAFT"`, 2026-08-18, 4,085 lines | 7 commits, all on 2026-10-22, 13:17 to 17:24 |
| Pushed? | Yes, `origin/main` | No. Only `main` exists on `origin` |
| Same as `/data/jayavardhan/projects/RaftKV/RAFT`? | Yes, every tracked file is identical | — |

**Original (`main`) contains:**
- Leader election with randomized timeouts and the election restriction.
- Log replication: one replicator thread per follower, batches of 64, fast backup by conflicting term.
- The current-term commit rule, plus a no-op entry on election.
- Reads that go through the log.
- Check-quorum: a leader steps down if it cannot hear a majority. Writes are refused early without a quorum.
- Persistence: term, vote and the whole log go to one file. Write a temp file, `fsync`, `rename`, then `fsync` the directory.
- `include/platform.h`: one place that maps sockets, fsync and rename to POSIX or Winsock.
- Timing profiles `lan` / `wifi` / `slow`, and `validate_timing` that raises unsafe values.
- Telemetry HTTP/JSON on port+1000, `scripts/dashboard.py`, `scripts/client.py` (with `bench`).
- `scripts/chaos_test.py`: 3 nodes, 100 writes, SIGKILL the leader, check failover, re-read all keys, restart, check catch-up.
- Per-node entry log file (`APPEND`/`COMMIT`/`DROP`), debug console mode, `kill`/`revive` shell commands.

**Upgrade (`upgrade-2026-10`) adds:**
- `d798369` Snapshots, InstallSnapshot, restart from snapshot. **This commit also holds both bug fixes**: the follower commit bound, and loading state before the port opens. Control characters are rejected.
- `1370db2` Pre-vote.
- `8239909` In-process fault injection (`FaultyTransport`), `partition_test.py`, the linearizability checker and its self-test, `linearizability_test.py` with a nemesis, and the `--unsafe-local-reads` negative control.
- `b601b14` `chaos_test.py` for any cluster size and multi-kill, JSON output, `chaos_matrix.py`.
- `1399db2` Sharding across several Raft groups: hash slots, `SHARD_FREEZE/INSTALL/DROP`, `shardkv.py`, `reshard_test.py`, `--unsafe-move` control.
- `4296681` Cross-compile check with `zig c++`, and snapshot info on the dashboard.
- `bad36f1` `run_all.sh`, all of `results/`, README sections 9–10, `compaction_bench.py`.

### Authorship: what anyone with the repo can see

- Every one of the 7 upgrade commits ends with `Co-Authored-By: Claude Opus 5.5` and a `Claude-Session:` link. The `main` commit has no such line.
- All 7 upgrade commits landed within about four hours on one day.
- The upgrade branch is not on `origin`. If you push it, these trailers become public.
- `main` is a single squashed commit. There is no development history to show how it was built.

You said you must defend the project "as its sole author". The git log does not support
that wording for the upgrade. If an interviewer sees the branch, the honest line is:
"The core Raft engine is mine. I used an AI assistant for the hardening pass, and I
reviewed and can explain every line." Do not claim the upgrade as unassisted work.
Your resume bullets describe only `main`-era features, which is the safer ground.

---

## 1. The pitch

### 60 seconds

I built a replicated key-value store on Raft, from scratch, in C++20. There are no
libraries, only the standard library and sockets. Each node is one executable. Nodes find
each other through a small config file, elect a leader, and replicate every write to a
majority before they acknowledge it. State goes to disk with fsync and an atomic rename,
so a crash never leaves a half-written file. Reads also go through the log, so a deposed
leader can never serve stale data. It runs on Linux, macOS and Windows, because all the
OS-specific code sits in one header. To test it, I wrote a chaos script. It kills the
leader with SIGKILL, checks that a new leader takes over, re-reads every acknowledged write,
restarts the dead node and checks that it catches up. A terminal dashboard shows roles,
terms, commit indexes and each follower's replication progress live.

### 3 minutes

**The problem.** One server is a single point of failure. Several copies need to agree on
one order of writes, even when machines crash or the network drops messages. Raft solves
this with a leader and a replicated log.

**Elections.** Every node starts as a follower. If it hears nothing from a leader for a
random 600–1200 ms, it stands for election in a new term. A node grants one vote per term,
and only to a candidate whose log is at least as up to date as its own. A majority of votes
makes a leader. The random timeout makes split votes rare. On Wi-Fi the default timings caused
constant re-elections, so I added timing profiles and a validator that refuses unsafe
combinations.

**Replication.** The leader runs one thread per follower. It sends heartbeats every 100 ms
and new entries right away, up to 64 per message. Each message carries the index and term of
the entry just before the batch. If the follower's log disagrees there, it says no and names
the first index of its conflicting term. The leader then jumps back a whole term per round trip.

**Commit.** An entry is committed when a majority stores it. The leader only counts replicas
for entries from its own term. Older entries commit indirectly behind them. This is the
Figure 8 rule from the paper. Each new leader appends a no-op so the rule cannot stall.

**Reads.** A `get` is a log entry like a `set`. That costs a disk write and a round trip.
In return, a leader cut off by a partition cannot answer reads at all, so it cannot serve
stale data.

**Durability.** Term, vote and log are written to a temp file, fsync'd, renamed over the
real file, and then the directory is fsync'd. Nothing is acknowledged before that.

**Availability.** A leader that cannot hear a majority steps down within 1.5 s. Writes
without a quorum are refused at once with a reason. This is the CP side of CAP.

**Testing.** The chaos test runs a full kill, failover, verify and catch-up cycle and prints
PASS or FAIL. The dashboard shows it live.

*(If asked "what next?" — this is where the upgrade fits.)* Later I added snapshots,
pre-vote, in-process fault injection, a Jepsen-style linearizability checker with negative
controls, and live resharding across several Raft groups. That work was AI-assisted (see §0).

---

## 2. Architecture

### Diagram

```
         client.py / raftkv --client / chaos_test.py          dashboard.py
                     |  TCP, 4-byte length + fields sep by \x1f    | HTTP GET, every 300 ms
                     v                                             v
+-------------------------------- one raftkv process (src/main.cpp:233) -------------------+
|                                                                                          |
|  TcpTransport  (src/transport.cpp)                     Telemetry (src/telemetry.cpp:38)  |
|    accept_loop :212 -> thread per connection            port+1000 -> status_json()       |
|    connection_loop :224 -> handler(msg)                                                  |
|        |                                                                                 |
|  [FaultyTransport src/fault.cpp:107  -- upgrade, only with --fault-injection]            |
|        |                                                                                 |
|        v                                                                                 |
|  RaftNode::handle_message  src/raft_node.cpp:993                                         |
|     RV  -> handle_request_vote :711      PV -> handle_pre_vote :883        (PV upgrade)  |
|     AE  -> handle_append_entries :770    IS -> handle_install_snapshot :915 (upgrade)    |
|     CLIENT -> handle_client :969 -> submit :619                                         |
|                                                                                          |
|  one mutex_ (raft_node.h:164) guards ALL Raft state                                      |
|                                                                                          |
|  threads:                                                                                |
|    ticker_loop :409     every 10 ms: quorum check (leader) / election timeout            |
|    replicator_loop :450 one per peer: AE or IS, heartbeat every heartbeat_ms             |
|    apply_loop :588      applies committed entries to StateMachine, takes snapshots       |
|    detached vote threads in start_election :374 / start_pre_vote :306                    |
|                                                                                          |
|  persist() :174 --> Storage::save  src/storage.cpp:276                                   |
|                       write_durably :212 = tmp file, fsync, rename, fsync(dir)           |
|                       platform.h:254 file_sync, :263 replace_file, :274 sync_directory   |
|                                                                                          |
|  StateMachine::apply  src/state_machine.cpp:242   std::map<string,string>               |
|                       (+ shard slots, :191, upgrade)                                     |
|                                                                                          |
|  outbound: TcpTransport::send :250 -> cached socket per peer, one RPC at a time per peer |
+------------------------------------------------------------------------------------------+
         data/raft_node_<id>.bin  (term, vote, log)     data/raft_node_<id>.snap (upgrade)
         data/node<id>.log        (human-readable APPEND / COMMIT / DROP lines)
```

### Key files

| File | What it holds |
|---|---|
| `src/main.cpp` | Flag parsing; wiring; `ClientRouter` :58; shell :146; startup order :367–381 |
| `include/raft_node.h` | Node state; `majority()` :151; `has_quorum()` :153 |
| `src/raft_node.cpp` | The whole algorithm (1,077 lines; 768 on `main`) |
| `include/timing.h` | Defaults :17–23, profiles :39–50, `validate_timing` :56–75 |
| `src/storage.cpp`, `include/platform.h` | Durable file format and OS calls |
| `src/transport.cpp` | Framing :120–138, server :212–233, client side :235–285 |
| `src/state_machine.cpp` | KV commands; snapshot encode/decode; shard commands |
| `src/fault.cpp` | Drop / delay / partition per peer (upgrade) |
| `scripts/chaos_test.py` | The resume's chaos test (phases at :193–258) |
| `scripts/dashboard.py` | Live monitor, `POLL_INTERVAL = 0.3` at :22 |
| `scripts/linearizability.py` | WGL checker, `check_register` :99 (upgrade) |

### One request, end to end: `python3 scripts/client.py set city mumbai`

1. `client.py:114` `command()` sends `CLIENT\x1fSET city mumbai` to its cached leader, or
   to the first node in the list.
2. Say it hit a follower. `transport.cpp:224` reads one frame and calls the handler.
   That calls `raft_node.cpp:993` `handle_message`, then `:969` `handle_client`, then
   `:619` `submit`.
3. `submit` takes `mutex_`. The role is not Leader, so it returns a redirect with
   `leader_address_locked()` (:636–644). The reply is `CLIENTR REDIRECT 10.0.0.2:5001`.
4. `client.py:144` remembers the leader and resends there.
5. On the leader, `submit` checks: alive, no control characters (:630, upgrade), role,
   then `has_quorum()` (:654). It appends `LogEntry{term, index, cmd}` (:668), calls
   `persist()` (:673; this means fsync while holding the mutex), wakes replicators (:674),
   then waits on `apply_cv_` until `last_applied_ >= index` or the deadline (:677–691).
6. Each `replicator_loop` (:450) wakes. It builds a batch from `next_index_[peer]`, at most
   64 entries (:509–512), with `prev_index/prev_term` and `commit_index_`. It unlocks and
   calls `transport_->send` (:537).
7. The follower runs `handle_append_entries` (:770). It checks the term, resets its timer,
   and checks consistency at `prev_index` (:824–846). It appends or truncates (:848–857)
   and moves `commit_index_` to `min(leaderCommit, last new entry)` (:862–865). It calls
   `persist()` and replies `AER term 1 0`.
8. Back on the leader (:554–575): it updates `match_index_`/`next_index_` and calls
   `maybe_advance_commit()` (:255). That finds the highest N with term == current term
   held by a majority, sets `commit_index_ = N` and wakes `apply_cv_`.
9. `apply_loop` (:588) applies index N to `StateMachine::apply` (`state_machine.cpp:242`).
   It stores `applied_results_[N] = {term, "OK"}` (:607).
10. `submit` wakes. It checks that the applied entry at its index has its own term (:698;
    if not, "replaced by a newer leader"). It returns `OK`.
11. The follower learns the new commit index on the next AE (a heartbeat within 100 ms)
    and applies it too.

A `get` takes exactly the same path. Only the command text differs.

---

## 3. Design decisions

Legend: **[doc]** = stated in the README, a code comment or a commit message (cited).
**[code]** = reconstructed from the code. The docs do not say why. Never present
a [code] reason as remembered history unless you actually remember it.

### Original (`main`)

**D1. Implement Raft from primitives in C++20, no third-party libraries.**
- Alternatives: etcd/raft, braft, NuRaft; Go.
- Chosen: plain C++20, standard library, OS sockets.
- Why: [doc] `Project.md` says "implemented from primitives (not just wrapping etcd/Raft
  libraries, to keep the consensus implementation itself as the contribution)". The README
  says "No libraries to install — just a C++ compiler and Python 3".
- Cost: you hand-roll framing, threading and persistence. Each is a place for bugs.

**D2. One mutex for all Raft state.**
- Alternatives: fine-grained locks; a single-threaded event loop with message queues (as in etcd/raft).
- Chosen: `mutex_` (`raft_node.h:164`) taken by every handler and loop.
- Why: [code] simplest way to keep invariants across term, log, commit index and roles.
- Cost: `persist()` does fsync while holding it (`raft_node.cpp:673`, `:876`), so every
  RPC on that node waits for the disk. `status()` also serialises the whole state
  under the lock to hash it (:1032).

**D3. One replicator thread per follower, blocking RPCs.**
- Alternatives: async I/O, pipelining several AEs in flight.
- Chosen: `replicator_loop` per peer (`raft_node.cpp:450`), one RPC in flight per peer.
- Why: [doc] README "The leader keeps one thread per follower, heartbeating every 100 ms
  and sending new entries immediately." The reason for blocking I/O is [code]: simplicity.
- Cost: no pipelining. Throughput is about one batch per round trip per follower. A lost RPC
  stalls that follower for a full `rpc_timeout`. This is why the validator demands
  `election_min ≥ rpc_timeout + heartbeat`.

**D4. Own wire format: 4-byte big-endian length, then text fields split by `\x1f`.**
- Alternatives: protobuf/gRPC, JSON.
- Chosen: `transport.cpp:102–138`. Numbers travel as decimal text. Entries are packed with
  `\x1e`/`\x1d` (`raft_node.cpp:18–43`).
- Why: [code] no dependencies. It is also endian-neutral, which helps the cross-OS claim.
  Python clients speak it in about 20 lines.
- Cost: commands must not contain those bytes. `main` did not check this; the upgrade
  rejects control characters (`raft_node.cpp:628–635`, commit `d798369`). Frames are capped
  at 32 MB (`transport.cpp:134`).

**D5. Election timeout random in [600, 1200] ms, heartbeat 100 ms, timer re-drawn on every reset.**
- Alternatives: the paper's 150–300 ms; a fixed timeout.
- Chosen: `timing.h:18–21`. A fresh random value on every reset (`raft_node.cpp:226, 756, 807`).
- Why: [doc] README "Each node picks a fresh random timeout between 600 and 1200 ms whenever
  it hears from a leader, which keeps two nodes from campaigning at the same moment."
  [code] The 600 floor follows from the validator rule `election_min ≥ rpc_timeout(400) +
  heartbeat(100)`: one lost RPC must not trigger an election.
- Cost: failover takes about 0.8–1.2 s instead of the paper's ~300 ms.

**D6. Timing profiles `lan` / `wifi` / `slow` and an auto-correcting validator.**
- Alternatives: one fixed set; adaptive timeouts from measured RTT.
- Chosen: `timing.h:39–50` (lan 100/600–1200, wifi 250/1500–3000, slow 500/3000–6000) and
  `validate_timing` `timing.h:56–75`, which raises values and prints why.
- Why: [doc] README §8: on Wi-Fi, "heartbeats then arrive late, followers campaign against a
  perfectly healthy leader … the term counter climbs into the thousands". The rule is
  "heartbeat ≪ election timeout ≪ time you are willing to wait for failover".
- Cost: slower failover on `wifi` (about 3 s). Every node must use the same profile. The
  dashboard warns on a mismatch (`dashboard.py:196–204`).

**D7. Commit only entries of the current term; append a no-op on election.**
- Alternative: count replicas for any entry (unsafe, Figure 8).
- Chosen: `maybe_advance_commit` `raft_node.cpp:255–276` stops at the first entry not from
  `current_term_` (:258). `become_leader` appends `NOOP -` (:246–250).
- Why: [doc] README "The leader only counts replicas for entries from its own term … Each new
  leader appends a no-op entry so this rule can never stall. This is what prevents a committed
  value from ever being overwritten." See §6 for Figure 8.
- Cost: one extra log entry per election.

**D8. Reads go through the log.**
- Alternatives: ReadIndex (thesis §6.4); leader leases; follower reads.
- Chosen: `GET` is submitted like any write (`submit` :619; `main.cpp:208`).
- Why: [doc] README "get goes through the log like a write, so you can never read a stale
  value from a leader that has already been deposed". The shell help says "linearizable read
  (goes through the log)" (`main.cpp:134`).
- Cost: every read pays an fsync on the leader and on a follower, plus a round trip, and it
  grows the log. See §6 and Q12 for why check-quorum alone does not make local reads safe.

**D9. Check-quorum: a leader steps down if a majority has not answered within 1.5 s.**
- Alternative: plain Raft, where an isolated leader keeps its title until it hears a higher term.
- Chosen: `ticker_loop` `raft_node.cpp:419–431`, using `reachable_nodes()` :69–76 over
  `last_contact_`.
- Why: [doc] README §7 "A leader nobody follows is useless, and keeping the title would only
  make clients wait for commits that can never happen."
- Cost: a slow-but-alive majority can cause a needless step-down. This is not a safety
  mechanism; it only improves liveness and error messages.

**D10. Refuse writes up front when there is no quorum.**
- Chosen: `submit` :652–664.
- Why: [doc] README §7 "Writes are refused before being appended, so a doomed request never
  enters the log".
- Cost: the leader's view of "reachable" can be up to 1.5 s stale, so this is a hint, not a
  guarantee.

**D11. Fast log backup: the follower returns the first index of its conflicting term.**
- Alternatives: decrement `nextIndex` by one; the full `ConflictTerm` + `ConflictIndex`
  scheme from the Students' Guide.
- Chosen: `raft_node.cpp:824–846` (follower); `:576–584` (leader sets `next = conflict`).
- Why: [doc] README "so the leader can rewind a whole term per round trip instead of one entry
  at a time".
- Cost: the leader does not search its own log for that term, so it may resend a few entries
  that already match. Harmless.

**D12. Persist by rewriting the whole state file: temp + fsync + rename + fsync(dir).**
- Alternatives: append-only WAL with checksums and segments (etcd, LogCabin); an embedded
  store (RocksDB, LMDB).
- Chosen: `storage.cpp:212–226` (`main:` same logic inside `Storage::save`).
  `platform.h:254–282` maps it to `fsync`/`rename`/dir-fsync on POSIX and
  `_commit`/`MoveFileExA(...WRITE_THROUGH)` on Windows.
- Why: [doc] README "written to a temp file, fsync'd, then renamed over the real file …
  the file can never be found half-written".
- Cost: [doc] "Each save still rewrites the whole (post-snapshot) log". On `main`, with no
  compaction, each write costs O(log size). The benchmark shows 235 → 125 writes/s over
  6,000 writes (`results/compaction_bench.json`, compaction-off arm). There are also 2
  fsyncs per persist and no group commit.

**D13. Telemetry on a separate port (raft port + 1000), its own thread.**
- Why: [doc] README §3 "so watching the cluster never slows down or interferes with consensus".
- Caveat [code]: `status_json()` still takes `mutex_` and hashes the whole state machine
  (`raft_node.cpp:1011, 1032`). So it does touch the consensus lock.

**D14. Redirect, not proxy, for non-leader requests.**
- Chosen: the follower replies `REDIRECT <leader>` (`raft_node.cpp:976–978`). Clients cache
  the leader (`client.py:144`, `main.cpp:58–96`).
- Why: [doc] README §4 "a follower answers with a redirect naming the leader, the client
  follows it and remembers it".

**D15. A `kill` / `revive` shell command as a soft crash.**
- Chosen: the `alive_` flag makes the node ignore all RPCs (`raft_node.cpp:994–999`).
  The chaos test uses real SIGKILL instead.
- Cost: `kill` does not lose in-memory state, so it is not a real crash test.

**D16. All OS code in one header.**
- Why: [doc] README "Everything OS-specific lives in `include/platform.h` … no other file has
  platform code."

### Upgrade (`upgrade-2026-10`)

**U1. Snapshots: full state-machine image, saved before the log is trimmed.**
- Chosen: `take_snapshot` `raft_node.cpp:186–206` every `--snapshot-every` (default 1000)
  applied entries. Separate `.snap` file via the same `write_durably`. On restart, leftover
  covered entries are dropped (:130–134).
- Why: [doc] commit `d798369`; README "Snapshots".
- Cost: [doc] "a snapshot is taken and sent in one piece while the node holds its lock". It is
  hex-encoded in one message, so a state over ~16 MB cannot be sent within the 32 MB frame cap.

**U2. Pre-vote.**
- Chosen: `start_pre_vote` :283–331; `handle_pre_vote` :883–913. A peer refuses while it has
  heard from a leader within `election_min_ms`.
- Why: [doc] commit `1370db2`: "a node that was partitioned away cannot depose a healthy
  leader when it returns". Measured in `results/partition_test*.json`.

**U3. In-process fault injection instead of iptables/tc.**
- Chosen: `FaultyTransport` (`src/fault.cpp`), with rules changed at runtime by `FAULT`
  messages. Client traffic is never blocked.
- Why: [doc] commit `8239909` "no root needed"; `fault.h:23–34`. (The machine has no sudo.)
- Cost: it does not model real TCP behaviour (reordering, half-open connections). A
  "dropped" reply means the handler already ran. That is realistic, but it is all one host.

**U4. Linearizability checker: WGL + Lowe memoisation, per key, with indeterminate ops.**
- Why: [doc] README §9; `linearizability.py:1–25`.
- Cross-checked against brute force on 3,000 random histories (I re-ran this: 1,880 ok,
  1,120 violation, 1,232 with indeterminate ops, 3,000/3,000 agree).

**U5. Negative controls (deliberate bugs behind flags).**
- `--unsafe-local-reads` (`raft_node.cpp:646–650`) and `reshard_test.py --unsafe-move`
  (`shardkv.py:148–153`).
- Why: [doc] README "A checker that never fails proves nothing".

**U6. Sharding: FNV-1a hash slots; ownership kept in each group's replicated state; moves are
three log commands.**
- Alternatives: a separate config service (like the shardctrler in MIT 6.5840); range
  partitioning; lazy (pull-on-demand) migration.
- Chosen: `state_machine.cpp:191–240`; coordinator `shardkv.py:143–165`.
- Why: [doc] README §10 "No separate configuration service is needed". "The freeze is ordered
  in the source's log before the install is ordered in the destination's, so there is never a
  moment when both groups accept requests for the slot."
- Cost: [doc] the coordinator's migration counter lives in memory. A slot is briefly
  unavailable (median 7.5 ms). Migration is eager: the whole slot moves in one entry.

**U7. Fixes** — see §4 stories 1 and 2.

---

## 4. Hardest problems and bugs (STAR)

Sources: README §9 "Bugs found along the way", commit messages, README §8 and §9.
**The repo does not say how either bug was found.** Prepare your own true account before
the interview. Do not invent one.

### Story 1 — Follower commit index past verified entries (fixed in upgrade, `d798369`)

- **Situation.** On `main`, a follower set its commit index to
  `min(leaderCommit, last_log_index())` (`main:src/raft_node.cpp:663`). Raft Figure 2 says
  `min(leaderCommit, index of last new entry)`.
- **Task.** Make sure a follower only commits entries that the current message proved equal
  to the leader's log.
- **Action.** The bound became the last entry received, or `prev_index` for an empty
  heartbeat (`raft_node.cpp:859–865`).
- **Result.** [doc] "When a far-behind follower held stale entries past a 64-entry batch,
  that could mark stale entries committed."

A concrete trace (reconstructed; learn it, interviewers love it):
1. Leader A in term 3 replicates 11..74 to everyone. Those commit.
2. A then appends 75..200 (term 3) but only follower B receives them. A crashes.
3. C wins term 4. It does not have 75..200. It writes 75..300 in term 4 and commits them with A's replacement.
4. B comes back. C sends `prev=300`. B has a gap, so it replies `conflict=201`.
5. C sends `prev=200 (t4)`. B has t3 there, so it replies with the first index of its term 3, which is 11.
6. C sends `prev=10` with entries 11..74 (term 3) and `leaderCommit=300`.
7. All 64 entries match, so nothing is truncated. Old code: `commit = min(300, 200) = 200`.
   B now applies its stale 75..200 as committed. That is a state-machine safety violation.
8. New code: `commit = min(300, 74) = 74`.

### Story 2 — Port opened before state was loaded (fixed in upgrade, `d798369`)

- **Situation.** On `main`, `transport.start()` ran at `main:src/main.cpp:338`, telemetry
  bound at `:347`, and only then `node.start()` loaded the disk state at `:352`.
- **Task.** Make sure no RPC can touch Raft state before the saved state is back in memory.
- **Action.** `RaftNode::start` now loads snapshot and log first and can fail with a reason.
  `main.cpp:367–373` runs before `transport.start` at `:375`.
- **Result.** [doc] README: "a node opened its port before loading its saved state, so an
  early AppendEntries could be overwritten by the load."

Worse than the README says (reconstructed): any RPC with a higher term in that window calls
`become_follower` and then `persist()`. `persist()` writes the *empty* in-memory log over
`raft_node_<id>.bin`. Then `start()` loads that file. The node has forgotten every entry it
had acknowledged. That can un-commit a committed entry. It is a durability bug, not just a
lost message. The window is a few milliseconds, but a restarting node gets a leader
heartbeat every 100 ms.

### Story 3 — Election storms on Wi-Fi (original)

- **Situation.** [doc] README §8: on Wi-Fi, RTTs over 100 ms made followers time out under a
  healthy leader. The log filled with `NOOP` entries and terms went into the thousands.
- **Task.** Keep the cluster stable on slow networks without losing fast failover on a LAN.
- **Action.** Named profiles (`timing.h:39–50`). A validator encodes the timing rules
  (`timing.h:56–75`). The dashboard shows each node's timings and flags mismatches. The docs
  tell you how to pick a profile from the RTT column.
- **Result.** Documented guidance only. **No measurement in the repo shows the before and
  after on Wi-Fi.** Tell this story only as you remember it.

### Story 4 — The stale-read control did not fire (upgrade)

- **Situation.** With `--unsafe-local-reads`, the checker flagged 0/20 runs at default
  timings (`results/lin_default_unsafe.json`).
- **Task.** Prove the test can catch a real stale read.
- **Action.** [doc] README §9 explains: a client's next write to the cut-off leader blocks
  until the old leader steps down (1.5 s), so few clients are still reading from it once the
  new leader commits something. The quorum window was widened to 4 s, and the workload sends
  20% of requests to a random node.
- **Result.** 13/20 flagged with the bug. The correct build was 20/20 clean on the same seeds
  and timings (`lin_wide_unsafe.json`, `lin_wide_safe.json`). [doc] The two changes "are not
  separated". Lesson: a passing run means "no violation in the interleavings produced".

### Story 5 — A returning node deposes a healthy leader (upgrade)

- **Situation.** Without pre-vote, an isolated follower keeps raising its term. When it
  returns, its higher term forces the leader to step down.
- **Action.** Pre-vote, plus an A/B test with interleaved runs (`partition_test.py`).
- **Result.** 3 nodes, 6 s isolation: pre-vote on gave a term rise of +0 and 0/5 deposed.
  Off gave +7 median (5–7) and 5/5 deposed. 5 nodes: +0, 0/5 vs +6, 5/5.

### Story 6 — Throughput decayed as the log grew (upgrade measures `main`'s behaviour)

- **Situation.** Each persist rewrites the whole log.
- **Result.** With compaction off: 235 → 125 writes/s over six 1,000-write blocks. With
  `--snapshot-every 1000`: 194 → 179, and the last block was 1.43× faster (paired median of
  3 pairs). Note the first block is *slower* with compaction on (194 vs 235). Snapshots cost
  something.

---

## 5. Numbers sheet

**Your two resume bullets contain no numbers.** Good: nothing to retract. But they make
claims that work like numbers ("zero data loss", "cross-platform"). Interviewers will
also ask "how fast?" and "how long is failover?". Every result below comes from
`results/`. That is a shared Linux server, loopback networking, measured 2026-10-22. Python
3.12.3, g++ 13.3.0. **No CPU, disk or load details are recorded.** I did not re-run the
cluster tests. I re-ran only the checker self-test.

### Claims in the bullets

| Claim | How the repo supports it | Caveats | Honest wording |
|---|---|---|---|
| "from-scratch Raft … in C++" | True. No consensus or RPC libraries. | Upgrade AI-assisted (§0). | Keep as is for `main`. |
| "leader election, log replication, conflict resolution" | `raft_node.cpp` election :333, replication :450, conflict backup :824–846 | `main` had the follower commit bug (§4.1) | Keep. Be ready to tell story 1. |
| "crash-safe persistent storage" | fsync + rename + dir fsync, `storage.cpp:212` | `main` had the port-before-load bug that could wipe the log on restart. `persist()` ignores write errors (`raft_node.cpp:181`). A corrupt `.bin` makes the node start empty, silently (`:120`). SIGKILL tests cannot prove fsync works (page cache survives). macOS `fsync` does not flush the drive cache (needs `F_FULLFSYNC`). | "crash-consistent storage (fsync + atomic rename)". Don't say "crash-safe" without the caveats. |
| "cross-platform cluster support" | `platform.h` Winsock/POSIX; endian-neutral text wire format. Upgrade: zig cross-compile for Windows, macOS x86/arm and Linux, 0 warnings (`results/cross_build.txt`) | [doc] "the Windows and macOS binaries were not run". **No log in the repo of a mixed-OS cluster run.** README sample lines show `192.168.1.x` peers, which hints at a multi-laptop run, but that is not evidence. | If you ran it: "ran a mixed macOS/Windows/Linux cluster over Wi-Fi". If not: "portable to Linux, macOS and Windows (single platform-abstraction header)". |
| "automated chaos-testing pipeline to inject node failures" | `main`: one script, one scenario (SIGKILL the leader once). Upgrade: matrix runner, partitions, lossy links, nemesis. | On `main` "pipeline" is generous: it is one test, not CI. | `main`-only: "an automated chaos test". With upgrade: "a fault-injection suite". |
| "verify leader failover" | `chaos_test.py:208–224` | Failover = first telemetry poll showing a LEADER among survivors | Keep. |
| "zero data loss" | Phase 3 re-reads all 100 acknowledged keys (`chaos_test.py:225–230`). 11/11 runs × 2 sizes × 2 binaries pass. Upgrade: 0 of 500 keys lost after SIGKILLing all 3 nodes, 10/10 (`snapshot_test.json`). | Sequential single client; SIGKILL is not power loss. | "no acknowledged write lost across leader crashes in N runs". |
| "automatic follower catch-up" | Phase 5 (`chaos_test.py:238–256`): restarted node reaches the survivors' commit index within 15 s | `main`'s phase 6 compares only commit index and term, not contents. The upgrade harness compares a state digest (`raftharness.py:146–159`). | Keep. |
| "live dashboard tracking cluster health and replication progress" | `dashboard.py`, polls every 300 ms; node table, per-follower match/next bars, events, quorum line | — | Keep. |

### Numbers you will be asked for

| Number | Source | How measured | Caveats | In repo? |
|---|---|---|---|---|
| **Failover ≈ 1.0 s median (3 nodes), 0.81 s (5 nodes, 2 killed)**; range 0.81–1.21 | `chaos_matrix.json` | SIGKILL the leader; time until a survivor's telemetry says LEADER; n=10 per arm after 1 warm-up | Values are quantized to ~0.2 s by the poll loop (0.81 / 1.01 / 1.21). Loopback only. 5 nodes is faster because 3 survivors race their timers (expected min of 3 uniforms on [600,1200] is 750 ms, vs 800 ms for 2). | Yes |
| Same on original code | `chaos_matrix.json` | Baseline binary built from `2a19135`, runs interleaved | 1.01 s and 0.81 s; no change | Yes |
| **~250 writes/s (3 nodes), ~200 (5 nodes)** | `chaos_matrix.json` (258 / 201 upgrade; 250 / 193 original) | 100 sequential SETs, 1 client, new TCP connection each, loopback | [doc] "dominated by one fsync per write … treat them as relative, not as a capacity claim". An earlier run of the original gave 223 / 184 (`chaos_baseline.json`), so expect ±15% noise. | Yes |
| "bench 500 → 446 writes/sec" | README §4 sample output only | Unknown | **No data behind it.** Do not quote it. | No |
| Writes/s under 20% loss + 5–30 ms delay | `partition_test.json` / `_5n` | 100 writes | 12.4 (3 nodes), 8.9 (5 nodes) | Yes |
| Linearizable 50/50 (5 nodes), 30/30 (3 nodes) | `lin_5n.json`, `lin_3n.json` | 6 clients, 8 keys, 20 s per run, nemesis every 0.8–2.5 s, snapshots every 40 | 180,462 + 91,875 acknowledged ops. Passing ≠ proof. | Yes (upgrade) |
| Stale-read control 13/20 caught (4 s window), 0/20 at default | `lin_wide_unsafe.json`, `lin_default_unsafe.json` | Same seeds as the safe arm | Workload and window changed together | Yes (upgrade) |
| Checker vs brute force 3,000/3,000 | `test_linearizability.py` | Random histories of 1–6 ops | `results/checker_selftest.txt` only shows "Ran 11 tests OK". I re-ran it: 1,880 / 1,120 / 1,232 confirmed. Small histories only. | Yes |
| Pre-vote: deposed 0/5 vs 5/5 | `partition_test*.json` | n=5 per arm | Small n, but the effect is all-or-nothing | Yes (upgrade) |
| Compaction: 1.43× last block | `compaction_bench.json` | 3 measured pairs | n=3 | Yes (upgrade) |
| Resharding 1.86× ops/s, p99 33 → 24 ms, slot unavailable 7.5 ms median | `reshard.json` | 3 groups × 3 nodes on one host, 8 clients, 16 slots, 10 runs | All 9 processes share one machine and disk. The gain is from 3 leaders fsyncing in parallel, not more hardware. With chaos, max unavailability was 1.18 s (`reshard_chaos.log`). | Yes (upgrade) |

Reproduce: `./scripts/run_all.sh` (about 75 min, no GPU). It rebuilds the baseline from
`2a19135`. It needs ports 7101–7861 free.

---

## 6. Prerequisite concepts

**Replicated state machine.** Each replica runs the same deterministic program over the same
ordered log. Same log, same state. Consensus is only about agreeing on the log order.
*Source:* Schneider, "Implementing Fault-Tolerant Services Using the State Machine Approach" (1990).

**Raft basics: terms, roles, the two RPCs.** Time is split into numbered terms, each with at
most one leader. RequestVote elects. AppendEntries replicates and heartbeats. Any message
with a higher term turns the receiver into a follower of that term.
*Source:* Ongaro & Ousterhout, "In Search of an Understandable Consensus Algorithm" (USENIX ATC 2014), Figure 2.

**Election timeouts.** A follower that hears nothing for a random timeout starts an election.
Randomness makes split votes rare. The rule is broadcastTime ≪ electionTimeout ≪ MTBF.
Here: heartbeat 100, rpc-timeout 400, election 600–1200 ms. On `wifi`: 250, 1000, 1500–3000.
The validator also requires `election_min ≥ rpc_timeout + heartbeat`, because one lost RPC
blocks that peer's replicator thread for a full rpc-timeout.
*Source:* Raft paper §5.2 and §5.6.

**Election restriction and Leader Completeness.** A voter refuses a candidate whose last log
entry is older: lower term, or same term and shorter log (`raft_node.cpp:747–748`). Every
committed entry is on a majority, and any winner needs a majority, so the two sets overlap.
The winner therefore has every committed entry. Leaders never need to fetch entries from
followers.
*Source:* Raft paper §5.4.1.

**Log Matching.** If two logs have an entry with the same index and term, the logs are
identical up to there. The `prevLogIndex/prevLogTerm` check enforces this by induction.
*Source:* Raft paper §5.3.

**Commit rule and Figure 8.** "Stored on a majority" is not enough for an entry from an
older term. The paper's Figure 8:
- (a) S1 leads term 2 and puts index 2 on S1 and S2.
- (b) S1 crashes. S5 wins term 3 with votes from S3, S4 and S5, and writes index 2 = term 3, locally only.
- (c) S5 crashes. S1 wins term 4 and copies its term-2 entry to S3. Now it is on a majority.
- (d) If S1 now treated it as committed and crashed, S5 could win term 5 (its last term 3 beats their 2) and overwrite index 2 everywhere.
- (e) The fix: a leader only commits by counting entries of its *own* term. Once a term-4 entry is on a majority, S5 cannot win, and everything before it is committed too.

Code: `raft_node.cpp:258`. The no-op at `:248` gives a new leader a current-term entry right away.
*Source:* Raft paper §5.4.2, Figure 8.

**Pre-vote and check-quorum.** Pre-vote: ask "would you vote for me?" before raising your term,
so a partitioned node cannot disrupt a healthy leader. Check-quorum: a leader steps down when
it cannot hear a majority.
*Source:* Ongaro, PhD thesis "Consensus: Bridging Theory and Practice" (2014), §9.6 and §6.2.

**Linearizability.** Each operation appears to take effect at one instant between its call
and its return. All clients see one order that respects real time. It is local: a history is
linearizable if and only if each object's sub-history is. That is why the checker splits by key.
Checking is NP-complete in general. WGL search with memoisation works on small histories.
*Sources:* Herlihy & Wing, "Linearizability" (TOPLAS 1990); Lowe, "Testing for Linearizability" (2017); Porcupine (Athalye) and Jepsen/Knossos.

**Indeterminate operations.** A request whose reply was lost may have happened or not. A
correct checker must allow both. That is why `ret: None` exists (`linearizability.py:11–13`)
and why timeouts are "info", not "fail" (`raftharness.py:212–253`).
*Source:* Kingsbury, Jepsen posts.

**Linearizable reads in Raft.** Options: (1) put reads in the log (this project); (2) ReadIndex:
record the commit index, confirm leadership with one heartbeat round, wait until applied, then
read; (3) leases: skip the heartbeat round if the lease is within bounded clock drift.
*Source:* Ongaro thesis §6.4.

**fsync and atomic rename.** `write()` only reaches the page cache. `fsync(fd)` forces data
and metadata to stable storage. `rename()` atomically replaces a name on POSIX within one
filesystem, but the rename is a directory change, so you must fsync the directory too. Order:
write tmp → fsync tmp → rename → fsync dir. A crash at any point leaves the old file or the new
one, never a mix. A process crash (SIGKILL) does not drop the page cache. Only power loss or a
kernel crash tests fsync.
*Sources:* Pillai et al., "All File Systems Are Not Created Equal" (OSDI 2014); the PostgreSQL "fsyncgate" discussion (2018).

**CAP and PACELC.** During a partition you must choose consistency or availability. Raft
chooses consistency: the minority side refuses. PACELC adds: Else (no partition) you trade
latency for consistency. Reads through the log pay that latency.
*Sources:* Gilbert & Lynch, "Brewer's Conjecture…" (2002); Abadi, "Consistency Tradeoffs in Modern Distributed Database System Design" (IEEE Computer 2012); Kleppmann, "A Critique of the CAP Theorem" (2015).

**Raft vs Paxos.** Both need majorities and use ballot/term numbers. Raft has a strong leader,
log entries flow only leader → follower, and the election restriction means a new leader
already has every committed entry. In Multi-Paxos, a new leader runs phase 1 and must learn
and re-propose values it did not have. Raft is roughly Multi-Paxos with those choices fixed.
*Sources:* Lamport, "Paxos Made Simple" (2001); Howard & Mortier, "Paxos vs Raft: Have we reached consensus on distributed consensus?" (PaPoC 2020).

**Snapshots and InstallSnapshot.** Replace the log prefix with a state image. A follower that
needs compacted entries gets the image.
*Source:* Raft paper §7.

**Sharding and migration.** Split keys across independent consensus groups. Move ownership so
that no two groups serve a key at once.
*Sources:* MIT 6.5840 Lab 5 (ShardKV); Corbett et al., "Spanner" (OSDI 2012) for the production version.

---

## 7. Fifty likely questions

### Raft core

**Q1. Walk me through an election.**
A follower's timer fires. With pre-vote on (upgrade), it first asks peers if they would vote at
term+1. Peers say no while they still hear a leader. With a majority yes, it becomes a candidate.
It bumps its term, votes for itself, persists, and sends RequestVote. A voter grants if the term
is current, it has not voted for someone else this term, and the candidate's log is at least as
up to date. A majority makes it leader. It appends a no-op and starts heartbeats.
`raft_node.cpp:283, 333, 711, 229`.

**Q2. Why random timeouts? What is the chance of a split vote?**
If two followers time out within about one RTT of each other, they split the vote. The window is
600 ms wide and loopback RTT is under 1 ms, so the chance per election is well under 1%. A split
vote just costs another timeout. The timer is re-drawn on every reset, so repeated collisions are
unlikely.

**Q3. Why 600–1200 ms and not the paper's 150–300?**
The validator requires `election_min ≥ rpc_timeout + heartbeat`. Each follower has one blocking
replicator thread. A lost RPC holds that thread for the whole 400 ms rpc-timeout, so the next
heartbeat can be ~500 ms late. A 300 ms timeout would trigger elections on one lost packet.
`timing.h:67`.

**Q4. Explain the Wi-Fi profiles.**
`wifi` is heartbeat 250, rpc 1000, election 1500–3000, quorum window 3500. `slow` doubles that.
Rule of thumb from the README: if RTT is regularly above a fifth of the heartbeat, go up one
profile. Cost: failover around 3 s instead of 1 s. All nodes must match.

**Q5. What happens if nodes use different profiles?**
Fast followers can time out under a slow leader whenever one heartbeat is lost. Pre-vote's
"leader alive" window is each node's own `election_min`, so they also disagree on when a leader
is dead. The dashboard turns red on mismatched timings. *(Hostile follow-up: "Why not negotiate
timings?" — Good idea, not done. It would be a cluster config entry in the log.)*

**Q6. State the commit rule and why it exists.**
The leader commits index N only if a majority has it and `log[N].term == currentTerm`. Older
entries commit indirectly. Without this, Figure 8 shows a "majority-replicated" entry being
overwritten by a later leader.

**Q7. Draw Figure 8.** See §6. Practise it on a whiteboard with five servers.

**Q8. Why the no-op at election?**
After a leader change, entries from older terms cannot commit until some current-term entry
commits. Without a no-op, they would wait for the next client write. The no-op also lets the
new leader learn its own commit index fast.

**Q9. How does a follower with a wrong log get fixed?**
The leader's AE fails the `prev` check. The follower returns the first index of its conflicting
term, or its log end plus one if there is a gap. The leader sets `nextIndex` there and retries
at once. When entries arrive that conflict, the follower truncates from the first conflict and
appends. It logs a `DROP` line.

**Q10. Why does the follower bound its commit index by the last new entry?**
Story 1. Entries after the batch have not been checked against the leader. They may be stale
from an old term.

**Q11. Why do reads go through the log?**
A leader can be deposed without knowing it. If it answered reads from memory, a client could
read an old value after a newer write was acknowledged by the new leader. That breaks
linearizability. Putting the read in the log means it only returns if a majority still accepts
this leader in this term.

**Q12. Your leader already steps down without a quorum. Isn't that enough for local reads?**
No. The quorum window is 1.5 s, but followers can elect a new leader after only 600 ms. For
about a second, two nodes both think they lead. The old one would serve stale reads. The
validator even forces `quorum_window ≥ election_max` (`timing.h:71`). That is the opposite of
what a safe lease needs (lease < election_min − clock drift). The upgrade measured this: local
reads were caught as non-linearizable in 13/20 runs.

**Q13. What does a read cost you, and how would you make it cheaper?**
One log append, an fsync on the leader and on a follower, a round trip, and log growth. Better:
ReadIndex. Note the commit index, confirm leadership with one heartbeat round to a majority,
wait until applied, then answer from memory. No disk write. Even better: batch many reads per
heartbeat round. Leases skip the round but depend on bounded clock drift.

**Q14. What exactly is persisted, and when?**
`currentTerm`, `votedFor` and the log. Before granting a vote, before replying success to AE,
before a leader counts its own entry, and before any reply after a term change.
`raft_node.cpp:349, 673, 766, 876`.

**Q15. Why fsync and then rename? Why fsync the directory?**
Writing in place could leave a torn file after a crash. Write a temp file, fsync it so its bytes
are durable, then rename. Rename is atomic, so readers see old or new. The rename itself lives in
the directory, so fsync the directory to make the new name durable.

**Q16. How did you test that fsync works?** *(hostile)*
Honest answer: the tests cannot. SIGKILL ends the process but the kernel page cache survives, so
the data reaches disk anyway. Testing fsync needs power-cut simulation: a VM hard reset,
`dm-flakey`, or a tool like ALICE/CrashMonkey. I did not do that.

**Q17. What if `fsync` fails?** *(hostile)*
`persist()` ignores the return value (`raft_node.cpp:174–182`). The node would carry on and could
acknowledge a write that is not durable. The fix: treat a failed persist as fatal and crash. That
is what PostgreSQL does since fsyncgate, because retrying fsync is not reliable.

**Q18. What if the `.bin` file is corrupt?** *(hostile)*
`Storage::load` returns false and `start()` carries on with an empty state (`raft_node.cpp:119–135`).
The node silently forgets its vote and log. It could then vote twice in a term. That is a safety
bug. Fix: refuse to start, as the snapshot path already does (:109–112). Add a checksum per
record.

**Q19. Why rewrite the whole log on every write?**
Simplicity and an easy atomicity proof. Cost: O(log length) per write. With no compaction,
throughput fell from 235 to 125 writes/s over 6,000 writes. A real system uses an append-only WAL
with per-record CRC, preallocated segments, and group commit.

**Q20. Is your vote safe across a quorum-loss step-down?** *(hostile, code-review find)*
`become_follower(current_term_)` resets `votedFor` to -1 in the same term (`raft_node.cpp:222, 426`).
In theory the node could vote again in that term. It is safe only because a leader always has its
own term-T no-op, so it refuses any term-T candidate (whose last term is at most T−1) on the
up-to-date check. Correct but fragile. The clean fix: keep `votedFor` when the term does not change.

### Engineering choices

**Q21. Why C++ and not Go?**
The spec allowed C/C++/Go. C++ needs no runtime and gives one static binary per OS. It is also
harder: no goroutines or channels, so threading and lifetimes are explicit. *(Honest add: Go
would have been faster to write. etcd/raft is Go.)*

**Q22. Why one global mutex?**
Raft's invariants span term, role, log and commit index together. One lock makes each handler
atomic with respect to the others. The cost is that fsync runs under the lock, so the node
serializes on its disk. The production answer is a single-threaded state machine with I/O done
outside it, the etcd/raft "Ready" pattern.

**Q23. Why threads instead of an event loop?**
Blocking code is easier to read and debug. The cluster size is 3–5, so the thread count is
small. One problem: the vote threads are detached and capture `this` (`raft_node.cpp:306, 374`).
On shutdown, a late reply could touch a destroyed node. It is fine for a process that exits, but
not for a library.

**Q24. Why a custom wire format?**
No dependencies, and Python clients are trivial. Decimal text plus a network-order length prefix
works across endianness and OS. Weak points: separator bytes are banned in commands; there are no
types, no versioning, no auth.

**Q25. What is your write throughput, and why is it low?** *(hostile)*
About 250 writes/s on loopback with one client. It is fsync-bound. Each write does two fsyncs on
the leader (file and directory) under the lock, then the same on a follower, one after the other,
with no batching. Plus a new TCP connection per request from the test client. etcd reaches
thousands per second with batching, pipelining, parallel leader disk writes (thesis §10.2.1) and
an append-only WAL.

**Q26. How would you get 10× throughput?**
- Group commit: one fsync for all pending entries.
- Let the leader write its disk in parallel with sending AE.
- Pipeline AEs instead of one in flight per peer.
- Append-only WAL instead of a full rewrite.
- Drop the directory fsync for appends to a preallocated file.
- Persistent client connections.
- Batch clients' requests.

**Q27. How is the client protected from duplicate writes on retry?** *(hostile)*
It is not. [doc] README: "No client sessions … a retried write can be applied twice." The fix is
the thesis §6.3: each client gets an ID and sequence numbers, and the state machine stores the
last result per client.

**Q28. How does the leader know a follower is down?**
Each replicator counts consecutive failed RPCs (`raft_node.cpp:537–544`). For quorum, `last_contact_`
holds the time of the last reply from each peer. The dashboard reads reachability separately
through telemetry.

**Q29. Why does the leader step down after losing quorum instead of waiting?**
So clients get a fast, clear "no quorum" error instead of timeouts, and so the client library
moves on to find the new leader.

**Q30. Can membership change?**
No. It is fixed by `cluster.conf` at startup. Adding a node needs joint consensus or
single-server changes (thesis ch. 4). Editing the config file on live nodes would be unsafe,
because two majorities could form.

### Testing

**Q31. What does the chaos test actually check?**
- Phase 1: 100 writes.
- Spot reads.
- SIGKILL the leader.
- A survivor becomes leader within 15 s.
- All 100 keys read back.
- 20 more writes.
- Restart the victim. Its commit index reaches the survivors' within 15 s.
- All nodes report the same commit index and term.

`chaos_test.py:193–268`.

**Q32. What does it NOT check?** *(hostile)*
- Concurrent clients.
- Partitions (on `main`).
- Power loss.
- That the logs are equal: `main` compares only commit index and term, not contents.
- Linearizability (on `main`).
- Long runs.

One scenario, many repeats.

**Q33. How is failover time measured, and why is it 0.81 / 1.01 / 1.21?**
From SIGKILL to the first poll that sees a LEADER among survivors. The poll loop sleeps 0.2 s,
so values snap to ~0.2 s steps. The real election happens about 600–1200 ms after the last
heartbeat.

**Q34. Why is 5-node failover faster than 3-node?**
Three survivors race their timers instead of two. The earliest of three uniform draws on
[600, 1200] averages 750 ms, versus 800 ms for two.

**Q35. Is it linearizable? How do you know?** *(upgrade)*
Within what was tested. 80 randomized runs (50 with 5 nodes, 30 with 3) with partitions, lossy
links and SIGKILLs. 272k acknowledged operations, no violation. The checker agrees with brute
force on 3,000 small histories, and it catches the planted stale-read bug. That is evidence, not
proof. No TLA+ or model checking.

**Q36. How does the checker handle a request that timed out?**
As indeterminate. A write may take effect any time after its call, or never. An indeterminate
read is dropped, because it has no effect and no observed output. Errors that occur before the
log append (no quorum, redirect, killed) count as definite failures and are left out.

**Q37. Why can you check per key?**
Linearizability is local (Herlihy & Wing). The whole history is linearizable if each object's
sub-history is. It also keeps the search small.

**Q38. How does fault injection work without root?** *(upgrade)*
A transport wrapper inside each node. A blocked peer's inbound RPCs get no reply. Outbound RPCs
sleep for the full timeout and then fail, like silence on a real network. Drops and delays are
random per message. Rules change at runtime through a `FAULT` message. Client traffic is never
blocked, so a client can still reach a stranded leader.

### Concepts

**Q39. Raft vs Paxos?**
Same safety core: majorities that overlap, plus ballot/term numbers. Raft fixes design choices
that Multi-Paxos leaves open: a strong leader, entries only flow from the leader, no holes in the
log, and only an up-to-date node can win. So a new leader never needs to learn missing entries.
Paxos allows any proposer at any time and is more flexible (e.g. EPaxos, Flexible Paxos), but it
is harder to get right. Howard & Mortier 2020 show they are very close.

**Q40. Where is your system in CAP?**
CP. During a partition the minority refuses reads and writes. The majority carries on. The
partition test shows an isolated old leader refusing a write with "no leader elected". In PACELC
terms it is PC/EC: consistency costs latency even without a partition.

**Q41. Define linearizability vs serializability vs sequential consistency.**
Linearizability: single objects, real-time order respected. Sequential consistency: one global
order that agrees with each client's own order, but not with real time. Serializability:
transactions over many objects behave like some serial order, real time not required. Strict
serializability is both serializable and linearizable.

**Q42. What happens in a 2-node cluster?**
The majority is 2, so it tolerates zero failures. The README says so. Use 3 or 5.

**Q43. Why odd sizes?**
Four nodes tolerate one failure, the same as three, but need a bigger majority. An even size adds
cost without fault tolerance.

**Q44. What breaks if clocks jump?**
Nothing in safety. All timers use `std::chrono::steady_clock` (`raft_node.cpp:13`), which is
monotonic. Wall-clock time is only used for log timestamps.

### Upgrade-specific

**Q45. How do snapshots interact with replication?**
If a peer's `nextIndex ≤ snapshot_index`, the leader sends the whole snapshot (`raft_node.cpp:466–506`).
The follower keeps entries after the snapshot if they match at `snap_index`, otherwise it clears
its log (:948–952). The snapshot is fsync'd before the log is trimmed. A crash between the two
leaves only redundant entries.

**Q46. How does resharding avoid two owners?**
Freeze in the source group's log, then install in the destination's, then drop. The freeze
commits before the coordinator sends the install, so the source stops serving before the
destination starts. Each step is idempotent per migration number, so a crashed coordinator can
retry. The `--unsafe-move` control installs first, and the checker flagged 5/5 runs.

**Q47. Why did throughput rise 1.86× after rebalancing?**
Three leaders fsync in parallel instead of one. It is all on one machine, so this shows the
mechanism, not scale-out. The coordinator is a single process with in-memory state.

### Reflection

**Q48. What would you do differently?**
- An append-only WAL with checksums from day one.
- Fail-stop on persist errors.
- An event-loop core so I/O is outside the lock.
- ReadIndex instead of log reads.
- Client sessions.
- A linearizability checker from the start, instead of a single-scenario chaos test.
- A tested startup order.

**Q49. What is next?**
- Membership changes.
- ReadIndex and leases.
- Batching and pipelining.
- A real-network test with `tc netem` on VMs.
- Power-loss testing.
- A TLA+ spec of the commit and snapshot logic.
- Lazy shard migration.

**Q50. What was the hardest bug?**
Story 1 or 2. Pick the one you can tell from memory. For story 2, add the worse case: an
early higher-term RPC persists an empty log over the file.

---

## 8. Questions where the honest answer is "it didn't work" or "I didn't do that"

Say it plainly, give the reason, then say what you would do. Do not apologise at length.

| Question | Honest answer | How to say it |
|---|---|---|
| "Did you build on your DECS KV server, as planned?" | The spec says to start from it and keep a before/after. The repo has no DECS code and no single-node baseline. | "I started fresh. The before/after in the repo is the original Raft code against the upgraded one, not DECS." |
| "Did you run it across Windows, Mac and Linux machines?" | Only if you did. The repo has compile-only proof for Windows and macOS, and the README says those binaries were not run. | If yes: give the setup. If no: "It is portable by design and cross-compiles cleanly. I tested clusters on Linux." |
| "Membership changes?" | Not implemented. | "Fixed membership. I'd add single-server changes from the thesis." |
| "Exactly-once writes?" | No client sessions. Retries can apply twice. | "Known gap. The tests avoid retries so the history stays honest." |
| "Is fsync really tested?" | No. SIGKILL cannot test it. | "Correct by construction, not by test. Power-loss testing is next." |
| "Did the stale-read control catch the bug?" (upgrade) | 0/20 at default timings. It needed a wider quorum window and a workload change. | "It first failed to catch it. That taught me how weak a passing run is." |
| "Eager vs lazy migration ablation? Replication-factor study? Throughput plot during resharding?" | All three were in `Project.md`. None done. Numbers only, no plot. | "I did eager migration only and measured before/during/after as numbers." |
| "Scalability as nodes are added?" | Only 3 vs 5 nodes, at fixed membership. | Same as above. |
| "Jepsen?" | Jepsen-style, in Python, on one host, in-process faults. Not Jepsen itself. | "Scoped-down Jepsen: same ideas, single host." |
| "Is 250 writes/s good?" | No. It is fsync-bound with no batching. | Q25–Q26. |
| "Did you write all of this yourself?" | `main` (as far as git shows). The upgrade is co-authored with an AI assistant per the commit trailers. | §0 wording. |
| "How did you find the commit-index bug?" | Not recorded anywhere. | Use your real memory. If unsure: "During a review against Figure 2 while adding snapshots." Only if true. |
| "Formal verification?" | None. | "Tested, not proved. A TLA+ model of commit plus snapshot would be my next step." |

---

## 9. Self-test exercises

Predict first. Then read the answer.

### Exercise 1 — Delete the current-term check in `maybe_advance_commit`

Remove `if (term_at(n) != current_term_) break;` at `src/raft_node.cpp:258`. Run
`chaos_test.py` and `linearizability_test.py --runs 10`. What happens?

<details><summary>Answer</summary>

Almost certainly everything still passes. Figure 8 needs a specific sequence: a leader
replicates an old-term entry to a majority, crashes before any current-term entry
commits, and a node with a newer-but-uncommitted entry wins. The nemesis rarely produces
this, and loopback elections are fast. The lesson: some safety rules are not covered by
random testing. You defend them by argument, or with a targeted test or a model checker.
Also note: with the no-op, a new leader almost always commits a current-term entry within
one round trip. That shrinks the window further.
</details>

### Exercise 2 — Remove the no-op in `become_leader`

Delete lines 246–250 of `src/raft_node.cpp` (keep the `replicate_cv_.notify_all()`).
Predict two effects. One is about liveness. One is about the vote reset in Q20.

<details><summary>Answer</summary>

1. **Liveness.** After failover, older-term entries stay uncommitted until a client sends
   something. In the chaos test, the first phase-3 `GET` is itself a current-term entry, so it
   commits everything and the test still passes. The dashboard would show `COMMIT < LOG` on the
   new leader until then.
2. **Safety.** The quorum-loss step-down resets `votedFor` in the same term (`:222, :426`).
   Without a term-T entry in its log, a stepped-down term-T leader can now grant a vote to
   another term-T candidate with an equal log. Two leaders in one term become possible. That
   breaks Election Safety, and through it Log Matching. So the no-op is quietly load-bearing
   here.
</details>

### Exercise 3 — Remove `file_sync` from `write_durably`

Make `src/storage.cpp:217` skip `file_sync(fd)`. Do `chaos_test.py` and `snapshot_test.py`
still pass?

<details><summary>Answer</summary>

Yes, both pass. Throughput rises a lot, maybe several times. SIGKILL kills the process, but the
kernel still holds the dirty pages and writes them out. Only power loss or a kernel crash would
show lost data. Your durability claim rests on code review, not on these tests. That is a
strong answer to give an interviewer.
</details>

### Exercise 4 — Local reads, two quorum windows

Run `linearizability_test.py --runs 20 --seed 1000 --node-args="--unsafe-local-reads"`,
then the same with `--node-args="--quorum-window 4000 --unsafe-local-reads"` and seed 3000.
Which one catches the bug more?

<details><summary>Answer</summary>

Recorded results: default window 0/20 caught; 4 s window 13/20 caught. A longer quorum window
means a cut-off leader keeps believing it leads for longer after the new leader is elected
(600–1200 ms). More stale reads land in that window. Note that with the default 1.5 s window
the bug is still real; the test just rarely sees it. Follow-up: what quorum window would make
local reads safe? Shorter than `election_min` minus clock drift and message delay, i.e. a
lease. `validate_timing` forbids this today (`timing.h:71`).
</details>

### Exercise 5 — Restore the old startup order

In `src/main.cpp`, move the `node.start(&start_error)` block (:369–373) to after
`telemetry.start()` (:384). Start a 3-node cluster, write 1,000 keys, then restart one
follower 50 times. What can go wrong, and how often?

<details><summary>Answer</summary>

Usually nothing. The window between the port opening and the state load is a few ms, and
heartbeats come every 100 ms. So expect a hit in a small fraction of restarts. When it hits:
an AE or RV with a higher term (any term above 0, since the node's in-memory term is 0) makes
the handler call `persist()`. That writes term + empty log over `raft_node_<id>.bin`. Then
`start()` loads that file. The follower has lost its log on disk. The leader will refill it, so
it looks healthy. But if this node was part of the majority that committed an entry, and the
leader then dies, that entry can be lost. Rare-but-catastrophic bugs like this are why startup
ordering belongs in code review.
</details>

---

## Appendix A — Code-review findings to know before an interviewer finds them

None of these are fixed in the repo. All are reconstructed from the code.

1. `persist()` ignores `Storage::save`'s result (`raft_node.cpp:181`). A disk error leaves the node acknowledging non-durable writes.
2. A corrupt or unreadable `.bin` means a silent empty start (`raft_node.cpp:120`). The node forgets its vote and log.
3. `votedFor` is reset on a same-term step-down (`raft_node.cpp:222` via `:426`). Safe only because of the no-op (Exercise 2).
4. Detached vote threads capture `this` (`raft_node.cpp:306, 374`). Possible use-after-free on shutdown.
5. fsync runs under the global mutex. `status()` hashes the whole state under it, every 300 ms per dashboard (`:1032`).
6. Two fsyncs per persist (file + directory). Leader and follower writes are sequential. No group commit.
7. The cached outbound socket keeps the timeout from when it connected (`transport.cpp:64, 235–248`). InstallSnapshot's `2× rpc_timeout` does not take effect on a reused socket.
8. `TcpTransport::send` retries once after a failed read (`transport.cpp:267–283`). The peer may handle the same RPC twice. Raft RPCs are idempotent, so this is safe, but say it if asked.
9. IPv4 literals only (`inet_pton`, `transport.cpp:45`). No hostnames, no TLS, no auth. `FAULT` messages are accepted from anyone when `--fault-injection` is on.
10. On macOS, `fsync` does not flush the drive's write cache. `F_FULLFSYNC` is needed for real durability.
11. A snapshot is held in memory and sent hex-encoded in one frame. With a 32 MB cap, state above ~16 MB cannot reach a lagging follower.
12. The `.bin` stores integers in host byte order (`storage.cpp:155–158`). That is fine locally, but the files do not move across architectures. Snapshots are explicitly little-endian.
