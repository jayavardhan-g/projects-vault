---
type: project
folder: projects/raftkv (duplicate in projects/RaftKV/RAFT)
repo: github.com/jayavardhan-g/Fault-Tolerant-Distributed-System and github.com/jayavardhan-g/raftkv
dates: committed 2026-08-18; upgraded 2026-10-22 (branch upgrade-2026-10, not yet pushed)
status: upgraded — snapshots, pre-vote, fault injection, linearizability checker, resharding
kind: distributed systems
stack: [C++20, Python 3 scripts]
tags: [project, systems, distributed, cpp]
---
# raftkv — a distributed key-value store on Raft (C++20)

**What.** Raft consensus written from scratch: leader election, log replication with conflict-term rewind, a commit rule plus no-op on
election, and reads that go through the log. It has a replicated KV store over TCP and crash-safe storage (fsync plus atomic rename).
It runs across Linux, macOS and Windows laptops in one cluster. The tooling is a live dashboard (status port = raft port + 1000), a Python client with
leader redirect, a per-node Raft log file, `--debug` tracing, and timing profiles (`lan`/`wifi`/`slow`) that fix constant
re-elections on Wi-Fi.

**Origin.** This is #10 on the [[Project strategy and resume plan]] shortlist, "Raft + dynamic resharding", as an extension of
[[DECS - HTTP key-value server]]. The plan is in `RaftKV/Project.md`. Resharding, compaction and pre-vote were added on 2026-10-22
(see below). Still open: fixed membership, no client sessions (a retried write can apply twice), whole-log rewrite per save.

**Verification.** `scripts/chaos_test.py` starts 3 nodes, writes 100 keys, kills the leader, checks failover and
that no data was lost, restarts the node, and confirms the nodes converge. It reports PASS or FAIL. The client `bench` showed ~446 writes/s across laptops over Wi-Fi.

**Duplication.** `projects/raftkv/` and `projects/RaftKV/RAFT/` are **byte-identical** (checked with `diff -rq`, ignoring `.git`) and are
pushed to two different GitHub repos. `RaftKV/` adds only the planning doc and an empty `Notes.md`. Pick one before you
link either from a resume.

**Docs in this vault** (copied 2026-10-14 from `/data/jayavardhan/projects/raftkv/` and `RaftKV/`):
- [[Sources/raftkv/README|README]]
- [[Sources/raftkv/Project|Project]]

**Source.** `raftkv/README.md` (27 KB guide: build, multi-laptop setup, dashboard, log format, troubleshooting), `RaftKV/Project.md` (the original spec).

## Upgrade 2026-10-22 (branch `upgrade-2026-10`, last commit bad36f1)

All numbers come from `results/` in the repo; `scripts/run_all.sh` regenerates them (~75 min, CPU only).

- **Chaos test** generalised (`--nodes`, `--kill`): 11/11 at 3 nodes and 11/11 at 5 nodes (leader + 1 follower killed); failover
  1.01 s / 0.81 s median; interleaved against the original commit, no measurable change (250 vs 258 writes/s at 3 nodes).
- **Log compaction**: snapshot file (fsync + rename) written before the log is trimmed; InstallSnapshot for lagging followers;
  restart from snapshot + log. `snapshot_test.py` 10/10. Write throughput without compaction decays 235 → 125 writes/s over
  6,000 writes; with `--snapshot-every 1000` it stays 165–194 (last block 1.43× paired).
- **Pre-vote**: follower isolated 6 s — with pre-vote its term rose +0 and the leader was deposed 0/5; without, +7 and 5/5
  (3 nodes; 5 nodes: +0 / 0/5 vs +6 / 5/5).
- **Fault injection** (`--fault-injection`): per-peer drop, delay, partition inside the transport, controlled by FAULT messages.
- **Linearizability**: WGL + memoisation per key, indeterminate ops; cross-checked against brute force on 3,000 histories.
  Randomized nemesis (partitions, leader isolation, lossy links, SIGKILL/restart, snapshots every 40): **50/50** runs linearizable
  at 5 nodes (180,462 acked ops, 91 kills, 181 partitions), 30/30 at 3 nodes. Negative control `--unsafe-local-reads`: flagged
  13/20 with a 4 s quorum window, **0/20 at default timings** — the workload rarely reads from a stale leader at default timings.
- **Resharding** (stretch goal from `RaftKV/Project.md`): `--shards N --shard-group G`; ownership is replicated state; slots move
  with SHARD_FREEZE / SHARD_INSTALL / SHARD_DROP (idempotent per migration number). 16 slots, 3 groups, rebalanced 16/0/0 → 6/5/5
  under 8 clients: 10/10 pass (ownership, data, linearizability), 10/10 with 38 SIGKILLs mid-move; 432 → 798 ops/s (1.86×
  paired); slot unavailable 7.5 ms median. Buggy move order (`--unsafe-move`) flagged 5/5.
- **Bugs fixed**: follower commit index was bounded by its own log end, not the last entry the leader verified (could commit
  stale entries past a 64-entry batch); the port opened before saved state was loaded.
- **Cross-platform**: `scripts/cross_build.sh` (zig 0.14.0) compiles and links for Windows x86_64, macOS x86_64/arm64, Linux,
  0 warnings — not executed on Windows/macOS.
- Env `raftkv` (Python 3.12.15, zig 0.14.0) only for the cross-build; tests use system g++ 13.3.0 and Python 3.12.3 stdlib.

**Duplicate repo.** `RaftKV/RAFT` (jayavardhan-g/raftkv) is still the 2026-08-18 code. Recommendation: archive that GitHub repo
(or replace its README with a pointer) and link only Fault-Tolerant-Distributed-System on the resume.
