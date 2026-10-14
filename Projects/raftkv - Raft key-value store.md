---
type: project
folder: projects/raftkv (duplicate in projects/RaftKV/RAFT)
repo: github.com/jayavardhan-g/Fault-Tolerant-Distributed-System and github.com/jayavardhan-g/raftkv
dates: committed 2026-08-18
status: done — resharding not built
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
[[DECS - HTTP key-value server]]. The plan is in `RaftKV/Project.md`. **Resharding was not built**. Known limits: no log compaction
or snapshots, fixed membership, no pre-vote.

**Verification.** `scripts/chaos_test.py` starts 3 nodes, writes 100 keys, kills the leader, checks failover and
that no data was lost, restarts the node, and confirms the nodes converge. It reports PASS or FAIL. The client `bench` showed ~446 writes/s across laptops over Wi-Fi.

**Duplication.** `projects/raftkv/` and `projects/RaftKV/RAFT/` are **byte-identical** (checked with `diff -rq`, ignoring `.git`) and are
pushed to two different GitHub repos. `RaftKV/` adds only the planning doc and an empty `Notes.md`. Pick one before you
link either from a resume.

**Docs in this vault** (copied 2026-10-14 from `/data/jayavardhan/projects/raftkv/` and `RaftKV/`):
- [[Sources/raftkv/README|README]]
- [[Sources/raftkv/Project|Project]]

**Source.** `raftkv/README.md` (27 KB guide: build, multi-laptop setup, dashboard, log format, troubleshooting), `RaftKV/Project.md` (the original spec).
