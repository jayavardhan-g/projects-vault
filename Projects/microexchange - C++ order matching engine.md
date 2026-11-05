---
type: project
folder: projects/microexchange
repo: github.com/jayavardhan-g/Micro-Exchange
dates: first commit 2026-08-14; upgraded 2026-10-22 (branch upgrade-2026-10)
status: done (upgrade branch not yet pushed)
kind: systems — low-latency C++
stack: [C++20, no dependencies; optional CMake]
env: microexchange (cmake 4.4.4, ninja 1.13.2, python 3.12.15; system g++ 13.3)
tags: [project, systems, cpp]
---
# microexchange — limit order matching engine in C++20

**What.** A price-time-priority limit order book with an ingestion thread and a matching thread. Between them is a
lock-free SPSC ring buffer (cache-line padded, with cached indices). It has a pooled order arena (zero allocations while matching),
a flat price-level array, intrusive lists with an id table for O(1) cancel, and `rdtsc` latency timing.
Since the 2026-10-22 upgrade it also has:
- multiple symbols (one book each), and market and IOC orders;
- a two-segment mmap WAL with periodic snapshots taken by a **follower replica** off the matching thread, and the WAL truncated behind each snapshot;
- an epoll TCP gateway with 32-byte binary frames, plus an open-loop latency client;
- CMake with four `ctest` suites.

**Key results, shared server** (2× Xeon Platinum 8570, load 9–18, 5 runs after a warm-up, median [min–max]; `results/`, `bench/upgrade_runs.sh`):
- **The laptop numbers do not reproduce.** The original binary runs at 3.44M ops/s at 2M commands, not 10.8M, and **18.1×** [12.0–26.8] paired vs naive, not 50×. This branch: 14.3× [12.7–15.2], 3.39M ops/s.
- The engine itself is not the limit: one thread with no queue applies the stream at 18.1M/s (original) and 17.0M/s (this branch). The cross-core ring handoff on the 2-socket mesh is the bottleneck.
- Rewrite cost per command, single thread: **0.935×** [0.85–0.95]. Two-thread pipeline: within noise. p99.9 8.6 µs on quiet CPUs 16/18 vs over 1 ms for the original's default CPUs 0/2, which are interrupt-heavy.
- Gateway round trip p50 **13.2 µs** one at a time, 16.7 µs at 100k/s, 29.9 µs at 1M/s; in-server p50 0.56–1.35 µs.
- Snapshots every 100k commands: 1.02× throughput, p50 unchanged, matching thread never waited. The first design (copy on the matching thread) stalled it for **1.6 ms** per snapshot.
- Correctness:
  - 983,976 → 3 allocations and identical books reproduce in every run.
  - The differential fuzz checks every trade against a reference model.
  - **kill -9 recovery is exact**: 100+ kills in the test runs, all recovered.
  - Six injected bugs were each caught by a test.

**Bugs found in the original.**
- The tombstone id table never reclaims slots, so a long-lived book eventually hangs on a cancel of an unknown id (regression test added).
- `compare` left the main thread pinned, so the naive latency phase ran both of its threads on CPU 0. That flattered the naive p50.
- Orders were dropped silently when the pool was full, and a duplicate id orphaned the first order.

**Negative results kept in the docs.**
- Pre-faulting the WAL and pools did not help tails.
- p99 at ≥1M/s swings 26 µs ↔ 483 µs between sessions on this shared box, so treat tails as indicative only.

**Limits.** One matching thread for all symbols. No FOK, stops or self-trade prevention. Durable against process crashes, not power loss (no per-record msync). The gateway has no cancel-on-disconnect, login or resend.

**State.** Upgrade done on branch `upgrade-2026-10` (last commit `1fc2869`). **Not pushed**: the machine has no GitHub credentials. Jayavardhan must run `git push -u origin upgrade-2026-10`.

**Docs in this vault** (copied 2026-10-14, before the upgrade; the repo's README/PROJECT are now newer):
- [[Sources/microexchange/README|README]]
- [[Sources/microexchange/PROJECT|PROJECT]]

**Source.** `README.md` ("Measured on a shared server"), `PROJECT.md` (sections 3.7, 3.9, 3.10, 4, 6), `results/upgrade_summary.md`.
