---
type: project
folder: projects/microexchange
repo: github.com/jayavardhan-g/Micro-Exchange
dates: first commit 2026-08-14
status: done
kind: systems — low-latency C++
stack: [C++20, no dependencies]
tags: [project, systems, cpp]
---
# microexchange — limit order matching engine in C++20

**What.** A price-time-priority limit order book with an ingestion thread and a matching thread. Between them is a
lock-free SPSC ring buffer (cache-line padded, with cached indices). It has a pooled order arena (zero allocations while matching),
a flat price-level array, intrusive lists with an id table for O(1) cancel, an mmap write-ahead log with
snapshot/replay, and `rdtsc` latency timing. `./microexchange` runs a demo, 40 unit checks, a fuzz test, a benchmark and crash recovery in about 1–2 s.

**Key results** (500k commands, vs the naive `baseline/naive_engine.hpp`, which uses a mutex queue, `std::map` and allocates per order):
- Throughput **1.6M → 13.0M ops/s (8.0×)**; p50 **875 → 83 ns**; heap allocations 983,976 → 3.
- The gap widens with book size: **50×** at 2M commands, because the naive engine's cancel is a linear scan and its pointer-chasing causes cache misses.
- The fuzz invariant `filled + resting + cancelled == submitted` holds over 200k random orders. Replay rebuilds an identical book.
- The p99 comparison is directional only, because the naive engine's sleeping condition variable dominates it.

**Limits.** One symbol, limit orders only, no network layer, and the snapshot is written only at the end of a run.

**State.** Done. It has a single commit. The `_to_delete/` folder (old CMake/Makefile, test binaries and WAL files) can be removed.

**Source.** `README.md`, `PROJECT.md` (full write-up and before/after tables).
