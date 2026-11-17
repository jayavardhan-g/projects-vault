# Project report: Micro Exchange, a C++20 limit-order matching engine

> **For the LLM reading this.** This report describes a project listed by G Jayavardhan (M.Tech CSE, IIT Bombay) as a
> self project, Summer 2026, upgraded in October 2026, on resume 5. It was written from
> `/data/jayavardhan/projects/microexchange`, **branch `upgrade-2026-10`** (HEAD `1fc2869`): headers, sources, tests,
> README, PROJECT.md (the long write-up) and `results/`.
>
> Labels:
> - **[CODE]**: `file:line`;
> - **[MEASURED]**: from README/PROJECT/results;
> - **[DOC]**: a reason written in the repo;
> - **[RECON]**: the report writer's reasoning.
>
> **Authorship note:** the first commit (`fb58991`, 2026-08-14, 2,719 lines) is authored as **"Tarunsai"**. It contains
> the ring buffer, object pool, order book, WAL, snapshot, naive baseline, benchmark harness and fuzz test, i.e. the
> original engine, plus the laptop results behind the resume numbers. The 7 later commits (2026-10-22, +2,461/−265
> lines) are Jayavardhan's: multiple symbols, market/IOC orders, the two-segment journal with follower snapshots, the
> TCP gateway, CMake, and the shared-server re-measurement. He should be ready to explain who wrote the August version.

---

## 1. Summary

A limit-order **matching engine** (the core of an exchange) in C++20 with **no external libraries**. Design:

```
producer / TCP gateway thread  ──SPSC lock-free ring──►  matching thread
                                                         │ 1. append Command to mmap'd WAL (write-ahead)
                                                         │ 2. apply to the symbol's OrderBook
                                                         │ 3. record latency (rdtsc) in a histogram
follower thread: tails the WAL, keeps a replica, writes snapshots (tmp+fsync+rename), frees WAL segments
```

Parts:
- **Order book:**
  - price-time priority;
  - a flat array of 65,536 price levels (integer ticks);
  - each level an **intrusive doubly-linked FIFO list** of `Order` nodes;
  - orders come from a pre-allocated **object pool** (no malloc while matching);
  - an **open-addressed id → Order\* table** (O(1) cancel, backward-shift deletion).
- **Order types:** limit, market, IOC. **Multiple symbols** (one book each).
- **Persistence:** a memory-mapped **write-ahead log** (two segments) plus **periodic snapshots taken by a follower thread** off the hot path. Crash recovery = snapshot + log tail.
- **Correctness:**
  - a conservation invariant;
  - 78 unit checks;
  - **differential fuzzing** against a `std::map` reference model (200k+ random orders);
  - **kill -9 crash tests**;
  - TCP gateway tests;
  - equivalence with the naive baseline;
  - 6 hand-injected bugs, all caught.
- **Baseline** (`baseline/naive_engine.hpp`): mutex queue, `std::map` of `std::list<shared_ptr<Order>>`, linear-scan cancel, an ofstream flushed per record.

**Headline numbers:**
- **Laptop** (Aug 2026, 4 cores): 10.8M ops/s and **50×** over naive at 2M commands.
- **Shared 2-socket server** (Oct 2026, median of 5): **3.44M ops/s and 18.1×** (original binary), **3.39M and 14.3×** (upgraded branch).
- The book alone, single-threaded: **18.1M commands/s**.
- So the **cross-core ring-buffer handoff is the bottleneck** on the server.

---

## 2. Resume bullets vs evidence (resume 5)

| Bullet | Verdict | Evidence |
|---|---|---|
| "Engineered a C++20 limit-order matching engine with price-time priority, a lock-free ring buffer, object pooling, WAL persistence, and crash recovery, verified via 200k+ order fuzzing and baseline checks" | **Supported.** | All components are in the code (§4). Fuzz: 200,000 random orders over 4 symbols (default run) against a reference model. `compare` checks book equality against the naive engine. |
| "Scaled to 10.8M ops/s at 2M commands by replacing O(n) cancellation and heap-based data structures with O(1) ID lookup and contiguous memory layouts, pushing throughput to 50× over the baseline" | **Laptop-only numbers; reword.** | PROJECT.md §6: laptop, single runs, 2M commands: naive 216,189/s vs optimised 10,816,352/s = **50.0×**. On the shared server (median of 5, paired): **3.44M ops/s and 18.1× [12.0–26.8]** for the same original code; 3.39M and **14.3×** for the upgraded branch with fixed CPU pinning. The README itself says "the laptop numbers do not reproduce here". Honest wording: "10.8M ops/s / 50× on a 4-core laptop; 3.4M ops/s / 14–18× on a shared 2-socket Xeon server, where the cross-core handoff limits throughput (the book alone does 18M/s)". Also note the engine was first committed by another author (see the authorship note at the top). |

---

## 3. Background (teach first)

### 3.1 Exchange basics
- **Limit order:** buy or sell Q at price ≤ P (buy) or ≥ P (sell). Whatever doesn't trade **rests** in the book.
- **Market order:** trade at any price. **IOC (immediate-or-cancel):** trade up to its limit, and the rest expires.
- **Order book:** bids (buy orders, highest first) and asks (sell orders, lowest first). The **spread** is best ask − best bid.
- **Matching:** an incoming buy at P crosses with asks priced ≤ P, best price first, and within a price **oldest first**. That is **price-time priority (FIFO)**. Each match is a trade between the **taker** (incoming) and the **maker** (resting).
- **Cancel:** remove a resting order by id.

### 3.2 Concurrency
- **SPSC ring buffer:** a fixed array with a head index (the producer writes) and a tail index (the consumer writes). Each side writes only its own index, so no locks are needed. Correctness needs memory ordering:
  - the producer writes the slot, then publishes `head` with a **release** store;
  - the consumer reads `head` with an **acquire** load, then reads the slot;
  - acquire/release guarantees the slot write is visible before the new head.
  - `relaxed` is fine for a thread reading its *own* index.
- **False sharing:** two variables on the same 64-byte cache line, written by different cores, cause the line to ping-pong between caches. Fix: `alignas(64)` to separate them.
- **Cached opposite index:** the producer keeps a private copy of `tail` and only re-reads the shared one when the queue *looks* full, which cuts cross-core cache traffic.
- **Busy polling vs blocking:** the consumer spins with `pause` instead of sleeping on a condition variable. Lower latency, one core burned.
- **Thread pinning:** `pthread_setaffinity_np` to chosen cores (avoid CPU 0, which takes interrupts).

### 3.3 Memory and data structures
- **Object pool:** allocate N objects once; acquire/release is a stack pop/push. No allocator calls or fragmentation on the hot path.
- **Flat array indexed by price tick:** O(1) level lookup, contiguous, cache-friendly. Compare with `std::map`, a red-black tree with pointer chasing and a node allocation per level.
- **Intrusive list:** the prev/next pointers live *in* the order, so unlinking a known node is O(1) with no allocation.
- **Open addressing with linear probing:** keys sit in one array, so cache-friendly. Deletion uses **backward shift** instead of tombstones, so the table never fills with dead slots as ids keep growing.

### 3.4 Durability
- **WAL (write-ahead log):** log the command *before* applying it. After a crash, replay the log to rebuild state.
- **mmap'd file:** an append is a memcpy into a `MAP_SHARED` mapping. The kernel page cache persists it even if the process dies (but not power loss without `msync`/`fsync`).
- **Snapshot:** the full state at sequence s. Recovery = load the snapshot + replay records after s. Then the log before s can be truncated.
- **Atomic file replace:** write `tmp` → `fsync(tmp)` → `rename(tmp, path)` → `fsync(dir)`.

### 3.5 Measuring latency
- `rdtsc` reads the CPU cycle counter (about 20 cycles of overhead), calibrated against the steady clock.
- **Coordinated omission:** a closed-loop benchmark under-reports latency when the system stalls, because the generator stops sending too. Fix: pace the producer at a target rate and measure from the *intended* send time.

---

## 4. Code walk

### 4.1 `Command`, `Order`, `Trade` (`include/engine/order.hpp`)
- `Command` = {type (New/Cancel), side, ord_type (Limit/Market/Ioc), symbol, id, price (ticks), qty, tsc}, **40 bytes, trivially copyable** (static_assert). The same struct goes into the ring buffer *and* the WAL by memcpy.
- `Order` = {id, price, qty, side, **prev, next**}: an intrusive list node.

### 4.2 `RingBuffer<T, Capacity>` (`include/core/ring_buffer.hpp`, 68 lines)
- The capacity must be a power of two (static_assert), so wrap is `& (Capacity−1)` instead of `%`.
- **`push`** (`:27-36`):
  - `head = head_.load(relaxed)`, its own index;
  - if `head − tail_cache_ >= Capacity`, refresh `tail_cache_ = tail_.load(acquire)`; if still full, return false;
  - write the slot;
  - `head_.store(head+1, release)`, which publishes the slot.
- **`pop`** (`:38-47`): the mirror image, using `head_cache_`.
- Indices only ever increase (`size_t`), so full is `head − tail == Capacity` and empty is `head == tail`.
- `alignas(64)` on `head_` and `tail_` (each followed by its owner's cached copy of the other index), so the producer's and consumer's hot data sit on separate cache lines. alignas also pads `sizeof` so neighbouring objects can't share the consumer's line.
- In use: `CommandQueue = RingBuffer<Command, 1<<16>`.

### 4.3 `ObjectPool<T>` (`include/core/object_pool.hpp`)
`storage_` (new T[cap]) plus a free stack of pointers. `acquire()` returns `top_ > 0 ? free_[--top_] : nullptr`. `release(p)` does `free_[top_++] = p`. On exhaustion it returns nullptr, so the order is refused rather than allocated.

### 4.4 `OrderBook` (`include/engine/order_book.hpp`, `src/order_book.cpp`)
- `bids_`, `asks_`: `std::vector<Level>(65536)`. Each `Level` = {head, tail, qty}.
- `best_bid_` (0 if empty), `best_ask_` (65536 if empty), `lo_price_` and `hi_price_` (the band ever used, so snapshots don't scan all 131k levels).
- Running totals: `resting_qty_`, `filled_qty_`, `cancelled_qty_`, `expired_qty_`.
- **`submit(id, side, price, qty, type)`** (`order_book.cpp:114-167`):
  1. A market order's price becomes an extreme (buy → 65535, sell → 1).
  2. **Reject** (count qty as expired) if id is 0, price is out of range, or it is a limit order with a **duplicate id**.
  3. **Buy side:** while qty > 0 and `best_ask_ <= price`: take the level at `best_ask_`, and while qty > 0 and the level is non-empty, match against `maker = lvl.head` (**the oldest order: time priority**). Then:
     - x = min(qty, maker.qty); reduce qty, maker.qty, lvl.qty and resting_qty_;
     - `filled_qty_ += 2x` (**both sides**, so the conservation sum balances);
     - optionally record the trade;
     - if the maker is exhausted, `erase_order(maker)`.
     - If the level empties, `raise_best_ask()` scans upward to the next non-empty level.
  4. The sell side is the mirror image.
  5. Remainder: a limit order **rests** (`rest()`), otherwise it **expires**. A failed rest (pool full) also expires.
- **`rest`** (`:90-112`): acquire a node from the pool, append it to the level's tail (FIFO), update level and resting totals, insert into the id table, widen the lo/hi band, update best bid/ask.
- **`cancel(id)`** (`:169-186`): `index_.find(id)` (O(1)) → subtract its qty from the level and resting totals, add it to cancelled → `erase_order` (O(1) unlink, erase from the id table, return the node to the pool) → if the level emptied at the best price, move best bid/ask.
- **`IdTable`** (`:26-67`):
  - keys and vals arrays of size **2× capacity rounded to a power of two**, so the load factor is ≤ 0.5;
  - hash = splitmix64 `mix(id) & mask`;
  - `insert` and `find` use linear probing;
  - **`erase`** uses **backward-shift deletion**: after emptying slot i, walk forward and move any entry j whose home slot is at or before i back into the hole. That keeps probe chains intact with no tombstones.
  - id 0 is reserved as the empty marker.
- **Complexity:**
  - submit = O(levels crossed + orders filled);
  - cancel = O(1) expected;
  - best-price updates scan empty levels (amortised small with dense prices).

### 4.5 `MatchingEngine` (`include/engine/matching_engine.hpp`)
- `apply(c)`: **`journal_->log(c)` first (write-ahead)**, then `ex_.apply(c)` (dispatch by symbol), then `++processed_`, then record the latency `rdtsc() − c.tsc` into the histogram.
- `run(q, producer_done)`: `pop`. If empty and the producer is done, stop. Otherwise `cpu_pause()` (busy-wait).

### 4.6 WAL (`include/persistence/wal_logger.hpp`)
- One segment = `WalHeader{magic "MXL2", record_size, base_seq, count}` + `max_records × Command`, in a file **sized up front and memory-mapped** (`MAP_SHARED`; Windows `CreateFileMapping`).
- **`append(c)`:** `records_[n] = c`, then `atomic_ref(count).store(n+1, release)`.
  - **Record before count:** a process killed at any instruction leaves a count that covers only complete records.
  - The release store lets the follower thread (which acquire-loads the count) read every record it covers.
- `reset(base)`: drop the count to 0 first, then set the new base, so a crash in between leaves an empty segment and never old records relabelled.
- Durability is against **process** crashes. Power loss would need `msync`.

### 4.7 Journal: two segments plus follower snapshots (`include/persistence/journal.hpp`)
- **Segment lifecycle:** **Active** → (the matching thread fills it) **Sealed** → (the follower snapshots past it) **Free** → Active. Each transition is owned by exactly one thread. The log never exceeds two segments.
- **Follower thread:** tails the log (shared memory + acquire loads), applies each command to its **own replica** of the exchange, and every `snapshot_every` commands writes the replica:
  - `Snapshot::write`: tmp file → `fsync` → `rename` → `fsync(dir)`;
  - then empties a sealed segment the snapshot fully covers.
- **Why [DOC + MEASURED]:** the first design copied the live books *on the matching thread*, which **stalled it 1.6 ms per snapshot** at 88k resting orders. Now the matching thread only waits if it fills a whole segment before the follower frees the other one. That never happened in the measured runs.
- **Recovery:** load the snapshot (its header records the sequence it covers) → replay later log records in sequence order. **Refuse to continue past a gap.**

### 4.8 TCP gateway (`include/net/gateway.hpp`, `src/gateway.cpp`, `include/net/protocol.hpp`)
- An epoll loop thread (accept, read frames, write reports) plus the matching thread, connected by **two SPSC rings** (commands in, `Outbound` reports out). Both threads busy-poll.
- **Wire protocol:** fixed **32-byte little-endian frames**, so parsing is `bytes / 32` plus memcpy (no length prefix, no text).
  - `ClientMsg`: type, side, ord_type, symbol, qty, seq, order_id (1 .. 2⁴⁰−1), price.
  - Replies: **Ack** (Rested / Filled / Expired / Cancelled / CancelRejected), **Fill** (to the maker's owner), **Reject** (BadSymbol / BadOrder).
- Listens on 127.0.0.1 only. Recovers its books from the journal on restart.

### 4.9 Timing (`core/tsc_clock.hpp`, `metrics/latency_histogram.hpp`)
`rdtsc` (x86) or `cntvct_el0` (ARM), calibrated once against `steady_clock`. The histogram is a flat array of 1 ns buckets, so recording a sample is one increment. p50, p99 and p99.9 are read off the bucket counts.

### 4.10 Naive baseline (`baseline/naive_engine.hpp`)
- mutex + condition-variable queue;
- `std::map<price, std::list<shared_ptr<Order>>>`;
- `unordered_map` id index, but **cancel walks the level list** to find the node;
- `ofstream` flushed on every record;
- latency samples pushed into a vector and sorted at the end.
- About **983,976 heap allocations** for 500k commands, vs **3** for the optimised engine.

---

## 5. Correctness [MEASURED + DOC]
- **Conservation invariant:** `filled + resting + cancelled + expired == submitted`. Filled counts both sides; tape volume is half of it.
- **78 unit checks:**
  - no-cross, exact and partial fills;
  - time priority within a level, price priority across levels;
  - cancel from the middle, cancel of an unknown id;
  - market sweeping levels, IOC stopping at its limit;
  - refusals;
  - the id table under churn vs `std::unordered_map`;
  - symbol isolation;
  - the snapshot round trip;
  - WAL replay;
  - the log staying within two segments.
- **Differential fuzzing:**
  - random limit, market and IOC orders plus cancels over 4 symbols, including 1 in 16 cancels naming the wrong symbol, duplicate ids, bad prices and unlisted symbols;
  - every trade must match the reference model (same maker, price, quantity, order);
  - every 1,000 ops: the invariant holds, walked totals equal cached ones, no zero-qty order is linked, no book is crossed;
  - every 10,000 ops: books equal the model, order by order.
- **Crash recovery** (`crash`): fork a child running the real 2-thread pipeline with snapshots every 20,000 commands, then **SIGKILL** it at random times *and* at each follower step (before writing a snapshot, after renaming it, after emptying a segment).
  - Pass: no gap; the recovered sequence equals what was logged (applied + at most 1 logged-but-unapplied); each book equals a clean replay of that prefix.
- **Gateway test:** two real TCP sessions trading (rest, fill, maker fill, IOC expiry, market on an empty book, cancel and cancel-gone, bad symbol, zero qty, duplicate id, a frame split across two writes), then restart and recover identical books.
- **Tests of the tests:** 6 injected bugs, all caught:
  - IOC remainder resting;
  - recovery ignoring the older segment;
  - truncating before the snapshot is durable (in both journal designs);
  - a wrongly labelled snapshot sequence;
  - the old tombstone id table.
- `ctest`: 4 suites in about 4 s.

---

## 6. Results

### 6.1 Laptop, August 2026 (single runs; the source of the resume numbers)
| commands | resting at end | naive | optimised | speedup |
|---|---|---|---|---|
| 100,000 | 16,076 | 2.09M/s | 20.3M/s | 9.7× |
| 500,000 | 81,117 | 1.62M/s | 12.97M/s | 8.0× |
| 1,000,000 | 162,158 | 716k/s | 13.1M/s | 18.3× |
| **2,000,000** | 324,182 | 216k/s | **10.8M/s** | **50.0×** |

p50 latency: 875 ns → 83 ns. Heap allocations: 983,976 → 3.

**Why the gap widens with book size [DOC]:**
- the naive engine's cancel is a linear scan of the level (longer levels → slower);
- its pointer-chasing structures (map nodes, list nodes, shared_ptr control blocks) fall out of cache at 324k orders.

So constant-factor work (no locks, no allocation) bought about 10× at every size, and the **algorithmic and layout** choices (O(1) cancel, flat arrays) kept the engine from degrading as the book grew.

**What each change fixes:**

| change | fixes |
|---|---|
| lock-free ring | mutex contention and sleep/wake |
| pool | allocator calls |
| flat price array | map lookups and node allocation |
| intrusive list + id table | O(n) cancel |
| mmap'd log | a syscall and flush per record |
| rdtsc histogram | measurement overhead |

### 6.2 Shared server, October 2026
2 × Xeon Platinum 8570 (224 hardware threads), shared (load 9–18), g++ 13.3 `-O3 -march=native`. Median of 5 runs after warm-up, with [min–max]. Speedups are medians of paired ratios. Threads pinned to CPUs 16 and 18.

**Original binary (rebuilt from `fb58991`):**

| commands | naive | optimised | paired speedup |
|---|---|---|---|
| 500k | 380k/s | 3.46M/s | 10.1× |
| 2M | 182k/s | 3.44M/s | **18.1×** [12.0–26.8] |

**Upgraded branch:**

| commands | resting | naive | optimised | speedup | naive p50 | optimised p50 |
|---|---|---|---|---|---|---|
| 100k | 16k | 515k/s | 3.22M/s | 5.6× | 4,777 ns | 421 ns |
| 2M | 324k | 236k/s | 3.39M/s | **14.3×** | 4,861 ns | 520 ns |

- **Book alone, one thread, no queue:** 18.1M cmds/s (original) vs 17.0M (branch) = 0.935×. That is the price of symbol dispatch, order types and the duplicate-id check.
- **So the bottleneck is the cross-core handoff through the ring buffer** on a 2-socket mesh, much slower than on a laptop. Next experiment: batch the ring's publishes and pops.
- Tail: p99.9 is 8.6 µs on CPUs 16/18 vs over 1 ms for the original on its default CPUs 0/2, because CPU 0 takes far more interrupts. That is why `--cpus` exists.
- **A fix:** the original harness left the main thread on CPU 0, so the naive engine's latency phase ran both threads on one core.
- **Symbols and order types** (1M commands): limit-only 1 symbol 3.01M/s; mixed with 16 symbols 3.83M/s. More symbols means shorter books.
- **Snapshots:** every 100k commands, 1.02× of the no-snapshot throughput; every 25k, 0.92×; p50 unchanged. The matching thread never waited for a free segment.
- **Through the TCP gateway** (4 symbols, mixed, journal on, loopback):

  | load | round-trip p50 | p99 | in-server p50 |
  |---|---|---|---|
  | one at a time | **13.2 µs** | 22.4 µs | 623 ns |
  | 100k/s | 16.7 µs | 24.3 µs | 559 ns |
  | 1M/s | 29.9 µs | 819 µs | 1,345 ns |

  The network path (two loopback TCP traversals plus syscalls) costs about 12–16 µs. Matching inside the server stays under 1.4 µs at p50.
- **Tails are unstable on a shared machine:** the same configuration gave p99 483 µs in one session and 26 µs twenty minutes later. Pre-faulting didn't help.

### 6.3 Measurement method [DOC]
- **Two phases:**
  1. flat-out throughput (latency there is queue backlog, so it isn't reported);
  2. **paced** at a target rate, with each command stamped with its **intended** send time, to avoid coordinated omission.
- p50 is the key latency. ms-level p99.9 is the OS stealing the core.

---

## 7. Design decisions
| Decision | Alternatives | Why | Source |
|---|---|---|---|
| Single-writer matching thread + SPSC ring | mutex queue; multi-threaded book | no locks; deterministic order | [DOC] |
| Cached opposite index in the ring | always load the shared index | fewer cross-core cache transfers | [DOC] |
| `alignas(64)` head/tail | packed | avoids false sharing | [DOC] |
| Object pool | new/delete per order | no allocation on the hot path (3 vs 984k allocations) | [MEASURED] |
| Flat 65,536-tick level array | `std::map` | O(1) level lookup, contiguous; costs 3 MB/symbol and limits the price range | [DOC] |
| Intrusive FIFO lists | `std::list<shared_ptr>` | O(1) unlink; time priority | [DOC] |
| Open addressing + backward shift | `unordered_map`; tombstones | cache-friendly; no dead-slot buildup (a tombstone bug was among the injected ones) | [DOC] |
| mmap'd preallocated WAL | write() + flush per record | append = memcpy; no metadata updates | [DOC] |
| Follower-thread snapshots | copy on the matching thread | removed the 1.6 ms stall | [MEASURED] |
| Two-segment journal | unbounded log | bounded disk, truncation behind snapshots | [DOC] |
| tmp + fsync + rename + dir fsync | overwrite in place | atomic, crash-safe snapshots | [DOC] |
| Fixed 32-byte binary frames | text/JSON | zero parsing cost | [DOC] |
| rdtsc + flat histogram | chrono + vector sort | measurement overhead | [DOC] |
| Busy polling | blocking waits | latency; costs a core | [DOC] |
| Pin to non-zero cores | default | CPU 0 interrupt noise | [MEASURED] |

---

## 8. Limitations [DOC]
- One matching thread for all symbols (next step: a thread per symbol group).
- No fill-or-kill, stop orders or self-trade prevention.
- Prices limited to `[1, 65536)` ticks; 3 MB per symbol.
- Durability against process crashes only (no per-record msync).
- No cancel-on-disconnect, login or resend.
- No replication (the follower replica is a natural hot standby).
- The cross-core handoff limits throughput on big servers.

---

## 9. Likely interview questions
1. *Walk through a buy that partially fills against two resting asks, then gets cancelled.*
   - submit → match `asks_[best_ask_].head` (oldest first), fill x₁ and erase that maker;
   - the level empties → `raise_best_ask`;
   - match the next maker partially;
   - the remainder rests at the tail of `bids_[price]`, inserted into the id table;
   - cancel → `index_.find` → subtract totals → unlink → pool release → maybe lower `best_bid`.
2. *Explain every memory order in `push`/`pop`.* → §4.2.
3. *What is false sharing, and how did you avoid it?* → §3.2, alignas.
4. *Why is cancel O(1)?* → id table + intrusive list.
5. *Backward-shift deletion: why not tombstones?* → probe chains never fill with dead entries as ids grow.
6. *How do you guarantee recovery is exact after kill -9?* → record-before-count release store, write-ahead ordering, snapshot covers seq, replay with gap check; tested by kills at each follower step.
7. *Why take snapshots on a follower thread?* → the 1.6 ms matching stall.
8. *The resume says 10.8M / 50×; what's real?* → laptop vs server (§2, §6.2); the cross-core handoff.
9. *Why is throughput on a 224-thread server lower than on a laptop?* → inter-core and inter-socket cache-line transfer latency per handoff; the book alone does 18M/s.
10. *How would you make it faster?* → batch ring operations; per-symbol matching threads; keep producer and consumer on the same physical core's SMT siblings or the same L3; huge pages; avoid the WAL on the hot path via group commit.
11. *What is coordinated omission?* → §3.5, §6.3.
12. *How does the fuzzer prove correctness?* → differential testing against a simple model plus invariants.
13. *Why `filled_qty_ += 2x`?* → both sides, so the conservation sum balances.
14. *What did you write vs the initial commit?* → be ready (see the authorship note at the top).

---

## 10. Numbers cheat sheet
| Number | Meaning |
|---|---|
| 40 B / 32 B | Command (WAL record) / wire frame size |
| 65,536 | price levels per side per symbol |
| 1 << 16 | ring buffer capacity |
| 10.8M ops/s, 50× | laptop, 2M commands (Aug) |
| 3.44M ops/s, 18.1× | server, original binary, 2M commands (median of 5) |
| 3.39M ops/s, 14.3× | server, upgraded branch |
| 18.1M cmds/s | book alone, single thread |
| 83 ns → ~420–520 ns | p50 laptop → server |
| 983,976 vs 3 | heap allocations, naive vs optimised (500k commands) |
| 1.6 ms | matching-thread stall per snapshot in the first design |
| 13.2 µs | gateway round-trip p50 (one at a time) |
| 200,000 | default fuzz orders (4 symbols) |
| 78 checks, 4 ctest suites, 6 injected bugs | testing |

---

## Appendix A: Key code, annotated

Real line numbers from `include/` and `src/` (branch `upgrade-2026-10`). `⋮` = left out; `◀` = plain-English note.

### Reading the code: C++ constructs
| Construct | Meaning |
|---|---|
| `template <typename T, size_t Capacity>` | compile-time parameters; `RingBuffer<Command, 65536>` is its own type with the constants folded in |
| `static_assert(cond, "msg")` | compile-time check (e.g. capacity is a power of two, `Command` is 40 bytes) |
| `std::atomic<size_t>` | a variable that is safe to read and write from two threads |
| `.load(std::memory_order_relaxed)` | atomic read with no ordering promise; fine for your OWN index |
| `.load(std::memory_order_acquire)` | reads after this see everything the other thread wrote before its matching release |
| `.store(v, std::memory_order_release)` | writes before this become visible before `v` does (publishes the slot) |
| `alignas(64)` | start this member on a new 64-byte cache line (avoids false sharing) |
| `std::atomic_ref<uint64_t>(x)` | treat a plain variable (here in mmap'd memory) as atomic for one operation |
| `x & (Capacity - 1)` | same as `x % Capacity` when Capacity is a power of two, but one instruction |
| `Order*` / `->` | a pointer to an Order / access a member through the pointer |
| `std::unique_ptr<T[]>` | an owned array, freed automatically |

### A.1 The lock-free SPSC ring (`core/ring_buffer.hpp`)
```cpp
19  template <typename T, size_t Capacity>
20  class RingBuffer {
21      static_assert(Capacity >= 2 && (Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
27      bool push(const T& item) {                               // ◀ PRODUCER thread only
28          const size_t head = head_.load(std::memory_order_relaxed);   // ◀ my own index: relaxed is enough
29          if (head - tail_cache_ >= Capacity) {                // ◀ looks full by my STALE copy of tail …
30              tail_cache_ = tail_.load(std::memory_order_acquire);    // ◀ … so refresh it (cross-core read, rare)
31              if (head - tail_cache_ >= Capacity) return false;       // ◀ really full
32          }
33          slots_[head & (Capacity - 1)] = item;                // ◀ write the slot FIRST
34          head_.store(head + 1, std::memory_order_release);    // ◀ then publish: the consumer sees the slot before the index
35          return true;
36      }
38      bool pop(T& out) {                                       // ◀ CONSUMER (matching) thread only
39          const size_t tail = tail_.load(std::memory_order_relaxed);
40          if (tail == head_cache_) {                           // ◀ looks empty by my stale copy of head
41              head_cache_ = head_.load(std::memory_order_acquire);    // ◀ acquire pairs with the release at :34
42              if (tail == head_cache_) return false;
43          }
44          out = slots_[tail & (Capacity - 1)];
45          tail_.store(tail + 1, std::memory_order_release);    // ◀ hand the slot back to the producer
46          return true;
47      }
⋮
57      std::vector<T> slots_;
59      alignas(kCacheLine) std::atomic<size_t> head_{0};        // ◀ producer's cache line …
60      size_t tail_cache_ = 0;                                  // ◀ … with its private copy of tail
64      alignas(kCacheLine) std::atomic<size_t> tail_{0};        // ◀ consumer's cache line
65      size_t head_cache_ = 0;
```
**What breaks if you change it:**
- Swap the order of `:33` and `:34`, and the consumer can read a slot before it is written.
- Make `:34` relaxed, and the CPU or compiler may reorder the same way.
- Drop `alignas`, and every push invalidates the consumer's cache line (false sharing).

### A.2 Pool, command and resting order (`core/object_pool.hpp`, `engine/order.hpp`)
```cpp
19          top_(capacity) {
20          for (size_t i = 0; i < capacity; ++i) free_[i] = &storage_[i];   // ◀ every node allocated ONCE
23      T* acquire() { return top_ > 0 ? free_[--top_] : nullptr; }       // ◀ pop the free stack; nullptr = full (order refused)
25      void release(T* p) { free_[top_++] = p; }                          // ◀ push back; no free()/delete
⋮
18  struct Command {                    // order.hpp: ring-buffer item AND WAL record (memcpy'd as-is)
    ⋮   OpType type; Side side; OrdType ord_type; uint8_t pad_; uint16_t symbol; uint64_t id; int64_t price; uint64_t qty;
27      uint64_t tsc;                   // ◀ cycle counter at submit → end-to-end latency
29  static_assert(sizeof(Command) == 40, "Command is the WAL record; keep it 40 bytes");
33  struct Order {
    ⋮   uint64_t id; int64_t price; uint64_t qty; Side side;
38      Order*   prev;                  // ◀ INTRUSIVE list links: the node IS the list entry
39      Order*   next;
```

### A.3 Matching with price-time priority (`src/order_book.cpp`)
```cpp
114 uint64_t OrderBook::submit(uint64_t id, Side side, int64_t price, uint64_t qty, OrdType type) {
117     if (type == OrdType::Market) price = side == Side::Buy ? kMaxPrice - 1 : 1;   // ◀ market = crosses at any price
118     if (id == 0 || price <= 0 || price >= kMaxPrice ||
119         (type == OrdType::Limit && index_.find(id))) {   // ◀ duplicate id for an order that could rest
120         expired_qty_ += qty; return 0;                   // ◀ refused: counted as expired
⋮
126     if (side == Side::Buy) {
127         while (qty > 0 && best_ask_ <= price) {          // ◀ PRICE priority: cheapest ask level first
128             Level& lvl = asks_[best_ask_];               // ◀ O(1): price IS the array index
129             while (qty > 0 && lvl.head) {
130                 Order* maker = lvl.head;                 // ◀ TIME priority: oldest order at this price
131                 const uint64_t x = std::min(qty, maker->qty);
132                 qty -= x; maker->qty -= x; lvl.qty -= x; resting_qty_ -= x;
136                 filled_qty_ += 2 * x;                    // ◀ both sides, so the conservation sum balances
140                 if (maker->qty == 0) erase_order(maker); // ◀ fully filled maker leaves the book
141             }
142             if (lvl.head == nullptr) raise_best_ask();   // ◀ level emptied: scan up to the next non-empty ask
143         }
⋮       // 144-162: the sell side, mirror image against bids_ / best_bid_
164     if (qty > 0 && (type != OrdType::Limit || !rest(id, side, price, qty)))
165         expired_qty_ += qty;                             // ◀ market/IOC remainder (or pool full) expires
⋮
90  bool OrderBook::rest(uint64_t id, Side side, int64_t price, uint64_t qty) {
91      Order* o = pool_.acquire();
92      if (!o) return false;                                // ◀ pool exhausted: refuse rather than allocate
93      *o = Order{id, price, qty, side, nullptr, nullptr};
95      Level& lvl = (side == Side::Buy ? bids_ : asks_)[price];
96      o->prev = lvl.tail;                                  // ◀ append at the TAIL: FIFO = time priority
97      if (lvl.tail) lvl.tail->next = o; else lvl.head = o;
98      lvl.tail = o;
102     index_.insert(id, o);                                // ◀ id → node, for O(1) cancel
⋮
169 bool OrderBook::cancel(uint64_t id) {
170     Order* o = index_.find(id);                          // ◀ O(1) expected (open addressing)
176     lvl.qty -= o->qty; resting_qty_ -= o->qty; cancelled_qty_ += o->qty;
179     erase_order(o);                                      // ◀ O(1) unlink via prev/next + pool release
⋮
82  void OrderBook::erase_order(Order* o) {
83      Level& lvl = (o->side == Side::Buy ? bids_ : asks_)[o->price];
84      if (o->prev) o->prev->next = o->next; else lvl.head = o->next;   // ◀ unlink from anywhere in the queue
85      if (o->next) o->next->prev = o->prev; else lvl.tail = o->prev;
86      index_.erase(o->id);
87      pool_.release(o);
```
**Worked example:** asks are 50@101 (#1), 30@101 (#2) and 40@102 (#3). A buy of 70@102 arrives:
- it fills 50 against #1 (the oldest at 101), and #1 is erased;
- it fills 20 against #2, leaving #2 with 10;
- `best_ask_` stays at 101.

That matches the README demo.

### A.4 Id table with backward-shift deletion (`src/order_book.cpp`)
```cpp
27  : keys_(round_up_pow2(capacity * 2), kEmpty), ...      // ◀ size ≥ 2× max orders → load factor ≤ 0.5
31  size_t IdTable::slot(uint64_t id) const { return mix(id) & mask_; }   // ◀ splitmix64 hash
33  void IdTable::insert(uint64_t id, Order* o) {
35      while (keys_[i] != kEmpty && keys_[i] != id) i = (i + 1) & mask_;   // ◀ linear probing
⋮
49  void IdTable::erase(uint64_t id) {
57      for (size_t j = (i + 1) & mask_; keys_[j] != kEmpty; j = (j + 1) & mask_) {
58          const size_t home = slot(keys_[j]);
59          if (((j - home) & mask_) >= ((j - i) & mask_)) {  // ◀ can entry j legally move back into hole i?
60              keys_[i] = keys_[j]; vals_[i] = vals_[j];       // ◀ shift it back, and the hole moves to j
62              i = j;
63          }
64      }
65      keys_[i] = kEmpty;                                    // ◀ no tombstones: probe chains stay short forever
```

### A.5 The matching thread and the write-ahead log
```cpp
14  using CommandQueue = RingBuffer<Command, 1u << 16>;            // engine/matching_engine.hpp
24  void apply(const Command& c) {
25      if (journal_) journal_->log(c, ex_);       // ◀ WRITE-AHEAD: log first …
26      ex_.apply(c);                              // ◀ … then mutate the book (dispatch by symbol)
29      const uint64_t cycles = rdtsc() - c.tsc;   // ◀ end-to-end latency in cycles → ns histogram
⋮
35  void run(CommandQueue& q, const std::atomic<bool>& producer_done) {
38      if (q.pop(c)) apply(c);
⋮       else if (producer_done && q.empty()) break;
43      else cpu_pause();                          // ◀ busy-spin (x86 `pause`), never sleeps
⋮
76  bool append(const Command& c) {                // persistence/wal_logger.hpp
77      const uint64_t n = header_->count;
78      if (n >= max_records_) return false;       // ◀ segment full → the journal rotates
79      records_[n] = c;                           // ◀ memcpy into the mmap'd file (MAP_SHARED, :162)
80      std::atomic_ref<uint64_t>(header_->count).store(n + 1, std::memory_order_release);
                                                   // ◀ count AFTER the record: a crash never counts a torn record,
                                                   //   and the follower's acquire-load sees complete records
```

### A.6 Snapshots on a follower thread (`persistence/journal.hpp`)
```cpp
122 void rotate(const Exchange& ex) {               // ◀ matching thread: the active segment is full
134     state_[active_].store(kSealed, std::memory_order_release);
135     if (state_[other].load(std::memory_order_acquire) != kFree) {
137         while (state_[other].load(std::memory_order_acquire) != kFree) cpu_pause();   // ◀ only waits if the follower is behind (never measured)
138     }
139     seg_[other]->reset(seq_);
140     state_[other].store(kActive, std::memory_order_release);
⋮
146 void follow(uint64_t applied) {                // ◀ follower thread
152     const uint64_t n = s.count();              // ◀ acquire: records [0, n) are complete
155     for (uint64_t k = applied - b; k < n; ++k) replica_->apply(s.at(k));   // ◀ its OWN copy of the books
167     if (applied > snap && (applied - snap >= every_ || urgent)) {
169         img_.capture(*replica_, applied);      // ◀ copy the REPLICA, not the live books (no stall on the matching thread)
170         write_or_die();                        // ◀ Snapshot::write: tmp → fsync → rename → fsync(dir)
171         snap = applied;
⋮
177     if (state_[i].load(std::memory_order_acquire) != kSealed || end_of(i) > snap) continue;
179     seg_[i]->reset(snap);                      // ◀ truncate a sealed segment fully covered by the durable snapshot
180     state_[i].store(kFree, std::memory_order_release);
```
Segment lifecycle: **Active → Sealed** (matching thread) **→ Free** (follower) **→ Active** (matching thread). Each transition is made by exactly one thread. Recovery = load the snapshot + replay later records in sequence order, refusing to continue past a gap.
