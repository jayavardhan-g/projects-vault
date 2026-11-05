> Prep pack for **Micro Exchange**, written 2026-11-05 from `microexchange/INTERVIEW_PREP.md`. The vault copy is the one to study and edit.

# Micro Exchange: interview prep

Branch read: `upgrade-2026-10` (HEAD `1fc2869`). `main` is the original (`fb58991`).
Sources: README.md, PROJECT.md, git log of both branches, every file under
`include/`, `src/`, `tests/`, `baseline/`, `bench/` and `results/`.
On 2026-11-05 I re-ran `./microexchange test`, `fuzz 200000`, `fuzz 1000000` and `crash 8` with the
existing binary. All passed. I did not re-run any benchmark. The server tables below come from `results/`.

---

## 0. Read this first

Five things to settle before any interview.

1. **10.8M ops/s and 50× are single laptop runs, and no raw output was kept.**
   PROJECT.md §6 says so: "No raw output from these laptop runs was kept, and
   they did not reproduce on a shared server." On the server, the same original
   binary gives 3.44M ops/s and 18.1× at 2M commands (median of 5 paired runs).
   The repo cannot back the resume numbers. Section 5 has honest wording.
2. **"The ring handoff is the bottleneck" is a hypothesis, not a measurement.**
   PROJECT.md §6 says "The likely mechanism, not yet profiled". The 18.1M/s
   single-thread run (`bench/book_only.cpp`) also leaves out the WAL append,
   two `rdtsc` calls and the histogram. So the 18.1M vs 3.44M gap is the whole
   pipeline, not only the ring. Say "most likely the cross-core handoff", not
   "proven to be".
3. **Two different "18.1" numbers.** 18.1× is the paired speedup at 2M commands.
   18.1M/s is the single-thread book rate at **1M** commands. They are unrelated.
   Do not mix them up under pressure.
4. **The resume's wording of the cancel fix is slightly off.** The naive engine
   already finds an order by id in O(1) (`std::unordered_map`,
   `baseline/naive_engine.hpp:60`). Its slow part is walking the price level's
   `std::list` to find the node (`baseline/naive_engine.hpp:108-119`). The fix
   is O(1) *unlinking* through an intrusive list. Also, "heap-based data
   structures" sounds like a binary heap. You mean node-based containers on the
   heap (`std::map`, `std::list`, `shared_ptr`).
5. **What GitHub shows.** `origin` (github.com/jayavardhan-g/Micro-Exchange) only has
   `main`. The upgrade branch has no upstream. On `main`:
   - the fuzzer only checks an invariant (no reference model);
   - "crash recovery" means replaying the log after a clean run (no kill -9);
   - the WAL count is a plain `++header_->count`, and appends stop silently when the log is full.

   The strong versions of "fuzzing" and "crash recovery" exist only on the
   unpushed branch. Also: the first commit's author field is
   `Tarunsai <dtarunsai08@gmail.com>`. It is one 2,719-line commit, so git holds no
   history of how the original was built. All upgrade commits carry
   `Co-Authored-By: Claude`. If you present yourself as sole author, expect a
   reviewer to see both. Decide now how you will explain it (see section 8).

---

## 1. Pitch

### 60 seconds

I built a limit order matching engine in C++20. It is the core of an exchange.
It pairs buyers with sellers by price, then by arrival time. One thread takes
orders in. Another thread matches them. A lock-free ring buffer joins the two.
The book never allocates while matching. Price levels are a flat array indexed by
tick. Each level is an intrusive linked list, so cancel is a pointer swap.
Every command goes to a memory-mapped write-ahead log before it touches the
book. A follower thread replays that log into a replica and takes snapshots, so
recovery after kill -9 is exact. I tested it with a differential fuzzer against a
simple reference book and with repeated kill -9 tests. On a laptop it ran 50×
faster than my naive first version. On a shared Xeon server it is 18× faster,
at 3.4M commands per second. The book alone does 18M per second on one core.
So the limit there is moving data between cores, not matching.

### 3 minutes

**The problem.** An exchange keeps resting orders that cannot trade yet. A new
order trades against the best opposite price first. At one price, the oldest
order goes first. Three operations must be fast: submit, cancel, best bid/ask.

**The first version.** I wrote the obvious version first and kept it as
`baseline/naive_engine.hpp`. It uses a mutex and condition variable queue and a
`std::map` of `std::list<shared_ptr<Order>>`. It allocates about twice per order
and flushes an `ofstream` on every record. It works. But it gets slower as the
book grows. Cancel walks the price level to find the node. Map and list nodes
are scattered across the heap, so each step is a cache miss.

**The fast version.** I changed six things.
1. A single-producer single-consumer ring buffer. It uses acquire/release
   atomics and no lock. Head and tail sit on separate cache lines, and each
   side caches the other's index.
2. A pre-allocated object pool for orders.
3. A flat array of price levels, indexed by integer tick.
4. An intrusive doubly linked list per level. An open-addressed id table gives
   cancel the node pointer directly.
5. A memory-mapped log. An append is one copy and one release store.
6. `rdtsc` timing and a flat histogram, so measuring costs almost nothing.

Both engines run in one binary on the same random command stream. A check
confirms they end with the same book.

**Correctness.** One invariant must always hold: filled + resting + cancelled +
expired = submitted. A fuzzer sends random limit, market and IOC orders and
cancels to the engine and to a `std::map` reference model. Every trade must
match exactly. A crash test forks the real two-thread engine and kills it with
SIGKILL. Some kills are random. Others fire inside each snapshot step. Then it
recovers and compares the books order by order with a clean replay. I also
injected six bugs by hand to check that the tests catch them.

**Upgrade.** Later I added multiple symbols, market and IOC orders, and a
two-segment log. I also added an epoll TCP gateway and thread pinning. My first
snapshot design copied the books on the matching thread. It stalled that thread
for 1.6 ms per snapshot at 88k resting orders. So I moved snapshots to a
follower thread. It replays the log into its own replica and snapshots that.

**Numbers, honestly.** On my laptop, one run gave 10.8M ops/s at 2M commands,
which was 50× the naive version. I re-ran everything on a shared two-socket
Xeon: five runs, a warm-up, and paired ratios. There it is 3.4M ops/s and 18×.
Allocations (983,976 vs 3) and identical books reproduce exactly. The book alone,
on one core, runs at 18M/s. So the matching code is not the limit on that
machine. The most likely cost is the cross-core handoff, which is much dearer on
a big server mesh than on a laptop. I have not profiled it yet. Batching the
ring is the next experiment.

---

## 2. Architecture

### Diagram (in-process benchmark path)

```
 PRODUCER THREAD (pinned, cpus[0])                 MATCHING THREAD (pinned, cpus[1])
 src/main.cpp:131 run_phase()                      src/main.cpp:140 consumer lambda
   |                                                 |
   | Workload::make  src/main.cpp:117                | MatchingEngine::run
   |  -> next_command  include/engine/workload.hpp:28|   include/engine/matching_engine.hpp:35
   | stamp c.tsc (intended time) src/main.cpp:159    |   q.pop(c)  matching_engine.hpp:38
   v                                                 v
 RingBuffer::push                                  RingBuffer::pop
 include/core/ring_buffer.hpp:27  ---- 65,536 ---> include/core/ring_buffer.hpp:38
   slot write  :33                  slots            acquire load of head_ :41
   release store head_ :34                           release store tail_  :45
                                                     |
                                                     v
                                                   MatchingEngine::apply  matching_engine.hpp:24
                                                     |
                     +-------------------------------+--------------------------+
                     | 1. log first  :25             | 2. then apply :26        | 3. timing :30
                     v                               v                          v
            Journal::log                     Exchange::apply             LatencyHistogram::record
            include/persistence/journal.hpp:84  include/engine/exchange.hpp:25   include/metrics/latency_histogram.hpp:18
                     |                               |
                     v                               v
            WalLogger::append                OrderBook::submit  src/order_book.cpp:114
            include/persistence/wal_logger.hpp:76   OrderBook::cancel  src/order_book.cpp:169
              records_[n] = c       :79             |
              release store count   :80             +- flat Level arrays bids_/asks_ (order_book.hpp:38, kMaxPrice :14)
                     |                              +- ObjectPool<Order>   include/core/object_pool.hpp:23,25
                     | (shared mmap)                +- IdTable (open addressing) src/order_book.cpp:33,40,49
                     v
 FOLLOWER THREAD  Journal::follow  journal.hpp:146
   reads count (acquire), applies records to replica_ Exchange
   every N commands: SnapshotImage::capture  snapshot.hpp:61
                     Snapshot::write tmp -> fsync -> rename -> fsync dir  snapshot.hpp:84-100
   frees a Sealed segment once a snapshot covers it  journal.hpp:179
   sleeps 200 us between batches  journal.hpp:187

 RECOVERY  recover()  journal.hpp:234 : load snapshot, replay later segments in seq order, refuse a gap (:249)
```

### Gateway path (`./microexchange serve`)

```
 client --TCP 32-byte ClientMsg--> gateway thread                    matching thread
                                   Gateway::run  src/gateway.cpp:336   Gateway::match_loop  :163
                                   epoll_wait(...,0) busy poll :347    in_.pop -> journal_->log :168
                                   read_session :314 -> on_frame :274  handle :122 -> Exchange::apply
                                   validate, build Command, id =       emit() :110 -> out_ ring
                                   session<<40 | client id :300
                                   in_.push (drains reports if full) <--- out_ ring of Outbound
                                   drain_reports :259 -> attach seq from per-session FIFO
                                   queue_report :222 -> flush :236 (send, EPOLLOUT if full)
 client <--32-byte ExecReport---------
```

### One command, end to end (buy 70 @ 102 against sells at 101)

1. `run_phase` (src/main.cpp:131) calls `w.make(i, rng)` (src/main.cpp:117). That calls
   `next_command` (include/engine/workload.hpp:28). It returns a 40-byte `Command`
   (include/engine/order.hpp:18, size checked at :29).
2. In the paced phase, the producer spins until the due time. Then it sets
   `c.tsc = due`, the *intended* send time (src/main.cpp:159).
3. `RingBuffer::push` (ring_buffer.hpp:27) reads its own `head_` with relaxed
   ordering. It checks fullness against its private `tail_cache_`. Only if that
   looks full does it reload `tail_` with acquire (:30). It copies the command
   into `slots_[head & mask]` (:33), then publishes with `head_.store(head+1, release)` (:34).
4. On the matching core, `pop` (:38) compares `tail` with its private
   `head_cache_`. If they match, it reloads `head_` with acquire (:41). The
   acquire pairs with step 3's release. So the slot copy at :44 sees the
   whole command. Then it frees the slot with `tail_.store(tail+1, release)` (:45).
5. `MatchingEngine::apply` (matching_engine.hpp:24) logs first: `Journal::log`
   (journal.hpp:84) → `WalLogger::append` (wal_logger.hpp:76). That is a struct copy into
   the mmap (:79) and a release store of the count (:80). If the segment is full,
   `rotate` (journal.hpp:122) seals it and switches to the other segment.
6. `Exchange::apply` (exchange.hpp:25) picks the book by `c.symbol` and calls
   `OrderBook::submit` (src/order_book.cpp:114).
7. `submit` validates the order (:116-122). A duplicate Limit id, a bad price or
   id 0 is counted as expired. Then the buy loop runs while `best_ask_ <= price`
   (:127). It takes `lvl.head`, the oldest order at the best ask (:130), and
   trades `min(qty, maker->qty)` (:131). A filled maker is unlinked by
   `erase_order` (:140 → :82). That is two pointer fixes, `IdTable::erase` with
   backward shift (:49), and `pool_.release` (:87). When a level empties,
   `raise_best_ask` (:142 → :74) scans up to the next non-empty level.
8. The leftover 10 rests through `rest` (:164 → :90). It takes a node from the pool
   (:91), appends it to the level's tail, inserts it into the id table (:102) and
   updates the best bid if needed.
9. Back in `apply`, the histogram records `rdtsc() - c.tsc` in ns
   (matching_engine.hpp:30). That is one array increment (latency_histogram.hpp:18).
10. The follower reads the new count with acquire. It applies the same command to
    its replica and, on schedule, writes a snapshot (journal.hpp:146-182).

---

## 3. Design decisions

Each entry: alternatives, choice, why, cost, source.
"Documented" cites a doc or commit. "Reconstructed" means I read it from the code only.

**D1. Two threads joined by an SPSC ring, not a mutex queue or one thread.**
- Alternatives: one thread doing intake and matching; mutex + condvar queue (the baseline); MPMC queue.
- Chosen: one producer, one consumer, lock-free ring.
- Why: intake never waits on matching. Exactly one writer per index, so no lock is needed.
- Cost: every command crosses cores. On the server that handoff is the likely bottleneck (3.4M/s vs 18.1M/s single thread). One producer only: several gateways would need a ring each or an MPSC queue.
- Source: documented (PROJECT.md §2, §3.2, §6).

**D2. Acquire/release memory ordering.**
- Alternatives: `seq_cst` everywhere; relaxed everywhere; a mutex.
- Chosen: each side loads its own index relaxed (only it writes it). It loads the other's index with acquire and publishes its own with release.
- Why: release on `head_` makes the slot write visible before the new head. Acquire on the consumer pairs with it. In the other direction, release on `tail_` and acquire in the producer stop the producer from overwriting a slot the consumer is still reading. On x86 (TSO) both compile to plain `mov`, so they only stop compiler reordering. A `seq_cst` store becomes `xchg` or `mov+mfence`, a full fence per operation.
- Cost: correctness rests on reasoning, not tests. The fuzzer is single-threaded. I know of no TSan run in the repo.
- Source: reconstructed from include/core/ring_buffer.hpp:27-47. The docs only say "only the right memory ordering" (PROJECT.md §3.2).

**D3. `alignas(64)` on `head_` and `tail_`.**
- Alternatives: no padding; 128-byte padding.
- Chosen: 64-byte lines. Each side's private cache shares the line of the index it owns: `tail_cache_` with `head_`, `head_cache_` with `tail_`.
- Why: avoid false sharing. The producer writing `head_` must not invalidate the line the consumer writes `tail_` on.
- Cost: none in memory terms. Not covered: the slots themselves. A `Command` is 40 bytes, so adjacent slots share lines. When the queue is nearly empty, producer and consumer touch the same slot lines. Intel's adjacent-line prefetcher pairs 64-byte lines, which is why some queues pad to 128.
- Source: documented (PROJECT.md §3.2, README). The slot-sharing point is reconstructed.

**D4. Cached indices.**
- Alternatives: read the other side's atomic on every op (Lamport's original queue).
- Chosen: the producer keeps `tail_cache_`, the consumer keeps `head_cache_`. Each refreshes only when its copy says full or empty.
- Why: most pushes then touch no shared line.
- Cost: it helps only the side that is rarely blocked. When the consumer keeps up, the queue looks empty on almost every pop. Then the consumer reloads `head_` almost every time. That is the regime the docs blame for the server slowdown.
- Source: documented (PROJECT.md §3.2, §6 "likely mechanism").

**D5. Power-of-two capacity (65,536 slots).**
- Alternatives: any size with `%`.
- Chosen: `index & (Capacity-1)`, enforced by `static_assert` (ring_buffer.hpp:21). Indices grow forever (size_t), so full = `head - tail >= Capacity`, and no slot is wasted.
- Cost: 65,536 × 40 B = 2.6 MB per ring.
- Source: mask documented (PROJECT.md §3.2). Size and monotonic indices reconstructed (matching_engine.hpp:14).

**D6. Spin with `pause`, never block.**
- Alternatives: condvar sleep (baseline), futex, `epoll_wait` with a timeout.
- Chosen: busy-poll with `_mm_pause` / `yield` (tsc_clock.hpp:30). The gateway calls `epoll_wait(…, 0)`.
- Why: a sleeping thread costs a wake-up. The doc credits the naive 2.2 ms p99 mostly to condvar sleep and wake.
- Cost: burns a full core per spinning thread, even when idle. On a shared machine that is unfriendly.
- Source: documented (PROJECT.md §6 p99 note, §3.10).

**D7. Flat price-level array, not `std::map`.**
- Alternatives: `std::map<price, level>`; sorted vector; hash map of levels; a ring of levels around the mid price.
- Chosen: `std::vector<Level>` of 65,536 entries per side, indexed by tick (order_book.hpp:14, :38).
- Why: finding a level is one index. No tree walk, no node allocation when a new price appears.
- Cost: 2 × 65,536 × 24 B ≈ 3 MB per symbol, whatever the use. Prices must be integer ticks in [1, 65536). Thousands of symbols would need a sparse index.
- Source: documented (PROJECT.md §3.4, §7).

**D8. Cached best bid/ask with a linear rescan.**
- Alternatives: bitmap of non-empty levels with find-first-set; heap of prices.
- Chosen: keep `best_bid_`/`best_ask_`. Rescan one step at a time only when the best level empties (src/order_book.cpp:74-80).
- Why: simple, and in this workload the next level is nearly always a step or two away (prices span 41 ticks).
- Cost: worst case is O(price range). If the ask side empties, `raise_best_ask` walks to 65,536. That is about 64k checks over 1.5 MB, a latency spike. A two-level bitmap with `ctz` would bound it.
- Source: rescan documented (PROJECT.md §3.4). Worst case reconstructed.

**D9. Intrusive doubly linked list per level.**
- Alternatives: `std::list` (separate nodes); `std::deque`/vector per level with lazy-delete tombstones.
- Chosen: `prev`/`next` inside `Order` (order.hpp:33).
- Why: append at tail keeps arrival order, which *is* time priority. Unlink from the middle is O(1) given the node, with no allocation.
- Cost: the orders in one level are not contiguous, so matching still chases pointers. They do come from one pooled arena. `Order` is 48 bytes, with 7 bytes of padding after `side`.
- Source: documented (PROJECT.md §3.4).

**D10. Open-addressed id table with backward-shift deletion.**
- Alternatives: `std::unordered_map`; tombstones (the first version); Robin Hood hashing.
- Chosen: linear probing. Capacity is 2 × max orders, rounded to a power of two, so load ≤ 0.5. Hash is the SplitMix64 finalizer (src/order_book.cpp:11). Deletion shifts later entries back (:49-67).
- Why: no allocation, cache-friendly probes, and no dead slots.
- Cost: sized once. Overfill is prevented only because the pool caps resting orders. Id 0 is reserved as the empty key.
- Source: documented (PROJECT.md §3.4, commit dd8f14e).

**D11. Object pool as a free stack.**
- Alternatives: `new`/`make_shared` (baseline); a custom allocator; growing slabs.
- Chosen: allocate `capacity` nodes once. `acquire`/`release` pop and push a pointer stack (object_pool.hpp:23-25).
- Why: zero allocations while matching (983,976 vs 3 measured). LIFO reuse hands back a cache-hot node.
- Cost: fixed capacity. A full pool now refuses the order (expired), where the first version dropped it silently. Not thread-safe; it does not need to be.
- Source: documented (PROJECT.md §3.3, §3.9, §6).

**D12. One 40-byte trivially copyable `Command` for queue and log.**
- Alternatives: separate wire, queue and log formats; a varint or protobuf log.
- Chosen: the same struct is copied into the ring and the WAL. Symbol and order type fit in old padding, so the record stayed 40 bytes.
- Cost: no versioning beyond a magic number and record size. No checksum. Host byte order only.
- Source: documented (PROJECT.md §3.1, §3.9).

**D13. Log before apply.**
- Alternatives: apply then log; log asynchronously on another thread.
- Chosen: `journal_->log` then `ex_.apply` (matching_engine.hpp:25-26).
- Why: the log always covers at least the state. Recovery is a replay.
- Cost: an append on the hot path.
- Source: documented (PROJECT.md §3.5).

**D14. Pre-sized mmap log, release-store count, no per-record sync.**
- Alternatives: `write()` per record (baseline uses `ofstream` + `flush` = one syscall per record); `write` + `fdatasync` group commit; `O_DIRECT`; io_uring.
- Chosen: `ftruncate` to full size, `MAP_SHARED`, store the record, then a release store of the count (wal_logger.hpp:76-81, :156-165).
- Why: no syscall per record. A process killed at any instruction leaves a count that covers only complete records. The page cache belongs to the kernel, so it survives the kill.
- Cost: no power-loss durability. `msync` runs only at the end of a run (`final_snapshot` → `sync`). `ftruncate` makes a sparse file, so the first touch of each page faults. If the disk is full, a write to a mapped page raises SIGBUS, not an error code (reconstructed; `posix_fallocate` would reserve the blocks). In the gateway, acks go out before anything is synced.
- Source: documented (PROJECT.md §3.6, §7). SIGBUS and ack points reconstructed.

**D15. Two log segments plus a follower replica for snapshots.**
- Alternatives: snapshot only at the end (original); copy the live books on the matching thread (built first in dd8f14e); `fork()` copy-on-write.
- Chosen: a follower replays the shared-memory log into its own `Exchange`, snapshots that, and frees sealed segments. Each segment state change has one owner: Active → Sealed (matching), Sealed → Free (follower), Free → Active (matching).
- Why: the copy on the matching thread stalled it 1.6 ms per snapshot at 88,390 orders.
- Cost: double the memory and a second core of CPU for the replica. If the follower falls a whole segment behind, the matching thread waits (counted as `full_waits`; zero in every measured run).
- Source: documented (PROJECT.md §3.7, commit 120cb4e).

**D16. Snapshot written as tmp + fsync + rename + fsync dir; abort on failure.**
- Alternatives: overwrite in place; skip the dir fsync.
- Chosen: snapshot.hpp:84-100. `write_or_die` aborts if a snapshot fails (journal.hpp:201).
- Why: a crash leaves either the old or the new snapshot. Truncating the log behind a snapshot that never reached disk would lose commands.
- Cost: no checksum, so a corrupted but well-sized file would load. On Windows the fsyncs are no-ops.
- Source: documented (README "What makes it correct", PROJECT.md §3.7). No-checksum reconstructed.

**D17. `rdtsc` and a 1 ns bucket histogram.**
- Alternatives: `std::chrono` per sample, storing and sorting every sample (baseline); HdrHistogram.
- Chosen: raw TSC calibrated once over 20 ms against `steady_clock` (tsc_clock.hpp:51). Histogram of 2^20 `uint32` buckets (latency_histogram.hpp:14).
- Why: `chrono` costs tens of ns, a big share of an 83 ns operation.
- Cost: 4 MB per histogram. Everything over ~1.05 ms clamps into the last bucket, so "p99.9 = 1,048,575" means "over 1 ms". `rdtsc` is not serializing (no `lfence`/`rdtscp`), so it can drift a few cycles. Start and end stamps come from different cores, which needs an invariant, synchronized TSC. This server has `constant_tsc` and `nonstop_tsc`.
- Source: documented (PROJECT.md §3.8). Serialization and cross-core points reconstructed.

**D18. Two benchmark phases and the coordinated-omission fix.**
- Alternatives: one flat-out run reporting latency.
- Chosen: phase 1 measures throughput only. Phase 2 paces the producer and stamps the *intended* send time (src/main.cpp:159).
- Why: in a flat-out run, latency is just queue backlog. Pacing with intended-time stamps keeps stalls visible.
- Cost: the latency phase is a separate run. `compare` measures latency on n/3 commands at only 100k/s (src/main.cpp:531-532).
- Source: documented (PROJECT.md §5, README "About the benchmark numbers").

**D19. One book per symbol, one matching thread.**
- Alternatives: one shared book keyed by symbol; a thread per symbol group.
- Chosen: `Exchange` holds `unique_ptr<OrderBook>` per symbol, each with its own pool and id table (exchange.hpp).
- Cost: one matching thread for all symbols. Per-symbol memory is fixed (3 MB of levels plus pool).
- Source: documented (PROJECT.md §3.9, §7).

**D20. Market and IOC never rest; refused orders count as expired.**
- Chosen: Market uses an extreme price (src/order_book.cpp:117). Remainders and refusals add to `expired_qty_`, which keeps the invariant exact.
- Cost: the duplicate-id check applies only to Limit orders (:119). An IOC or Market order can reuse a resting order's id. The gateway would then report the *old* order's leaves as `Rested` (src/gateway.cpp:138, :156-160). The reference model copies the same rule, so the fuzzer cannot catch this.
- Source: documented (PROJECT.md §3.9). Duplicate-IOC gap reconstructed, not tested.

**D21. Gateway: epoll busy-poll, fixed 32-byte frames, session-tagged ids.**
- Alternatives: FIX or text; length-prefixed frames; thread per connection; `io_uring`; kernel bypass.
- Chosen: `ClientMsg`/`ExecReport` are 32 bytes each (protocol.hpp). Exchange id = `session << 40 | client_id` (src/gateway.cpp:300). A per-session FIFO of `seq`s. A client 4 MB behind is dropped. Listens on 127.0.0.1 only.
- Why: parsing is `bytes / 32` plus a memcpy. A fill routes to its owner straight from the maker id, with no lookup.
- Cost: no login, resend or cancel-on-disconnect. Also, session numbers restart at 0 on every gateway start (`next_gen_`, src/gateway.cpp:194). After a recovery, a new session can get the same number as a pre-restart session whose orders still rest. It would then receive that session's fills and clash on its ids. This is reconstructed; the gateway test restarts but never trades afterwards.
- Source: documented (PROJECT.md §3.10, §7). Restart-collision reconstructed.

**D22. Differential fuzzing against a reference model.**
- Alternatives: invariant-only property fuzzing (what `main` has); hand-written cases only.
- Chosen: the same random stream goes to the engine and a `std::map`-of-`std::list` model. Every trade must match. The invariant is checked every 1,000 ops and full books are compared every 10,000 (tests/fuzz_correctness.cpp:161-231).
- Cost: the model shares the engine's *spec*. A spec mistake made in both (like D20) is invisible. Single-threaded, so it does not test the ring or follower.
- Source: documented (PROJECT.md §4, commit dd8f14e).

**D23. Pinning with `pthread_setaffinity_np`, then unpin.**
- Chosen: `--cpus a,b` / `MX_CPUS`. `unpin_thread` restores the startup mask after a phase (affinity.hpp).
- Why: the original left the main thread pinned to CPU 0. The naive latency phase then ran both its threads on one core. CPU 0 also takes more interrupts.
- Source: documented (PROJECT.md §3.10, README, commit dd8f14e).

**D24. Paired, repeated measurement.**
- Chosen: 5 measured runs after a discarded warm-up. Sizes rotate within a run, arms are interleaved, and speedup is the median of per-run ratios (bench/summarize.py).
- Why: a shared machine drifts. Pairing cancels slow drift.
- Source: documented (commits b9d6f42, 52aaa9a).

**D25. Counting `operator new` to prove zero allocations.**
- Chosen: a global replacement counts every allocation (src/main.cpp:34).
- Cost: only `new` is counted. A direct `malloc` would slip through, but the engine has none.
- Source: documented (PROJECT.md §6).

**D26. No external libraries; build with one compiler command.**
- Source: documented (README). Cost: no Google Benchmark, no HdrHistogram, no test framework.

---

## 4. Hardest problems, as STAR stories

**S1. The headline numbers did not reproduce.** (documented: commit b9d6f42, README, PROJECT.md §6)
- Situation: my resume said 10.8M ops/s and 50×. Both came from single laptop runs with no saved output.
- Task: find out whether they hold on other hardware, and why not if they don't.
- Action: I wrote `bench/reproduce.sh`. It rebuilds the original commit, runs `compare` at four sizes plus `bench`, six rounds, and drops round 0. `bench/summarize.py` reports median, min and max, and the median of per-run paired speedups. Then I wrote `bench/book_only.cpp`. It applies the same stream to the book on one core, with no queue, log or timing.
- Result: on a 2 × Xeon Platinum 8570 the original binary gave 3.44M ops/s and 18.1× at 2M commands. The allocation count (983,976 vs 3) and identical books reproduced in 20/20 runs. The book alone ran at 18.1M/s. So the matching code was not the limit, and the gap is in the two-thread pipeline. I relabelled the laptop numbers in the docs.

**S2. Snapshots stalled the matching thread.** (documented: commits dd8f14e → 120cb4e, PROJECT.md §3.7, results/capture_stall_dd8f14e.csv)
- Situation: the first periodic-snapshot design copied the books on the matching thread, then a writer thread saved the copy.
- Task: snapshot without hurting latency.
- Action: I measured the longest copy per run (`bench/capture_stall.sh`): 1.6 ms median [1.2–1.7] at 88,390 resting orders. I pre-faulted the buffer to rule out page faults. That did not help. The commit message says the cost was about 8 ns per order of pointer chasing. Then I redesigned: a follower thread replays the shared-memory log into its own replica and snapshots that. Records are published with release stores and read with acquire.
- Result: the matching thread copies nothing. Snapshots every 100k commands cost 1.02× [0.91–1.56] throughput, with p50 unchanged. No run ever waited for a free segment.

**S3. A race between the follower and the matching thread.** (documented: commit 120cb4e, PROJECT.md §3.7)
- Situation: an early draft let the follower free "any segment that is not active".
- Task: make segment reuse safe.
- Action: I saw that a caught-up follower could empty the segment the matching thread had just switched into. I replaced the rule with a state machine. Each transition belongs to exactly one thread: matching seals and activates, the follower frees.
- Result: 40/40 SIGKILL recoveries were exact. Injected bugs were caught: freeing before the snapshot is durable, skipping a segment in recovery, and a snapshot with the wrong sequence.

**S4. The id table could hang forever.** (documented: commit dd8f14e, PROJECT.md §3.4)
- Situation: the first `IdTable` deleted with tombstones and never reclaimed them.
- Task: make it safe for a long-lived book.
- Action: I worked out that after enough distinct ids, no empty slot remains. A lookup of a missing id then never stops. I switched to backward-shift deletion. I also added a test that cycles 100,000 ids through a 16-order book and checks a random workload against `std::unordered_map`.
- Result: no tombstones at all. The regression test hangs on the old code and passes on the new.

**S5. The benchmark harness was unfair to the baseline.** (documented: commit dd8f14e, README, PROJECT.md §3.10)
- Situation: the original harness pinned the producer and never unpinned it. Threads created later inherit their creator's mask.
- Task: make the comparison fair.
- Action: I added `unpin_thread()` after each phase, plus `--cpus` and `MX_CPUS`. I re-ran on CPUs 0/2 to isolate the effect, and on CPUs 16/18.
- Result: the naive latency phase no longer runs both threads on CPU 0. Moving off CPU 0 also took the tail from over 1 ms to 8.6 µs p99.9 (results/regression.csv).

**S6. The follower slowed the matching thread by just reading.** (documented: commit 120cb4e, journal.hpp:184-187)
- Situation: the follower polled the segment header continuously.
- Task: stop it from hurting the hot path.
- Action: polling kept pulling the header's cache line, which holds `count`, away from the matching core on every append. I made the follower read in 200 µs batches, about 1,000 records at 4M/s.
- Result: the follower no longer contends per append. A whole-segment margin means it can always catch up.

**S7. Testing the tests.** (documented: PROJECT.md §4, commits dd8f14e, 120cb4e)
- Action: I injected six bugs by hand: IOC remainders resting, recovery ignoring the older segment, truncating before the snapshot is durable (in both designs), a wrong snapshot sequence, and the old tombstone table.
- Result: each one failed a test.

**S8. Tail latency would not sit still.** (documented: commit 52aaa9a, README "How much to trust the tails", results/prefault.csv)
- Situation: at 1M commands/s, p99 swung wildly.
- Action: I pre-faulted the log and pools to rule out page faults.
- Result: no improvement. The same binary and settings gave p99 483 µs in one session and 26 µs twenty minutes later. I did not keep the change, and the docs call p99 directional.

**S9. The original log could drop records silently.** (reconstructed from `git diff main upgrade-2026-10`; commit dd8f14e says the new log "never drops a record")
- On `main`, `WalLogger::append` returned silently when the log was full. The count was a plain `++header_->count` with no ordering. Now `append` returns false and the journal rotates. The count is published with a release store, so another thread or a post-kill reader sees only complete records.

---

## 5. Numbers sheet

### Resume numbers

| claim | where it came from | how measured | repo supports it? |
|---|---|---|---|
| **10.8M ops/s at 2M commands** | PROJECT.md §6 laptop table (10,816,352 /s) | `compare 2000000`, throughput phase, WAL on, 1 symbol, limit orders, 1 in 8 commands a cancel. 4-core laptop, GCC `-O3 -march=native`, Aug 2026. **Single run, raw output not kept.** | **No.** The server gives 3.44M [3.10–4.89M] with the original binary and 3.39M with the branch. |
| **50× over baseline** | same table: 10,816,352 / 216,189 = 50.0 | single laptop run, ratio of the two engines in one process | **No.** The server gives 18.1× [12.0–26.8] (original binary, CPUs 0/2, 5 paired runs) and 14.3× [12.7–15.2] (branch, CPUs 16/18). |
| 2M commands | stream length; ends with 324,182 resting orders | — | Yes (deterministic seed). |
| O(1) ID lookup / O(n) cancel | code | — | Partly. Both engines look up by id in O(1) (hash). The baseline's O(n) is the level-list walk, with n = orders at that price. The fix is O(1) *unlink*. Hash lookup is expected O(1), not worst case. |
| 200k+ order fuzzing | `./microexchange` check step: `run_fuzz(200000, …)` | 200,000 *operations*, about 3/4 new orders, so about 150k orders | Yes, if you say "operations". I re-ran 200k and 1M today; both pass (1M takes 0.7 s). On `main` it is invariant-only; on the branch it is differential. |
| baseline checks | `compare`: identical final book | filled, resting, cancelled, count, best bid/ask | Yes, 5/5 at every size, on both binaries. |
| crash recovery | branch: `crash` test, SIGKILL | 600k-command stream, 4 symbols, snapshots every 20k | Yes on the branch (8/8 today; 40/40 in commit 120cb4e). On `main` it is only replay after a clean run. |
| lock-free ring, object pool, WAL | code | — | Yes. |

### Other numbers you may be asked about

| number | source | setup | caveat |
|---|---|---|---|
| 18.1M commands/s single thread | results/book_only.csv | original engine, **1M** commands, 1 core (taskset), no queue, no log, no timing; 10 paired rounds after warm-up | Leaves out the WAL append, two `rdtsc` and the histogram, not just the ring. The branch does 17.0M (0.935× [0.85–0.95]). |
| 3.44M ops/s, 18.1× at 2M | results/repro_fb58991_summary.md | original binary, CPUs 0/2, load 8.7–10.2, 5 runs | Optimised spread is 3.10–4.89M, wide. |
| 3.39M ops/s, 14.3× at 2M | results/compare_new_* | branch, CPUs 16/18, load 10.2–17.9 | Part of the drop from 18.1× is the pinning fix. Load was also higher. Effects not separated. |
| 983,976 vs 3 allocations | every run at 500k | counting `operator new` | Reproduces exactly. The 3 are the thread, the ring storage and the histogram. |
| p50 83 ns (laptop) | PROJECT.md §6 | `compare` latency phase: 500k/3 commands at 100k/s, log off | Single run. Server: 431 ns (original), 522 ns (branch). |
| p99 833 ns vs 2.23 ms (laptop) | PROJECT.md §6 | as above | The docs call it directional. Naive and optimised use different clocks (chrono vs rdtsc). |
| 1.6 ms snapshot stall | results/capture_stall_dd8f14e.csv | dd8f14e, 1M mixed commands, 4 symbols, snapshot every 100k, 88,390 resting | Median of 5 (+ warm-up), range 1.2–1.7 ms. |
| gateway 13.2 µs RTT p50 | results/gateway.csv | loopback, one at a time, pinned 16/18 + client 20 | Most of it is two loopback TCP trips and syscalls. In-server p50 is 623 ns. |
| p99.9 8.6 µs vs >1 ms | results/regression.csv | CPUs 16/18 vs 0/2 | The histogram ceiling is ~1.05 ms. |

### Why the laptop and the server differ

- **The engine is not slower; the handoff is (most likely).** On the server, the book alone does 18.1M/s, about 55 ns per command. The full pipeline does 3.44M/s, about 290 ns per command. The extra ~235 ns per command is the ring handoff plus the WAL append, two `rdtsc` calls and the histogram. Two-thread throughput stays flat at 3.2–3.4M from 16k to 324k resting orders, while the book grows 20×. If the book were the limit, throughput would fall with size. It doesn't.
- **What a handoff costs.** With the consumer caught up, almost every `pop` sees "empty" in its cache. It reloads `head_` from the producer's core, a cache miss. Then it reads the slot the producer just wrote, often another miss. Every `push` writes a slot line the consumer just read, so the producer must invalidate the consumer's copy. That is at least two cache-line transfers per command. Commonly reported core-to-core latency is a few tens of ns on a client CPU and around 100 ns or more on large server meshes. I did not measure either machine.
- **Which cores.** I checked `lscpu` on this server: 2 sockets, 56 cores each, 4 NUMA nodes. CPUs 0, 2, 16 and 18 are all on NUMA node 0. So the handoff is core to core *within one socket's mesh*, not across sockets. Don't say "it crosses sockets".
- **The baseline is the same speed on both machines.** Naive is 216k/s on the laptop and 182–236k/s on the server at 2M. So the ratio shrinks along with the optimised side.
- **Shared machine.** Other users' load was 9–18. The laptop was a single unrepeated run, so its noise is unknown.

Say it like this: *"On my 4-core laptop, one run gave 10.8 million ops a second, 50× my naive version. I later re-ran it properly on a shared Xeon server, five paired runs. There it was 3.4 million and 18×. The book alone does 18 million a second on one core. So the drop is in passing each command between cores, which is far more expensive on a big server mesh. I haven't profiled that yet, so I call it the likely cause."*

### Honest resume wording

Bullet 2, conservative:
> Removed size-dependent slowdowns by replacing an O(level-length) cancel scan and node-based `std::map`/`std::list`/`shared_ptr` storage with O(1) intrusive unlinking, a tick-indexed flat price array and a pre-allocated pool; 18× faster than the naive baseline at 2M commands on a shared 2-socket Xeon (3.4M ops/s through the two-thread pipeline, 18M ops/s single-threaded), with zero heap allocations per order.

Bullet 2, if you want the laptop figure:
> … 18× (3.4M ops/s) over the naive baseline at 2M commands on a shared Xeon server, measured over 5 paired runs; up to 50× (10.8M ops/s) in a single run on a 4-core laptop.

Bullet 1 (true on the upgrade branch; push it first):
> Engineered a C++20 limit-order matching engine (price-time priority; multiple symbols; limit, market and IOC orders) with a lock-free SPSC ring buffer, object pooling, and a memory-mapped write-ahead log with snapshots taken off the matching thread; verified by 200k-operation differential fuzzing against a reference model, repeated kill -9 recovery tests, and book-equality checks against a naive baseline.

If only `main` is public, say "property fuzzing (200k operations)" and "log replay", not "crash recovery".

Note on the 50×: the bullet credits it to cancel and layout alone. But the baseline also had a mutex/condvar queue, a syscall per log record and per-sample `chrono` plus `push_back`. The ~10× at small books comes mostly from those constant-factor changes. The *growth* from 10× to 50× with book size is what cancel and layout bought. PROJECT.md §6 says exactly this.

---

## 6. Prerequisite concepts

**Limit order book, price-time priority.** Resting orders sit by price level. A new order trades against the best opposite price first. Within a level, first in, first out. Trades happen at the resting (maker) price.
Sources: L. Harris, *Trading and Exchanges* (2003), ch. 6; Gould et al., "Limit order books", *Quantitative Finance* 13(11), 2013.

**Order types.** Limit rests what is left over. Market takes any price and never rests. IOC trades up to its limit, and the rest is cancelled. Not built here: FOK (all or nothing, immediately), stop orders, amend/replace.
Source: Harris (2003), ch. 4.

**C++ memory model, acquire/release.** A release store "publishes" all earlier writes. An acquire load that reads that value sees them. Without the pair, the compiler or CPU may reorder, and a non-atomic slot read becomes a data race (UB). `seq_cst` adds a single global order, at the cost of a full fence on x86 stores.
Sources: Boehm & Adve, "Foundations of the C++ Concurrency Memory Model", PLDI 2008; Williams, *C++ Concurrency in Action*, 2nd ed., ch. 5.

**x86-TSO vs ARM.** x86 never reorders load-load, store-store or load-store. It only lets a later load pass an earlier store (store buffer). So acquire/release cost nothing on x86. ARM is weaker and needs `ldar`/`stlr` or barriers. Code that "works on x86" with relaxed atomics can fail on ARM.
Source: Sewell et al., "x86-TSO", CACM 2010.

**SPSC ring buffer.** One writer per index means no compare-and-swap is needed. Both push and pop are wait-free (bounded steps). The program as a whole still blocks, because callers spin when the ring is full. Cached indices (Rigtorp) cut shared reads.
Sources: Lamport, "Specifying Concurrent Program Modules", TOPLAS 1983; Thompson et al., "Disruptor" white paper (LMAX, 2011).

**Cache lines, coherence, false sharing.** Cores move memory in 64-byte lines under a MESI-style protocol. A write needs the line exclusive, which invalidates other copies. Two unrelated variables on one line make the line bounce between cores. That is false sharing. The fix is padding, `alignas(64)` or `std::hardware_destructive_interference_size`.
Source: Drepper, "What Every Programmer Should Know About Memory", 2007, §3.3 and §6.4.

**NUMA and core-to-core latency.** Big servers connect cores through a mesh and split memory into NUMA nodes. A cache line moving between cores costs more as distance grows. A laptop has few cores and short paths. Pinning both threads on one node keeps the cost lowest.
Source: Drepper (2007) §5; Intel's optimization reference manual (Xeon mesh / sub-NUMA clustering sections).

**Intrusive containers.** The links live inside the object, so there is no separate node allocation. Unlinking with a pointer to the object is O(1). One object can belong to only as many lists as it has link fields.
Source: Boost.Intrusive docs; Linux kernel `include/linux/list.h`.

**Open addressing, linear probing, backward-shift delete.** Keys live in one array, and a collision probes the next slot. Tombstone deletion leaves dead slots that clog the table. Backward shift moves later keys of the run back into the hole, so the table stays tombstone-free.
Source: Knuth, *TAOCP* Vol. 3, §6.4, Algorithm R.

**Object pools and free lists.** Pre-allocate N objects and keep a stack of free ones. Acquire and release are O(1), with no allocator lock and no fragmentation. LIFO reuse keeps hot memory hot.
Source: Drepper (2007); Stroustrup, *The C++ Programming Language*, 4th ed., §11.2 on allocation costs.

**Write-ahead logging and recovery.** Log the intent before changing state. After a crash, load the last checkpoint and replay the log after it. A deterministic state machine plus an ordered log gives an exact rebuild.
Sources: Mohan et al., "ARIES", TODS 1992; Schneider, "Implementing Fault-Tolerant Services Using the State Machine Approach", ACM Comput. Surv. 1990.

**fsync, msync, rename atomicity.** Writes land in the page cache first, which survives a process crash but not power loss. `fsync`/`msync(MS_SYNC)` force them to disk. Atomic replace is: write tmp, fsync tmp, rename, fsync the directory. Group commit batches many records per fsync.
Sources: Pillai et al., "All File Systems Are Not Created Equal", OSDI 2014; DeWitt et al., "Implementation Techniques for Main Memory Database Systems", SIGMOD 1984 (group commit).

**mmap for persistence.** Stores become plain memory writes, and the kernel writes pages back on its own schedule. You lose control of write-back order. I/O errors arrive as SIGBUS. Page faults show up on first touch.
Source: Crotty, Leis & Pavlo, "Are You Sure You Want to Use MMAP in Your DBMS?", CIDR 2022.

**TSC timing.** `rdtsc` reads a per-core cycle counter in ~20 cycles. With invariant TSC (`constant_tsc`, `nonstop_tsc`) it ticks at a fixed rate and is synced across cores. It is not serializing, so precise micro-timing uses `lfence; rdtsc` or `rdtscp`.
Source: Paoloni, "How to Benchmark Code Execution Times on Intel IA-32 and IA-64", Intel white paper, 2010; Intel SDM Vol. 3, "Time-Stamp Counter".

**Coordinated omission and percentiles.** If the load generator waits for slow replies, it sends fewer requests during a stall and under-counts the stall. The fix is a fixed schedule, measuring from the *intended* send time. p99 of n samples rests on the top 1% only, so it needs many samples and a quiet machine.
Sources: Gil Tene, "How NOT to Measure Latency" (talk, 2015); HdrHistogram docs.

**Benchmark methodology.** Repeat runs, drop warm-up, report spread, pair the arms. Small setup changes (core choice, environment) can move results more than the change being tested.
Sources: Georges et al., "Statistically Rigorous Java Performance Evaluation", OOPSLA 2007; Mytkowicz et al., "Producing Wrong Data Without Doing Anything Obviously Wrong!", ASPLOS 2009.

**epoll and busy polling.** `epoll` reports which sockets are ready. A timeout of 0 never sleeps, which trades a whole core for no wake-up latency. `TCP_NODELAY` turns off Nagle batching for small frames.
Source: `man 7 epoll`; Kegel, "The C10K problem".

---

## 7. Fifty likely questions

**Ring buffer and concurrency**

1. **Why SPSC and not a mutex queue?** One writer per index, so no lock and no CAS. A mutex costs an atomic RMW per push and pop. A condvar adds a sleep/wake that can take milliseconds under load. That wake-up is what dominated the baseline's 2.2 ms p99.
2. **Walk me through the memory ordering in `push`.** Load my own `head_` relaxed, since only I write it. Check fullness against `tail_cache_`. If it looks full, reload `tail_` with acquire. Write the slot. Then store `head_+1` with release. The release orders the slot write before the publish.
3. **And in `pop`?** Load my own `tail_` relaxed. If `tail == head_cache_`, reload `head_` with acquire. That pairs with the producer's release, so the slot read sees the full command. Read the slot. Store `tail_+1` with release. That orders the read *before* the producer may reuse the slot.
4. **Why is the acquire in `push` needed? The producer only reads an index.** Without it, the producer could see the new `tail_` and overwrite the slot while the consumer's read of it is not yet ordered. Release on `tail_` plus acquire on the reload forbids that. It is the write-after-read side.
5. *Hostile:* **"On x86 acquire/release is free, so why not relaxed?"** Relaxed is a data race on the non-atomic slot in C++: undefined behaviour. The compiler may then move the slot read above the index load. And on ARM the CPU reorders for real. Acquire/release costs nothing on x86 anyway.
6. **Why not `seq_cst`?** It adds a global total order I don't need. On x86 a `seq_cst` store is `xchg` or `mov+mfence`, a full fence on every push and pop.
7. **Is it lock-free or wait-free?** Each `push`/`pop` is wait-free: bounded steps, it never retries. The callers spin when the ring is full or empty. So the system as a whole can block if the other thread stops.
8. **What is false sharing, and where did you prevent it?** Two independent variables on one 64-byte line make the line bounce between cores. `head_` and `tail_` each sit on their own line, with the owner's private cache beside it.
9. *Hostile:* **"You padded the indices. What about the slots?"** I didn't pad them. A `Command` is 40 bytes, so slots share lines. When the queue is nearly empty, both cores touch the same lines. That is true sharing, not false, and you can't avoid it with one item per slot. Batching, or 64-byte slots, would cut it.
10. **Why cache the other side's index?** Reading the other atomic means a cross-core miss. With a cache, a push usually touches only lines the producer owns. It refreshes only when the ring looks full.
11. **When does the caching *not* help?** When the consumer keeps up. Then the ring looks empty on almost every pop, and each pop reloads `head_`. That is the likely reason throughput falls on the server.
12. **Why 64 bytes and not 128?** 64 is the x86 line size. Intel's spatial prefetcher fetches line pairs, so 128 can be better. I didn't measure it. C++17's `std::hardware_destructive_interference_size` is the portable answer.
13. **Why spin instead of sleep?** Wake-up latency. The cost is a full core per spinning thread. On a shared box I would add a back-off, or wake through a futex after N empty spins.
14. **How would you support several producers?** One SPSC ring per producer, with the consumer polling them in turn. That keeps single-writer indices. Or an MPSC queue with an atomic fetch-add on head plus a per-slot sequence number (Vyukov style).

**Order book**

15. **Why a flat array and not `std::map`?** A level lookup becomes one index, with no tree walk, rebalancing or node allocation. Price is an integer tick, so it works as an index.
16. *Hostile:* **"3 MB per symbol for 41 used prices is wasteful."** Yes. That's the trade. For thousands of symbols or wide price ranges I would use a window of levels around the mid price, or a sparse map from tick to level slot. The idea stays the same. PROJECT.md §7 lists it.
17. **How do you find the next best price after a level empties?** I step one tick at a time to the next non-empty level. That's cheap here, since the next level is usually adjacent.
18. *Hostile:* **"What's your worst case?"** If one side empties completely, `raise_best_ask` scans up to 65,536 levels. That is tens of µs. A 64-bit bitmap per 64 levels with `ctz`, plus a top-level summary word, would make it a few instructions.
19. **Why intrusive lists?** Append at the tail keeps arrival order, which is time priority. Unlink is O(1) given the node. No list nodes to allocate.
20. **Why not a vector per level?** FIFO matching would be contiguous and fast. But cancel from the middle becomes O(n) or needs tombstones and compaction. The workload has 1 in 8 commands cancelling.
21. **How does cancel work?** Hash the id in `IdTable`, get the `Order*`, fix the counters, unlink (two pointer writes), backward-shift delete from the table, return the node to the pool. If the level emptied and it was the best, rescan.
22. *Hostile:* **"Your resume says O(n) cancel was replaced with O(1) lookup. The naive version already had a hash map."** Correct. The naive version found the order in O(1). Then it walked the price level's `std::list` to find the node, which costs O(orders at that price). At 324k resting orders that is thousands of nodes per cancel. My fix is O(1) unlinking, because the table hands back the intrusive node itself.
23. **Why linear probing and not `unordered_map`?** No allocation per insert, and probes stay in one or two cache lines. Load ≤ 0.5 keeps runs short. The `unordered_map` in the naive version allocates a node per insert.
24. **What was the tombstone bug?** Deletes left tombstones that were never reused. A long-lived book cycling ids would end with no empty slots. A lookup for a missing id then probes forever. Backward-shift deletion removes tombstones entirely.
25. **What does the pool do when full?** It refuses the order whole and counts it as expired. The original silently dropped it.
26. **Why count filled quantity twice?** Each trade removes x from the buyer *and* x from the seller. Counting both sides makes filled + resting + cancelled + expired = submitted exact. Tape volume is half of it.
27. **How do market orders work without a price?** A market buy gets price 65,535 and a market sell gets 1, so they cross anything. Their leftover expires and never rests.

**Persistence and recovery**

28. **Why log before applying?** So the log always covers the state. If the process dies mid-apply, replay redoes that command deterministically.
29. **Why mmap and not `write()`?** No syscall per record. An append is a struct copy and a release store. The baseline's `ofstream::flush` per record is a `write` syscall each time.
30. *Hostile:* **"Is your WAL durable?"** Against a process crash, yes: the page cache belongs to the kernel and survives SIGKILL, and the tests prove it. Against power loss, no. I `msync` only at the end of a run. In the gateway, acks go out before any sync. A real venue needs group commit: batch records, one `fdatasync`, ack only after it.
31. **What would fsync per record cost?** One device flush per record. That is tens of µs on a good NVMe and more on others, so throughput would drop to tens of thousands per second. Group commit spreads one flush over many records.
32. **Why the release store on `count`?** Two reasons. First, a reader that loads `count` with acquire is sure to see every record below it. The follower relies on that. Second, the record is written before the count. So whatever instant the kill lands, the count covers only complete records.
33. *Hostile:* **"Pavlo says don't use mmap for databases."** His points are lost control of write-back order, I/O errors as SIGBUS, and TLB shootdowns on eviction. For an append-only log that I never read on the hot path, with process-crash durability as the goal, those are acceptable. For power-loss durability I'd move to `write` + `fdatasync` or `O_DIRECT`. I'd also `posix_fallocate` so a full disk can't SIGBUS me mid-trade.
34. **How do snapshots stay consistent?** The follower snapshots its replica at a known log sequence. The file goes to `snapshot.tmp`, gets fsynced, is renamed over `snapshot`, and the directory is fsynced. A segment is freed only after a snapshot covering it is durable.
35. **Why a follower replica and not copy-on-write `fork()`?** I measured the in-thread copy: 1.6 ms. Fork would also stall the matching thread, for page-table copying and CoW faults on every page it then writes. The replica costs memory and a core, but the matching thread does no extra work.
36. **What if the follower falls behind?** The matching thread can fill its active segment while the other is still Sealed. Then it spins until the follower frees it. That's counted as `full_waits`. It never happened in a measured run, because each segment holds two snapshot intervals.
37. **What does the kill -9 test prove?** The child runs the real two-thread pipeline and is SIGKILLed at random times, or inside each snapshot step. Recovery must find no gap. The recovered sequence must equal what was applied, or one more (logged, not yet applied). Every book must equal a clean replay, order by order.
38. *Hostile:* **"Kill -9 isn't a crash test; power loss is."** Agreed. It tests process-crash consistency only. Power loss would need fault injection below the page cache (dm-flakey, or a VM snapshot without flush) plus record checksums. I have neither.

**Measurement**

39. **Why `rdtsc`?** It is about 20 cycles, versus tens of ns for `clock_gettime` per sample. Calibrated once against `steady_clock`.
40. *Hostile:* **"The stamp is taken on one core and read on another. Is that valid?"** Only with an invariant, synchronized TSC. This server reports `constant_tsc` and `nonstop_tsc`. `rdtsc` is also not serializing, so a few cycles of skew are possible. That is fine for 400 ns medians, not for 10 ns claims.
41. **What is coordinated omission, and how did you handle it?** A closed-loop generator sends less during a stall and hides it. I stamp each command with its *scheduled* send time. Both the in-process paced phase and the TCP client are open loop.
42. **Why is p99 only directional?** It moves too much to quote as a fact. The same binary measured 483 µs, then 26 µs twenty minutes later. The machine is shared (load 9–18). The histogram clamps at ~1 ms. The laptop numbers were single runs. And the naive and optimised engines use different clocks. I quote p50 with ranges, and p99 only as "better or worse".
43. **Why median of paired ratios, not ratio of medians?** Both engines run back to back in one process. Pairing cancels drift in machine load between runs.
44. **Why does throughput stay flat as the book grows, while naive falls?** Cancel is O(1), and pooled nodes and flat levels stay cache-friendly. The naive version's cancel scan and pointer chasing grow with the book.

**Scaling, failures, what next**

45. **How would you scale beyond one matching thread?** Books are independent. So shard symbols across matching threads, each with its own ring and log. The gateway routes by symbol. Cross-symbol orders would break that.
46. **What would you try first to raise the two-thread throughput?** Batching. The consumer pops everything up to `head_` in one acquire. The producer publishes `head_` every k items or when idle. That turns per-command cache transfers into per-batch ones. I'd confirm with `perf c2c` first.
47. **What fails if the gateway restarts?** Books recover exactly. But session numbers restart at zero, so a new session can collide with an old one whose orders still rest. It would get their fills. I'd persist the session counter in the snapshot, or add real session logins.
48. **What happens when a client stops reading?** Reports queue in a 4 MB buffer, then the gateway disconnects it. Its resting orders stay in the book; there is no cancel-on-disconnect.
49. **What would you do differently?** Keep raw output from every benchmark, laptop included. Profile before explaining. Add a bitmap for best-price search. Add checksums to log records and snapshots. Run TSan on the concurrent paths. Make the reference model check rules against a written spec, not copy them.
50. **What's next?** Ring batching, then `perf c2c` to confirm the handoff theory. Group-commit durability. Self-trade prevention, FOK and amend. Per-symbol sharding. A hot standby seeded from the follower's replica.

---

## 8. Questions where the honest answer is "it didn't work" or "I didn't do that"

**"Can you reproduce 10.8M ops/s?"** No. Say: "That was one run on my 4-core laptop, and I didn't keep its output. When I reran it properly on a shared Xeon server, it was 3.4M and 18×. I've updated the docs to label the laptop numbers." Then explain why (section 5). Owning this beats defending it.

**"Did you profile the handoff?"** No. Say: "The single-thread result rules out the book. The cross-core handoff is my leading explanation, but I haven't run `perf c2c` or a ping-pong microbenchmark. The single-thread run also leaves out the log and timing, so I can't split the gap yet."

**"Did batching fix it?"** Not built. It's the documented next step.

**"Did pre-faulting fix the tails?"** No. Say: "I tried pre-faulting the log and pools to rule out page faults. Tails didn't improve. The same binary varied from 483 µs to 26 µs between sessions, so I dropped the change and stopped quoting p99 as a number."

**"Is it durable to power loss?"** No. Say: "Process crashes only. No per-record msync, and acks go before any sync. Group commit is how I'd close that, at a latency cost."

**"Did you test on real market data?"** No. The stream is synthetic: uniform prices over 41 ticks around 1000, uniform sizes, 1 in 8 commands a cancel (many naming orders already gone). Real flow is bursty, and cancels cluster near the touch.

**"Did you run ThreadSanitizer or a model checker?"** Nothing in the repo shows it. If you didn't, say: "No. The concurrency is two well-known patterns, a single-writer ring and release/acquire publication of the log count. I tested it with kill -9 and equality checks, but TSan is the next thing I'd add."

**"Self-trade prevention, FOK, amend?"** Not built. Only New and Cancel exist. A modify is cancel plus new, which loses queue position.

**"Does the gateway speak FIX?"** No. It uses fixed 32-byte binary frames on localhost. No login, resend or cancel-on-disconnect.

**"Did you measure it on ARM?"** The code has a `cntvct_el0` path, but the repo has no ARM measurements. Say so.

**"Are there bugs you know of?"** Two, both from reading the code and neither tested. (1) Gateway session numbers restart at zero after a restart, so a new session can collide with recovered orders. (2) An IOC or Market order reusing a resting order's id is not refused, and the gateway then acks it with the old order's leaves. The fuzzer copies the same rule, so it can't see (2).

**"Who wrote this?"** Decide your answer before the interview. The facts a reviewer can see: the first commit is authored "Tarunsai". The upgrade commits are co-authored with an AI assistant (Claude). A good answer is short and true. For example: "I designed it and I can defend every line. I used an AI coding assistant for parts of the upgrade, and the commits say so." If Tarunsai is your own alternate account, just say so. If someone else wrote part of the original, name what they did. Do not claim sole authorship if the history says otherwise. A reviewer who checks will trust nothing else you say.

---

## 9. Self-test exercises

Write your prediction before reading the answer.

**E1. Make the consumer's head reload relaxed.**
Change ring_buffer.hpp:41 to `head_.load(std::memory_order_relaxed)`. What happens on this x86 server? On ARM? Do the tests catch it?

> Answer: on x86 you will very likely see no change. TSO keeps load order in hardware, and the compiler rarely hoists the slot read here. But in C++ it is now a data race, so behaviour is undefined. A compiler may legally read the slot before the index. On ARM the CPU can reorder for real, and the consumer may read a half-written `Command`. The tests probably won't catch it on x86. The fuzzer is single-threaded, and the crash and bench tests compare final books, so a rare torn read would show only as an occasional failure. Lesson: memory-ordering bugs need reasoning or TSan, not just tests.

**E2. Put `head_` and `tail_` on one cache line.**
Remove `alignas(kCacheLine)` from `tail_` (ring_buffer.hpp:64). Predict the effect on `bench --no-wal` throughput (two threads, pinned), and on `bench/book_only.cpp`.

> Answer: `head_`, `tail_cache_`, `tail_` and `head_cache_` now share one line. Every push writes it and every pop writes it, so it bounces between the cores on every operation. Even reading `tail_cache_` on a push now misses after a pop. Expect two-thread throughput to drop clearly. I won't guess the size; measure it with 5 paired runs. `book_only` doesn't use the ring, so it is unchanged. That contrast is the point.

**E3. Stamp the actual send time in the paced phase.**
Change src/main.cpp:159 from `c.tsc = due;` to `c.tsc = rdtsc();`. Predict the effect on p50 and on p99/p99.9 at 1M/s.

> Answer: p50 barely moves, because the producer is usually on schedule. p99 and p99.9 fall, sometimes a lot. When the matching thread stalls, the ring fills and the producer blocks in `push`. Commands sent after the stall then carry late stamps, so the wait vanishes from the numbers. That is coordinated omission. The original choice is right.

**E4. Widen the price range.**
Set `kMaxPrice = 1 << 20` (order_book.hpp:14). Predict memory, start-up time, matching speed and snapshot time for `bench --symbols 16`.

> Answer: each book's levels grow from ~3 MB to ~48 MB (2 × 2^20 × 24 B), so 16 symbols take ~768 MB just for levels. Start-up slows, because the vectors are zero-filled and every page is touched. Matching speed is about the same, since only ~41 prices are used. One caveat: a side that empties now rescans up to 1M levels, so rare spikes get 16× worse. Snapshots don't change, because `for_each_order` scans only `[lo_price_, hi_price_]`. Market orders now use price 2^20 − 1. The wire check `price >= kMaxPrice` follows automatically.

**E5. Apply before logging.**
Swap the two lines in `MatchingEngine::apply` (matching_engine.hpp:25-26) so the book changes before the log append. Does `./microexchange crash 40` still pass? What changes in what recovery guarantees?

> Answer: usually it still passes, because the window between apply and log is a few ns and a random kill rarely lands there. But the child updates `applied` after `engine.apply` returns. A kill between `ex_.apply` and `journal_->log` can't make `applied` too high. Over many kills, though, a kill after apply and before log loses that command from the log. With the gateway, an ack based on applied-but-unlogged state would point at a command recovery doesn't know about. The test checks `r.seq == applied || applied + 1`, which allows logged-not-applied and catches applied-not-logged only if `applied` was published. So the test is weak against this exact bug. The real lesson: write-ahead matters once something outside sees the state, such as acks, fills or market data. Log before you let anyone see the effect.
