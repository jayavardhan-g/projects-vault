# microexchange — how it works

A limit order matching engine in C++20: the piece of software an exchange runs to
pair buyers with sellers. This document explains what the project does, how each
part works, and — at the end — measures the finished engine against the
straightforward first version of the same thing.

Contents

1. [What an order book has to do](#1-what-an-order-book-has-to-do)
2. [The shape of the system](#2-the-shape-of-the-system)
3. [The parts, one by one](#3-the-parts-one-by-one)
4. [Correctness](#4-correctness)
5. [How the numbers are measured](#5-how-the-numbers-are-measured)
6. [Before and after](#6-before-and-after)
7. [Limits and next steps](#7-limits-and-next-steps)

---

## 1. What an order book has to do

A limit order says *"buy 70 shares at 102 or better"*. The exchange holds orders
that cannot trade yet (**resting** orders) and matches them against new arrivals.
Two rules decide who trades first:

- **Price priority** — the best price wins. Highest bid, lowest ask.
- **Time priority** — at the same price, whoever arrived first trades first.

Run `./microexchange demo` to see both:

```
  book before            sell 50@101, 30@101, 40@102   buy 25@100
  incoming               buy 70 @ 102
  fill  50 @ 101   buy #5 / sell #1
  fill  20 @ 101   buy #5 / sell #2
  best bid / ask         100 / 101
```

The buyer was willing to pay 102, but got filled at 101 — the resting price, not
their own limit. The 50-lot from order #1 filled completely before order #2 got
anything, even though both sat at 101, because #1 arrived first. The remaining 10
of order #2 stays in the book and becomes the new best ask.

Three operations matter, and they all have to be fast:

| operation | what it does |
| --------- | ------------ |
| **submit** | match against the other side, rest whatever is left |
| **cancel** | pull a specific resting order out of the book |
| **best bid / ask** | what is the top of the book right now |

## 2. The shape of the system

Two threads. One takes orders in, one matches them. Everything between them is a
queue that never locks.

```
   ingestion thread                        matching thread
   ----------------                        ---------------
   build Command                           pop Command
        |                                       |
        v                                       v
   ring buffer  ----------------------->   write-ahead log   (append, memcpy)
   (lock-free)                                  |
                                                v
                                           order book        (match, rest, cancel)
                                                |
                                                v
                                           latency histogram
```

Splitting the work this way means the thread accepting orders never waits on the
thread matching them. The write-ahead log comes *before* the book is touched, so
if the process dies mid-match, the log already knows what was supposed to happen.

Source layout:

```
include/core/        ring_buffer.hpp  object_pool.hpp  tsc_clock.hpp
include/engine/      order.hpp  order_book.hpp  matching_engine.hpp  workload.hpp
include/persistence/ wal_logger.hpp  snapshot.hpp
include/metrics/     latency_histogram.hpp
src/                 order_book.cpp   main.cpp
tests/               test_order_book.cpp   fuzz_correctness.cpp
baseline/            naive_engine.hpp     <- the first version, kept for comparison
```

## 3. The parts, one by one

### 3.1 The order (`engine/order.hpp`)

Two plain structs. `Command` is what crosses the queue — trivially copyable, so
it can be memcpy'd into the ring buffer and into the log without any conversion
step. `Order` is what rests in the book, and it carries its own `prev`/`next`
pointers.

```cpp
struct Order {
    uint64_t id;
    int64_t  price;
    uint64_t qty;
    Side     side;
    Order*   prev;   // links this node into its price level
    Order*   next;
};
```

Those two pointers are the whole trick behind the book, explained in 3.4.

### 3.2 The ring buffer (`core/ring_buffer.hpp`)

A single-producer, single-consumer queue with no mutex. The producer owns `head_`,
the consumer owns `tail_`, and because there is exactly one of each, no locking is
needed — only the right memory ordering.

Two details make it fast:

**Cache-line separation.** A CPU moves memory in 64-byte lines. If `head_` and
`tail_` sat next to each other, every write by the producer would invalidate the
line the consumer is reading, and the two cores would ping-pong that line back and
forth. That is *false sharing* — the variables are independent, but the hardware
cannot tell. `alignas(64)` puts them on separate lines:

```cpp
alignas(64) std::atomic<size_t> head_{0};
size_t tail_cache_ = 0;      // producer's private guess at tail_

alignas(64) std::atomic<size_t> tail_{0};
size_t head_cache_ = 0;      // consumer's private guess at head_
```

**Cached indices.** The producer only needs `tail_` to know whether the queue is
full. So it keeps a stale copy and only re-reads the real one when its copy says
"full". Most pushes therefore touch no shared memory at all.

Capacity is a power of two, so wrapping is `index & (Capacity - 1)` instead of a
division.

### 3.3 The object pool (`core/object_pool.hpp`)

Calling `new` for each resting order would be a system-level allocation on the
hot path — hundreds of nanoseconds, unpredictable, and it fragments memory. So all
the `Order` nodes are allocated once, at startup, and handed out from a stack of
free pointers:

```cpp
T* acquire() { return top_ > 0 ? free_[--top_] : nullptr; }
void release(T* p) { free_[top_++] = p; }
```

Both are an index bump. This is what makes "zero allocations while matching" true,
and section 6 measures it.

### 3.4 The order book (`engine/order_book.hpp`, `src/order_book.cpp`)

The heart of the project. Two design choices:

**Price levels are a flat array.** Prices are integer ticks in `[1, 65536)`, so
`bids_[price]` and `asks_[price]` are a direct index — no map lookup, no tree
walk, no allocation when a new price is touched. Best bid and best ask are cached
in two variables and only re-scanned when a level empties.

**Each level is an intrusive doubly-linked list.** "Intrusive" means the list
pointers live inside `Order` itself rather than in separate container nodes. So:

- adding to the back of a level is a pointer swap — arrival order is preserved,
  which *is* time priority
- removing a node is a pointer swap, no search, no iterator, no deallocation

```cpp
void OrderBook::erase_order(Order* o) {
    Level& lvl = (o->side == Side::Buy ? bids_ : asks_)[o->price];
    if (o->prev) o->prev->next = o->next; else lvl.head = o->next;
    if (o->next) o->next->prev = o->prev; else lvl.tail = o->prev;
    index_.erase(o->id);
    pool_.release(o);
}
```

Cancel needs to get from an order id to its node without searching. That is an
open-addressed hash table (`IdTable`) sized once at construction, so a cancel is:
hash the id, read the pointer, unlink. All O(1), no allocation.

Matching walks the far side from the best price outward, taking from the front of
each level until the incoming order is filled or the price no longer crosses:

```cpp
while (qty > 0 && best_ask_ <= price) {
    Level& lvl = asks_[best_ask_];
    while (qty > 0 && lvl.head) {
        Order* maker = lvl.head;              // oldest at this price
        const uint64_t x = std::min(qty, maker->qty);
        ...                                   // partial fills fall out naturally
        if (maker->qty == 0) erase_order(maker);
    }
    if (lvl.head == nullptr) raise_best_ask();
}
if (qty > 0) rest(id, side, price, qty);      // leftover joins the book
```

### 3.5 The matching engine (`engine/matching_engine.hpp`)

The consumer loop, and the only place the ordering of side effects is decided:

```cpp
void apply(const Command& c) {
    if (wal_) wal_->append(c);        // write-ahead: log before mutating state
    apply_command(book_, c);
    if (hist_) hist_->record(...);    // time from submit to done
}
```

Log first, then change the book. That ordering is what makes the log *ahead* of
the state, and therefore what makes recovery possible.

### 3.6 The write-ahead log (`persistence/wal_logger.hpp`)

The naive way to log is `ofstream::write` plus a flush. That is a system call per
record, and when the file grows the filesystem has to update its metadata — an
unpredictable stall in the middle of trading.

Instead the file is created at its full size up front and mapped into memory, so
appending is:

```cpp
records_[header_->count] = c;
++header_->count;
```

A memcpy and an increment. The operating system writes the pages back on its own
schedule. Records are fixed-size `Command` structs behind a small header holding a
magic number and the record count, which is all the framing a fixed-size record
needs.

The file mapping is the one place the code is genuinely platform-specific:
`mmap`/`ftruncate` on macOS and Linux, `CreateFileMapping`/`MapViewOfFile` on
Windows. Everything above that line is identical.

### 3.7 Snapshot and recovery (`persistence/snapshot.hpp`)

Replaying a log from the beginning gets slower forever, so a snapshot writes the
whole book plus the log position it covers. Recovery is then:

```cpp
const uint64_t seq = Snapshot::load(book, snapshot_path);   // state as of record `seq`
for (uint64_t i = seq; i < log.size(); ++i)                 // catch up
    apply_command(book, log[i]);
```

Orders are written in price-time order and restored in the same order, without
matching, so time priority survives the restart. `./microexchange replay` rebuilds
the book both ways and checks they agree:

```
  replay  data/microexchange.wal
         from the log alone     1,000,000 records in 0.058 s
           resting orders       162,158
           best bid / ask       989 / 1008
         from snapshot + tail   0 records replayed
           resting orders       162,158
           best bid / ask       989 / 1008
         both recovery paths rebuild the same book        pass
```

### 3.8 Timing (`core/tsc_clock.hpp`, `metrics/latency_histogram.hpp`)

Measuring nanoseconds with `std::chrono` costs tens of nanoseconds per call, which
is significant when the thing being measured takes 83 of them. So timestamps come
straight from the CPU cycle counter — `rdtsc` on x86, `cntvct_el0` on ARM — and
are converted to nanoseconds once, using a factor calibrated at startup.

Storing every sample and sorting is also too slow, so the histogram is a flat
array of 1 ns buckets up to ~1 ms. Recording is one increment; a percentile is a
walk over the buckets afterwards.

## 4. Correctness

**The invariant.** Every unit of quantity that enters the book must end up in
exactly one of three places:

```
filled + resting + cancelled == submitted
```

Filled volume is counted on both sides of each trade, which is what makes the sum
balance (a 50-lot trade removes 50 from the buyer's order *and* 50 from the
seller's). The tape volume an exchange would publish is half of it.

**Unit tests** — 40 checks: orders that do not cross both rest; exact fills;
partial fills with the remainder resting; time priority within a level; price
priority across levels; cancelling, including from the middle of a queue and
cancelling an id that never existed; snapshot round trip preserving time priority;
and a WAL replay reproducing a live book.

**Property fuzzing** — a random stream of new orders and cancels, checking every
1,000 operations that the invariant holds, that totals recomputed by walking the
intrusive lists match the cached counters, that no zero-quantity order is still
linked, and that the book is never crossed.

**Equivalence** — `./microexchange compare` runs the identical command stream
through the optimised engine and the original naive one and checks the resulting
books are identical. Whatever the optimisations did, they did not change what the
engine decides.

## 5. How the numbers are measured

The benchmark runs two phases, because throughput and latency are different
questions.

**Phase 1 — throughput.** The producer sends as fast as it can. This measures how
many commands per second the engine can absorb. Latency here is meaningless: the
queue fills up, so most of the measured time is orders waiting in line.

**Phase 2 — latency.** The producer paces itself to a fixed rate, and each command
is stamped with the time it was *supposed* to be sent, not the time it actually
went out. If the engine stalls and the producer falls behind schedule, that
lateness shows up in the numbers instead of quietly disappearing. This is the
*coordinated omission* problem, and stamping the intended time is the standard fix.

The p50 is the number to trust: one command through the ring buffer, the log and
the matching path. A p99 in the millisecond range is the operating system taking
the core away, not the engine — it moves between runs on a busy laptop and
disappears on a quiet machine with spare cores.

## 6. Before and after

`baseline/naive_engine.hpp` is the first version of this project, kept in the
repository on purpose. It is not a strawman — it is the reasonable, obvious
implementation:

| the first version | what it became |
| ----------------- | -------------- |
| `std::mutex` + `condition_variable` queue | lock-free SPSC ring buffer |
| `std::map<price, std::list<shared_ptr<Order>>>` | flat price array + intrusive list |
| `make_shared` per resting order | pre-allocated object pool |
| linear scan through a level to find a cancel | O(1) unlink through an id table |
| `ofstream` flushed on every record | memory-mapped append |
| `chrono` clock, every sample stored and sorted | `rdtsc` + bucket histogram |

Both are compiled into the same binary and driven by the same random command
stream, so this is a like-for-like measurement, not two different workloads.

```sh
./microexchange compare 500000
```

### Results

500,000 commands, 4-core laptop, GCC, `-O3 -march=native`:

```
                                 before            after
  throughput               1,617,069 /s    12,969,888 /s     8.0x faster
  wall time                     0.309 s          0.038 s
  latency p50                    875 ns            83 ns    10.5x lower
  latency p99              2,227,668 ns           833 ns
  heap allocations              983,976                3    none per order

         resulting book         81,117 resting orders, bid/ask 1004/1011
         both engines produced exactly the same book      pass
```

| metric | before | after | change |
| ------ | ------ | ----- | ------ |
| throughput | 1,617,069 ops/s | 12,969,888 ops/s | **8.0x** |
| latency p50 | 875 ns | 83 ns | **10.5x lower** |
| latency p99 | 2.23 ms | 833 ns | see note below |
| heap allocations during the run | 983,976 | 3 | ~2 per order → none per order |
| wall time for 500k commands | 0.309 s | 0.038 s | **8.1x** |

The three remaining allocations in the optimised run are the thread, the ring
buffer's storage and the histogram — all before the first order is processed. Not
one order allocates. The allocation count is real, not estimated: global
`operator new` is replaced with a counting version in `src/main.cpp`.

**The p99 needs a word of honesty.** The naive engine's 2.2 ms tail is mostly its
condition variable: when the queue runs dry the consumer sleeps, and waking a
sleeping thread costs milliseconds if the scheduler is busy. The lock-free version
spins instead, so it never pays that. It is a real difference in design, but the
exact multiple swings a lot between runs, so treat the p50 as the honest headline
and the p99 as directional.

### It gets wider as the book grows

The same comparison at different sizes — the interesting column is the last one:

| commands | resting orders at the end | before | after | speedup |
| -------- | ------------------------- | ------ | ----- | ------- |
| 100,000 | 16,076 | 2,091,413 /s | 20,332,952 /s | 9.7x |
| 500,000 | 81,117 | 1,617,069 /s | 12,969,888 /s | 8.0x |
| 1,000,000 | 162,158 | 716,043 /s | 13,079,468 /s | 18.3x |
| 2,000,000 | 324,182 | 216,189 /s | 10,816,352 /s | **50.0x** |

The optimised engine stays roughly flat between 10 and 20 million operations per
second no matter how big the book gets. The naive one falls off a cliff — from 2.1
million to 216 thousand, a **10x slowdown on itself** — because two of its costs
grow with the book:

- **cancelling is a search.** With no way to get from an order id to its list
  node, cancel walks the price level from the front. Longer levels, slower
  cancels. The optimised version unlinks in constant time no matter how long the
  level is.
- **pointer chasing gets expensive.** `std::map` nodes, `std::list` nodes and
  `shared_ptr` control blocks are scattered across the heap. While the book is
  small they stay in cache; at 324,000 resting orders they do not, and every hop
  becomes a cache miss. The flat price array and pooled orders keep related data
  close together.

That is the real lesson of the project: the constant-factor work — no locks, no
allocation, cycle-counter timing — bought roughly an order of magnitude, and it
bought the same order of magnitude at every size. The **algorithmic and layout**
choices — O(1) cancel and contiguous memory — are what stopped the engine from
degrading as the book filled up. On a book with a million resting orders, the
first version would be unusable and this one would not have noticed.

### Where each change shows up

| change | what it fixes | visible in |
| ------ | ------------- | ---------- |
| lock-free ring buffer | mutex contention, sleep/wake latency | p50, p99 |
| object pool | allocator calls on the hot path | allocations, p50 |
| flat price array | map lookup and node allocation per new price | throughput |
| intrusive list + id table | O(n) cancel scan | scaling with book size |
| memory-mapped log | a syscall and a flush per record | throughput |
| rdtsc + bucket histogram | measurement cost inside the measurement | p50 accuracy |

## 7. Limits and next steps

What this is not, yet:

- **One symbol.** A real exchange runs many books, usually one thread per symbol
  group.
- **Limit orders only.** No market, IOC, or fill-or-kill orders; no self-trade
  prevention.
- **Prices are ticks in `[1, 65536)`.** The flat array is what buys the O(1)
  lookup, and it costs 3 MB of levels. A wider price range would want a sparse
  index over the same idea.
- **The snapshot is written at the end of a run**, not periodically in the
  background.
- **No network layer.** Orders come from a generator, not a socket. A real
  ingestion path would add parsing and a session protocol in front of the ring
  buffer.

The pieces that would come next, roughly in order: order types, self-trade
prevention, periodic background snapshots, then multiple symbols.
