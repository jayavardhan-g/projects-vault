# microexchange

A limit order matching engine in C++20 — the kind of thing an exchange runs to
pair buyers with sellers. Written to explore how low-latency trading systems are
built: a lock-free queue between two threads, an order book that never allocates
while matching, a memory-mapped write-ahead log, and latency measured with the
CPU cycle counter.

No libraries, no build tools. Just a C++ compiler.

## Run it

**macOS / Linux**

```sh
./build.sh
./microexchange
```

**Windows**

```
build.bat
microexchange
```

That's the whole thing. `./microexchange` with no arguments builds a book, runs
every test, benchmarks a million orders and recovers from the log — about two
seconds:

```
  microexchange   C++20 limit order matching engine

  [1/5]  matching demo
         book before            sell 50@101, 30@101, 40@102   buy 25@100
         incoming               buy 70 @ 102
         fill  50 @ 101   buy #5 / sell #1
         fill  20 @ 101   buy #5 / sell #2
         best bid / ask         100 / 101
         oldest order at 101 filled first                 pass

  [2/5]  unit tests
         40 checks on matching, cancel, snapshot, replay  pass

  [3/5]  fuzz  200,000 random orders
         filled + resting + cancelled == submitted        pass

  [4/5]  benchmark  1,000,000 commands
         throughput             12,136,947 ops/s
         latency p50            83 ns
         latency p99            2,375 ns
         book balances after 1,000,000 commands           pass

  [5/5]  crash recovery
         replayed               1,000,000 log records
         rebuilt book identical to the live one           pass

  everything passed in 1.2 s
```

If the compiler is missing: macOS `xcode-select --install`, Ubuntu
`sudo apt install g++`, Windows install *Build Tools for Visual Studio* or
*MSYS2*. Needs g++ 10+, clang 13+, or MSVC 2022.

## The other commands

Everything above is also available on its own:

```sh
./microexchange demo             # one worked matching example
./microexchange test             # unit tests
./microexchange fuzz 1000000     # random orders + invariant check
./microexchange bench 1000000    # full benchmark table
./microexchange replay           # rebuild the book from disk
./microexchange compare 500000   # this engine vs the naive first version
./microexchange help
```

`bench` takes `--rate N` (ops/s for the latency phase) and `--no-wal`.
`fuzz` takes an operation count and a seed.
Log and snapshot files are written to `data/` — delete that folder any time.

`PROJECT.md` is the long version: how every part works, and a measured
before/after against the first, straightforward implementation.

## How it reads

```
build.sh, build.bat        one command per platform
PROJECT.md                 full write-up
include/
  core/
    ring_buffer.hpp        lock-free SPSC queue, cache-line padded
    object_pool.hpp        pre-allocated arena for Order nodes
    tsc_clock.hpp          cycle counter + calibration
  engine/
    order.hpp              Command / Order / Trade structs
    order_book.hpp         price-time priority book
    matching_engine.hpp    consumer loop: queue -> log -> book -> metrics
  persistence/
    wal_logger.hpp         memory-mapped write-ahead log
    snapshot.hpp           snapshot save/load + recovery
  metrics/
    latency_histogram.hpp  p50 / p99 / p99.9
src/
  order_book.cpp           the matching implementation
  main.cpp                 command line front end
tests/
  test_order_book.cpp      matching, cancel, snapshot, replay
  fuzz_correctness.cpp     random order stream + invariant check
baseline/
  naive_engine.hpp         the first version, kept to measure against
```

Start with `src/order_book.cpp` — that is where matching actually happens.

## What makes it fast

**Lock-free SPSC ring buffer.** One thread feeds orders in, one thread matches.
Power-of-two capacity so the index wrap is a mask. `head_` and `tail_` are
`alignas(64)` on separate cache lines, so the producer writing `head_` never
invalidates the line the consumer is reading `tail_` from. Each side also caches
the other's index and only refreshes it when the queue looks full or empty,
which removes most of the cross-core traffic.

**No allocation while matching.** Every `Order` node comes from a pool allocated
once at startup; `acquire()` and `release()` are an index bump. Price levels are
a flat array indexed by tick, so finding a level is one index — no map, no tree,
no node allocation.

**Intrusive lists.** Each price level is a doubly-linked list whose `prev`/`next`
pointers live on the `Order` struct itself. Adding to the back and removing from
anywhere are pointer swaps — O(1) — and arrival order is preserved, which is what
gives time priority. Cancels find their node through an open-addressed id table
sized up front, so they never allocate either.

**Memory-mapped log.** The log file is sized to its full length up front and
mapped into memory, so appending a record is a memcpy — no filesystem metadata
work in the middle of trading.

**Cycle-counter timing.** Timestamps come from `rdtsc` (`cntvct_el0` on ARM),
calibrated once against the steady clock. The histogram is a flat array of 1 ns
buckets, so recording a sample is a single increment.

## What makes it correct

**Snapshot and replay.** A snapshot writes every resting order in price-time
order along with the log position it covers. Recovery loads the snapshot and
replays only the records after it. `./microexchange replay` rebuilds the book
both ways — straight from the log, and snapshot plus tail — and checks the two
agree.

**The invariant.** Every quantity that enters the book has to end up somewhere:

```
filled + resting + cancelled == submitted
```

The fuzzer throws random orders and cancels at the book and checks this every
1,000 operations, along with recomputing the totals by walking the intrusive
lists, confirming no zero-quantity order is still linked, and confirming the book
is never crossed.

Filled volume is counted on both sides of each trade, which is what makes the sum
above balance; the tape volume an exchange would report is half of it.

## Before and after

`./microexchange compare` runs the same order stream through this engine and
through `baseline/naive_engine.hpp` — the obvious first version, with a mutex
queue, a `std::map` book and one allocation per order. On 500,000 commands:

| metric | before | after |
| ------ | ------ | ----- |
| throughput | 1,617,069 ops/s | 12,969,888 ops/s |
| latency p50 | 875 ns | 83 ns |
| heap allocations | 983,976 | 3 |

Both produce exactly the same book. The gap widens as the book grows — at 2
million commands it is 50x. `PROJECT.md` has the full table and the reasoning.

## About the benchmark numbers

Two phases, because they measure different things. The first runs the producer
flat out and reports throughput — latency there is just queue backlog, so it is
not reported. The second paces the producer at a target rate and stamps each
command with its *intended* send time, so a slow engine shows up as latency
instead of hiding behind the queue (the coordinated omission problem).

The p50 is the number that matters: one command through the ring buffer, the
write-ahead log and the matching path. A p99.9 in the millisecond range means the
machine lost the core for a moment — that is the operating system, not the
engine. Run it on a quiet machine with a few spare cores for a fair tail.
