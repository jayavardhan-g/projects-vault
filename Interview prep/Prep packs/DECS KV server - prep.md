> Prep pack for **DECS KV server**, written 2026-11-05 from `decs/INTERVIEW_PREP.md`. The vault copy is the one to study and edit.

# Interview prep: High Performance HTTP Key-Value Server (CS744 / DECS)

Repo: `/data/jayavardhan/projects/decs`, branch `main`, one commit (`82de05e "http server"`, 2026-08-18).
Public remote: `github.com/jayavardhan-g/High-Performance-HTTP-Key-Value-Server`.
All paths below are relative to the repo root. Line numbers are for that commit.

**What the repo has.** A README, the source, the k6 scripts, raw CSVs, plots and the logs of the last run.
**What it does not have.** No PROGRESS log, no RESULTS doc, no written analysis, no commit history.
The config values used in the runs (pool size, cache size, http_threads) are typed at stdin and never saved.
So most "why" answers below are **reconstructed from the code**. Only a few things are documented.

---

## 0. Read this first: the five things that matter

1. **The 11K "CPU saturation" claim holds, with care.** The `cpu` column in `results/final tests/*.csv` was
   **added by hand**. No script writes it. `script.sh:36` writes nine columns; `backup1/` has the same numbers with no
   `cpu` column. It reads 100% on get-popular at 750 and 1000 VUs while TPS sits at 11.1K. The values cap at 100, so
   it is a single-core reading. Most likely it is the core the server was pinned to. Say "I read it per core from htop".
2. **The 3.1K "I/O saturation" claim is not supported.** The disk was not saturated. In the only disk log in the repo
   (put-only, 1500 VUs), the real device `sdd` averaged **6% busy** (max 13%), about 60 flushes/s, 0.7 ms per flush.
   The server core was at 75%. So writes are **database-bound**, not disk-bound. Say "database-bound" instead.
3. **Every request pays a ~43 ms floor, and it is Nagle's algorithm plus delayed ACK.** cpp-httplib leaves
   `TCP_NODELAY` off (`httplib.h:145`). It writes headers and body in two `write()` calls (`httplib.h:7728`, `:7735`).
   k6's `http_req_receiving` averages 42.7 ms (`summary.json`). This one fact explains the shape of every curve.
4. **There are real correctness bugs.** The cache can go stale under concurrent writes (cache-aside race). DELETE and
   `/nocache/val` shadow `conn`, so a DB error leaks a pool connection (`server.cpp:192`, `:233`). Know them cold.
5. **The Postgres password is hard-coded** (`server.cpp:37`, also `:28-29`). It is also inside the committed
   binaries `project/server` and `project/server_debug`. The repo is public. See section 8 for what to say.

Two smaller flags:
- The **800 threads** value is not recorded anywhere. `http_threads` is read from stdin (`server.cpp:67-72`).
  The data is consistent with ~800 (see 5.3), but the repo cannot prove it.
- `results/final tests/results_mixed.csv`, 1000-VU row: TPS is `4332.79987830445`. Both backups have
  `4414.79987830445`. The 11 decimal digits match, so the integer part was edited by hand. It is not on your resume.
  Do not quote the mixed 1000-VU number. Fix the CSV or quote 4.4K at 500 VUs.

---

## 1. Pitches

### 60-second pitch

I built a key-value store that speaks HTTP, in C++, for my systems course.
A client sends GET, POST or DELETE with an integer key.
The server is cpp-httplib with a fixed pool of worker threads.
In front of PostgreSQL sits an in-memory LRU cache, guarded by a mutex.
Reads check the cache first and fall back to the database. Writes go to the database first, then the cache.
Database connections come from a small blocking pool that I wrote.
Then I load-tested it with k6, from 10 to 1000 virtual users, across five workloads.
I pinned Postgres to its own core so I could tell server CPU from database cost.
Cache-hit reads plateaued at about 11K requests per second, with the server core at 100%. That is CPU-bound.
Writes plateaued at about 3.1K per second, with the server core at only 75% and the disk mostly idle.
So writes are bound by the database tier, not by my server.

### 3-minute deep explanation

**The system.** There are three tiers: the load generator, the HTTP server, and PostgreSQL.
The server is one C++ file, `project/server.cpp`. At start-up it asks for the DB host, the pool size, the cache size and
the number of HTTP threads. It creates a table `kv_store(id INT PRIMARY KEY, value TEXT)`.

**The request path.** cpp-httplib has one thread that calls `accept()`. Each new connection becomes a job in a thread
pool. The worker thread then owns that connection for its whole keep-alive life: up to 100 requests, or 5 idle seconds.
So the pool size caps how many connections are served at once. That is why the pool is large: hundreds of
threads, set to about 800 for the runs.

A GET parses the id and checks the cache. On a hit it returns at once. On a miss it takes a DB connection from my
pool, runs a SELECT in a transaction, returns the connection, fills the cache, and replies.
A POST runs an UPSERT (`INSERT ... ON CONFLICT DO UPDATE`), commits, then writes the cache. That is write-through.
A DELETE deletes the row, then erases the key from the cache.

**The cache.** It is a classic O(1) LRU: a doubly linked list for recency, plus a hash map from key to list node.
A get splices the node to the front. A put on a full cache evicts the tail. One mutex guards it all.

**The pool.** It is a queue of `pqxx::connection*` with a mutex and a condition variable. `acquire()` blocks until a
connection is free. `release()` pushes it back and wakes one waiter.

**The experiments.** `script.sh` runs k6 at fixed VU counts for 40 s each. It records throughput and latency from k6,
and CPU and disk from vmstat and iostat. k6 is closed-loop with zero think time: each VU sends its next request as
soon as the last one returns.
- **get-popular** loops over 500 keys, so it is all cache hits. It shows the server's own ceiling.
- **get-only** reads uniform random keys from 20K, so most reads miss and go to Postgres.
- **put-only** writes uniform random keys. **mixed** is 70% GET, 20% POST, 10% DELETE.

**The finding.** Whole-machine CPU never passed 35%, because it was a 12-CPU WSL2 machine.
So I pinned Postgres to core 1 and read CPU per core.
get-popular hit 100% on the server's core and flattened at 11.1K TPS. Latency then grew in a straight line with VUs.
That is a saturated closed system, exactly as Little's law predicts.
put-only flattened at 3.1K TPS with the server core at 75%, so the limit moved into the database tier.

**What I learned.** The low-load latency was 43 ms even for cache hits. That turns out to be Nagle's algorithm meeting
delayed ACKs, because the library writes headers and body separately. Setting `TCP_NODELAY` would remove it.
I would also move to an event-driven server, use prepared statements, and test open-loop to see true tail latency.

---

## 2. Architecture

```
  k6 (same WSL2 box, closed loop, N VUs, 0 think time)
  load_gen/get_popular.js:19  get_only.js:14  put_only.js:13-15  mixed.js:16-36
        |  HTTP/1.1 keep-alive, localhost:1234
        v
  +----------------------------------------------------------------------------+
  | server process  project/server.cpp  (built: g++ server.cpp -lpqxx -lpq      |
  |                 -lpthread, no -O flag => -O0)                               |
  |                                                                            |
  |  main()  server.cpp:26                                                      |
  |   stdin config: db host :32-33, pool size :39-42, cache size :43-46,       |
  |                 http_threads :67-69                                         |
  |   conn_str with password=<redacted>  :36-37                                       |
  |                                                                            |
  |  accept thread: Server::listen_internal  httplib.h:8025                     |
  |     accept4() httplib.h:8046  (listen backlog 5, httplib.h:180)            |
  |     enqueue(process_and_close_socket)  httplib.h:8075                       |
  |                 |                                                          |
  |                 v                                                          |
  |  ThreadPool(http_threads)  httplib.h:870, set at server.cpp:70-72          |
  |   worker owns ONE connection: keep-alive loop httplib.h:3458               |
  |     (max 100 requests httplib.h:53, 5 s idle httplib.h:45,                  |
  |      polls select() every 10 ms httplib.h:3423)                            |
  |   process_request httplib.h:8359 -> routing -> handler lambda              |
  |                 |                                                          |
  |   +-------------+-------------+-----------------+                          |
  |   | GET /val :81 | POST /save :131 | DELETE /delete :174 | GET /nocache/val :215 |
  |   +-------------+-------------+-----------------+                          |
  |        |  cache.get :99                                                    |
  |        v                                                                    |
  |  KVCache (KVCache.h:12)  list + unordered_map, one std::mutex              |
  |     get :21 (splice to front :28)  put :34 (evict tail :38-42)  erase :48 |
  |        | miss                                                               |
  |        v                                                                    |
  |  LibpqxxPool (DBPool.h:12) queue<pqxx::connection*> + mutex + condvar      |
  |     acquire :31 (blocks, no timeout :33)   release :39                      |
  |        |  pqxx::work: BEGIN / SELECT|UPSERT|DELETE / COMMIT                 |
  +--------|-------------------------------------------------------------------+
           v  TCP (libpq)
  PostgreSQL, one backend process per pooled connection
     all postgres processes pinned to CPU 1 by project/dbpin.sh:2 (re-run every 1 s)
     table kv_store(id INT PRIMARY KEY, value TEXT NOT NULL)  server.cpp:52-55
     WAL on the WSL2 virtual disk (sdd in iostat)

  Measurement: load_gen/script.sh  vmstat 1 (:42), iostat -dx 1 (:45),
               k6 --summary-export (:51), jq (:67-72) -> results_<workload>.csv (:78)
               cpu column typed in by hand later; plotted by load_gen/cpu_plot.py:50-69
```

### One GET traced end to end (cache miss, then hit)

1. A k6 VU runs `get_only.js:14`. It picks `id` uniform in 1..KEYSPACE (20000 from `script.sh:9`).
   It sends `GET /val?id=1234` over its keep-alive connection.
2. If the connection is new, the accept thread takes it at `httplib.h:8046`. It wraps it in a job at `httplib.h:8075`.
3. A free pool worker (`httplib.h:870`) runs `process_and_close_socket` (`httplib.h:8562`). It enters the keep-alive
   loop (`httplib.h:3458`). `keep_alive()` waits in `select()` for bytes (`httplib.h:3423`).
4. `process_request` (`httplib.h:8359`) parses the request line, headers and query string. Routing calls the
   lambda at `server.cpp:81`.
5. The handler checks for `id` (`:83`) and parses it with `stoi` (`parse_id`, `:18`). A bad id gets 400 (`:91-95`).
6. `cache.get(1234, v)` (`:99`) locks the mutex (`KVCache.h:22`) and looks up the map (`:23`). A miss returns false.
7. `pool.acquire()` (`server.cpp:106`) locks the pool mutex. It waits on the condition variable until the queue is
   not empty (`DBPool.h:33`), then pops a connection.
8. `db::work txn{*conn}` (`server.cpp:107`) sends `BEGIN`. `exec_params` (`:109`) sends the SELECT with `$1`.
   `txn.commit()` (`:110`) sends `COMMIT`. That is three round trips to Postgres for one read.
9. `pool.release(conn)` (`:111`) pushes the connection back and wakes one waiter (`DBPool.h:39-43`).
10. If no row, reply 404 (`:113-116`). A 404 is **not** cached. If found, `cache.put` (`:119`) inserts at the
    front of the LRU list and evicts the tail if full (`KVCache.h:38-44`). Reply 200 (`:120`).
11. `write_response_core` (`httplib.h:7679`) writes the status line and headers (`:7728`). Then it writes the body
    in a second call (`:7735`). Nagle holds the body until the header segment is ACKed. The client delays that ACK
    by about 40 ms. This is the 43 ms floor.
12. The worker goes back to step 3 on the same connection. After 100 requests it sends `Connection: close` and
    the socket closes. k6 opens a new one, which waits in the job queue for a free worker.
13. The next GET for 1234 hits the cache at step 6. `splice` moves the node to the front (`KVCache.h:28`).
    The handler returns at `server.cpp:100-101` without touching Postgres.

---

## 3. Design decisions

Legend: **[doc]** = documented in a file or commit. **[code]** = reconstructed from the code; the reason was never
written down. Nothing here is invented history. If you remember the real reason, use yours.

| # | Decision | Alternatives | Chosen, and why | Cost | Source |
|---|---|---|---|---|---|
| 1 | HTTP library | Raw sockets + epoll, Boost.Beast, Drogon, Crow | cpp-httplib 0.27.0: one header, blocking API, easy routing | Thread-per-connection model; no epoll; default Nagle on; backlog 5 | [code] `httplib.h:11`, README lists it |
| 2 | Concurrency model | Event loop (epoll) + small worker pool; thread per request | Fixed worker pool, size from stdin (`server.cpp:67-72`) | A worker is pinned to a connection for its keep-alive life. Need ≥1 thread per concurrent connection, so hundreds of threads | [code] |
| 3 | Pool size ~800 | Default (`max(8, cores-1)` = 11 on 12 CPUs, `httplib.h:165`) | Larger than the VU count in most runs, so connections do not queue | 800 stacks (8 MB virtual each), context switches, 10 ms polling wake-ups per idle thread | [code]; value not recorded |
| 4 | Read path | Read-through library, no cache | Cache-aside: check cache, on miss read DB and fill (`:99-121`) | Race with writers can leave stale data (section 4, B1) | [code] |
| 5 | Write path | Write-back (cache first, flush later); write-around (invalidate) | Write-through: DB commit, then `cache.put` (`:158-162`) | Every write waits for a commit; two concurrent writes can reorder in the cache | [code] |
| 6 | Eviction | FIFO, LFU, CLOCK, ARC, TTL | Exact LRU: `std::list` + `unordered_map` (`KVCache.h:14-15`) | Every get mutates the list, so even reads need the exclusive lock | [code] |
| 7 | Cache locking | Sharded locks, `shared_mutex`, lock-free | One `std::mutex` (`KVCache.h:16`) | Serialises all cache ops; on one pinned core it matters less | [code] |
| 8 | Capacity unit | Bytes | Count of entries (`KVCache.h:13`) | Memory unbounded if values are large | [code] |
| 9 | Negative results | Cache "not found" | Not cached (`:113-116`) | Every GET of a missing key hits Postgres | [code] |
| 10 | DB connections | New connection per request; one per thread; pgBouncer | Own fixed pool, mutex + condvar (`DBPool.h`) | No timeout, no health check, no RAII guard, unfair wake-ups | [code] |
| 11 | Pool size | Equal to threads | Separate stdin value (`:39-42`), far below thread count | Threads queue on `acquire()` when the DB is slow | [code]; value not recorded |
| 12 | Transactions | `pqxx::nontransaction`; prepared statements | `pqxx::work` per query, `exec_params` (`:107-110`) | BEGIN + query + COMMIT = 3 round trips; parse/plan on every call | [code] |
| 13 | Write SQL | SELECT then INSERT/UPDATE | UPSERT `INSERT ... ON CONFLICT (id) DO UPDATE` (`:155-156`) | Atomic in one statement; none really | [code] |
| 14 | Schema | Key-value in a NoSQL store | `id INT PRIMARY KEY, value TEXT` (`:52-55`); B-tree on the PK | Integer keys only | [code] |
| 15 | Database | SQLite, Redis, RocksDB | PostgreSQL via libpqxx | Process per connection; WAL commit cost per write | README build line; course likely required it (not documented) |
| 16 | Config input | CLI flags, env vars, config file | Interactive `cin` prompts | Values used in runs were never recorded | [code] |
| 17 | Baseline endpoint | none | `/nocache/val` skips the cache (`:215`) | No k6 script uses it, so no cache-vs-no-cache number exists | [code] |
| 18 | Load model | Open loop (`constant-arrival-rate`) | Closed loop: fixed VUs, no sleep (`options.vus`) | Hides queueing delay (coordinated omission); TPS is capped by VUs / latency | [code] all `*.js` |
| 19 | Workload split | One mixed workload | get-popular (500 hot keys) vs get-only (uniform 20K) vs put-only vs mixed | Separates cache-bound from DB-bound behaviour | [code] `get_popular.js:13`, `script.sh:9` |
| 20 | Core pinning | cgroups / cpusets; separate machines | `taskset -cp 1` on every postgres pid, looped each second (`dbpin.sh`) | Puts all Postgres backends, WAL writer and checkpointer on one core. Needs sudo | [doc] `dbpin.sh` |
| 21 | CPU metric | `mpstat -P ALL`, `pidstat` | vmstat whole-machine idle (`script.sh:59-60`) + hand-read per-core value | Whole-machine number hides one hot core; hand values are coarse | [doc] `script.sh`; README "Use htop" |
| 22 | Disk metric | Per-device `%util` | Mean `%util` over all devices (`script.sh:63`) | Averages 4 devices, 3 idle, so it reads ~4x low | [doc] `script.sh:63` |
| 23 | Run length | Longer runs, repeats | 40 s per point, 3 s cool-down, one run per point | No error bars; no warm-up discard | [doc] `script.sh:8,81` |

---

## 4. Hardest problems, as STAR stories

There is no PROGRESS log and only one commit. These stories are **reconstructed from artifacts in the repo**.
Each one names its evidence. Fill in the details from your own memory. Do not add events you do not remember.

### Story 1: Making the bottleneck visible with core pinning (the resume story)

- **Situation.** Throughput flattened at 11K for cached reads and 3.1K for writes. But the whole-machine CPU from
  vmstat stayed at 25–35% (`cpu_util_pct`). On a 12-CPU WSL2 box, one full core is only 8%. So the average said
  "plenty of CPU left", which was wrong.
- **Task.** Find out which resource saturated, for each workload.
- **Action.** I pinned every Postgres process to CPU 1 with `taskset` (`dbpin.sh`). It loops every second because new
  backends can fork later. I watched per-core CPU and recorded the server core's value per run (`cpu` column, added
  to the CSVs). I wrote `cpu_plot.py` to print the core value on each TPS point. I kept the hot-key workload
  (`get_popular.js`) separate from the uniform one, so cache cost and DB cost do not mix.
- **Result.** Cached reads: core at 100% from 750 VUs, TPS flat at 11.1–11.2K. CPU-bound. Writes: core at 75% at
  3.1K TPS, and the disk under 15% busy. So writes are limited inside the database tier.
- **Evidence.** `dbpin.sh`; `cpu` column in `results/final tests/*.csv` but not in `backup1/`; `cpu_plot.py:50-69`.
- **Be careful.** No script in the repo pins the **server**. If you pinned it by hand (`taskset -c 0 ./server`), say so.
  Two facts support it: the `cpu` value never goes above 100, and vmstat's run queue was ~400–1000 while the machine was 80% idle.

### Story 2: The load generator was not sending what I thought

- **Situation.** Early k6 scripts did not match the server's API. The comments in the scripts still show the fixes.
  "FIX: Append ID to URL string" (`get_only.js:15`). "Your C++ server expects 'id' and 'val' as query parameters"
  (`mixed.js:27`). "Use http.del shortcut" (`mixed.js:34`).
- **Task.** Make sure every request was valid before trusting any throughput number.
- **Action.** I made debug twins of the scripts (`*_debug.js`). With `DEBUG=true` they log URL, status and body as JSON.
  I also built `server_debug`, which prints the method of each request (its strings include `GET `, `POST `,
  `DELETE `, `GET NC`; the plain `server` binary does not).
- **Result.** Requests hit the right routes with the right parameters. Then the real runs started.
- **What you can add.** Which bug you actually saw (for example 400s from a missing `id`). Only if you remember it.
- **Still open.** `get_only.js:16` assigns to `res` without `let`/`const`. In a k6 ES module that likely throws a
  ReferenceError after each request. The request still counts, but check your old k6 output for a flood of errors.

### Story 3: Pushing past 1000 VUs broke the server

- **Situation.** In `backup2/`, at 1250 VUs, put-only fell from 3.0K to 2.4K TPS. The last run in the repo used 1500 VUs.
  It gave 1.86K TPS, 140 requests failed by 60 s timeout, and 134 iterations were cut off (`k6_run.log`, `summary.json`).
  Max latency was 59 s. Max TCP connect time was 5.1 s.
- **Task.** Understand why throughput fell instead of staying flat.
- **Action (reconstructed).** The cause is the threading model. Each worker owns a connection for up to 100 requests.
  With ~800 workers and 1500 VUs, ~700 connections wait in the job queue. Some wait so long they time out.
  The 5 s connect times fit SYN retries when the listen backlog of 5 (`httplib.h:180`) overflows.
- **Result.** The final results stop at 1000 VUs, inside the range where the server degrades gracefully.
- **Honest note.** This explanation is my reading of the code and logs. It was not tested by changing the thread count.

### Story 4: Getting WSL2 to talk to PostgreSQL

- **Situation.** The server first had a fixed connection string to the Windows host (`server.cpp:27-29`:
  `host=Nani.mshome.net` and `host=172.23.32.1`). WSL2 sits behind a NAT, and that IP can change.
- **Action.** The notes at `server.cpp:2-3` show how the host IP was found (`/etc/resolv.conf`, `ip route`).
  The host became a prompt at start-up (`:31-37`).
- **Result.** By the final runs Postgres ran inside WSL2. `dbpin.sh` finds postgres processes with `pgrep`, and iostat
  shows WAL writes on the WSL disk during put-only.
- **Strength.** Weak as a "hardest bug". Use it only if asked about environment problems.

### Story 5: Results that move between runs

- **Situation.** There are three result sets: `backup1/`, `backup2/`, `final tests/`. get-only at 1000 VUs reads 3361 TPS in
  backup1 and 3593 in backup2. `final tests` uses the 3593 row.
- **Lesson.** Single runs at the saturation point vary by ~7%. Report a range, or repeat runs and give a mean and spread.
- **Do not** present `final tests` as one clean run. It is backup1 plus a re-run row plus the hand-typed `cpu` column.

---

## 5. Numbers sheet

**Test machine (from logs).** WSL2, kernel `6.6.87.2-microsoft-standard-WSL2`, host "Nani", **12 CPUs**, ~5 GB free RAM
(`disk.log` line 1, `cpu.log`). Compiler Debian GCC 14.3.0 (binary strings). Built with no `-O` flag, so `-O0`.
k6 ran on the same machine as the server and Postgres. Date of the last run: 2025-11-25 (`k6_run.log`).
Each point is one 40 s run. No warm-up was discarded. p50 and p99 are `null` in every CSV: the script asks k6 for
`p(50)` and `p(99)`, but k6's summary export only has `med`, `p(90)` and `p(95)` by default.

### 5.1 Resume numbers

| Resume number | Repo evidence | Supported? | Honest wording |
|---|---|---|---|
| **800-thread architecture** | `http_threads` read from stdin (`server.cpp:67-72`). The value is never saved. Only indirect evidence (5.3). | **Not provable from repo** | "configurable worker pool, run with ~800 threads" only if you remember 800. Explain it as "one thread per keep-alive connection", not as parallelism. |
| **thread-safe LRU cache** | `KVCache.h`: list + map, splice on hit, evict tail, one mutex. | **Yes** | True. But the cache plus DB together is not linearizable (B1). |
| **PostgreSQL persistence** | UPSERT and DELETE in transactions (`server.cpp:150-160`, `:191-197`). | **Yes** | True. Durability is whatever Postgres settings were; not recorded. |
| **20K-key workload** | `script.sh:9` `KEYSPACE=20000`, passed to k6 (`:51`). Scripts default to 10000 if not passed. get-popular uses 500 keys. | **Yes, with caveats** | "uniform random keys over a 20K keyspace". The DB may not have held all 20K keys: `client_small` fills only keys 0–2999 (30 threads × 100), and 404s are not cached. |
| **1K virtual users** | Final CSVs run 10, 50, 100, 175, 250, 500, 750, 1000 VUs. backup2 adds 1250. One run at 1500. | **Yes** | "10 to 1000 VUs, closed loop, four workloads" (plus a partial delete-only run to 250). |
| **CPU saturation at 11K TPS (cache-heavy reads)** | get-popular: 11,130 TPS at 750 VUs, 11,187 at 1000, `cpu` = 100 at both. 10,184 at 500 (`cpu` 80). | **Yes, if `cpu` is the server core** | "the pinned server core reached 100% and TPS flattened at ~11.1K". Don't say "the machine's CPU saturated": whole-machine CPU was 26%. |
| **I/O saturation at 3.1K TPS (write-heavy)** | put-only: 3,171 TPS at 750 VUs, 3,046 at 1000, `cpu` = 75. CSV disk util 2.1–2.4% (4-device mean). Real device `sdd` 6% mean, 13% max, in the 1500-VU log. | **Not supported** | "writes plateaued at ~3.1K TPS, bound in the database tier: the server core was at 75% and the disk under 15% busy". |
| **core pinning** | `dbpin.sh` pins Postgres to CPU 1. No script pins the server. | **Partly** | "pinned Postgres to a dedicated core and read per-core CPU". Add "pinned the server by hand" only if true. |

### 5.2 Full final results (`project/load_gen/results/final tests/`)

TPS / avg latency (ms) / p95 (ms) / `cpu` (hand-typed core %) / `cpu_util_pct` (whole machine).

| VUs | get-popular | get-only | put-only | mixed |
|--:|---|---|---|---|
| 10 | 225 / 43.9 / 49 / 4 / 9.3 | 226 / 43.0 / 48 / 6 / 9.5 | 225 / 44.4 / 51 / 5 / 11.1 | 228 / 43.7 / 48 / 8 / 9.9 |
| 100 | 2260 / 43.4 / 48 / 9 / 9.2 | 2213 / 42.7 / 47 / 43 / 17.2 | 1616 / 61.1 / 72 / 33 / 16.3 | 2193 / 45.1 / 52 / 30 / 16.7 |
| 250 | 5183 / 47.4 / 48 / 25 / 11.5 | 3562 / 65.0 / 95 / 93 / 32.0 | 2477 / 99.6 / 116 / 60 / 20.8 | 4096 / 60.4 / 89 / 90 / 31.7 |
| 500 | 10184 / 47.9 / 52 / 80 / 23.9 | 3641 / 130.8 / 153 / 97 / 32.0 | 2880 / 170.3 / 192 / 72 / 23.2 | 4386 / 112.2 / 157 / 95 / 34.7 |
| 750 | 11130 / 64.7 / 68 / 100 / 25.8 | 3620 / 197.2 / 960 / 98 / 31.9 | 3171 / 231.1 / 256 / 75 / 24.5 | 4382 / 168.3 / 173 / 97 / 33.7 |
| 1000 | 11187 / 84.8 / 76 / 100 / 26.4 | 3593 / 249.7 / 1213 / 98 / 31.7 | 3046 / 297.9 / 340 / 75 / 24.9 | (edited row, don't quote) |

### 5.3 Derived numbers you can defend

**Little's law holds in every row.** N = X × R. get-popular at 1000 VUs: 11,187 × 0.0848 s = 949 in flight.
put-only at 1000: 3,046 × 0.298 = 908. The gap to 1000 is k6's own time between requests.
At 10 VUs: 10 / 0.0434 s = 230 TPS, which matches 225. So at low load, TPS is just VUs / 43 ms.

**Server CPU per request (utilization law, S = U / X, if `cpu` is the server core).**
- get-popular at saturation: 1.00 / 11,150 ≈ **90 µs** per request.
- get-only: 0.98 / 3,620 ≈ **270 µs** per request (cache miss + libpqxx work + three round trips).
- put-only: 0.75 / 3,170 ≈ **240 µs** per request, and the core is not full.
- At 100–175 VUs, get-popular costs only ~30–50 µs per request. Cost per request roughly doubles as concurrency
  grows: more threads, more context switches (~60K/s in `cpu.log`), more idle polling. (The low-end `cpu` values are
  coarse integers, so treat this as a trend.)

**The knee matches the operational laws.** In a closed system the knee is at N* ≈ D / S_max, where D is the no-load
response time and S_max is the bottleneck's service time.
- get-popular: 0.043 s × 11,150/s ≈ **480 VUs**. The data bends between 500 and 750. ✓
- get-only: 0.043 × 3,620 ≈ **155 VUs**. Latency starts rising after 175. ✓
- put-only: 0.044 × 3,100 ≈ **136 VUs**. Latency rises gently from 50 VUs. The soft knee hints at a shared resource
  with batching (group commit) or contention, not one clean queue.

**The 43 ms floor is network, not compute.** In `summary.json` (the 1500-VU put run) `http_req_receiving` averages
42.7 ms. Its median is 43.2 ms. "Receiving" is the time from first byte to last byte. That gap is the body segment held
by Nagle until the client's delayed ACK. Waiting (time to first byte) is the real server time.

**Disk during writes (`disk.log`, 1500-VU put run, device `sdd`, 58 active seconds).** 122 writes/s, 1.0 MB/s,
61 flushes/s, 0.69 ms per flush, 6.3% mean util, 13.2% max. 1,857 commits/s over ~61 flushes/s means about
30 commits share each flush. That is Postgres group commit working. The disk had plenty of headroom.

**Thread pool of ~800: indirect evidence only.** At get-popular 750 VUs, avg (64.7) < p95 (67.7): no heavy tail.
At 1000 VUs, avg (84.8) > p95 (76.2): a few very slow requests. That fits a pool between 750 and 1000. Rough check:
200 extra connections wait for a worker. Connections turn over at about 11,187 / 100 ≈ 110 per second, so each new
connection waits ~2 s. About 1 request in 100 is the first on its connection. That adds ~20 ms to the mean, which
matches. This is consistent with 800. It is not proof.

### 5.4 Caveats to volunteer before they are found

- `cpu` column: typed in by hand, from htop (`README.md`, "Other tools"), no timestamps, integers.
- `cpu_util_pct`: whole-machine average. It also includes vmstat's first since-boot line and idle seconds after k6
  stops, so it reads a little low.
- `disk_util_pct`: mean over sda, sdb, sdc, sdd. Only sdd does work, so it reads ~4x low.
- TPS counts every response, including 404s and 500s. The CSV has no error rate. In get-only and mixed, some keys
  were missing, so some reads were 404s.
- k6, the server and Postgres all shared one 12-CPU machine. k6 used CPU too.
- One run per point. Re-runs varied ~7% at saturation (get-only 1000 VUs: 3361 vs 3593).
- Cache size, pool size and thread count were not recorded.
- `-O0` build. Absolute numbers would rise with `-O2`.

---

## 6. Prerequisite concepts

**Thread-per-connection vs event loop.** A blocking server gives each connection a thread. It is simple, but threads
cost memory and context switches, and idle keep-alive connections hold threads. An event loop (epoll) lets a few
threads watch thousands of sockets and only do work when bytes arrive. cpp-httplib is the first kind.
Sources: Dan Kegel, "The C10K problem"; Welsh et al., "SEDA" (SOSP 2001).

**LRU cache in O(1).** A hash map gives O(1) lookup from key to node. A doubly linked list keeps recency order.
A hit splices the node to the head. An insert into a full cache drops the tail. LRU fails on cyclic scans larger than the
cache: every access misses. Sources: Megiddo & Modha, "ARC" (FAST 2003), for LRU's weak spots; Redis docs on approximate LRU.

**Cache-aside vs write-through vs write-back.** Cache-aside: the app reads the cache, and on a miss reads the DB and
fills the cache. Write-through: every write goes to the DB and the cache at once. Write-back: writes go to the cache and
flush later, which is fast but loses data on a crash. Cache-aside has a known stale-set race. Facebook solves it with
leases. Source: Nishtala et al., "Scaling Memcache at Facebook" (NSDI 2013), section 3.2.1.

**Connection pooling.** Opening a Postgres connection forks a backend and authenticates, which takes milliseconds.
A pool keeps N open and lends them out. Too few, and threads queue. Too many, and the DB thrashes on its cores.
A common rule is about (cores × 2) + spindles. Source: HikariCP wiki, "About Pool Sizing"; PostgreSQL docs, "Connections and Authentication".

**Little's law.** In any stable system, the average number inside (L) equals arrival rate (λ) times average time
inside (W). It needs no assumptions about distributions. For load tests, in-flight requests = TPS × latency.
Source: J. D. C. Little, "A Proof for the Queuing Formula L = λW" (1961); Lazowska et al., *Quantitative System Performance*, ch. 3.

**Utilization law and bottleneck analysis.** U = X × S: a resource's busy fraction is throughput times service time
per request. The resource with the largest S saturates first and caps X at 1/S. Before that knee, latency stays flat.
After it, latency grows in proportion to N in a closed system. Source: Jain, *The Art of Computer Systems Performance
Analysis*, ch. 33; Lazowska et al., ch. 3 and 5.

**Closed-loop vs open-loop load.** Closed: N users, each sends the next request only after the last returns. The system
slows the load as it slows down, so throughput flattens and latency hides queueing. Open: requests arrive at a set rate
regardless. That shows real queue growth and tail latency. Sources: Schroeder, Wierman & Harchol-Balter, "Open Versus
Closed: A Cautionary Tale" (NSDI 2006); Gil Tene, "How NOT to Measure Latency" (coordinated omission).

**Nagle's algorithm and delayed ACK.** Nagle holds a small segment while an earlier one is still unACKed. Delayed ACK
waits up to ~40 ms (Linux) before sending an ACK alone. A server that writes headers and body separately meets both and
stalls ~40 ms per response. `TCP_NODELAY` turns Nagle off. Sources: RFC 896 (Nagle); RFC 1122 §4.2.3.2 (delayed ACK).

**WAL, fsync and group commit.** Postgres writes changes to the write-ahead log. A commit returns only after its WAL is
flushed with fsync. Many commits waiting at once share one flush: group commit. `synchronous_commit=off` skips the wait,
at the risk of losing the last few commits on a crash. Sources: PostgreSQL docs, "WAL Configuration" and "Asynchronous
Commit"; Mohan et al., "ARIES" (TODS 1992).

**CPU affinity.** `sched_setaffinity` (the `taskset` tool) limits which cores a thread may run on. `taskset -c 0 cmd`
starts a process whose threads all inherit the mask. `taskset -p` on a running pid changes only that thread, unless
you add `-a`. Pinning removes noise and makes per-core utilization meaningful. Source: `man 1 taskset`, `man 2 sched_setaffinity`.

**Tail latency.** At high fan-out or under load, the slow 1% dominates user experience. Mean hides it. Average above
p95 is a sign of a few huge outliers. Source: Dean & Barroso, "The Tail at Scale" (CACM 2013).

**HTTP/1.1 keep-alive.** One TCP connection carries many requests in sequence. It saves handshakes. But in a blocking
server it also pins a worker thread while the connection is idle. Source: RFC 9112 §9.

---

## 7. Fifty likely questions, with crisp answers

### Architecture and design

1. **Walk me through a GET.** Parse id, check LRU under a mutex. Hit: reply. Miss: borrow a pooled connection, run the
   SELECT in a transaction, return the connection, fill the cache, reply. `server.cpp:81-128`.
2. **Why cpp-httplib?** One header, no build system, blocking code that is easy to reason about. For a course project
   that was the fastest path to a working server. The cost is its thread-per-connection model.
3. **Why 800 threads? You only have 12 cores.** The threads are not for parallelism. In cpp-httplib a worker owns a
   keep-alive connection until it closes. To serve ~1000 concurrent VUs without queueing connections, I need about
   that many workers. Most of them sit blocked in `select()` or on the DB pool.
4. **Hostile: isn't 800 threads just bad design?** For this library, it is the price of the model. The better fix is
   an event loop: a few threads on epoll, with DB calls handed to a small worker pool. I'd do that next.
5. **Why an LRU and not LFU or CLOCK?** LRU is simple, O(1), and fits recency-heavy workloads. CLOCK would avoid
   moving nodes on every hit, so reads could share a lock. LFU suits stable hot sets but adapts slowly.
6. **Is your LRU really O(1)?** Yes. `unordered_map::find` is O(1) on average. `list::splice`, `pop_back` and
   `emplace_front` are O(1). `KVCache.h:21-46`.
7. **Why one mutex for the cache?** Simplicity. Every `get` changes the list, so a reader-writer lock does not help.
   To scale: shard by key hash into, say, 16 LRUs with 16 locks. Or use CLOCK, where a hit only sets a bit.
8. **Why write-through and not write-back?** The DB is the source of truth. Write-back would ack writes that a crash
   could lose. Write-through keeps the cache warm for read-after-write.
9. **Why update the cache after the DB, not before?** If the DB write fails, the cache never holds a value the DB
   lacks. The order is DB commit (`:159`), then `cache.put` (`:162`).
10. **Hostile: is the cache consistent with the database?** Not under concurrency. A GET can read an old row, a POST
    then commits and caches a new value, and then the GET caches the old value over it. Two POSTs can also land in
    the cache in the opposite order from the DB. A fix: on writes, delete from the cache instead of putting. Add
    versions or leases so a stale fill cannot overwrite a newer one, as Facebook does with memcache leases.
11. **Why UPSERT?** One atomic statement covers insert and update. No race between a SELECT and an INSERT.
12. **Why your own pool, not pgBouncer?** It is small and in-process. A mutex, a condition variable and a queue
    (`DBPool.h`). pgBouncer adds a network hop, but handles many more clients and health checks.
13. **What happens if the pool is empty?** `acquire()` blocks on the condition variable with no timeout
    (`DBPool.h:33`). The HTTP thread waits. Under heavy load threads pile up there.
14. **How did you size the DB pool?** It was a start-up input. I did not record the value used. In principle it should
    be close to Postgres's useful parallelism, here one pinned core, not to the thread count.
15. **Why a transaction for a single SELECT?** It was the default libpqxx pattern. It costs BEGIN and COMMIT round
    trips. `pqxx::nontransaction` or a prepared statement would cut the read to one round trip.
16. **Why not cache 404s?** Not done. It means missing keys always hit the DB. A short-lived negative entry fixes
    that, but DELETE and POST must then clear it.
17. **What is `/nocache/val` for?** A baseline that skips the cache. I never ran a k6 script against it, so I have no
    cache-vs-no-cache number. get-popular vs get-only is the comparison I have.

### Measurement and results

18. **How did you measure throughput and latency?** k6's summary export: `http_reqs.rate` and `http_req_duration`.
    `script.sh` runs each VU level for 40 s and appends one CSV row.
19. **How did you measure CPU?** vmstat gives whole-machine idle. Because Postgres was pinned to one core and the
    machine had 12, I also read the server's core from htop and added it to the CSVs as `cpu`.
20. **Hostile: your CPU column says 26%. How is that saturation?** 26% is the average over 12 CPUs. One full core is
    8%. The server's core alone read 100% while TPS stopped growing. That is saturation of that core.
21. **Hostile: how do I know the `cpu` column is real?** It was read by eye from htop, not logged by a script. I'd
    log it with `mpstat -P ALL 1` or `pidstat -t` next time. It is consistent with the rest: it rises in step with TPS,
    and stops at 100 exactly where TPS flattens.
22. **Why does throughput plateau while latency keeps rising?** In a closed loop, X = N / R. Once a resource is
    saturated, X is capped at 1/S. Adding users only adds queueing, so R = N / X_max grows in a straight line.
    get-popular: 1000 / 11,187 = 89 ms predicted, 85 ms measured.
23. **Explain Little's law with your data.** In-flight = TPS × latency. At 1000 VUs on put-only, 3,046 × 0.298 s ≈ 908
    requests in flight, close to the 1000 VUs.
24. **Why is latency 43 ms even at 10 users?** Nagle's algorithm plus delayed ACK. The library writes headers and body
    separately with `TCP_NODELAY` off. k6's "receiving" phase is ~43 ms. Setting `svr.set_tcp_nodelay(true)` fixes it.
25. **Hostile: so your low-load numbers are an artifact?** Yes, the latency floor is. The saturated throughput is not,
    because at saturation the server core is busy and the floor only sets where the knee is.
26. **What limits writes at 3.1K?** Not the server: its core is at 75%. Not disk bandwidth: under 15% busy. It is
    the database tier. Likely candidates: Postgres's single pinned core running all backends, the DB pool size, and
    per-commit WAL flush waits. I did not measure the Postgres core, so I can't say which one.
27. **Hostile: your resume says I/O saturation.** That wording is too strong. The disk was mostly idle. The correct
    claim is "database-bound". If I re-ran it, I'd read the Postgres core and `pg_stat_activity` wait events.
28. **Why are cache-miss reads (3.6K) so close to writes (3.1K)?** Both pay the DB round-trip path. Reads miss the
    cache most of the time because the key space (20K) is much larger than the cache, and absent keys are never cached.
    get-only seems bound by the server core (98%), at ~270 µs of server CPU per miss.
29. **Why is mixed (4.4K) faster than get-only (3.6K)?** My guess: POSTs fill the cache (write-through), so later GETs
    hit more often, and deletes of absent keys are cheap. I did not measure the hit rate, so this is a hypothesis.
30. **What was the cache hit rate?** Not measured. get-popular is near 100% after the first pass over 500 keys.
    get-only is about cache size / 20K for keys that exist. I'd add hit/miss counters.
31. **What is p99?** Not recorded. k6's summary export has no p99 by default, so that column is null. I'd set
    `summaryTrendStats` to include `p(99)`.
32. **Why does get-only have a p95 of 1.2 s at 1000 VUs?** A heavy tail. Likely threads waiting on the DB pool. A
    `condition_variable` does not wake waiters in FIFO order, so some wait much longer. Plus connections queued for a
    worker. Not measured directly.
33. **What happened at 1500 VUs?** Throughput fell to 1.86K, and 140 requests hit k6's 60 s timeout. More connections
    than workers queue for minutes. Also, a listen backlog of 5 causes SYN retries (5 s connect times).
34. **Was k6 the bottleneck?** Unlikely but not ruled out. It ran on the same 12-CPU machine. The machine stayed
    65–75% idle at peak, so k6 had room. The clean fix is to run k6 on another machine or other pinned cores.
35. **How repeatable are the numbers?** One run per point. Re-runs at 1000 VUs differed by about 7%. I'd do 3–5 runs
    and report a mean and spread.
36. **Why closed-loop?** k6's default and the course setup. It is fine for capacity. For latency under a given
    arrival rate, I'd use `constant-arrival-rate`, which avoids coordinated omission.
37. **Why get-popular with 500 keys?** To make every read a cache hit, so I measure the server path alone. Comparing
    it with get-only shows what the database costs.
38. **What does core pinning buy you?** It keeps Postgres and the server off each other's cores. Then per-core
    utilization tells you which tier is saturated. Without it the scheduler spreads load and averages hide the hot spot.
39. **Why does `dbpin.sh` loop every second?** Postgres forks new processes over time (autovacuum workers, new
    backends). A loop catches them. It needs sudo because those processes belong to the `postgres` user.

### Failure modes and bugs

40. **What if Postgres restarts?** Pooled connections break. Each request on a broken one returns 500, and the
    connection goes back to the pool still broken. No reconnect or health check exists.
41. **Any resource leaks?** Yes. In DELETE and `/nocache/val`, `conn` is declared twice (`server.cpp:190/192`,
    `:231/233`). On a DB exception the catch block sees the outer `nullptr` and never releases. After pool-size errors,
    every DB request blocks forever. The fix is an RAII guard that releases in its destructor.
42. **What if cache size is 0?** `put` sees size 0 ≥ 0 and calls `items.back()` on an empty list
    (`KVCache.h:38-39`). That is undefined behaviour, most likely a crash.
43. **What does `?id=12abc` do?** `stoi` reads 12 and ignores the rest, so it works as key 12. `-1` is rejected
    because it is the error sentinel, but `-5` is accepted. Out-of-range values return 400.
44. **Security problems?** Hard-coded DB password and superuser login, no auth on the API, no TLS, values in query
    strings with no size limit, and error messages that echo database errors to the client.
45. **What breaks at 10x load?** Threads: 8,000 threads won't fit well. Cache: one mutex becomes a hot lock on many
    cores. DB: one Postgres core and one pool. I'd go event-driven, shard the cache, add read replicas or shard the DB.

### What next and what differently

46. **What would you do differently?** Turn on `TCP_NODELAY`. Build with `-O2`. Use an epoll server. Use prepared
    statements and `nontransaction` reads. An RAII pool guard. A sharded cache. Log per-core CPU with a script.
    Record every config value in the CSV. Repeat runs. Add an open-loop test with p99.
47. **How would you scale it out?** Make the HTTP tier stateless, then add nodes behind a load balancer. The cache can
    then be a shared Redis or memcached tier, or a local cache with invalidation. Shard the DB by key range or hash.
48. **How would you make writes faster?** Batch commits, or set `synchronous_commit=off` if losing the last few
    writes is acceptable. Give Postgres more than one core. Use prepared statements. Size the pool to the DB's cores.
49. **How would you test the cache race?** Many threads write and read the same key. After writes stop, compare the
    cached value with the DB row. Any difference is a stale entry. A stress test with random sleeps makes it show up fast.
50. **What's the link to your Raft project?** This server is a single node. Raft is the next step: replicate the
    key-value log across nodes so a crash does not lose availability.

---

## 8. Questions where the honest answer is "it didn't work" or "I didn't do that"

Say it plainly, give the reason, then say what you would do. Do not apologise at length.

- **"Was the disk saturated?"** — "No. I wrote I/O saturation, but the disk was under 15% busy. The write limit is in
  the database tier. I didn't measure the Postgres core, so I can't name the exact cause. I'd measure it next."
- **"Did you pin the server too?"** — If yes: "Yes, by hand with taskset; it isn't in the scripts." If you don't
  remember: "Postgres was pinned by a script. I don't have a record of how the server was pinned."
- **"What was the cache hit rate / pool size / cache size?"** — "I didn't record them. The server read them from
  stdin. That was a mistake. Now I'd write every config value into the results file."
- **"Did you compare with and without the cache?"** — "I had a no-cache endpoint but didn't load-test it. The
  get-popular vs get-only gap (11K vs 3.6K) is the closest comparison, though the two also differ in key pattern."
- **"p99?"** — "Not captured. k6's export left it null. I have p90 and p95."
- **"Why 43 ms at low load?"** — "I didn't know at the time. Looking back, it's Nagle plus delayed ACK. One
  `set_tcp_nodelay(true)` call would fix it."
- **"Is the cache consistent?"** — "Not under concurrent writes to the same key. Here is the race, and here is the fix."
- **"Did you test failures?"** — "No. And there's a connection leak on the DELETE error path that a failure test
  would have caught."
- **"How many times did you run each point?"** — "Once, and some points twice. Re-runs varied about 7%."
- **"The password in your public repo?"** — "That is a lab credential for a local Postgres in my WSL VM. It was never
  exposed to a network. Still, it should not be in source control, and it is also in the compiled binaries I committed.
  The fix is to read it from an environment variable or `~/.pgpass`, use a non-superuser role, rotate the password,
  and remove it from git history with `git filter-repo`." Only say "I've rotated and scrubbed it" if you have.
  (Your projects vault already has this as an open thread.)
- **"The mixed workload's 1000-VU number?"** — If it comes up: "That row in the final CSV was edited by hand; the two
  raw runs say 4.4K. I'd quote the raw value." Better: fix the CSV before interviews.

---

## 9. Self-test exercises

Write your prediction first. Then read the answer.

### Exercise 1: turn on TCP_NODELAY
Add `srv.set_tcp_nodelay(true);` after `Server srv;` (`server.cpp:66`). Re-run get-popular at 10, 100 and 1000 VUs.
Predict latency at 10 VUs, TPS at 10 VUs, and peak TPS.

<details><summary>Answer</summary>

Latency at 10 VUs drops from ~43 ms to about 1 ms or less, because the body no longer waits for a delayed ACK.
TPS at 10 VUs jumps from ~225 to several thousand, since a closed loop gives TPS ≈ VUs / latency.
Peak TPS barely moves at first, because the server core is still the bottleneck at 100%. It may rise a little,
because idle threads spend less time polling. The knee moves to far fewer VUs: N* = D / S drops with D.
With 1000 VUs, latency is still ~N / X_max ≈ 90 ms, because saturated queueing dominates.
</details>

### Exercise 2: shrink the HTTP pool
Start the server with `http_threads = 100`. Run get-popular at 1000 VUs with today's code (Nagle on).
Predict TPS and the tail.

<details><summary>Answer</summary>

Only 100 connections are served at once. Each request still takes ~43 ms, so the ceiling is about
100 / 0.043 ≈ 2.3K TPS, far below 11K. The other 900 connections wait in the job queue. A connection is held for
100 requests (≈ 4.3 s). So ~23 connections free up per second, and a new one waits ~900 / 23 ≈ 40 s.
Expect a huge tail and some 60 s k6 timeouts. This is the same effect as the 1500-VU run, only larger.
</details>

### Exercise 3: cache size 0
Start the server with cache size 0. Send one POST, then one GET.

<details><summary>Answer</summary>

The POST commits to Postgres, then calls `cache.put`. In `KVCache.h:38` the check `items.size() >= capacity`
is `0 >= 0`, so it is true. Line 39 calls `items.back()` on an empty list. That is undefined behaviour, most likely
a segfault, and the whole server dies after the DB write already succeeded. Fix: return early if `capacity == 0`.
</details>

### Exercise 4: kill Postgres during deletes
Set pool size 4. Run delete-only. Stop Postgres for 5 seconds, then start it again. Then send GETs for keys
that are cached and keys that are not.

<details><summary>Answer</summary>

Each DELETE that throws inside `try` leaks its connection. `server.cpp:192` declares a new `conn` that shadows the
one at `:190`, so the catch at `:207` sees `nullptr` and never releases. After 4 failures the pool is empty.
Every later request that needs the DB blocks forever in `acquire()` (`DBPool.h:33`), even after Postgres is back.
Cached GETs still work, because they return before touching the pool. Uncached GETs, POSTs and DELETEs hang.
The HTTP workers fill up with blocked threads, so soon even cache hits wait. Fix: remove the inner declaration, or
better, an RAII guard. Also note: even without the leak, broken connections return to the pool and keep failing.
</details>

### Exercise 5: cache smaller than the hot set
Set cache size 400. Run get-popular (keys 1–500 in a loop). Predict the hit rate in two cases:
(a) all VUs walk the keys in lockstep; (b) VUs are at random points in the loop.

<details><summary>Answer</summary>

(a) Lockstep: between two uses of key k, all 499 other keys are used. That is more than 400, so k has already been
evicted. LRU on a cyclic scan larger than the cache gets almost 0% hits. TPS falls toward the DB-bound level (~3.6K).
(b) Random phases: with 1000 VUs spread over the loop, key k is touched about twice per 500 requests. Only ~250
distinct keys come between uses, which is less than 400, so most accesses hit. In practice VUs start together and
drift apart, so the real result falls in between. This is why reuse distance, not just working-set size,
decides LRU hit rates. CLOCK has the same weakness; ARC or a scan-resistant policy handles (a) better.
</details>

---

## Appendix: quick file map

| File | What it is |
|---|---|
| `project/server.cpp` | Whole server: config prompts, routes, handlers |
| `project/KVCache.h` | LRU cache (README calls it `kvcache.h`; `server.cpp:10-11` includes lower-case `dbpool.h`/`kvcache.h`, which only works on a case-insensitive filesystem) |
| `project/DBPool.h` | Blocking libpqxx connection pool |
| `project/dbpin.sh` | Pins all postgres processes to CPU 1, every second (uses sudo) |
| `project/client.cpp`, `client_small.cpp` | Hand-written C++ test clients; `client_small` option 6 fills keys 0–2999 |
| `project/httplib.h` | cpp-httplib 0.27.0, vendored |
| `project/server`, `server_debug` | Committed binaries (contain the password string) |
| `project/load_gen/script.sh` | Sweep driver: k6 + vmstat + iostat -> CSV. Currently set to put-only, 1500 VUs only |
| `project/load_gen/*.js` | k6 workloads; `*_debug.js` log each response |
| `project/load_gen/cpu_plot.py`, `csv_plot.py` | Plots TPS and latency; `cpu_plot.py` annotates the `cpu` column |
| `project/load_gen/{cpu,disk,k6_run}.log`, `summary.json` | Logs of the last run only (put-only, 1500 VUs) |
| `project/load_gen/results/final tests/` | The results you quote |
| `project/load_gen/results/backup1/`, `backup2/` | Earlier runs; backup2 has 1250 VUs |
