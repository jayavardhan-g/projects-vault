# Project report: High-performance HTTP key-value server (CS744, Design and Engineering of Computing Systems)

> **For the LLM reading this.** This report describes a course project by G Jayavardhan (M.Tech CSE, IIT Bombay) for
> **CS744: Design and Engineering of Computing Systems** (instructor Prof. Mythili Vutukuru, Autumn 2025). It appears
> on resume 5, and he must defend it in interviews.
>
> It was written from `/data/jayavardhan/projects/decs`:
> - one commit, `82de05e`, 2026-08-18;
> - `project/server.cpp`, `KVCache.h`, `DBPool.h`, clients, `load_gen/` k6 scripts and result CSVs.
>
> Labels:
> - **[CODE]**: `file:line`;
> - **[MEASURED]**: from the result CSVs in `load_gen/results/final tests/`;
> - **[RECON]**: the report writer's reasoning or findings from reading the code.

---

## 1. Summary

A multi-tier key-value store with an HTTP interface.

```
k6 load generator (closed loop, N virtual users)
        │ HTTP/1.1
        ▼
C++ server (cpp-httplib 0.27.0), thread pool of N http_threads (e.g. 800)
        │                         │
   in-memory LRU cache       libpqxx connection pool (M connections)
   (KVCache: list+hashmap,        │
    one mutex)                    ▼
                           PostgreSQL table kv_store(id INT PK, value TEXT)
```

Routes:

| route | behaviour |
|---|---|
| `GET /val?id=` | cache first; on a miss, SELECT from PostgreSQL, then fill the cache |
| `POST /save?id=&val=` | UPSERT into PostgreSQL, then update the cache (write-through) |
| `DELETE /delete?id=` | DELETE from the database, then erase from the cache |
| `GET /nocache/val` | bypass the cache |

The server reads its parameters from stdin at start-up: DB host, pool size, cache size, HTTP threads.

**Load testing:**
- k6 scripts for five workloads: `get_only`, `get_popular`, `put_only`, `delete_only`, `mixed` (70/20/10 GET/POST/DELETE).
- Keyspace 20,000. Virtual users 10 → 1,000. 40 s per point.
- `vmstat` and `iostat` sample CPU and disk while the test runs.
- `dbpin.sh` pins all PostgreSQL processes to CPU core 1 with `taskset`, isolating the database's CPU from the server's.

**Findings [MEASURED]:**
- Cache-friendly reads (`get_popular`: each VU cycles over the same 500 hot keys) plateau at about **11.1–11.2k TPS** at 750–1,000 VUs, with the monitored core at **100%**. That is CPU saturation.
- Writes (`put_only`) plateau at about **3.0–3.2k TPS** with that core at about 75% and disk utilisation 2–3%.
- Random reads over 20k keys (`get_only`) plateau at about **3.6k TPS** with the core at 98%.

**Stack:** C++, cpp-httplib, PostgreSQL, libpqxx, k6, Linux/WSL2.

---

## 2. Resume bullets vs evidence (resume 5)

| Bullet | Verdict | Evidence |
|---|---|---|
| "Engineered a high-throughput, multi-tier concurrent C++ HTTP key-value store featuring an 800-thread architecture, thread-safe LRU cache, and PostgreSQL persistence across a 20K-key workload" | **Supported.** | `server.cpp:66-72`: `srv.new_task_queue = [] { return new ThreadPool(http_thread); }`, where the thread count is read at start-up (800 is the configured value; the code has no hard-coded 800). `KVCache.h` is a true O(1) LRU (list + unordered_map) behind one `std::mutex`. `KEYSPACE=20000` in `script.sh`. |
| "Stress tested for different kinds of loads across 1K virtual users while isolating CPU and I/O bottlenecks" | **Supported.** | VU sweep 10 → 1,000 over 5 workloads; vmstat and iostat sampling; `dbpin.sh` core pinning. |
| "Isolated server and database resources via CPU core pinning, identifying CPU saturation at 11K TPS for cache-heavy reads versus I/O saturation at 3.1K TPS for write-heavy workloads" | **11K CPU claim: supported. "I/O saturation": partly.** | `get_popular`: 11,130 / 11,187 TPS at 750 / 1,000 VUs, monitored core 100%. Whole-machine vmstat CPU is only about 26%, because it averages all cores and the work is pinned. `put_only`: 3,171 / 3,046 TPS, monitored core 75%, **iostat disk utilisation only about 2–3%**. So the write ceiling is **not CPU and not disk bandwidth**. It is most plausibly **per-commit WAL flush (fsync) latency and row-lock contention** in PostgreSQL. Calling it "I/O-bound (commit/fsync latency)" is defensible; "disk saturated" is not. Note also that `dbpin.sh` uses `sudo`, so it was run where he had root (WSL2), not on the shared server. |

---

## 3. Background (teach first)

### 3.1 Server concurrency models
- **Thread-per-request / thread pool (this project):** a fixed pool of worker threads. Each accepted connection is handled by one worker, which blocks on I/O (here, the database).
  - Simple.
  - Threads cost memory (stack) and context switches.
  - Concurrency is capped by the pool size.
- **Event loop (epoll, nginx, Node):** a few threads multiplex many sockets with non-blocking I/O. Efficient, but blocking calls (a synchronous DB driver) stall the loop, so you need async DB drivers or offload threads.
- **Why 800 threads [RECON]:** with a synchronous libpqxx call per request, a thread is stuck while it waits for the database. Many threads keep the CPU busy while others wait (overlap I/O). Past the saturation point, more threads only add queueing and context switches.

### 3.2 Caching
- **LRU:** a doubly linked list ordered by recency plus a hash map key → list iterator.
  - `get` moves the item to the front (O(1) `splice`);
  - `put` inserts at the front and evicts the tail when full.
- **Write-through:** write the database, then update the cache. Here POST does this. GET-miss does cache-aside fill.
- **Consistency races** between concurrent readers and writers (§6.2).

### 3.3 Connection pools
Opening a PostgreSQL connection costs a TCP handshake, authentication and a backend process. A pool keeps M open connections. Requests borrow one and return it. If none is free, they wait on a condition variable. M bounds database concurrency.

### 3.4 Closed-loop load and Little's law
- **k6 runs a closed loop:** each virtual user sends one request, waits for the response, then sends the next.
- So **throughput X = N / (R + Z)** (N users, response time R, think time Z = 0 here). That is **Little's law**: N = X·R.
- Before saturation, X rises linearly with N. After saturation, X flattens and R grows linearly with N (pure queueing).
- **Utilisation law:** U = X·S (S = service time per request at the bottleneck). The bottleneck resource reaches U ≈ 100% at the plateau.

### 3.5 PostgreSQL write path
- Each committed transaction writes WAL (write-ahead log) records and, with `synchronous_commit=on`, **fsyncs the WAL** before acknowledging.
- With many concurrent small writes, throughput is limited by fsync latency (partly hidden by group commit) and by locking on the same rows (UPSERT on a random key in a 20k keyspace).
- A disk can be latency-bound while its %util looks low, especially on a virtualised disk (WSL2).

---

## 4. Code walk

### 4.1 `KVCache.h` (57 lines)
- `std::list<std::pair<int,std::string>> items` (front = most recent) and `std::unordered_map<int, list::iterator> index`. One `std::mutex mtx`.
- `get(key, &value)` (`:21-32`): lock; find; `items.splice(items.begin(), items, it->second)` moves the node to the front **without reallocating**, so the iterator stays valid; copy the value out.
- `put` (`:34-46`): lock. If the key exists, erase its node. Else, if full, evict `items.back()` and erase its map entry. Then `emplace_front` and store the new iterator.
- `erase` (`:48-55`).
- **Every operation, including `get`, takes the exclusive lock**, because `get` mutates the list. So cache reads are serialised. [RECON] At 11k TPS this global mutex is a likely contributor to CPU saturation. Alternatives:
  - sharding (N independent LRUs by key hash);
  - CLOCK / approximate LRU with read-mostly locking;
  - per-shard reader-writer locks.

### 4.2 `DBPool.h` (53 lines): `LibpqxxPool`
- Holds a `std::queue<pqxx::connection*>`, a mutex and a condition variable. The constructor opens `size` connections and exits on failure.
- `acquire()`: `cv.wait(lock, [&]{ return !pool.empty(); })`, then pop. This blocks without a timeout.
- `release(conn)`: push, unlock, `cv.notify_one()`.
- The destructor deletes all connections.

### 4.3 `server.cpp` (258 lines)
- Reads the DB host, pool size, cache size and HTTP thread count from `cin`. Builds the connection string (`dbname=decs user=postgres password=kali host=…`). **The password is hard-coded in source** [RECON: should come from env or config].
- Creates `kv_store(id INT PRIMARY KEY, value TEXT NOT NULL)` if missing.
- `Server srv; srv.new_task_queue = [] { return new httplib::ThreadPool(http_thread); }`. The pool size is configurable; the resume's 800.
- **GET `/val`** (`:81-128`): validate the id (`stoi` with try/catch; −1 means invalid) → `cache.get` → hit: return. Miss: acquire a connection → `pqxx::work` → `exec_params("SELECT value FROM kv_store WHERE id=$1")` → commit → release → 404 or `cache.put` + return.
- **POST `/save`** (`:131-171`): UPSERT `INSERT … ON CONFLICT (id) DO UPDATE SET value=$2` → commit → release → `cache.put`.
- **DELETE `/delete`** (`:174-212`): DELETE → if no rows were affected, 404; else `cache.erase`.
- **GET `/nocache/val`** (`:215-254`): a DB read with no cache, for comparison.
- Listens on `0.0.0.0:1234`.
- Prepared/parameterised statements (`exec_params`) prevent SQL injection.

### 4.4 Load generation (`load_gen/`)
- `script.sh`: for each workload and each VU count, it
  1. starts `vmstat 1` and `iostat -dx 1`;
  2. runs `k6 run --summary-export=summary.json` with VUS, BASE_URL, KEYSPACE and DURATION;
  3. kills the monitors;
  4. computes the average CPU utilisation (100 − idle, vmstat column 15) and disk %util;
  5. reads throughput (`http_reqs.rate`) and latency percentiles from `summary.json` with `jq`;
  6. appends a CSV row.
- Workloads:
  - `get_only.js`: a uniformly random key in 1..20,000 (many cache misses → DB-bound);
  - **`get_popular.js`**: `key = (__ITER % 500) + 1`, so each VU cycles over the same 500 hot keys (cache-heavy);
  - `put_only.js`: random-key POSTs;
  - `delete_only.js`;
  - `mixed.js`: 70% GET, 20% POST, 10% DELETE.
- `dbpin.sh`: an infinite loop running `pgrep -u postgres | xargs taskset -cp 1` every second, pinning every PostgreSQL backend to core 1. New backends get pinned within 1 s.
- `csv_plot.py`, `cpu_plot.py`: plots.
- The result CSVs' last column `cpu` is **not written by `script.sh`**. It appears to have been added separately, most likely the utilisation of the monitored or pinned core (from htop). [RECON]

---

## 5. Results (final tests) [MEASURED]

| workload | VUs at plateau | plateau TPS | avg latency at 1,000 VUs | monitored core `cpu` | vmstat whole-machine CPU | disk %util |
|---|---|---|---|---|---|---|
| **get_popular** (500 hot keys) | 750–1,000 | **11,130–11,187** | 84.8 ms | **100%** | ~26% | 0.02% |
| get_only (random over 20k) | 250–1,000 | ~3,560–3,640 | 249.7 ms (p95 1,213 ms) | 93–98% | ~32% | 0.02% |
| mixed (70/20/10) | 500–1,000 | ~4,330–4,390 | 221.9 ms | 95–98% | ~33% | 7–14% |
| **put_only** | 750–1,000 | **3,046–3,171** | 297.9 ms | 72–75% | ~24% | 2–3% |
| delete_only (to 250 VUs only) | not reached | 5,456 at 250 | 43.9 ms | — | — | 0.07% |

**Linear region:** at 10/50/100 VUs every workload gives about 225 / 1,120 / 2,230 TPS with about **43 ms** average latency. That is exactly Little's law: 10 / 0.0434 s ≈ 230 TPS.

**Reading the curves:**
- `get_popular` rises linearly to 500 VUs (10.2k TPS at 47.9 ms) and then flattens at about 11.1k. Latency then grows (64.7 ms at 750, 84.8 ms at 1,000). That is the saturation signature: the monitored core is at 100%.
- `get_only` saturates much earlier (about 3.6k at 250 VUs). Misses go to PostgreSQL on the pinned core (98%), and tail latency explodes (p95 0.96–1.2 s).
- `put_only` plateaus at about 3.1k with the core *not* saturated (75%) and disk %util low. The ceiling is in commit and locking latency (§3.5).

**Why vmstat says 26% while the "cpu" column says 100%:** vmstat averages across *all* cores of the machine. With the work pinned to one or a few cores, one saturated core barely moves the machine average. Per-core monitoring (htop, `mpstat -P ALL`) is what reveals the bottleneck.

### 5.1 The ~43 ms latency floor [RECON, worth knowing]
- At low load every request (cache hit, DB read, write or delete) takes about 43 ms, even though a cache hit should take microseconds.
- A constant floor independent of the work points to the **network stack**:
  - cpp-httplib 0.27.0 ships with `#define CPPHTTPLIB_TCP_NODELAY false` (`httplib.h:145`), so **Nagle's algorithm is on** for the server's sockets;
  - with the client's delayed ACKs (about 40 ms on Linux), a response written in two pieces (headers, then body) can stall about 40 ms;
  - WSL2's virtual network adds its own overhead.
- The quick test: rebuild with `-DCPPHTTPLIB_TCP_NODELAY=true` and re-run. If the floor drops to about 1 ms, the low-VU throughput rises by about 40×.
- This floor also explains why throughput below saturation is simply VUs/0.043 s.

---

## 6. Correctness issues found by reading the code [RECON]

### 6.1 Connection leaks on exceptions in DELETE and no-cache GET
`server.cpp:190-192` (and `:231-233`):
```cpp
db::connection *conn = nullptr;
try {
  db::connection *conn = pool.acquire();   // shadows the outer conn
  ...
} catch (...) {
  if (conn) pool.release(conn);             // outer conn is still nullptr → never released
```
- The inner declaration **shadows** the outer variable. If the query throws (DB down, timeout), the connection is **never returned** and the pool shrinks permanently. Eventually `acquire()` blocks forever (there is no timeout), so the server hangs.
- Fix: assign rather than redeclare (`conn = pool.acquire();`), or better, an RAII guard that releases in its destructor.
- Related: in GET and POST, `pool.release(conn)` runs inside the try but `conn` isn't reset to nullptr. If something after the release throws (e.g. `cache.put` with bad_alloc), the catch releases the same connection **twice**, and two threads could then share one connection.

### 6.2 Cache/database race: stale values
- **GET-miss vs POST:**
  1. thread A misses, reads the old value v1 from the DB;
  2. thread B UPSERTs v2 and does `cache.put(v2)`;
  3. thread A then does `cache.put(v1)`.
  The cache now holds stale v1 indefinitely, until it's evicted or rewritten.
- **GET-miss vs DELETE:** a reader can re-insert a value that was just deleted.
- Fixes:
  - version numbers per key;
  - delete-then-fill with a lease (Facebook's memcache lease idea);
  - per-key locks;
  - invalidate on write instead of updating.

### 6.3 Others
- Mixed-case includes: `#include "dbpool.h"`/`"kvcache.h"` while the files are `DBPool.h`/`KVCache.h`. This only builds on a case-insensitive filesystem (e.g. a Windows drive under WSL).
- `get_only.js` assigns to an undeclared `res` in an ES module (strict mode), which throws *after* the request is sent. Throughput is still counted, but each iteration logs an error.
- Hard-coded DB password; configuration via stdin (not scriptable).

---

## 7. Design decisions
| Decision | Alternatives | Why | Source |
|---|---|---|---|
| cpp-httplib thread pool | epoll event loop (Boost.Asio) | simple; blocking libpqxx fits thread-per-request | [RECON] |
| Large thread count (800) | ≈ #cores | overlap DB waits; closed-loop VUs need concurrent handlers | [RECON] |
| LRU, list + hashmap, one mutex | sharded LRU, CLOCK | simple, O(1) | [CODE] |
| Write-through on POST, erase on DELETE | invalidate-only | reads after writes hit the cache | [CODE] |
| Connection pool with condition variable | connect per request | connection set-up is expensive; bounds DB concurrency | [CODE] |
| UPSERT | SELECT then INSERT/UPDATE | one round trip; atomic | [CODE] |
| Pin PostgreSQL to one core | no pinning | separates DB CPU from server CPU so the bottleneck is attributable | [CODE] |
| `get_popular` (500 hot keys) vs `get_only` (20k uniform) | single read workload | contrasts cache-bound vs DB-bound reads | [CODE] |
| Closed-loop k6 sweep | open-loop arrival rate | easy saturation curves (but hides queueing delay that open loop would show) | [RECON] |

---

## 8. Limitations and improvements [RECON]
- Single global cache mutex → shard it.
- Synchronous DB → async pipelines or an event loop.
- No timeouts on pool acquire; connection-leak bugs (§6.1).
- Cache consistency races (§6.2).
- The Nagle/TCP_NODELAY floor (§5.1).
- Measurements come from a single WSL2 machine, with client and server on the same host (client CPU competes).
- One 40 s run per point; no repetitions or variance.
- "I/O saturation" not proven by disk metrics. Next step: measure PostgreSQL `pg_stat_wal`, fsync time, lock waits; try `synchronous_commit=off` to see whether the write ceiling moves (if it jumps, the limit was commit fsync).

---

## 9. Likely interview questions
1. *Draw the architecture and trace a GET miss.* → §4.3.
2. *Is your cache really LRU? Complexity?* → yes; O(1) with list + map + `splice`.
3. *Why does `get` need an exclusive lock?* → it mutates recency order. Discuss sharding.
4. *Why 800 threads? What happens with 8 or 8,000?* → too few: idle CPU while waiting on the DB. Too many: context switching and memory, no throughput gain past saturation.
5. *Explain the throughput curves with Little's law.* → §3.4, §5.
6. *Why does vmstat show 26% CPU at "CPU saturation"?* → machine-wide average vs one pinned core.
7. *Why do writes cap at about 3.1k when neither CPU nor disk %util is maxed?* → commit fsync latency and row locks (§3.5). How to prove it: `synchronous_commit=off`, `pg_stat` views.
8. *Why is latency 43 ms at low load?* → §5.1: Nagle + delayed ACK (TCP_NODELAY false in httplib), WSL2.
9. *Find a bug in your server.* → §6.1 shadowed `conn` leak; §6.2 stale-cache race.
10. *How would you make it scale 10×?* → shard the cache; async I/O; read replicas; batch writes and group commit; tune the PG pool size (pool ≈ cores × 2 rule of thumb); TCP_NODELAY.
11. *Write-through vs write-back vs cache-aside?* → trade-offs: consistency, write latency, durability.
12. *Open vs closed loop?* → closed loop self-throttles (coordinated omission); open loop exposes queueing at a fixed arrival rate.

---

## 10. Numbers cheat sheet
| Number | Meaning |
|---|---|
| ~11.1–11.2k TPS | `get_popular` plateau (750–1,000 VUs), monitored core 100% |
| ~3.6k TPS | `get_only` plateau (cache misses → DB), core 98% |
| ~4.4k TPS | `mixed` plateau |
| ~3.0–3.2k TPS | `put_only` plateau, core 75%, disk %util 2–3% |
| ~43 ms | low-load latency floor (all workloads) |
| 230 TPS at 10 VUs | Little's law: 10 / 0.0434 s |
| 20,000 | keyspace |
| 500 | hot keys in `get_popular` |
| 10 → 1,000 VUs, 40 s | sweep |
| 800 | configured HTTP worker threads |
| core 1 | PostgreSQL pinned via `taskset` |
| 0.27.0 | cpp-httplib version (TCP_NODELAY default false) |

---

## Appendix A: Key code, annotated

Real line numbers from `project/` (`server.cpp`, `KVCache.h`, `DBPool.h`, `load_gen/*.js`). `⋮` = left out; `◀` = plain-English note.

### Reading the code: C++ / library constructs
| Construct | Meaning |
|---|---|
| `std::list<std::pair<int,std::string>>` | doubly linked list of (key, value) nodes; iterators stay valid while other nodes move |
| `std::unordered_map<int, list::iterator>` | hash map key → the node's position in the list |
| `items.splice(items.begin(), items, it)` | move node `it` to the front in O(1), no copy, no allocation |
| `std::lock_guard<std::mutex> lock(mtx)` | locks now, unlocks automatically at the end of the scope (RAII) |
| `std::unique_lock` + `cv.wait(lock, pred)` | sleep until `pred()` is true; releases the mutex while sleeping |
| `cv.notify_one()` | wake one waiting thread |
| `[&](const Request& req, Response& res) {…}` | lambda capturing outer variables by reference (here the cache and pool) |
| `srv.new_task_queue = [] { return new ThreadPool(n); }` | cpp-httplib: handle connections with a pool of n worker threads |
| `pqxx::work txn{*conn}` | libpqxx transaction on a connection; `txn.commit()` ends it |
| `exec_params(sql, a, b)` | parameterised query (`$1`, `$2`): no SQL injection |

### A.1 The LRU cache (`KVCache.h`)
```cpp
12  class KVCache {
13    size_t capacity;
14    std::list<std::pair<int, std::string>> items;       // ◀ front = most recently used
15    std::unordered_map<int, std::list<std::pair<int, std::string>>::iterator> index;   // ◀ key → node
16    std::mutex mtx;                                      // ◀ ONE lock for everything
⋮
21    bool get(int key, std::string &value) {
22      std::lock_guard<std::mutex> lock(mtx);             // ◀ exclusive even for reads (get mutates order)
23      auto it = index.find(key);
24      if (it == index.end()) return false;               // ◀ miss
28      items.splice(items.begin(), items, it->second);    // ◀ O(1) move-to-front = "recently used"
29      value = it->second->second;
30      return true;
32    }
34    void put(int key, const std::string &value) {
35      std::lock_guard<std::mutex> lock(mtx);
36      if (index.count(key)) {
37        items.erase(index[key]);                         // ◀ update: drop the old node
38      } else if (items.size() >= capacity) {
39        auto last = items.back();                        // ◀ full: evict the LEAST recently used (tail)
40        index.erase(last.first);
41        items.pop_back();
42      }
43      items.emplace_front(key, value);
44      index[key] = items.begin();
```

### A.2 The connection pool (`DBPool.h`)
```cpp
20  LibpqxxPool(int size, const std::string &connStr) : conn_string(connStr) {
21    for (int i = 0; i < size; i++) {
23        pool.push(new pqxx::connection(conn_string));   // ◀ open all connections up front
⋮
31  pqxx::connection *acquire() {
32    std::unique_lock<std::mutex> lock(mtx);
33    cv.wait(lock, [&] { return !pool.empty(); });       // ◀ block (no timeout) until a connection is free
34    pqxx::connection *conn = pool.front();
35    pool.pop();
36    return conn;
⋮
39  void release(pqxx::connection *conn) {
40    std::unique_lock<std::mutex> lock(mtx);
41    pool.push(conn);
42    lock.unlock();                                       // ◀ unlock BEFORE notify (the woken thread can take the lock at once)
43    cv.notify_one();
```

### A.3 The GET handler: cache, then the database (`server.cpp`)
```cpp
70  srv.new_task_queue = [http_thread] { return new httplib::ThreadPool(http_thread); };   // ◀ 800 workers in the tests
⋮
81  srv.Get("/val", [&](const Request &req, Response &res) {
90      int id_int = parse_id(req.get_param_value("id"));  // ◀ stoi in try/catch; −1 = invalid → 400
99      if (cache.get(id_int, cached_value)) {             // ◀ HIT: no database work at all
100       res.set_content(cached_value, "text/plain");
101       return;
102     }
104     db::connection *conn = nullptr;
106       conn = pool.acquire();                            // ◀ MISS: borrow a connection (may block)
107       db::work txn{*conn};
108       db::result r = txn.exec_params("SELECT value FROM kv_store WHERE id = $1", id_int);
110       txn.commit();
111       pool.release(conn);                               // ◀ (conn is NOT reset to nullptr here; see §6.1)
113       if (r.empty()) { res.status = 404; ... }
118         string db_val = r[0][0].as<string>();
119         cache.put(id_int, db_val);                      // ◀ fill the cache (races with a concurrent POST: §6.2)
```

### A.4 POST (UPSERT, write-through) and the DELETE bug (`server.cpp`)
```cpp
151       conn = pool.acquire();
152       db::work txn{*conn};
155       const string sql = "INSERT INTO kv_store (id, value) VALUES ($1, $2) "
156                          "ON CONFLICT (id) DO UPDATE SET value = $2";   // ◀ insert-or-update in one statement
158       txn.exec_params(sql, id_int, val);
159       txn.commit();                                     // ◀ PostgreSQL WAL flush (fsync) here: the write ceiling
160       pool.release(conn);
162       cache.put(id_int, val);                           // ◀ write-through: the cache gets the new value
⋮
190     db::connection *conn = nullptr;                     // ◀ DELETE handler
191     try {
192       db::connection *conn = pool.acquire();            // ◀ BUG: a NEW variable shadows the outer conn
⋮
206     } catch (const std::exception &e) {
207       if (conn) pool.release(conn);                     // ◀ outer conn is still nullptr → connection LEAKED on error
```

### A.5 The workloads that produced the two plateaus (`load_gen/*.js`)
```javascript
// get_popular.js: cache-heavy reads (≈11.1k TPS, monitored core 100%)
13  const SEQ_MAX = 500;
19    const key = (__ITER % SEQ_MAX) + 1;   // ◀ each virtual user cycles over the same 500 hot keys
21    http.get(`${BASE}/val?id=${key}`);

// put_only.js: write-heavy (≈3.1k TPS, core ~75%, disk %util 2–3%)
13    const id = Math.floor(Math.random() * (parseInt(__ENV.KEYSPACE || '10000'))) + 1;   // ◀ KEYSPACE=20000 from script.sh
14    const payload = { id: String(id), val: `val-${__VU}-${Math.random().toString(36).slice(2,8)}` };
15    const res = http.post(`${BASE}/save`, payload);   // ◀ form body; cpp-httplib parses it into params
```
`__ITER` is k6's per-virtual-user iteration counter, and `__VU` is the virtual-user id. Each VU loops with no think time, which makes this a closed loop.
