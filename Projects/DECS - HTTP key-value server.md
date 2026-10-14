---
type: project
folder: projects/decs
repo: github.com/jayavardhan-g/High-Performance-HTTP-Key-Value-Server
dates: course project (before 2026-08); uploaded 2026-08-19
status: done
kind: coursework — systems (Design and Engineering of Computing Systems)
stack: [C++, cpp-httplib, libpqxx/PostgreSQL, k6 (JS), Python plots]
tags: [project, systems, cpp, coursework]
---
# DECS — multi-tier HTTP key-value server

**What.** This was the course project for *Design and Engineering of Computing Systems*. It is a C++ HTTP KV server with a worker
thread pool, an in-memory cache (`KVCache.h`) in front of PostgreSQL through a connection pool (`DBPool.h`), and clients plus a
closed-loop k6 load generator. Five workloads were tested: put-only, get-only, get-popular, delete-only and mixed. CPU and disk utilisation were logged
alongside to find the bottleneck.

**Key results** (`project/load_gen/results/final tests/`, 10–1000 virtual users):

| workload | peak throughput | at VUs | avg latency | CPU |
|---|--:|--:|--:|--:|
| get-popular (cache hits) | ~11,200 req/s | 1000 | 85 ms | 26% |
| mixed | ~4,400 req/s | 500 | 112 ms | 35% |
| get-only | ~3,600 req/s | 500 | 131 ms | 32% |
| put-only | ~3,200 req/s | 750 | 231 ms | 24% |

CPU never saturated, so the limit sits elsewhere: the DB, the pool size or the load generator. The repo does not include a written analysis.

**Follow-ups.** [[raftkv - Raft key-value store]] was planned explicitly as its distributed extension.

**Note.** `server.cpp` hard-codes a local Postgres connection string with a password, and that file is in the public repo.
It is a lab credential, but it is worth scrubbing.

**Source.** `README.md`, `project/server.cpp`, `project/load_gen/` (scripts, CSVs, plots, `summary.json`).
