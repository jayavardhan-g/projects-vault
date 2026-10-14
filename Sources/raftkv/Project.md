# Fault-Tolerant Distributed KV Store with Raft + Dynamic Resharding

**Priority:** #10 (strong standalone Systems/Infrastructure flagship if a second systems-heavy project beyond 02/06 is wanted)
**Category:** Distributed Systems
**Resume(s):** Systems/Infrastructure
**Overlap classification:** Strong extension (of DECS KV server)
**Scores:** Difficulty 7/10 · Technical depth 8/10 · Resume impact 7/10 · Research potential 5/10 · Novelty 5/10 · Interview value 8/10 (distributed-systems correctness questions are a classic, high-value systems interview topic)
**Compute:** low — runs on commodity VMs/containers, no GPU
**Timeline:** 6–8 weeks

**One-line pitch:** A Raft-replicated, dynamically resharded distributed KV store that takes the single-node DECS multi-tier KV server all the way to real consensus and live rebalancing under load — closing the exact "read about Raft, never implemented it" gap visible against the strongest systems resumes in the senior comparison set.

---

## Full Project Specification

**Description & core problem.** The DECS course project (multi-tier KV server) was single-node with a worker-thread pool — solid systems fundamentals, but no consensus, replication, or resharding. This project builds a distributed KV store with real Raft-based leader election and log replication, plus dynamic resharding (redistributing key ranges across nodes as they join/leave or as load becomes imbalanced) under live load.

**Why it matters / fit.** This directly closes the gap identified in the senior-resume overlap analysis (main report §3.1) — you've read about Raft/Dynamo/GFS in DECS but never implemented consensus; the two strongest systems resumes in the comparison set both have hands-on distributed-systems implementation, not just paper exposure, and this closes that exact gap credibly.

**Existing skills reused:** concurrent-server design, worker-thread pools, TCP socket programming, throughput/latency/capacity analysis and closed-loop load generation — all directly from the DECS KV-server project.

**New skills gained:** Raft consensus implementation (leader election, log replication, safety under partition), live resharding/rebalancing under load, distributed-systems testing (fault injection, partition simulation).

**What makes it non-trivial:** correctness under node failure and network partition is genuinely hard to get right (split-brain avoidance, log-consistency guarantees) and resharding *while the system is live and serving traffic* (not just at startup) requires careful handling of in-flight requests during migration.

**Existing baseline(s):** the original single-node DECS KV-server as the "before" reference point for both correctness scope and performance.

**Proposed contribution:** a working Raft-replicated KV store with live resharding, evaluated for both correctness under injected faults (partition, node crash, leader failure) and performance (throughput/latency) under resharding events.

**Experiments:** throughput/latency before, during, and after a resharding event; correctness under injected leader failure and network partition (does the system remain linearizable / does it recover correctly); scalability as nodes are added.

**Ablations:** resharding strategy (eager vs. lazy key migration); replication factor sensitivity.

**Metrics:** throughput, p99 latency (including during resharding), time-to-recovery after injected failure, correctness under a fault-injection test suite (Jepsen-style, scoped down).

**Dataset(s):** N/A — synthetic KV workloads (reuse the Put/Get workload mix methodology from the original DECS project: Put-all, Get-all, Get-popular, Get+Put).

**Models/frameworks:** implemented from primitives (not just wrapping etcd/Raft libraries, to keep the consensus implementation itself as the contribution) in C/C++/Go; your own closed-loop load generator adapted from the DECS project.

**Open-source potential:** moderate.

**Paper potential:** low (this is a solid-engineering, not research-novel, project — appropriate, since not every project needs paper potential).

**Risks:** consensus correctness bugs are notoriously easy to introduce and hard to catch without rigorous fault-injection testing — budget real time for this rather than treating it as an afterthought.

**Lower-compute version:** smaller cluster (3 nodes instead of 5+), simpler fault-injection suite.

**Stretch goals:** add linearizability checking (e.g., via a model checker or a scoped Jepsen-style test harness) for a stronger correctness claim.

**How it differs from existing work:** explicitly goes "significantly beyond" the original KV server per Intro.md's own constraint on basic-KV-store recommendations — consensus and live resharding are new capabilities, not present in the course project.

**Resume bullets:**
- *"Implemented a Raft-based distributed KV store with dynamic live resharding, maintaining correctness under injected leader failure and network partition while sustaining X req/s during rebalancing."*
- *"Extended a single-node multi-tier KV server (DECS coursework) into a fault-tolerant distributed system with consensus-based replication, validated via a fault-injection test suite."*

---

## Getting Started: Key References

- **"In Search of an Understandable Consensus Algorithm"** (Ongaro & Ousterhout — the original Raft paper) — implement from this, not from a library, to keep the consensus logic itself as your contribution.
- **Dynamo** and **Google File System** papers — already covered in DECS coursework; revisit specifically for how they handle partitioning/resharding and replication trade-offs.
- **Jepsen** (Kyle Kingsbury) — the standard methodology reference for distributed-systems fault-injection testing; scope a much smaller version of this approach for your own test suite rather than attempting full Jepsen-grade rigor.
- Your own DECS multi-tier KV-server code and load-generator — this is your literal starting point; the single-node version should still exist in the repo as a documented "before."

## Suggested Milestones (6–8 weeks)

1. **Weeks 1–2** — Stand up the single-node baseline (reuse/adapt the DECS KV server) and implement Raft leader election on top of it across multiple nodes.
2. **Weeks 3–4** — Implement log replication and get correctness tests passing under normal operation (no faults yet).
3. **Weeks 5–6** — Implement dynamic resharding under live load, handling in-flight requests during key-range migration carefully.
4. **Week 7** — Build and run the fault-injection suite (partition, node crash, leader failure).
5. **Week 8** — Write-up; stretch goal: add a scoped linearizability checker.

## Suggested Repo Structure

```
distributed-kv-raft/
├── node/                   # single-node KV server (from DECS)
├── raft/                     # leader election, log replication
├── resharding/                # dynamic key-range migration under load
├── faultinjection/              # partition/crash/leader-failure test suite
├── loadgen/                     # closed-loop load generator (from DECS)
├── results/
└── README.md
```

## Definition of Done

- [ ] Multi-node cluster with working Raft leader election and log replication
- [ ] Live resharding implemented and correctness-tested under load
- [ ] Throughput/p99-latency plot before/during/after a resharding event
- [ ] Fault-injection suite results (partition, crash, leader failure) with pass/fail per scenario
- [ ] README documenting the "before" (DECS single-node) vs. "after" (this project) explicitly

## Related Projects in This Portfolio

- Pairs naturally as a second, independent Systems/Infrastructure-resume flagship alongside **02-disaggregated-prefill-decode-serving** and **06-ebpf-llm-observability** — see Portfolio 2 in the main report, §8.
