---
type: note
tags: [os, interview, scheduling, virtualization]
source: iitb_slides/scheduling.pdf
---
# 05 · CPU Scheduling

> **In one line:** scheduling is the *policy* of which ready process runs next (the *mechanism* is the context
> switch from [[04 - Trap Handling and Context Switching]]); the story is a ladder — FIFO → SJF → SRTF → RR —
> each fixing the last one's flaw, ending at **MLFQ** (approximate SJF without knowing job lengths) and Linux's
> **CFS** (weighted fair sharing).

Prereqs: [[02 - Processes]], [[04 - Trap Handling and Context Switching]].

---

## 1. Policy vs mechanism

- **Mechanism** = *how* to switch (save old context, restore new) — [[04 - Trap Handling and Context Switching]].
- **Policy** = *which* ready process to run next, on a given core. That's this note.

The scheduler only runs when a process is in kernel mode on a trap — and **not even on every trap**. The key
enabler for forcing switches is the **timer interrupt**: a periodic interrupt that drags the OS back into control
even if the running process never makes a syscall. Without it, **preemptive** scheduling is impossible.

- **Non-preemptive (cooperative):** switch only on *voluntary* events — the process blocks or exits.
- **Preemptive:** switch *involuntarily* too — yank the CPU from a still-runnable process so it can't starve
  others. **All modern OSes are preemptive**, using the timer interrupt.

## 2. Goals (and the key metrics)

Two headline metrics, usually in tension:

```
process created ──────────────── first runs ─────────────── completes
        └──────────── response time ────────┘
        └────────────────── turnaround (completion) time ───────────────┘
```

- **Turnaround time** = completion − arrival. Minimized by finishing short jobs first.
- **Response time** = first-run − arrival. Minimized by giving everyone a quick slice (good for *interactive* jobs).
- Also: **utilization** (keep the CPU busy), **fairness** (incl. priorities), **low overhead** (decision cost +
  not too many switches — a switch costs ~1 µs).

**CPU burst** = one continuous stretch of CPU use before a process blocks (on I/O) or exits. Schedulers reason in
bursts: an I/O-bound process has many short bursts; a CPU-bound one has long bursts.

## 3. The policy ladder

### FIFO (First-In-First-Out) — non-preemptive
Run processes in arrival order, each to completion/block.

> [!bug] The convoy effect
> A long job that arrives first makes everyone behind it wait. If A(100), B(10), C(10) arrive together, B and C
> wait ~100 units behind A → terrible average turnaround. Short jobs stuck behind a long one = **convoy effect**.

### SJF (Shortest Job First) — non-preemptive
Pick the ready process with the **smallest CPU burst** next (store PCBs in a heap → extract-min).

- **Provably optimal average turnaround** *if all jobs arrive together*.
- Flaws: (1) it assumes you **know burst lengths in advance** (you usually don't); (2) being non-preemptive, a
  short job that arrives *just after* a long job started still waits for it to finish.

### SRTF (Shortest Remaining Time First) — preemptive SJF
A newly arrived job **preempts** the running one if its burst is shorter than the running job's *remaining* time.
Fixes SJF flaw (2): short jobs no longer wait behind a long job that's already running.

```
Example: P1 burst 5 @t0, P2 burst 3 @t1, P3 burst 2 @t3
FIFO:  P1[1-5] P2[6-8] P3[9-10]
SJF :  P1[1-5] P3[6-7] P2[8-10]        (at t5 pick shortest remaining ready burst)
SRTF:  P1[1] P2[2-4] P3[5-6] P1[7-10]  (P2 preempts P1 at t1, P3 preempts at t3)
```

### Round Robin (RR) — preemptive
Each process runs for a fixed **time quantum/slice**, then goes to the back of the queue. Enforced by the timer.

- Great **response time** and **fairness**; **bad turnaround** (nobody finishes early — everyone is interleaved).
- Slice tuning: **too small** → context-switch overhead dominates; **too big** → degenerates toward FIFO, poor
  responsiveness. (**xv6 is a simple RR scheduler**.)

### Weighted Fair Queueing (WFQ) → Linux CFS
RR but slices are proportional to a **weight/priority**. Real systems can't honor exact slices (timer misaligned,
process blocks early), so the practical trick: **track each process's accumulated runtime and always run the one
that has used the least of its fair share**, compensating surplus/deficit later. **Linux CFS** is this: it keeps
runnable tasks in a **red-black tree** keyed by virtual runtime and runs the one with the least.

## 4. MLFQ — the practical star

**The problem it solves:** we want SJF's turnaround (favor short jobs) *and* RR's response time, **without
knowing burst lengths ahead of time.**

**The design:**
```
Q3 (highest) ──► run RR here first
Q2           ──►
Q1           ──►
Q0 (lowest)  ──► only if everything above is empty
```
- Multiple priority queues; **always run from the highest non-empty queue**; **RR within a level**.
- **Rule: a job that uses up its whole time slice at a level is demoted** one level; a job that yields/blocks
  before its slice ends **stays** at its level.

**Why this works (the clever part):** CPU-bound jobs keep burning full slices → sink to low priority.
I/O-bound/interactive jobs block quickly (before their slice is up) → stay high. So **interactive jobs
automatically get priority over CPU hogs — and we never needed to know burst lengths.** It approximates SJF.

**Two fixes needed:**
- **Starvation:** low-priority CPU jobs could never run. Fix: **periodically reset everyone to the top priority**.
- **Gaming:** a job that yields *just before* its slice ends stays high forever. Fix: account **total** time used
  at a level (across runs), not per-run, before demoting.
- Lower levels often get **longer slices** (long jobs run less often but for longer when they do).

Parameters (how many levels, slice sizes, reset interval) have **no universal answer** — each OS ships tuned
defaults for its workload.

## 5. Multicore scheduling

- A scheduling decision is made **per core**.
- **Affinity:** prefer to keep a process on the **same core** — avoids cross-core coordination and keeps its data
  warm in that core's **cache**.
- But stay flexible: if a core is overloaded, **load-balance** by migrating processes to idle cores. Affinity vs
  balance is the core tension.

---

## Interview questions

> [!question] Q1. Difference between turnaround time and response time? Which policy optimizes which?
> Turnaround = completion − arrival (optimized by SJF/SRTF — finish short jobs fast). Response = first-run −
> arrival (optimized by RR — give everyone a quick slice). They conflict: RR has great response but poor
> turnaround because it finishes nothing early.

> [!question] Q2. Why is SJF optimal but impractical?
> It provably minimizes average turnaround when all jobs arrive together, but it requires knowing each job's CPU
> burst in advance (unknowable in general) and, being non-preemptive, lets short jobs wait behind an
> already-running long job. SRTF fixes the second; MLFQ fixes the first by *approximating* it.

> [!question] Q3. How does MLFQ approximate SJF without knowing job lengths?
> It observes behavior: jobs that exhaust their time slice (CPU-bound) get demoted; jobs that block early
> (I/O-bound/interactive) stay high. Short/interactive jobs thus float to high priority automatically — so the
> scheduler favors them like SJF would, using only observed behavior.

> [!question] Q4. What problems does MLFQ have and how are they fixed?
> **Starvation** of low-priority jobs → periodic priority boost (reset all to top). **Gaming** by yielding just
> before the slice ends to stay high → account total time at a level, not per-burst. Tuning the level count and
> slices → workload-specific defaults.

> [!question] Q5. How do you pick the RR time quantum?
> Trade-off: too small → context-switch overhead (~1 µs each) wastes CPU; too large → poor responsiveness and it
> behaves like FIFO. Pick a slice large enough to amortize the switch cost but small enough that interactive jobs
> feel responsive (typically a few ms).

> [!question] Q6. Why is a timer interrupt essential for preemptive scheduling?
> The OS can only run when the CPU is in kernel mode. A CPU-bound process that never makes a syscall would never
> enter the kernel, so the OS could never take the CPU back. The periodic timer interrupt forces a trap into the
> kernel regardless, letting the scheduler preempt.

> [!question] Q7. What does Linux use, roughly?
> A weighted-fair-queueing variant — **CFS (Completely Fair Scheduler)** — that tracks each task's virtual
> runtime in a red-black tree and runs the task with the least, scaled by priority/weight. (Newer kernels use
> EEVDF, a refinement of the same fairness idea.)

## Five things to remember
1. Scheduling = **policy**; the context switch is the **mechanism**. The **timer interrupt** enables preemption.
2. **Turnaround** (favor short jobs: SJF/SRTF) vs **response time** (favor quick slices: RR) are in tension.
3. Ladder: FIFO (convoy effect) → SJF (optimal but needs burst lengths) → SRTF (preemptive SJF) → RR (fair, responsive, poor turnaround).
4. **MLFQ** = multiple priority queues + demote-on-full-slice → approximates SJF by behavior; needs a **priority boost** to avoid starvation.
5. **Linux ≈ weighted fair sharing (CFS)**; multicore adds **affinity vs load-balancing**.
