---
type: note
tags: [os, interview, xv6, concurrency, synchronization]
source: iitb_slides/xv6-sleep-wakeup.pdf
---
# 29 · xv6 Sleep and Wakeup

> **In one line:** `sleep(chan, lock)` blocks a process on a **channel** and `wakeup(chan)` marks all sleepers on
> that channel RUNNABLE — xv6's version of condition variables — and the **lock passed to `sleep` is what
> prevents the lost-wakeup race** (sleep atomically releases it after committing to sleep).

Prereqs: [[08 - Condition Variables]] (the missed-wakeup problem), [[28 - xv6 Locks]], [[27 - xv6 Switch]] (`sched`). Example ties to [[10 - Inter-Process Communication]] (pipes).

---

## 1. The need

The kernel must let a process **block** until an event, then **wake** it — e.g. a process does a blocking disk
`read`, sleeps, and is woken when the disk interrupt signals data is ready. This is exactly a **condition
variable** ([[08 - Condition Variables]]), implemented in the kernel as **sleep/wakeup**.

```
P1 (wants to block):  read() → sleep(chan, lock) → sched()   [gives up CPU]
P2 (event happens) :  disk interrupt handler → wakeup(chan)   [marks P1 RUNNABLE; no immediate switch]
P1 later scheduled :  resumes right after sched() in sleep, returns
```
Like `signal`, `wakeup` only makes the sleeper **RUNNABLE** — it runs when the scheduler next picks it
([[08 - Condition Variables]]).

## 2. The channel

How does `wakeup` know whom to wake? By a **channel** — any pointer value both sides agree on:
- `sleep(chan, lock)` records `chan` in the sleeper's `struct proc` and blocks.
- `wakeup(chan)` scans `ptable` and wakes **every** process SLEEPING on that `chan` (like CV **broadcast**, not
  single `signal` — [[08 - Condition Variables]]).
- The `chan` is just an address known to both — e.g. for a disk read, the address of the buffer; for `wait`, the
  parent's `struct proc`; for a pipe, the address of a pipe counter.

## 3. The lost-wakeup race and how the lock fixes it

This is the **missed-wakeup problem** from [[08 - Condition Variables]], in the kernel. Without care:
```
P1: check "disk not ready"
    --- P2 runs: data ready; wakeup(chan) → wakes nobody (P1 not asleep yet) ---
P1: sleep(chan)   ← sleeps forever; the wakeup already happened
```
**Fix (same as CVs):** `sleep(chan, lk)` and `wakeup(chan)` are both called **holding the same lock `lk`**.
`sleep` only **releases `lk` after** it has acquired `ptable.lock` and committed the process to SLEEPING — so
`wakeup` (which needs `lk`) cannot run in the gap between P1's check and P1's sleep.

```c
void sleep(void *chan, struct spinlock *lk) {
  acquire(&ptable.lock);     // get ptable.lock BEFORE releasing lk
  release(lk);               // now safe to drop the condition lock
  p->chan = chan;            // record channel
  p->state = SLEEPING;
  sched();                   // give up CPU (ptable.lock held across switch — [[28 - xv6 Locks]])
  // ... woken; resumes here ...
  p->chan = 0;
  release(&ptable.lock);
  acquire(lk);               // re-acquire the condition lock before returning
}
void wakeup(void *chan) {    // caller holds the same lk (→ which holds ptable.lock path)
  acquire(&ptable.lock);
  for (p in ptable) if (p->state == SLEEPING && p->chan == chan) p->state = RUNNABLE;
  release(&ptable.lock);
}
```
- Just like a CV's `wait`, `sleep` **atomically releases the lock and blocks**, and **re-acquires it on wakeup**.
- `ptable.lock` is the bridge: acquiring it before releasing `lk` makes the "release lock + go to sleep" step
  atomic with respect to `wakeup`.
- Callers must still re-check their condition in a **`while` loop** after `sleep` returns (spurious/early
  wakeups, broadcast) — the [[08 - Condition Variables]] golden rule.

> [!note] Optimization when the condition lock *is* `ptable.lock`
> If the lock protecting the condition is already `ptable.lock` (e.g. `wait`/`exit`), `sleep`/`wakeup` don't need
> to acquire it twice — they use `wakeup1`, the version that assumes `ptable.lock` is already held.

## 4. Example: `wait`/`exit`
- A parent calling `wait` with children still running calls `sleep(chan = curproc, lock = ptable.lock)`.
- A child in `exit` holds `ptable.lock` and calls `wakeup(parent)` using the parent's `struct proc` as the
  channel. The parent wakes, rescans for the ZOMBIE child, and reaps it ([[25 - xv6 Process System Calls]]).
- Why does the **parent** free the zombie's memory? Because at `exit` the child is still using its own kernel
  stack and page table — they can't be freed until it's off the CPU.

## 5. Example: pipes (the canonical producer–consumer)

xv6 anonymous pipes ([[10 - Inter-Process Communication]]) use sleep/wakeup for the bounded-buffer pattern of
[[08 - Condition Variables]]:
- A shared buffer with `nread`/`nwrite` counters, protected by the **pipe's spinlock**.
- **Reader** sleeps if the pipe is **empty** (channel = address of `&nread`); the **writer** `wakeup`s it after
  writing.
- **Writer** sleeps if the pipe is **full** (channel = address of `&nwrite`); the **reader** `wakeup`s it after
  reading.
- The pipe lock passed to `sleep` guarantees atomicity (no lost wakeup). This is producer–consumer with two
  "condition variables," exactly as in [[08 - Condition Variables]] §4.

---

## Interview questions

> [!question] Q1. What are `sleep` and `wakeup` in xv6, and what concept do they implement?
> `sleep(chan, lock)` blocks the calling process on a channel and yields the CPU; `wakeup(chan)` marks all
> processes sleeping on that channel RUNNABLE. Together they are the kernel's **condition variable** mechanism —
> `sleep` ≈ CV `wait`, `wakeup` ≈ CV `broadcast`.

> [!question] Q2. What is the channel and why does `wakeup` wake *all* sleepers on it?
> The channel is any pointer value both the sleeper and waker agree on (e.g. a buffer address). `wakeup` wakes
> every process sleeping on that channel — like a CV broadcast — because it can't tell which one(s) can actually
> proceed; each re-checks its condition on waking (hence the `while` loop).

> [!question] Q3. Explain the lost-wakeup problem and how `sleep` avoids it.
> If a process checks its condition, then (before actually sleeping) another process makes the condition true and
> calls `wakeup`, the wakeup is lost and the first process sleeps forever. `sleep` takes the condition lock
> (held by both sleeper and waker) and only **releases it after acquiring `ptable.lock` and marking itself
> SLEEPING**, so `wakeup` can't slip into the gap. It re-acquires the lock on return.

> [!question] Q4. Why must callers re-check the condition in a loop after `sleep` returns?
> Because `wakeup` is a broadcast and other processes may have consumed the condition before this one runs, and
> wakeups can be spurious. Re-checking in a `while` loop ensures the process only proceeds when the condition is
> truly satisfied (same as condition variables).

> [!question] Q5. In `wait`/`exit`, why does the parent free the zombie child, not the child itself?
> At `exit` the child is still running on its own kernel stack and page table, so it can't free them. It becomes a
> ZOMBIE and wakes the parent; the parent, in `wait`, frees the child's kernel stack and page table and recycles
> its pid once the child is off the CPU.

## Five things to remember
1. `sleep(chan, lock)` blocks on a **channel**; `wakeup(chan)` makes **all** sleepers on it RUNNABLE — xv6's **condition variable** (broadcast semantics).
2. The **lock passed to `sleep`** (held by both sleeper and waker) prevents the **lost-wakeup** race; `sleep` releases it only after committing to sleep, re-acquires on wakeup.
3. `sleep` holds **`ptable.lock`** across `sched()` ([[28 - xv6 Locks]]); callers re-check the condition in a **`while`** loop.
4. **Channel** = any agreed pointer (buffer address, parent's `struct proc`, pipe counter).
5. Used by **`wait`/`exit`** (parent reaps the zombie) and **pipes** (reader/writer sleep on empty/full) — producer–consumer in the kernel.
