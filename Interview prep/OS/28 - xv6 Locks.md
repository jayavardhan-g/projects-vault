---
type: note
tags: [os, interview, xv6, concurrency, locks]
source: iitb_slides/xv6-locks.pdf
---
# 28 · xv6 Locks

> **In one line:** xv6 has no user threads, but the **kernel** still races (multiple cores in the kernel, and
> nested traps on one core), so it protects shared structures with **spinlocks** built on the `xchg` atomic — and
> crucially **disables interrupts** on the local core while holding one, to avoid deadlocking against its own
> interrupt handler.

Prereqs: [[07 - Locks]] (spinlocks, test-and-set), [[23 - xv6 Process]], [[26 - xv6 Trap]]. Pairs with [[27 - xv6 Switch]], [[29 - xv6 Sleep and Wakeup]].

---

## 1. Why the xv6 kernel needs locks at all

xv6 has **single-threaded user processes** sharing no memory, so **no user-space locks** (no pthread mutex). But
the **kernel** has real concurrency ([[06 - Threads]]'s race conditions apply to OS data):
- **Multiple cores** can be in kernel mode at once, touching the same structures (e.g. `ptable`, the free list).
- **Even on one core**, a process in kernel mode can be interrupted, and the **trap handler** may touch the same
  data the interrupted kernel code was using.

Solution: **spinlocks** around kernel critical sections.

## 2. The spinlock, via `xchg`

```c
void acquire(struct spinlock *lk) {
  pushcli();                         // disable interrupts (see §3)
  while (xchg(&lk->locked, 1) != 0)  // test-and-set: atomically set 1, return old
    ;                                // spin while old value was 1 (held)
}
void release(struct spinlock *lk) {
  xchg(&lk->locked, 0);
  popcli();                          // re-enable interrupts
}
```
- **`xchg`** is x86 test-and-set ([[07 - Locks]] §3): atomically set the lock to 1 and return the old value. Old
  value 0 → the lock was free and is now ours; old value 1 → held, keep spinning.
- A **spinlock** busy-waits — fine because xv6 kernel critical sections are short ([[07 - Locks]] spin vs sleep).
  (xv6 also has a **sleeplock** built on top of spinlocks, for long holds like disk I/O.)

## 3. Why disable interrupts while holding a kernel spinlock

> [!important] The self-deadlock
> Suppose a core holds spinlock **L** in kernel code, then an **interrupt** fires and its handler also tries to
> `acquire(L)`. The handler spins forever — the only code that could release L is the interrupted code, which
> can't run until the handler returns. **Deadlock on a single core.** Fix: **disable interrupts on the local core
> while holding the lock**, so no handler can run and contend for L.

Details:
- Interrupts are disabled **only on the local core** — it's fine for *other* cores to spin for L (the holder is
  still running there and will release it; [[07 - Locks]]'s "multicore spinning is OK").
- Interrupts are disabled **before** attempting to acquire (in `pushcli`), not after — otherwise there's a
  vulnerable window between acquiring and disabling where an interrupt could sneak in.
- **User-space** locks (pthread mutex) don't need this — kernel interrupt handlers don't contend for user locks.

**`pushcli` / `popcli`** handle **nested** locks:
- `pushcli` disables interrupts on the first acquire and **increments a counter** on later ones.
- `popcli` **decrements**; it re-enables interrupts only when the count hits 0 (all locks released).
So interrupts stay off until the **last** lock is released.

## 4. `ptable.lock` held across a context switch (the tricky one)

`ptable` is protected by `ptable.lock`; normally you acquire it, change `ptable`, release. But during a **context
switch** the `ptable` is in flux the *whole* time (one process leaving, another entering), so who releases the
lock?

**The discipline:** the process **giving up** the CPU acquires `ptable.lock`; the process **switched in**
releases it.
```
P1: acquire(ptable.lock) → sched → swtch → scheduler → swtch → P2: release(ptable.lock)
```
- Every function that calls `sched()` to yield the CPU (**`yield`**, **`sleep`**, **`exit`**) does so **holding
  `ptable.lock`**.
- Every place a process **resumes** after being switched in **releases** `ptable.lock` (in `yield` after the
  switch, in `sleep` on wakeup, and — for a brand-new process — in **`forkret`**, whose whole job is to release
  the lock on the first run; [[27 - xv6 Switch]]).
- The scheduler itself **loops with the lock held**, releasing it briefly between iterations (and when no
  RUNNABLE process is found, it releases the lock and **enables interrupts** so processes can become runnable —
  otherwise nothing could ever wake).

This is why holding a lock *across* a switch is correct here: the lock is handed off from the outgoing process,
through the scheduler, to the incoming process.

---

## Interview questions

> [!question] Q1. xv6 has no user threads — why does it still need locks?
> Its **kernel** is concurrent: multiple cores can execute kernel code simultaneously on shared structures
> (`ptable`, free list), and on a single core a trap handler can interrupt kernel code mid-update and touch the
> same data. These are race conditions on OS data, so kernel critical sections need spinlocks.

> [!question] Q2. How is an xv6 spinlock implemented?
> With the `xchg` atomic (test-and-set): `acquire` spins on `xchg(&lk->locked, 1)` until it returns 0 (was free);
> `release` sets it back to 0. `acquire`/`release` also disable/re-enable local interrupts via `pushcli`/`popcli`.

> [!question] Q3. Why disable interrupts while holding a kernel spinlock? Only locally?
> To avoid self-deadlock: if an interrupt handler on the same core tries to acquire a lock the interrupted code
> already holds, it spins forever (the holder can't run to release it). Disabling local interrupts prevents that.
> Only the **local** core needs it — other cores may spin safely because the holder is still running.

> [!question] Q4. Why disable interrupts *before* acquiring the lock, not after?
> If you acquired first and disabled after, an interrupt in that gap could run a handler that tries to take the
> same lock → deadlock. Disabling first (in `pushcli`) closes the window.

> [!question] Q5. How is `ptable.lock` handled across a context switch?
> The process giving up the CPU acquires `ptable.lock` before `sched()`; it stays held through the switch to the
> scheduler and into the next process, which releases it. For a brand-new process that never acquired it,
> `forkret` releases it on the first run. The scheduler loops holding the lock, releasing briefly each iteration.

> [!question] Q6. What are `pushcli`/`popcli` for?
> Nested-lock-safe interrupt control: `pushcli` disables interrupts and counts how many locks are held; `popcli`
> decrements and re-enables interrupts only when all locks are released — so interrupts stay off for the whole
> span that any spinlock is held.

## Five things to remember
1. No user threads in xv6, but the **kernel races** (multiple cores + nested traps) → **spinlocks** on shared structures.
2. Spinlock = **`xchg`** test-and-set; spin while old value is 1; short critical sections (a **sleeplock** exists for long holds).
3. **Disable local interrupts while holding a kernel spinlock** to avoid self-deadlock with an interrupt handler; only local (other cores may spin).
4. **`pushcli`/`popcli`** nest correctly — interrupts re-enabled only when the **last** lock is released.
5. **`ptable.lock` is held across the switch**: outgoing process acquires, incoming releases (new process via **`forkret`**).
