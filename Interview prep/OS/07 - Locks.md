---
type: note
tags: [os, interview, concurrency, locks, synchronization]
source: iitb_slides/locks.pdf
---
# 07 · Locks

> **In one line:** a lock gives mutual exclusion over a critical section; you can't build it correctly in plain
> software (the check-then-set is itself a race), so real locks are built on a **hardware atomic instruction**
> (test-and-set / compare-and-swap), and the design choice that matters is **spin vs sleep**.

Prereqs: [[06 - Threads]] (race conditions). Next: [[08 - Condition Variables]], [[09 - Semaphores]]. Real code: [[28 - xv6 Locks]].

---

## 1. What a lock is

A lock is a variable with two operations, `acquire`/`lock` and `release`/`unlock`, such that **once a thread
holds the lock, no other thread can acquire it** until the holder releases. Wrap the critical section:

```c
pthread_mutex_t m;
// ...
pthread_mutex_lock(&m);      // acquire
counter = counter + 1;       // critical section — now safe
pthread_mutex_unlock(&m);    // release
```

If T1 holds `m`, T2's `lock(&m)` **stops** until T1 unlocks. This serializes the critical section → the lost
update from [[06 - Threads]] can't happen.

**Goals of a lock implementation:**
1. **Mutual exclusion** (the whole point).
2. **Fairness** — every waiter eventually gets in; no starvation.
3. **Low overhead** — acquiring, releasing, and *waiting* shouldn't burn resources.

Locks are needed in both **user space** (the pthreads library) and the **kernel** (OS data structures also race)
— separate implementations, same idea.

## 2. Why you can't do it in plain software

The obvious attempt:
```c
int isLocked = 0;
void acquire() { while (isLocked == 1) ; /* wait */  isLocked = 1; }
void release() { isLocked = 0; }
```

This is **broken** — the check and the set are two separate steps, so it has the *same* race it's trying to
prevent:
```
isLocked = 0
T1: while(isLocked==1) → false, exits loop
--- context switch (before T1 sets isLocked=1) ---
T2: while(isLocked==1) → false, exits loop
T2: isLocked = 1 ; enters critical section
--- resume T1 ---
T1: isLocked = 1 ; enters critical section    ← BOTH inside! mutual exclusion broken
```

The root problem: user programs **have no control over when a context switch happens**, so they can't guarantee
"check and set" runs without interruption. We need the *hardware* to make check-and-set a single, uninterruptible
step.

## 3. Hardware atomic instructions

The CPU provides instructions that **read-and-modify a memory location in one indivisible step** — no context
switch, no other core, can interleave in the middle.

**test-and-set (TAS)** — `test_and_set(addr, new)`: write `new` into `*addr` and **return the old value**, atomically.
```c
int isLocked = 0;
void acquire() { while (test_and_set(&isLocked, 1) == 1) ; }  // spin while old value was 1 (busy)
void release() { isLocked = 0; }
```
If it returns 1, the lock was already taken (and we just wrote 1 again, harmlessly) → keep spinning. If it returns
0, the lock was free and we just atomically grabbed it → done. **No race is possible**, because the test and the
set are one hardware step.

**compare-and-swap (CAS)** — `cas(addr, expected, new)`: if `*addr == expected`, set it to `new` and return true;
else leave it and return false — all atomic.
```c
void acquire() { while (!cas(&isLocked, 0, 1)) ; }  // succeed only if it was 0 (free)
```
CAS is more general than TAS and is the workhorse of lock-free data structures (compare, e.g., the Micro Exchange
ring buffer in your projects). **All modern locks are built on these.** Pure-software algorithms (Peterson's,
Dekker's) exist but don't play well with modern CPUs/compilers (reordering, caches).

## 4. Spinlock vs sleeping mutex — the real design choice

The locks above are **spinlocks**: a waiter **busy-waits** in a `while` loop, burning CPU until the lock frees.

The alternative: a **sleeping mutex** — the waiter **blocks** (the OS puts it to sleep and switches to another
thread); when the lock is released the OS wakes a waiter.

```
Spinlock:                              Sleeping mutex:
  acquire → spin, spin, spin...          acquire → if busy, go to sleep (yield CPU)
  (wastes CPU while waiting)             (OS runs someone else; wakes you on release)
  no context-switch cost                 context-switch cost to sleep & wake
```

| | Spinlock | Sleeping mutex |
|---|---|---|
| Waiter behavior | busy-waits (burns CPU) | blocks (frees CPU) |
| Overhead | none, if wait is brief | two context switches (sleep + wake) |
| Best when | critical section is **short** | critical section is **long** |

**Rules of thumb:**
- **Short** critical section → **spinlock** (spinning a few cycles beats a ~1 µs context switch).
- **Long** critical section → **sleeping mutex** (don't waste a core spinning for milliseconds).

> [!warning] Spinlocks are dangerous on a single core
> On one core, if the lock **holder** is preempted while other threads spin for the lock, those threads waste the
> *entire* time slice spinning for a lock that can't be released until the holder is rescheduled. On multicore
> it's less bad: the holder can run on another core and finish while others spin. This is why kernel spinlocks
> are held only very briefly, usually with preemption/interrupts disabled ([[28 - xv6 Locks]]).

A note on **fairness**: the simple TAS/CAS spinlocks above are **not fair** — a thread can be unlucky and keep
losing the race, starving. Fair variants (ticket locks, MCS locks, queued mutexes) hand the lock out in order.

## 5. Using locks well

- **One lock per shared structure.** Every thread touching it must acquire that lock first.
- **Coarse-grained** (one big lock) = simple but kills parallelism (everything serializes). **Fine-grained** (many
  small locks) = more parallelism but harder to get right (deadlock risk, more overhead).
- **Lock for reads too**, not just writes — you don't want to read half-updated data. **Reader-writer locks** let
  many readers in concurrently but give writers exclusive access.
- **Check third-party libraries are thread-safe** before calling them from multiple threads.
- **Minimize locking** — locks *serialize* and erase the gains of parallelism. Classic optimization: each thread
  updates a **local** counter and only occasionally merges into the shared global under the lock (sharding).

> [!note] Hazards locks introduce (good follow-up territory)
> **Deadlock** (each thread holds a lock the other needs — avoid by always acquiring locks in a fixed global
> order), **livelock**, **priority inversion** (a low-priority holder blocks a high-priority waiter — fixed by
> priority inheritance), and **convoying**. Locks solve races but create their own failure modes.

---

## Interview questions

> [!question] Q1. Why can't you implement a lock with a simple flag variable in C?
> Because "check the flag, then set it" is two operations; a context switch between them lets two threads both
> see the flag free and both set it — the exact race you're trying to prevent. You can't stop the switch from user
> code, so you need a hardware instruction that does check-and-set atomically.

> [!question] Q2. How does test-and-set build a lock? How does CAS?
> TAS atomically writes 1 and returns the old value; `acquire` spins while the returned old value is 1 (busy) and
> proceeds when it's 0 (it just grabbed a free lock). CAS atomically sets to `new` only if the value equals
> `expected`; `acquire` spins until `cas(&lock, 0, 1)` succeeds (it was free and is now ours). Both make the
> check-and-set a single uninterruptible step.

> [!question] Q3. Spinlock vs sleeping mutex — when do you use each?
> Spinlock for very short critical sections (spinning a few cycles is cheaper than a ~1 µs context switch) and in
> contexts where you can't sleep (e.g. interrupt handlers). Sleeping mutex for long critical sections, so waiters
> yield the CPU instead of burning it. Spinning for a long time wastes a whole core.

> [!question] Q4. Why are spinlocks especially bad on a uniprocessor?
> If the lock holder is preempted, every spinning waiter wastes its full time slice spinning for a lock that
> literally cannot be released until the holder is rescheduled — no forward progress. On multicore the holder can
> keep running on another core, so spinning may pay off.

> [!question] Q5. Coarse-grained vs fine-grained locking?
> Coarse (one big lock) is simple and deadlock-free but serializes everything, killing parallelism. Fine (many
> small locks) allows concurrency on disjoint data but risks deadlock and adds overhead. Start coarse; go finer
> only where contention is measured.

> [!question] Q6. Do the basic spinlocks guarantee fairness? What if not?
> No — TAS/CAS spinlocks let any waiter win the next race, so a thread can starve. Fair locks (ticket/MCS locks)
> queue waiters and grant the lock in arrival order.

> [!question] Q7. Name hazards that locks introduce.
> Deadlock (fixed by a global lock-ordering discipline or lock-free structures), priority inversion (fixed by
> priority inheritance), livelock, and convoying. Also plain performance loss from over-serialization — minimize
> critical-section size and use sharding/reader-writer locks.

## Five things to remember
1. A lock enforces **mutual exclusion**; goals = mutual exclusion + fairness + low overhead.
2. You **can't** build a correct lock in plain software — check-then-set is itself a race.
3. Real locks use a **hardware atomic**: **test-and-set** or **compare-and-swap** (one indivisible read-modify-write).
4. **Spinlock** (busy-wait, short sections, cheap) vs **sleeping mutex** (block, long sections); spinning is dangerous on one core if the holder is preempted.
5. Discipline: one lock per structure, lock reads too, keep critical sections small, beware **deadlock** (fixed ordering) and **priority inversion**.
