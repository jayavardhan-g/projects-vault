---
type: note
tags: [os, interview, concurrency, synchronization]
source: iitb_slides/semaphores.pdf
---
# 09 · Semaphores

> **In one line:** a semaphore is an integer counter with two atomic operations — **`down`/wait** (decrement,
> block if it goes negative) and **`up`/post** (increment, wake a waiter) — and it's the Swiss-army primitive:
> initialize it to **1** and it's a lock, to **0** and it's a signal, to **N** and it counts resources.

Prereqs: [[07 - Locks]], [[08 - Condition Variables]] (semaphores do the same jobs differently). Pairs with [[06 - Threads]].

---

## 1. Definition

A semaphore holds an internal **counter** (initialized to a chosen value) and exposes exactly two operations:

- **`down(s)`** (a.k.a. `wait`, `P`): decrement the counter by 1; if the result is **negative**, **block** the
  caller.
- **`up(s)`** (a.k.a. `post`, `V`): increment the counter by 1; if any thread is blocked, **wake one** of them.

> [!important] You cannot read or test the counter directly
> There's no "if counter > 0 then down". You can *only* `down` and `up`. This restriction is what makes the
> atomicity guarantees hold — and why the **initial value** is where all the design lives.

Both `up` and `down` are **atomic** — implemented with internal locking — so you never need an extra lock to
protect the counter itself. (But a semaphore used for *signalling* does **not** give you mutual exclusion over
your shared data — for that you still need a lock / binary semaphore.)

Intuition: the counter = "number of units of a resource available." `down` = take one (wait if none);
`up` = return one (wake someone who was waiting). A negative counter's magnitude = how many threads are waiting.

## 2. Three uses, set by the initial value

### Init = 1 → a lock (binary semaphore)
```c
sem s = 1;
down(s);            // first thread: counter 1→0, proceeds; others: 0→-1, block
// critical section
up(s);              // counter 0→1, wakes a waiter
```
Counter toggles 1↔0 → mutual exclusion. This is a **binary semaphore**.

### Init = 0 → signalling / ordering (T1 → T2)
Want T2 to run only after T1 finishes? Init `s = 0`:
```c
// T1                     // T2
do work;                  down(s);   // if T1 not done (counter 0→-1), block
up(s);  // counter 0→1     do work;   // runs only after T1's up
```
- If **T2 first**: `down` makes counter −1 → T2 blocks until T1's `up` wakes it. ✔
- If **T1 first**: `up` makes counter 1 → T2's `down` finds 1→0 and proceeds without blocking. ✔
Either order works. (This is the semaphore analog of a CV `signal`.)

### Init = N → count N resources
E.g. N free buffer slots, or "allow up to N threads in at once." `down` takes a unit (blocks when all N are
used); `up` returns one.

## 3. Semaphore vs condition variable

| | Condition variable | Semaphore |
|---|---|---|
| `wait` ≈ | `wait(cv, mutex)` | `down(s)` |
| `signal` ≈ | `signal(cv)` | `up(s)` |
| Needs an external mutex? | **yes** (missed-wakeup) | no — `up`/`down` are self-atomic |
| Remembers past signals? | **no** — a `signal` with no waiter is lost | **yes** — `up` increments the counter; a later `down` still sees it |
| Broadcast? | `broadcast` wakes all | **no broadcast** — must `up` N times |
| State | just a wait queue (condition is your own variable) | a built-in counter (can replace some int/bool flags) |

The "remembers signals" row is the big practical difference: with a CV, if you `signal` before anyone `wait`s,
the wakeup is **lost**; with a semaphore, the `up` bumps the counter, so a later `down` still succeeds. That's why
semaphore solutions often need fewer helper variables.

## 4. Worked example: producer–consumer with semaphores

Cleaner than the CV version — the counters *are* the slot/item counts:

```c
sem empty = N;   // free slots            sem filled = 0;  // items ready
// Producer                               // Consumer
down(empty);   // block if buffer full    down(filled);  // block if buffer empty
produce item;                             consume item;
up(filled);    // signal an item          up(empty);     // signal a free slot
```

No explicit counter variables, no mutex needed *if* producing/consuming the slot is itself atomic. **If the
buffer is a shared structure** you still need mutual exclusion — add a binary semaphore `mutex = 1`:

```c
// Producer                               // Consumer
down(empty);                              down(filled);
down(mutex);                              down(mutex);
add item to buffer;                       remove item from buffer;
up(mutex);                                up(mutex);
up(filled);                               up(empty);
```

> [!bug] Order of the two `down`s matters — deadlock if swapped
> If you take the mutex *before* the resource semaphore:
> ```c
> // Producer (DEADLOCK)        // Consumer (DEADLOCK)
> down(mutex);                  down(mutex);
> down(empty);  // may block    down(filled); // may block
> ```
> A producer can grab `mutex`, then block on `down(empty)` because the buffer is full — **still holding `mutex`.**
> The consumer that would free a slot can't even enter: it blocks on `down(mutex)`. Deadlock.
> **Unlike a CV's `wait`, `down` does *not* release any held lock when it blocks.** So always take the *resource*
> semaphore first, the *mutex* second. (This asymmetry — CV `wait` releases its mutex, semaphore `down` releases
> nothing — is a favorite exam gotcha.)

## 5. Worked example: two threads rendezvous

Both threads must finish step 1 before either starts step 2. Two semaphores init 0:
```c
// T1                         // T2
step1;                        step1;
up(T1done);                   up(T2done);
down(T2done);                 down(T1done);
step2;                        step2;
```
Each says "I'm done with step 1" (`up`), then waits for the other (`down`).
**Swap `up` and `down` and you deadlock:** if both do `down` first, each waits for an `up` that never comes.

## 6. Guidelines

- **Pay attention to the initial value** — it defines the semaphore's role (1 = lock, 0 = signal, N = resource count).
- Use **separate** semaphores for signalling vs mutual exclusion; a signalling semaphore gives no mutual exclusion.
- **Ensure every blocking `down` has a matching `up` that can run** — else the thread sleeps forever.
- Remember `down` **does not release** other locks/semaphores you hold — mind the ordering to avoid deadlock.
- There's **no broadcast**: to release N waiters, call `up` N times (or use the "each woken thread wakes the next"
  chain pattern).

---

## Interview questions

> [!question] Q1. What is a semaphore and what are its two operations?
> An integer counter with atomic `down`/wait (decrement; block if negative) and `up`/post (increment; wake a
> waiter). You can't read the counter directly. The initial value sets its role.

> [!question] Q2. How do you use a semaphore as a lock? As a signal?
> As a **lock**: init to 1 (binary semaphore); `down` before the critical section, `up` after. As a **signal**
> (T1→T2): init to 0; T1 does `up` when done, T2 does `down` before proceeding — works in either arrival order.

> [!question] Q3. Semaphore vs condition variable — key differences?
> A semaphore is self-atomic (no external mutex) and **remembers signals** via its counter (an `up` with no
> waiter isn't lost — a later `down` still sees it), but has no broadcast. A CV needs an associated mutex, **loses**
> a signal that arrives with no waiter, and supports broadcast. CVs separate "the condition" (your variable) from
> "the wait queue"; semaphores fuse a count with the wait queue.

> [!question] Q4. In producer–consumer, why must you `down(resource)` before `down(mutex)`, not the reverse?
> Because `down` doesn't release held resources when it blocks. If you hold `mutex` and then block on
> `down(empty)` (buffer full), the consumer that would free a slot can't acquire `mutex` to make progress →
> deadlock. Taking the resource semaphore first means you only grab the mutex when you can actually proceed.

> [!question] Q5. A thread blocks on `down`. Does it release other locks it holds?
> No. This is the crucial contrast with a CV's `wait`, which atomically releases its mutex while sleeping. A
> blocked `down` keeps everything it holds, so careless ordering deadlocks.

> [!question] Q6. How do you wake N threads with a semaphore (no broadcast)?
> Call `up` N times, or use a chaining pattern where each woken thread does one `up` to wake the next. There's no
> single broadcast operation like a CV has.

## Five things to remember
1. Semaphore = atomic **counter** + `down`(decrement, block if negative) + `up`(increment, wake one).
2. **Init value = role:** 1 → lock (binary semaphore), 0 → signal/ordering, N → count resources.
3. It **remembers** signals (counter persists) and needs **no external mutex** — unlike a CV; but it has **no broadcast**.
4. `down` **does not release** other held locks — order resource-then-mutex to avoid deadlock; swapping `up`/`down` order deadlocks rendezvous.
5. Producer–consumer: `empty=N`, `filled=0` (+ a `mutex=1` binary semaphore if the buffer itself is shared).
