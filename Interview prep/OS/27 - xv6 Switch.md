---
type: note
tags: [os, interview, xv6, scheduling, context-switch]
source: iitb_slides/xv6-switch.pdf
---
# 27 · xv6 Switch (scheduling & context switching in code)

> **In one line:** xv6 switches via a **per-CPU scheduler thread** — a process gives up the CPU in `sched()`,
> which calls **`swtch`** to swap to the scheduler, which picks the next RUNNABLE process and `swtch`es into it;
> `swtch` saves/restores a tiny **`context`** (callee-save regs + eip), distinct from the **trapframe**.

Prereqs: [[04 - Trap Handling and Context Switching]] (the generic switch), [[23 - xv6 Process]], [[26 - xv6 Trap]], [[22 - xv6 Introduction]] (calling convention). Pairs with [[28 - xv6 Locks]].

---

## 1. The two-step switch via the scheduler thread

xv6 never switches **directly** process→process. Every switch goes **through** the per-CPU **scheduler thread**:

```mermaid
flowchart LR
    P1["process P1 (kernel mode)"] -->|sched() → swtch| S["scheduler thread"]
    S -->|picks RUNNABLE P2, swtch| P2["process P2 (kernel mode)"]
    P2 -->|later: sched() → swtch| S
```

- A process gives up the CPU in **`sched()`**, which calls `swtch` to switch to the **scheduler** thread.
- **`scheduler()`** loops over `ptable`, finds a RUNNABLE process, and calls `swtch` to switch into it.
- So a P1→P2 switch is really **P1 → scheduler → P2**.

**When does a process call `sched()`?** Three cases ([[04 - Trap Handling and Context Switching]] §6):
- **`yield`** — timer interrupt, it has run long enough (preemption);
- **`exit`** — it's terminating (becomes ZOMBIE);
- **`sleep`** — it's blocking on an event ([[29 - xv6 Sleep and Wakeup]]).

## 2. `context` vs `trapframe` (don't confuse them)

Both live on the kernel stack; `struct proc` points to each:

| | **trapframe** (`p->tf`) | **context** (`p->context`) |
|---|---|---|
| Saved when | CPU enters kernel on a **trap** | process **switches** to another (in `swtch`) |
| Holds | full user register set | just callee-save regs + `eip` |
| PC meaning | where user code was when it trapped | where kernel code was when `swtch` was called |
| Restored by | `iret` (return to user) | `swtch` (resume kernel) |

So a switched-out process's kernel stack holds **both**: the trapframe (how to get back to user mode) and the
context (how to resume the kernel code it was in) — exactly the "two contexts" of [[04 - Trap Handling and Context Switching]].

## 3. `swtch` — the actual switch

```asm
swtch(struct context **old, struct context *new):
    # old kernel stack already has return addr + caller-save regs (from the `call`)
    push callee-save regs (ebx, ebp, esi, edi)   # complete the old context on the stack
    mov esp → *old        # save pointer to old context into *old
    mov new → esp         # ← THE SWITCH: stack pointer now points at new's context
    pop callee-save regs  # restore new process's callee-save regs
    ret                   # pop new's return addr → resume where new left off
```

The whole context switch is that **`mov new → esp`** — swapping which kernel stack the CPU stands on ([[04 - Trap Handling and Context Switching]] §7).

> [!note] Why save only callee-save registers (not all)?
> Because `swtch` is reached by a normal `call`, the **caller-save** registers (`eax/ecx/edx`) and the **return
> address** are **already** on the stack (C calling convention — [[22 - xv6 Introduction]]). `swtch` only needs to
> add the callee-save registers to form a complete `context`. That's why `context` is so small.

**`swtch` arguments:** `swtch(&old_context_ptr, new_context_ptr)` — the first is the **address** of a pointer (so
`swtch` can write the saved context back into it), the second is the value to switch to. From `scheduler`:
`swtch(&cpu->scheduler, p->context)`; from `sched`: `swtch(&p->context, cpu->scheduler)`.

## 4. Making a brand-new process runnable: `allocproc` + `forkret`

Problem: `swtch` *resumes* a process where it stopped — but a **newly forked process never ran**. So xv6
**hand-crafts** its kernel stack to look like it was switched out at a chosen point:

```
new process kernel stack (built by allocproc):
   [ trapframe ]            ← copied from parent → returns to user just after fork
   [ address of trapret ]   ← so after forkret returns, it runs trapret → iret to user
   [ context, eip = forkret ] ← so swtch "returns" into forkret
```
- `allocproc` finds an UNUSED slot, marks it **EMBRYO**, allocs a pid + kernel stack, and lays out the stack
  above.
- When the scheduler `swtch`es in, the CPU "returns" into **`forkret`**, which **releases `ptable.lock`** (see
  below), then returns into **`trapret`**, which `iret`s to user mode — so the child emerges in user space just
  after `fork`, like its parent ([[25 - xv6 Process System Calls]]).

**`init`** (the first process) is created the same hand-crafted way; its trapframe is set so it starts at
`initcode.S`, which `exec`s the init program.

## 5. The `ptable.lock`-across-switch subtlety

The scheduler holds **`ptable.lock`** while switching. The rule: the process that *gives up* the CPU acquires the
lock, and the process that is *switched in* releases it — so the lock is held across the P1→scheduler→P2 switch:
```
P1: acquire(ptable.lock) → swtch → scheduler → swtch → P2: release(ptable.lock)
```
That's why a brand-new process needs **`forkret`** — something has to release the lock on the *first* run, since
the new process didn't acquire it itself. (Full reasoning in [[28 - xv6 Locks]].)

---

## Interview questions

> [!question] Q1. How does a context switch work in xv6?
> A process calls `sched()`→`swtch` to switch to the per-CPU **scheduler thread**; the scheduler finds a RUNNABLE
> process and `swtch`es into it. `swtch` pushes callee-save registers to finish the old `context`, saves its
> pointer, sets `esp` to the new context (the actual switch), pops the new callee-save registers, and `ret`s into
> the new process's kernel code.

> [!question] Q2. Why switch process→scheduler→process instead of directly?
> It centralizes scheduling: each process only ever switches to/from the scheduler thread, which owns the
> `ptable` scan and policy. It keeps the per-process code simple and the locking discipline uniform (acquire
> before leaving, release after arriving).

> [!question] Q3. Difference between `context` and `trapframe`?
> The **trapframe** is the full user register set saved on a trap (restored by `iret` to return to user mode). The
> **context** is the small set (callee-save regs + eip) saved by `swtch` when switching (restored by `swtch` to
> resume kernel code). A switched-out process's kernel stack holds both.

> [!question] Q4. Why does `swtch` save only callee-save registers?
> Because it's called like a normal function: the C calling convention already placed caller-save registers and
> the return address on the stack. `swtch` only adds the callee-save registers to complete a valid `context`,
> which is why the context is minimal.

> [!question] Q5. How can the scheduler `swtch` into a process that has never run?
> `allocproc` hand-crafts the new process's kernel stack: a copied trapframe, the address of `trapret`, and a
> `context` whose saved `eip` points at `forkret`. So `swtch` "returns" into `forkret` (which releases
> `ptable.lock`), which returns into `trapret`, which `iret`s to user mode — making the new process emerge in user
> space just after `fork`.

> [!question] Q6. What is `forkret` for?
> It runs once, as the entry point of every newly scheduled process, to **release `ptable.lock`** (which the
> switching discipline leaves held) before the process returns to user mode via `trapret`.

## Five things to remember
1. Switches go **process → per-CPU scheduler thread → process**; a process yields in **`sched()`**, the scheduler picks the next RUNNABLE.
2. **`swtch`** saves callee-save regs + eip into a **`context`**, swaps **`esp`** to the new context (the switch), restores, and `ret`s.
3. **`context`** (saved on switch, small) ≠ **`trapframe`** (saved on trap, full user regs); both on the kernel stack, pointers in `struct proc`.
4. A process calls `sched()` on **yield (timer) / exit / sleep**; the timer gives preemptive round-robin.
5. New processes get a **hand-crafted** kernel stack (trapframe + context→`forkret`→`trapret`); **`forkret`** releases **`ptable.lock`** on first run.
