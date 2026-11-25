---
type: note
tags: [os, interview, xv6, processes]
source: iitb_slides/xv6-process.pdf
---
# 23 · xv6 Process (`struct proc`, the process table)

> **In one line:** xv6's PCB is **`struct proc`** (page table, kernel stack, trapframe/context pointers, state,
> pid, open files); all processes live in a **fixed-size array `ptable`**; and the scheduler loops over `ptable`
> running each **RUNNABLE** process in turn.

Prereqs: [[02 - Processes]] (the generic PCB), [[22 - xv6 Introduction]]. Next: [[24 - xv6 Memory]], [[27 - xv6 Switch]].

---

## 1. `struct proc` — xv6's PCB

xv6's version of the [[02 - Processes|PCB]]. The fields that matter:

```c
struct proc {
  uint sz;                  // size of process memory (bytes)
  pde_t *pgdir;             // page table (page directory) — see [[24 - xv6 Memory]]
  char *kstack;             // this process's kernel stack
  enum procstate state;     // UNUSED, EMBRYO, SLEEPING, RUNNABLE, RUNNING, ZOMBIE
  int pid;                  // process id
  struct proc *parent;      // parent (for wait/exit)
  struct trapframe *tf;     // saved user registers on a trap  → [[26 - xv6 Trap]]
  struct context *context;  // saved kernel registers on a switch → [[27 - xv6 Switch]]
  void *chan;               // sleep channel → [[29 - xv6 Sleep and Wakeup]]
  struct file *ofile[NOFILE]; // open files (fd = index) → [[16 - Files]]
  struct inode *cwd;        // current directory
};
```

Key fields and why:
- **`pgdir`** — the process's page table (virtual→physical map); the MMU uses it, loaded into `cr3` when this
  process runs ([[24 - xv6 Memory]]).
- **`kstack`** — a *separate kernel stack*, in kernel memory, **not** accessible to user code (the OS doesn't
  trust the user stack — [[04 - Trap Handling and Context Switching]]).
- **`tf` and `context`** — two pointers to saved register state on the kernel stack: the **trapframe** (user
  state saved on a trap) and the **context** (kernel state saved on a switch). This two-level save is exactly the
  "save context twice" idea from [[04 - Trap Handling and Context Switching]].
- **`state`** — the process states ([[02 - Processes]]), plus xv6-specific `EMBRYO` (being created) and `ZOMBIE`
  (exited, awaiting reap).
- **`ofile[]`** — the per-process fd array ([[16 - Files]]); 0/1/2 = stdin/out/err by default.

## 2. `ptable` — the fixed-size process table

```c
struct { struct spinlock lock; struct proc proc[NPROC]; } ptable;
```
- A **fixed array** of `NPROC` process slots (real kernels use dynamic structures; xv6 keeps it simple — see the
  data-structure trade-off in [[02 - Processes]] §6).
- Protected by **`ptable.lock`**: any code touching `ptable` must hold it, because multiple cores can be in the
  kernel at once ([[28 - xv6 Locks]]).

## 3. The scheduler loop

Each CPU has a dedicated **scheduler thread** that loops over `ptable` forever, running RUNNABLE processes:

```c
void scheduler(void) {
  for (;;) {
    acquire(&ptable.lock);
    for (p = ptable.proc; p < &ptable.proc[NPROC]; p++) {
      if (p->state != RUNNABLE) continue;
      // switch TO p:
      c->proc = p;
      switchuvm(p);          // load p's page table into cr3
      p->state = RUNNING;
      swtch(&c->scheduler, p->context);  // context switch → p  ([[27 - xv6 Switch]])
      // ... p ran, then switched back to here ...
      c->proc = 0;
    }
    release(&ptable.lock);
  }
}
```
- It's a plain **round-robin** scan ([[05 - CPU Scheduling]]) — no priorities.
- `switchuvm(p)` installs `p`'s page table; `swtch` does the actual register swap into `p`.
- When `p` gives up the CPU it switches **back** to this scheduler thread, which continues the scan. The
  scheduler holds `ptable.lock` across the switch — a subtle point explained in [[28 - xv6 Locks]] and [[27 - xv6 Switch]].

---

## Interview questions

> [!question] Q1. What's in xv6's `struct proc`?
> The PCB fields: `pgdir` (page table), `kstack` (kernel stack), `state`, `pid`, `parent`, `tf` (trapframe
> pointer — saved user regs), `context` (saved kernel regs), `chan` (sleep channel), `ofile[]` (open files),
> `cwd`, `sz`. It's xv6's concrete version of a generic PCB.

> [!question] Q2. Why does each process have a separate kernel stack?
> So kernel-mode execution never uses the untrusted user stack. The user could set up a malicious/tiny stack;
> the kernel stack lives in kernel memory, inaccessible from user mode, and holds the trapframe and saved context
> ([[04 - Trap Handling and Context Switching]]).

> [!question] Q3. Trapframe vs context in `struct proc`?
> The **trapframe** holds the user registers saved when the CPU entered the kernel on a trap (restored by
> return-from-trap). The **context** holds the kernel registers saved when the process was switched out (restored
> when it's scheduled again). Both sit on the kernel stack; `struct proc` keeps a pointer to each.

> [!question] Q4. How does xv6 schedule, and what data structure does it use?
> A fixed array `ptable[NPROC]`; each CPU's scheduler thread loops over it round-robin, running each RUNNABLE
> process via `swtch`, with `ptable.lock` held. Simple RR, no priorities — contrast real kernels' dynamic
> structures and MLFQ/CFS ([[05 - CPU Scheduling]]).

## Five things to remember
1. **`struct proc`** = xv6 PCB: `pgdir`, `kstack`, `state`, `pid`, `parent`, `tf`, `context`, `chan`, `ofile[]`.
2. **`tf`** (user regs, saved on trap) and **`context`** (kernel regs, saved on switch) both live on the **kernel stack** — the two-level save.
3. Each process has a **private kernel stack** (user stack is untrusted).
4. All processes live in the **fixed array `ptable`**, protected by **`ptable.lock`**.
5. The **scheduler thread** round-robins over `ptable`, `swtch`ing into each RUNNABLE process.
