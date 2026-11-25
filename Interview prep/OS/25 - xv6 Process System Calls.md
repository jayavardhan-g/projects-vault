---
type: note
tags: [os, interview, xv6, processes, syscalls]
source: iitb_slides/xv6-process-syscall.pdf
---
# 25 · xv6 Process System Calls (fork/exit/wait/exec in code)

> **In one line:** xv6 implements the Unix process calls from [[03 - Process System Calls]] concretely — a user
> wrapper traps in (`int`, syscall number in `eax`), the kernel dispatches on that number, and `fork`/`exit`/
> `wait`/`exec` manipulate `struct proc` and `ptable`.

Prereqs: [[03 - Process System Calls]] (the semantics), [[23 - xv6 Process]], [[26 - xv6 Trap]] (how the trap dispatches), [[27 - xv6 Switch]] (how a new process is made runnable).

---

## 1. User side → kernel side

- User programs call library functions declared in **`user.h`** (xv6 has no standard libc).
- The real syscall is made in **`usys.S`**: a macro puts the **syscall number in `eax`** and runs **`int
  T_SYSCALL`** (the trap instruction — [[26 - xv6 Trap]]).
- The trap lands in the kernel, which reads `eax` and calls the matching kernel function via the `syscalls[]`
  table. The **return value is placed back in `eax`.**

> [!warning] Same name, two functions
> The user `fork()` (in the library, just traps) and the kernel `fork()` (does the real work at higher privilege)
> share a name — don't confuse them.

## 2. `fork` (kernel side)

```
sys_fork / fork():
  np = allocproc();            // new struct proc + pid + kernel stack  ([[27 - xv6 Switch]])
  copyuvm(np, curproc);        // copy the whole memory image           ([[24 - xv6 Memory]])
  *np->tf = *curproc->tf;      // copy the trapframe (so child resumes exactly like parent)
  np->tf->eax = 0;             // but child's return value = 0
  copy open files, cwd;        // dup the fd array
  np->state = RUNNABLE;        // scheduler will run it later
  return np->pid;              // parent gets the child's pid
```
The trick behind "[[03 - Process System Calls]] child returns 0, parent returns pid": the child's **trapframe is
copied** from the parent (so it returns from the trap to the *exact same instruction*), and only **`eax` is
overwritten with 0** in the child. Same code, same place, different return value.

## 3. `exit`

```
exit():
  close all open files; drop cwd;
  wakeup(parent);                 // in case parent is in wait()  ([[29 - xv6 Sleep and Wakeup]])
  reparent children to init;      // orphans adopted by init
  curproc->state = ZOMBIE;        // dead but not reaped
  sched();                        // give up CPU forever — never returns
```
The process becomes a **ZOMBIE** and calls `sched()` to yield the CPU; it's never scheduled again. Its memory
(kernel stack, page table) **can't be freed by itself** — those are in use *right now* — so the **parent** frees
them in `wait`. This is exactly why [[03 - Process System Calls]] says a process can't clean up its own memory.

## 4. `wait`

```
wait():
  loop:
    scan ptable for a child of curproc;
    if a ZOMBIE child found: free its kstack + page table + pid, return its pid;
    if no children at all: return -1;
    else: sleep(curproc, &ptable.lock);  // block until a child exits, then rescan
```
Matches the semantics in [[03 - Process System Calls]]: reap a dead child immediately, block if children are
still running (woken by the child's `exit` → `wakeup`), return −1 if there are no children. The parent is what
actually frees the zombie's memory.

## 5. `exec` (overview)

```
exec(path, argv):
  load the new executable from disk into a NEW page table/image;
  build a fresh stack; copy argv onto it;
  switch the process to the new page table (new memory image);
  (on any error, keep the OLD image and return -1)
  return → process now runs the new program from its entry point
```
`exec` **replaces** the memory image in place (same `struct proc`, same pid), so after it returns the process is
running completely different code — the mechanism behind [[03 - Process System Calls]] §4's "no return on
success." If anything fails, it reverts to the old image so the caller can handle the error.

---

## Interview questions

> [!question] Q1. How does xv6 make a user `fork()` call actually enter the kernel?
> The user library's `fork` (via `usys.S`) puts the syscall number in `eax` and executes `int T_SYSCALL`. That
> trap switches to kernel mode, saves a trapframe, and the kernel's trap handler dispatches on `eax` to the kernel
> `fork()`. The result is returned in `eax`.

> [!question] Q2. In xv6, how does the child get return value 0 and the parent the child's pid?
> `fork` copies the parent's **trapframe** into the child (so the child will return from the trap to the same
> instruction as the parent), then sets the child's **`tf->eax = 0`**. The parent's `fork` returns the child's
> pid normally. Same code, same resume point, different `eax`.

> [!question] Q3. Why can't an exiting process free its own memory?
> At `exit` time the process is still running on its own **kernel stack** and **page table** — it can't free the
> very memory it's using. So it becomes a ZOMBIE and yields; the **parent**, running in `wait`, frees the zombie's
> kernel stack and page table and recycles the pid.

> [!question] Q4. What does `exec` do to the process?
> It loads a new executable into a fresh memory image/page table, builds a new stack with the arguments, and
> switches the process to it — keeping the same `struct proc`/pid. The old image is gone, so on success nothing
> after `exec` runs; on error it reverts to the old image and returns −1.

## Five things to remember
1. User wrapper (`usys.S`) sets **syscall number in `eax`** + `int T_SYSCALL`; kernel dispatches on `eax`, returns value in `eax`.
2. **`fork`** = `allocproc` + `copyuvm` + copy trapframe, then **child `tf->eax = 0`**, parent returns child pid.
3. **`exit`** closes files, wakes the parent, reparents orphans to init, marks **ZOMBIE**, calls `sched()` (never returns).
4. **`wait`** reaps a ZOMBIE child (freeing its kstack/page table/pid), blocks if children still run, returns −1 if none — the parent frees the zombie.
5. **`exec`** replaces the image in place (same pid); reverts to the old image on error.
