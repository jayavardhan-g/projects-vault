---
type: note
tags: [os, interview, foundations, architecture]
source: iitb_slides/os-background.pdf
---
# 01 · Computer Architecture Background

> **In one line:** before you can reason about an OS you need the hardware model it manages — a CPU that fetches
> instructions via the **PC**, a **memory hierarchy** from registers down to disk, and a process image split into
> **code / data / stack / heap**, with function calls driving the stack.

Prereqs: comfort with C. Pairs with [[00 - OS Introduction]]. Next: [[02 - Processes]].

---

## 1. The pieces of a computer system

- **Hardware:** CPU (registers, ALU, caches), main memory (DRAM), I/O devices + secondary storage (disk, SSD, NIC).
- **Software:** user programs (your code) and system software (the OS).
- High-level code is **compiled** to a binary executable of machine instructions the CPU can run. The OS itself is
  written in C.

**Running a C program:** `.c` → (compiler: preprocess → compile → assemble → link) → executable on disk →
loaded into RAM when you run it → CPU fetches and executes its instructions.

## 2. CPU and the ISA

Every CPU exposes an **Instruction Set Architecture (ISA)** — the instructions it understands and the registers
it has (e.g. Intel = x86, Apple/phones = ARM, xv6 = RISC-V).

- **Registers** are tiny, ultra-fast storage inside the CPU:
  - **Special**: the **Program Counter (PC)** holds the address of the *next* instruction; the **stack pointer**
    tracks the top of the stack.
  - **General-purpose**: operands, intermediate values.
  - Width = 32-bit or 64-bit (that's what "64-bit CPU" means).
- Common instruction kinds: `load` (memory→register), `store` (register→memory), arithmetic/logic (`add`,
  `cmp`), `jump` (change PC), `call` (invoke a function).

**The basic fetch-decode-execute cycle:**
```
every clock cycle:  fetch instruction at PC → decode → fetch operands → execute → update PC → repeat
```
PC normally increments to the next instruction; a `jump`/`call` sets it elsewhere. Real CPUs add **pipelining**
(several instructions in flight) and many tricks to raise instructions-per-cycle — but this simple model is enough
for OS reasoning.

## 3. The memory / storage hierarchy

Fast memory is small and expensive; big memory is slow and cheap. So machines stack layers:

```
        size            latency        volatile?
registers   ~KB         < 1 ns          yes      (inside CPU)
L1/L2/L3    few MB      1–10 ns         yes      (CPU caches, HW-managed)
DRAM        few GB      ~100 ns         yes      (main memory)
SSD/HDD     TB          ~10µs / ~1 ms   NO        (persistent storage)
```

Rules of thumb worth memorizing (the "latency numbers"):
- Register access ≪ 1 ns; L1 ≈ 1 ns; DRAM ≈ 100 ns; disk seek ≈ milliseconds.
- **DRAM is ~100× slower than cache; disk is ~10,000× slower than DRAM.** This gap is *why* caching, paging, and
  buffering exist throughout the OS.
- **Caches are transparent to software** and managed by hardware — your code just "accesses memory" and doesn't
  know whether it hit cache or DRAM. Only the **disk is non-volatile**; everything above it is lost on power-off.

## 4. The memory image of a running program

When a program runs, its DRAM image has four regions:

```
 high addr ┌────────────────────┐
           │       stack        │  ← grows DOWN; local vars, args, return addrs
           │         ↓          │
           │                    │
           │         ↑          │
           │        heap        │  ← grows UP; malloc'd memory
           ├────────────────────┤
           │  data (globals)    │  ← compile-time: global/static vars
           ├────────────────────┤
           │  code (text)       │  ← the compiled instructions
 low addr  └────────────────────┘
```

- **Code**: the instructions.
- **Data**: globals/statics — allocated **when the executable is loaded**, exist for the whole run.
- **Stack**: function args, return addresses, locals — pushed on call, popped on return.
- **Heap**: dynamic memory from `malloc` — allocated at runtime when you ask, freed when you (in C) say `free`.

## 5. When is memory allocated? (worked example)

```c
int g;                      // global
int increment(int a) {      // a: on stack when increment is called
    int b;                  // b: on stack
    b = a + 1;
    return b;
}
int main() {
    int x, y;               // x, y: on stack when main runs
    x = 1;
    y = increment(x);
    int *z = malloc(40);    // z (the pointer): on stack
                            // the 40 bytes it points to: on the heap
}
```

- `g` → **data** segment, reserved when the program is loaded.
- `x, y, a, b, z` → **stack**, when their function is entered; gone when it returns.
  (Why not reserve `a`/`b` at program start? You don't know if/how many times `increment` runs.)
- The 40 bytes from `malloc` → **heap**, at the moment `malloc` runs; you must `free` them or you **leak**.

## 6. Pointers vs the thing they point to

A pointer variable stores an **address**. Two separate allocations: space for the pointer (often on the stack)
and space for the pointee (declared or `malloc`'d separately).

- The address a pointer holds is a **virtual/logical** address, not the real DRAM location (see
  [[11 - Memory Introduction]] / [[13 - Paging]]).
- Classic bug: using a pointer before pointing it at valid memory, or using `z` after `free(z)` (dangling pointer).

**Stack vs heap, the C contract:**
- Heap memory from `malloc` must be **explicitly freed**, or it leaks (lives until the process exits).
- Stack memory is **automatically** reclaimed when the function returns — which is also why you must never return
  a pointer to a local variable.

## 7. What happens on a function call

This is the mechanical heart of the stack, and it mirrors what a **system call** has to do (compare with
[[04 - Trap Handling and Context Switching]]):

```
          before call                 after call (new frame on top)
          ┌───────────┐               ┌───────────┐  ← new top of stack
          │   ...      │               │ local vars│
 old top→ │           │               ├───────────┤
                                       │ saved regs│
                                       │ return PC │  ← where to resume in caller
                                       ├───────────┤
                                       │  arg N     │
                                       │  ...       │
                                       │  arg 1     │  (args pushed in reverse by convention)
                               old top→└───────────┘
```

Steps on a call:
1. Push **arguments** on the stack (reverse order, by calling convention).
2. Push the **return address** (old PC) so control can come back.
3. Jump: set PC to the function's code (`call` in x86).
4. Save some **register context**; allocate **local variables**.
5. The function runs using this new **stack frame**.
6. On `return`: pop the frame — restore registers, pop the return address, jump back.

> [!note] Why this matters for the OS
> A **system call** must do the same bookkeeping (save context, change the PC) **but cannot trust the user's stack
> or let the user pick where to jump** — so the hardware provides a *trap instruction*, a *kernel stack*, and the
> *IDT*. That's the whole subject of [[04 - Trap Handling and Context Switching]].

---

## Interview questions

> [!question] Q1. Walk me through the memory layout of a C process.
> Four regions: **code/text** (instructions), **data** (globals/statics, fixed at load time), **heap** (grows up,
> `malloc`/`free`), **stack** (grows down, function frames). Heap and stack grow toward each other. Pointers hold
> virtual addresses into this space.

> [!question] Q2. Stack vs heap — when do you use each, and who cleans up?
> **Stack**: automatic, fast (just move the stack pointer), scoped to a function, size known at compile time, freed
> on return. **Heap**: dynamic, size/lifetime decided at runtime, survives across function returns, but you must
> `free` it in C or it leaks. Use the stack for short-lived locals of known size; the heap for data that outlives
> the creating function or whose size is only known at runtime.

> [!question] Q3. Why is there a memory hierarchy instead of one big fast memory?
> Physics and cost: fast memory (SRAM/registers) is expensive and small; cheap memory (DRAM, disk) is large but
> slow. The hierarchy + **locality** (programs reuse recent data) lets you get *most* of the speed of the top
> layer at *most* of the capacity of the bottom. The ~100× (cache↔DRAM) and ~10,000× (DRAM↔disk) gaps motivate
> caching, paging and buffering everywhere in the OS.

> [!question] Q4. What's the difference between a logical/virtual address and a physical address?
> A virtual address is what the program sees and what a pointer prints — it starts at 0 for every process. The
> physical address is the real DRAM location. The OS + MMU translate virtual→physical on every access, which is
> how isolation and the "each process owns all of memory" illusion are achieved ([[13 - Paging]]).

> [!question] Q5. What does the Program Counter do, and what changes it?
> The PC holds the address of the next instruction. It auto-increments each cycle, and is changed explicitly by
> `jump`/`call`/`return`, and *implicitly* by the hardware on a **trap** (it's set from the IDT to OS code).

## Five things to remember
1. CPU runs **fetch-decode-execute** driven by the **PC**; the **ISA** defines its instructions and registers.
2. Memory hierarchy: registers → caches → DRAM → disk; **~100×** cache↔DRAM, **~10⁴×** DRAM↔disk; only disk is non-volatile.
3. Process image = **code · data · heap(↑) · stack(↓)**.
4. **Stack** = automatic, function-scoped; **heap** = manual (`malloc`/`free`), outlives the function; leaking is forgetting to free.
5. A **function call** (push args, return addr, save regs, jump, run, pop) is the blueprint for how a **syscall/context switch** works.
