---
type: note
tags: [os, interview, xv6]
source: iitb_slides/xv6-intro.pdf
---
# 22 · xv6 Introduction

> **In one line:** xv6 is MIT's tiny teaching OS (this course uses the **x86, 32-bit** version) — small enough to
> read end to end, so it's where the abstractions from the earlier notes become real C and assembly; you need a
> little x86 (registers, `mov`/`push`/`call`, the calling convention) to read it.

Prereqs: [[01 - Computer Architecture Background]], [[04 - Trap Handling and Context Switching]]. The xv6 series: [[23 - xv6 Process]] → [[24 - xv6 Memory]] → [[25 - xv6 Process System Calls]] → [[26 - xv6 Trap]] → [[27 - xv6 Switch]] → [[28 - xv6 Locks]] → [[29 - xv6 Sleep and Wakeup]].

---

## 1. What xv6 is

- A simple Unix-like OS for **teaching** (MIT PDOS; `github.com/mit-pdos/xv6-public` for x86, and a RISC-V
  version). This course uses the **x86 32-bit** version.
- Small, readable code with simple exercises; runs inside the **QEMU** emulator (or real hardware). Far simpler
  than Linux, but the **core concepts are identical** — it's the bridge from theory to real kernel code.

## 2. Why some of the OS is in assembly

The OS is mostly **C** (compiled, loaded at boot), but a few parts **must** be assembly: low-level actions that
can't be expressed in C — the trap entry/exit, the context switch, boot code, setting special registers. So you
need to read a little x86 to follow xv6.

## 3. x86 refresher (just enough to read xv6)

**Registers:**
- General-purpose: `eax, ebx, ecx, edx, esi, edi` (hold data; `eax` also carries syscall number / return value).
- Stack: `ebp` (base of current stack frame), `esp` (top of stack).
- `eip` — the program counter (next instruction).
- Control: `cr3` holds the physical address of the current page table ([[24 - xv6 Memory]]).
- Segment registers: `cs, ds, ss, …` (the `cs` ring bits reveal privilege level — relevant to [[20 - Virtualization]]).

**Instructions:**
```asm
mov %eax, %ebx      ; ebx = eax
mov (%eax), %ebx    ; ebx = *(eax)           (dereference)
mov 4(%eax), %ebx   ; ebx = *(eax + 4)       (offset)
push %eax / pop %eax ; push/pop, adjust esp
jmp addr            ; eip = addr
call fn             ; push return addr, jump to fn
ret                 ; pop return addr into eip
```

## 4. The function-call / calling convention (the key to traps & switch)

This is the blueprint for both trap handling ([[26 - xv6 Trap]]) and context switching ([[27 - xv6 Switch]]):

```
On `call fn`:
  push caller-save regs (eax, ecx, edx)   ← caller may need these later
  push args (reverse order)
  call → pushes return address (old eip)
  push old ebp; set ebp = esp             ← new stack frame base
  push local vars + callee-save regs (ebx, esi, edi)
  ... run fn ...
On `ret`:
  pop frame, restore ebp, pop return addr → eip
```

- **Caller-save** (`eax, ecx, edx`): the caller saves them if needed; the callee may clobber them. Return value
  goes in **`eax`**.
- **Callee-save** (`ebx, esi, edi`): the callee must preserve them (save on entry, restore on exit).
- Why this matters: `swtch` only needs to save **callee-save** registers, because the caller-save ones and the
  return address are **already** on the stack from the `call` — a detail that confuses people until they know the
  convention ([[27 - xv6 Switch]] §"why only some registers").

## 5. Booting xv6

Same chain as [[02 - Processes]] §7: BIOS → boot loader (`bootasm.S`/`bootmain.c`, fits the first sector) →
loads the kernel → `main()` runs → the kernel creates the **first user process `init`** (via a hand-crafted
`struct proc` — [[27 - xv6 Switch]] §init), which `exec`s the init program, which forks the shell.

---

## Interview questions

> [!question] Q1. Why is part of an OS written in assembly if C is "good enough"?
> Some operations can't be expressed in C: saving/restoring the exact CPU register state on a trap or context
> switch, executing privileged instructions like `iret`/setting `cr3`, and the earliest boot code before a C
> runtime exists. The bulk is C; these few mechanisms are assembly.

> [!question] Q2. What's the difference between caller-save and callee-save registers, and why does it matter for context switching?
> Caller-save (`eax/ecx/edx`) may be clobbered by a callee, so the caller saves them if needed; callee-save
> (`ebx/esi/edi`) must be preserved by the callee. `swtch` only needs to save callee-save registers plus the
> stack/PC, because the `call` into `swtch` already pushed caller-save registers and the return address — so the
> saved "context" is deliberately small.

> [!question] Q3. What does `cr3` hold and when does it change?
> The physical address of the current process's page directory (top-level page table). The MMU uses it for
> translation; the OS reloads it on every context switch / page-table change so the right address space is active
> ([[24 - xv6 Memory]], [[13 - Paging]]).

## Five things to remember
1. xv6 = small readable **teaching OS** (x86 32-bit here), runs in QEMU; same concepts as Linux, far less code.
2. Mostly **C**, with **assembly** for trap entry/exit, context switch, boot, and privileged instructions.
3. Know the x86 basics: `eax` (syscall no./return), `esp`/`ebp`, `eip`, `cr3` (page table), `int`/`iret`.
4. The **calling convention** (caller- vs callee-save) is the key to understanding `swtch` and trap frames.
5. Boot chain: **BIOS → boot loader → kernel main → init → shell** (first process is hand-crafted).
