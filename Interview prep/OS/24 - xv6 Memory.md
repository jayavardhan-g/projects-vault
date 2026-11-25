---
type: note
tags: [os, interview, xv6, memory, paging]
source: iitb_slides/xv6-memory.pdf
---
# 24 · xv6 Memory

> **In one line:** xv6 is 32-bit with a **two-level page table**; each process's address space has user stuff low
> and the **kernel mapped identically high (from KERNBASE = 2GB)**; free RAM is a **linked free list** served by
> `kalloc`/`kfree`; and `fork` eagerly **copies** the whole image (`copyuvm`) rather than copy-on-write.

Prereqs: [[13 - Paging]], [[23 - xv6 Process]]. (No demand paging — compare [[14 - Demand Paging]].)

---

## 1. The paging setup

- 32-bit → 4 GB virtual address space per process; **4 KB pages** → VA = **20-bit page number + 12-bit offset**.
- **Two-level page table** ([[13 - Paging]] §4): 10 bits index the **page directory** → 10 bits index an
  **inner table** → 12-bit offset. The page directory's physical address sits in **`cr3`**.
- **PTE flag bits:**
  - `PTE_P` — present/valid (clear → access faults),
  - `PTE_W` — writable (clear → read-only),
  - `PTE_U` — user-accessible (clear → kernel only).

## 2. The address-space layout (memorize this picture)

```
 0          ┌────────────────────────┐
            │ user code/data          │  ← from the executable
            │ guard page (unmapped)   │  ← catches stack overflow
            │ user stack              │
            │ user heap (grows up)    │
 KERNBASE   ├────────────────────────┤
  (2GB)     │ KERNEL code/data        │
            │ free pages (for users)  │  ← [KERNBASE, KERNBASE+PHYSTOP] → [0, PHYSTOP]
            │ device memory           │
 4GB        └────────────────────────┘
```

- **Low half**: per-process user memory (code, guard page, stack, heap).
- **High half (from KERNBASE = 2 GB)**: the **kernel**, mapped **identically in every process's page table**
  (the "OS in every address space" idea — [[13 - Paging]] §6). `[KERNBASE, KERNBASE+PHYSTOP]` maps linearly to
  physical `[0, PHYSTOP]`.
- The kernel pages are `PTE_U`-clear → user mode can't touch them; a trap flips to kernel mode and they become
  accessible. So on a trap, **no page-table switch is needed** — the kernel is already mapped.

> [!note] A quirk: 2 GB max physical memory
> Because physical memory is mapped into the high half starting at 2 GB (and VAs max out at 4 GB), xv6 can only
> use **PHYSTOP ≤ 2 GB** of physical RAM. Each physical page ends up with (up to) two virtual addresses — one in
> the kernel's direct map, one in a user mapping.

## 3. Free-memory management: `kalloc` / `kfree`

The kernel can't use libc `malloc`; it manages free RAM itself:
- At boot, all RAM in `[0, PHYSTOP]` not used by the kernel is collected into a **free list** — a linked list
  where **each free page stores the pointer to the next free page inside itself** (the embedded free list from
  [[12 - Memory System Calls]]).
- **`kalloc()`** — pop the first page off the free list and return it.
- **`kfree(p)`** — push page `p` back onto the head of the free list.
- xv6 allocates only **page-sized** chunks; all other kernel structures are **fixed size** (e.g. `ptable`), so it
  doesn't need a general variable-size allocator. (Real kernels add **slab**/**buddy** allocators — [[12 - Memory System Calls]].)

## 4. Building a page table

Three functions, worth knowing by name:
- **`setupkvm()`** — allocate a new page directory and install the **kernel mappings** (from the `kmap` table) via
  `mappages`. Every page table starts here; user mappings are added afterward.
- **`mappages(pgdir, va, size, pa, perm)`** — add PTEs mapping a VA range to a PA range with permissions; for each
  page it calls `walkpgdir` to find the PTE slot and fills in `pa | perm | PTE_P`.
- **`walkpgdir(pgdir, va, alloc)`** — walk the two levels to return a pointer to the PTE for `va`, optionally
  allocating the inner table if missing. This *is* the software page-table walk from [[13 - Paging]].

## 5. `fork` memory copy: `copyuvm` (eager, not COW)

`fork` builds the child's image with **`copyuvm`**: make a fresh page table, then for **each** page of the parent
— read its PTE, allocate a **new** physical frame for the child, **copy the bytes**, and map it in the child.

> [!important] xv6 forks eagerly; real kernels use copy-on-write
> xv6 makes two full separate copies immediately. Copy-on-write (not in xv6, but easy to add, and the standard in
> Linux): the child's page table points at the **parent's** frames marked **read-only**; on the first write by
> either, the MMU **faults**, the OS copies just that page, and both get private copies lazily. COW makes
> `fork`+`exec` nearly free. (This is the implementation behind [[03 - Process System Calls]] §2's "copy-on-write
> under the hood.")

## 6. Growing memory: `sbrk` → `growproc` → `allocuvm`

- `sbrk(n)` (called by `malloc`) grows the heap via **`growproc`** → **`allocuvm`**, which allocates `n` bytes of
  new frames and maps them with user permissions (`deallocuvm` shrinks and frees).
- xv6 has **no `mmap`** ([[12 - Memory System Calls]]) and **no demand paging** ([[14 - Demand Paging]]) — pages
  are allocated eagerly, not on first touch.
- Any page-table change must **reload `cr3`** (and flush the TLB) — done on context switches too ([[13 - Paging]] §5).

---

## Interview questions

> [!question] Q1. Describe xv6's address space layout.
> Low addresses hold per-process user memory (code, a guard page, stack, heap). From **KERNBASE (2 GB)** up, the
> **kernel** is mapped — identically in every process — with `[KERNBASE, KERNBASE+PHYSTOP]` mapping to physical
> `[0, PHYSTOP]`. Kernel PTEs are kernel-only, so user mode can't touch them, but a trap can reach the kernel with
> no address-space switch.

> [!question] Q2. Why map the kernel into every process?
> So that on a trap the CPU can jump straight to kernel code (already mapped at high VAs) without switching page
> tables, and the kernel can access user memory. One physical kernel copy is shared via identical high-half PTEs
> in every page table; permission bits keep user mode out ([[13 - Paging]] §6).

> [!question] Q3. How does xv6 manage free physical memory?
> A linked **free list** of 4 KB pages, with each free page storing the next pointer inside itself. `kalloc` pops
> a page; `kfree` pushes one back. Only page-sized allocations; other kernel structures are fixed-size, so no
> general allocator is needed.

> [!question] Q4. How does xv6 implement `fork`'s memory copy, and how do real kernels differ?
> `copyuvm` eagerly allocates a new frame for every parent page and copies the bytes — two full images
> immediately. Real kernels use **copy-on-write**: share parent frames read-only, duplicate a page only on the
> first write (via a page fault), making `fork` cheap.

> [!question] Q5. What do `setupkvm`, `mappages`, and `walkpgdir` do?
> `setupkvm` creates a page directory with the kernel mappings installed. `mappages` adds PTEs for a VA→PA range
> with permissions. `walkpgdir` walks the two-level table to return (and optionally allocate) the PTE for a given
> VA — the software page-table walk.

## Five things to remember
1. 32-bit, **two-level** page table (`cr3` → directory → inner table → offset); PTE flags **P/W/U** (present/writable/user).
2. Layout: **user low, kernel high from KERNBASE (2 GB)**; kernel mapped **identically in every process**, kernel-only perms → traps need no page-table switch (max ~2 GB phys).
3. Free RAM = **linked free list** of pages; **`kalloc`/`kfree`** pop/push; only page-sized allocs (fixed-size structures otherwise).
4. `fork` uses **`copyuvm`** (eager full copy); real kernels use **copy-on-write** (share read-only, copy on fault).
5. `setupkvm`/`mappages`/`walkpgdir` build page tables; `sbrk`→`allocuvm` grows the heap; **no `mmap`, no demand paging**.
