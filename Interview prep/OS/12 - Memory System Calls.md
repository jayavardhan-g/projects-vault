---
type: note
tags: [os, interview, memory, syscalls, allocators]
source: iitb_slides/memory-syscall.pdf
---
# 12 · Memory System Calls (`malloc`, `brk`, `mmap`)

> **In one line:** a program grows its address space with **`brk`/`sbrk`** (extend the heap) or **`mmap`** (map
> pages anywhere); `malloc` is a **library** layer on top that carves those OS-given pages into variable-sized
> chunks using a **free list** with splitting/coalescing — which is where **external fragmentation** lives.

Prereqs: [[11 - Memory Introduction]], [[01 - Computer Architecture Background]] (stack vs heap). Then: [[13 - Paging]], [[14 - Demand Paging]].

---

## 1. The layers: user → library → OS

```
your code:   p = malloc(40);        ← C library call (not a syscall)
              │
C library:    carve 40 bytes from the heap's free list;
              if the heap is too small, ask the OS for more pages:
              │
OS syscalls:  brk/sbrk  (extend heap end)   or   mmap (map new pages)
              │
OS:           add valid pages to the address space (physical frames on demand)
```

Key point: **`malloc`/`free` are C-library functions, not system calls.** They manage the heap *in user space*
and only drop into the kernel (`brk`/`mmap`) when they need more raw pages. Most programs never call
`brk`/`mmap` directly.

## 2. How the address space changes

- Created at **`fork`**, replaced at **`exec`**.
- You **don't** need a syscall to use memory that's already a valid part of your address space but not yet backed
  by a physical frame — the OS allocates the frame automatically on the **page fault** ([[14 - Demand Paging]]).
- You **do** need a syscall to *add new pages* / grow the space:

**`sbrk`/`brk`** — move the **program break**, the end of the heap. `sbrk(n)` grows the heap by `n` bytes.
```
 low ─ code ─ data ─[ heap →→→ ][ ... gap ... ][ ←←← stack ]─ high
                          ↑ program break (moved by brk/sbrk)
```

**`mmap`** — map new pages at (preferably) any address, not just the heap end:
```c
char *buf = mmap(NULL, length, PROT_READ|PROT_WRITE, flags, fd, offset);
buf[0] = ...;            // physical memory allocated on demand (first touch → page fault)
munmap(buf, length);     // release it
```
- Returns the start virtual address; the region is **contiguous in virtual** space (not necessarily physical).
- Two flavors: **anonymous** (zero-filled, e.g. heap memory) or **file-backed** (initialized from a file, e.g.
  mapping an executable/shared library, or doing file I/O by memory access).
- `prot` = permissions (read/write/exec); `flags` = **private** (copy-on-write, changes not shared) vs **shared**
  (changes visible to other mappers — a form of [[10 - Inter-Process Communication|shared memory]]).

**Stack growth is automatic**: when the stack pointer crosses into a new page, a page fault occurs and the OS
extends the stack — no syscall needed.

## 3. `malloc`/`free` and their classic bugs

```c
int *z = malloc(40);   // 40 bytes from the heap
// ... use z ...
free(z);               // you MUST free it (C has no GC)
```
- Allocate the **correct size** before use.
- **Free exactly once**: forget → **memory leak** (lives until the process exits); free twice or use after free →
  corruption. (GC'd languages automate reclamation; C doesn't.)

## 4. Designing a heap allocator (the interesting part)

Given a big block of pages, satisfy *variable-sized* `malloc`/`free`. Core data structure: a **free list** — a
linked list of free chunks, initialized to the whole heap, updated on every alloc/free.

- **malloc**: search the free list for a suitable chunk; **split** a bigger chunk if needed (want 1 byte from a
  10-byte chunk → split into 1 + 9).
- **free**: add the chunk back; **coalesce** adjacent free chunks into a bigger one (so you can later satisfy large
  requests).

**Which free chunk to pick?** (request `malloc(15)` against chunks of sizes 20 and 30)
| Policy | Rule | Result |
|---|---|---|
| **First fit** | first chunk that fits | fast; can fragment the front |
| **Best fit** | smallest chunk that fits | picks the 20 → least waste, but slow scan + tiny leftovers |
| **Worst fit** | largest chunk | picks the 30 → leftover stays big/usable |

**Headers & the embedded free list** (implementation details worth knowing):
- Each allocated chunk has a hidden **header** storing its **size** (so `free` knows how much to release) and a
  **magic number** to detect overflow/corruption from the previous chunk. So `malloc(100)` really uses ~108 bytes.
- The free list is **embedded** inside the free chunks themselves — each free chunk stores a pointer to the next
  free chunk, so the allocator only remembers the list head.

### External fragmentation (worked)
4096-byte heap (4088 usable after an 8-byte header). Three `malloc(100)` → each takes 108 bytes; one 3764-byte
chunk remains. Free the **middle** one → the free list now has **two non-adjacent 100-byte chunks** plus the big
one: you **can't** satisfy a 3800-byte request even though the total free space exceeds it. That scattering is
**external fragmentation**. Freeing *all three* makes them adjacent → a smart allocator **coalesces** them back
into one big chunk.

## 5. Specialized allocators

Variable-size allocation is expensive (complex structures, split/coalesce, fragmentation). If the workload
allocates **fixed sizes**, specialize:

- **Segregated free lists**: separate free lists per popular chunk size → no split/coalesce for those sizes, less
  fragmentation; odd sizes fall back to the general allocator.
- **Slab allocator**: pre-carve a slab into chunks of one fixed size, ready to hand out — even cheaper, avoids
  per-alloc overhead. Heavily used **inside the kernel** for fixed-size objects (e.g. PCBs, inodes).
- **Buddy allocation**: allocate in **powers of two** (request 7000 B → give 8 KB). Two adjacent equal power-of-two
  blocks ("buddies") merge into the next power of two → **very cheap coalescing**; costs internal fragmentation.

---

## Interview questions

> [!question] Q1. Is `malloc` a system call?
> No. `malloc`/`free` are **C-library** functions that manage the heap in user space via a free list. They only
> invoke syscalls (**`brk`/`sbrk`** to extend the heap, or **`mmap`** to map pages) when the heap needs more raw
> memory from the OS. This keeps most allocations fast (no kernel crossing).

> [!question] Q2. Difference between `brk`/`sbrk` and `mmap`?
> `brk`/`sbrk` move the single "program break" to grow/shrink the heap — contiguous, at the heap end only. `mmap`
> maps pages at (preferably) any address, supports **file-backed** and **anonymous** mappings and **shared** vs
> **private** semantics. Modern `malloc` uses `sbrk` for small allocations and `mmap` for large ones.

> [!question] Q3. How does a heap allocator satisfy variable-sized requests?
> It keeps a **free list** of free chunks. `malloc` searches it (first/best/worst fit), **splits** a larger chunk
> if needed; `free` returns the chunk and **coalesces** adjacent free chunks. Headers record each chunk's size (so
> `free` knows the size) plus a magic value for corruption detection; the free-list pointers are embedded in the
> free chunks.

> [!question] Q4. First fit vs best fit vs worst fit?
> First fit: take the first chunk that fits — fastest, but fragments the front of the list. Best fit: smallest
> fitting chunk — minimizes immediate waste but is slow and leaves tiny unusable slivers. Worst fit: largest chunk
> — keeps the leftover big and reusable, but can waste large chunks. First fit is the common practical default.

> [!question] Q5. What is external fragmentation at the heap level, and how is it mitigated?
> Free memory split into scattered chunks so a request can't be satisfied despite enough total free space.
> Mitigations: **coalescing** adjacent free chunks on `free`, choosing fit policy carefully, and **specialized
> allocators** (segregated lists, slab, buddy) that avoid variable-size management for common sizes.

> [!question] Q6. When would you use a slab or buddy allocator?
> **Slab**: when you repeatedly allocate/free many objects of the *same fixed size* (kernel objects like inodes,
> task structs) — pre-carved chunks make alloc/free O(1) with no fragmentation. **Buddy**: when you want cheap
> coalescing of larger blocks (the Linux physical-page allocator) — power-of-two sizing makes merging trivial, at
> the cost of internal fragmentation.

## Five things to remember
1. `malloc`/`free` are **library** functions over the heap; they call **`brk`/`sbrk`** (extend heap) or **`mmap`** (map pages) only when needed.
2. `mmap` maps pages anywhere, **anonymous or file-backed**, **private or shared**; stack growth is automatic via page faults.
3. Heap allocators use a **free list** with **split** (on alloc) and **coalesce** (on free); chunks carry a **header** (size + magic).
4. Fit policies: **first fit** (fast), **best fit** (least waste, slow), **worst fit** (big leftovers) — all fight **external fragmentation**.
5. Specialized allocators: **segregated lists**, **slab** (fixed-size objects), **buddy** (power-of-two, easy coalescing).
