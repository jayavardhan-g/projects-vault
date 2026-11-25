---
type: index
tags: [os, interview, index]
source: Downloads/iitb-os-slides (Mythili Vutukuru, IIT Bombay OS course)
created: 2026-11-25
---
# Operating Systems — interview prep

Concept-by-concept notes built from the IIT Bombay OS slides (Mythili Vutukuru). Each note teaches
**basics → advanced**, with diagrams, worked examples, and an **Interview questions** block at the end.
Read top to bottom the first time; after that, jump to any concept.

> [!tip] How to use these
> Every note has the same shape: **In one line → Why it exists → the mechanism with a diagram → a worked
> example → trade-offs → Interview questions → Five things to remember.** Skim the diagram and the five
> things for revision; read the body in full the first time.

## The three big ideas (OSTEP)
The whole course hangs off three jobs the OS does. Keep these in your head as the spine:

| Idea | One line | Notes that cover it |
|---|---|---|
| **Virtualization** | Give each program the illusion it owns the whole CPU and all of memory | [[02 - Processes]], [[04 - Trap Handling and Context Switching]], [[05 - CPU Scheduling]], [[11 - Memory Introduction]], [[13 - Paging]], [[14 - Demand Paging]] |
| **Concurrency** | Run many programs/threads at once, correctly and safely | [[06 - Threads]], [[07 - Locks]], [[08 - Condition Variables]], [[09 - Semaphores]], [[10 - Inter-Process Communication]] |
| **Persistence** | Store data that outlives the process and survives crashes | [[15 - IO Devices]], [[16 - Files]], [[17 - Filesystems]] |

## Modules

### 0 · Foundations
- [[00 - OS Introduction]] — what an OS is, user/kernel mode, syscalls, interrupts, the three big ideas
- [[01 - Computer Architecture Background]] — CPU/ISA, memory hierarchy, stack vs heap, what a function call does

### 1 · Processes & the trap/switch mechanism
- [[02 - Processes]] — the process abstraction, states, the PCB
- [[03 - Process System Calls]] — `fork`, `exec`, `wait`, `exit`; how the shell works; redirection & pipes
- [[04 - Trap Handling and Context Switching]] — traps, the IDT, kernel stack, how a context switch really works

### 2 · Scheduling
- [[05 - CPU Scheduling]] — FIFO, SJF, STCF, RR, MLFQ, lottery; turnaround vs response time

### 3 · Threads & concurrency
- [[06 - Threads]] — threads vs processes, the shared address space, race conditions
- [[07 - Locks]] — mutual exclusion, spinlocks, hardware atomics, the costs
- [[08 - Condition Variables]] — waiting for a condition, the `while`-loop rule, producer/consumer
- [[09 - Semaphores]] — counting/binary semaphores, how they build everything else
- [[10 - Inter-Process Communication]] — pipes, shared memory, message passing, signals

### 4 · Memory & virtual memory
- [[11 - Memory Introduction]] — address spaces, base & bounds, segmentation
- [[12 - Memory System Calls]] — `malloc`/`free`, `brk`/`sbrk`, `mmap`, fragmentation, allocators
- [[13 - Paging]] — page tables, TLB, multi-level tables
- [[14 - Demand Paging]] — page faults, swapping, replacement policies (LRU, clock), thrashing

### 5 · I/O & storage
- [[15 - IO Devices]] — device controllers/drivers, polling vs interrupts, DMA
- [[16 - Files]] — the file abstraction, file descriptors, the open-file table
- [[17 - Filesystems]] — inodes, directories, the block layout, crash consistency & journaling

### 6 · Networking
- [[18 - Network IO Sockets]] — the socket API, TCP/UDP, client & server flow
- [[19 - Network IO in Linux]] — the kernel network stack, `epoll`, softirqs, NAPI, zero-copy

### 7 · Virtualization & performance
- [[20 - Virtualization]] — VMs, hypervisors, trap-and-emulate, containers
- [[21 - Performance]] — latency vs throughput, Little's law, bottleneck analysis, measurement traps

### 8 · xv6 internals (how a real teaching OS does it)
- [[22 - xv6 Introduction]] — the codebase map, booting
- [[23 - xv6 Process]] — `struct proc`, the process table, `scheduler`
- [[24 - xv6 Memory]] — RISC-V paging, kernel/user page tables, `kalloc`
- [[25 - xv6 Process System Calls]] — `fork`/`exec`/`wait` in real code
- [[26 - xv6 Trap]] — trampoline, trapframe, `usertrap`/`kerneltrap`
- [[27 - xv6 Switch]] — `swtch`, `context`, how scheduling is coded
- [[28 - xv6 Locks]] — spinlocks, sleeplocks, why interrupts are disabled
- [[29 - xv6 Sleep and Wakeup]] — the sleep/wakeup primitive, lost-wakeup problem

### 9 · Advanced deep-dives
- [[30 - DPDK Deep Dive]] — poll-mode drivers, hugepages, why bypass the kernel
- [[31 - Network Stack and Kernel Bypass]] — the cost of the kernel path, XDP, io_uring, DPDK vs kernel

## A 30-minute revision path
Can't read everything? Do this order: [[00 - OS Introduction]] → [[02 - Processes]] →
[[04 - Trap Handling and Context Switching]] → [[05 - CPU Scheduling]] → [[07 - Locks]] →
[[08 - Condition Variables]] → [[13 - Paging]] → [[14 - Demand Paging]] → [[17 - Filesystems]].
These are the nine topics interviewers return to most.
