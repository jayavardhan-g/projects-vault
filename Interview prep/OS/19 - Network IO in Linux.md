---
type: note
tags: [os, interview, networking, kernel]
source: iitb_slides/network-io-linux.pdf
---
# 19 · Network I/O in Linux (the kernel network stack)

> **In one line:** a received packet travels NIC → **DMA into kernel memory** → **interrupt** → a split
> **top-half / bottom-half (ksoftirq)** handler that does TCP/IP processing into an **`sk_buff`** → **copy to
> the user socket buffer**; the overheads (interrupts, context switches, copies) motivate **NAPI polling** and
> **kernel-bypass** techniques.

Prereqs: [[18 - Network IO Sockets]], [[15 - IO Devices]] (DMA, interrupts). Deep dives: [[30 - DPDK Deep Dive]], [[31 - Network Stack and Kernel Bypass]].

---

## 1. What happens below the socket

Your app sends/receives via a socket; underneath, the kernel's **network stack** does the work. This note follows
the **receive (RX) path** — the harder, more interview-relevant direction.

## 2. The RX path, step by step

```mermaid
flowchart TD
    N["NIC receives packet"] -->|DMA| K["packet in kernel memory (RX ring)"]
    K -->|interrupt| TH["TOP HALF: ack IRQ, minimal work, schedule bottom half"]
    TH --> BH["BOTTOM HALF (ksoftirq): alloc sk_buff, TCP/IP processing"]
    BH --> MAP["map packet to socket by (IP, port)"]
    MAP -->|app calls read()| U["copy payload: kernel memory → user memory"]
```

1. **NIC → DMA → kernel memory.** The driver configures the NIC via **MMIO**; the NIC **DMAs** incoming packets
   into kernel memory ([[15 - IO Devices]]) and raises an **interrupt**.
2. **Top half (hard IRQ).** NIC interrupt processing is a lot of work (header parsing, routing, congestion
   control), so it's **split in two** to avoid hogging the interrupted process. The **top half** just
   acknowledges the interrupt, does minimal work, and **schedules the bottom half**. The NIC and kernel share
   **TX/RX rings** — circular arrays of pointers to packet buffers; the NIC DMAs a packet, updates the RX ring,
   interrupts.
3. **Bottom half (`ksoftirq`).** A kernel thread, scheduled when the CPU is free, processes **all** packets
   collected in the RX ring since last time. For each it allocates an **`sk_buff`** (socket buffer — a struct
   with pointers to the packet's headers/fields) and runs the **TCP/IP stack**: parse/check headers, IP routing,
   TCP reliability + congestion control. Top half + ksoftirq can run on multiple cores.
4. **Map to socket.** Headers (destination **port**) identify which socket the packet belongs to; the payload is
   queued on that socket.
5. **Copy to user.** When the app calls `read`/`recv`, the payload is **copied from kernel memory to user
   memory**.

## 3. The overheads (why the stack is "slow")

Every packet pays:
- **Interrupt handling** + user↔kernel mode transitions,
- **Context switch** from the application to `ksoftirq`,
- **`sk_buff` allocation**,
- **a copy** from kernel space to user space.

At high packet rates these dominate. On a 100 Gbps NIC with small packets, you have only a **few nanoseconds per
packet**, but a single DRAM access is **hundreds of ns** — so a multi-threaded app on the kernel stack tops out
around **~15 Gbps**, far below line rate.

## 4. The interrupt storm → NAPI

At high load, interrupts fire **continuously**, each saying the same thing ("packets to process") → **excessive
interrupt overhead**. A naive fix is pure **busy-polling** `ksoftirq` (trade CPU for throughput/latency).

**NAPI** (the real Linux solution) is a **hybrid interrupt + polling** scheme (the same idea as
[[15 - IO Devices]] §3):
- Top half acks the interrupt, schedules the bottom half, and **disables further interrupts**.
- The bottom half runs with a **budget** (max packets per round) and **polls** the RX ring.
- Budget exhausted → run again; ring drained → **stop polling and re-enable interrupts**.

So under light load you get interrupts (low latency, no wasted CPU); under heavy load you get polling (no
interrupt storm). Best of both.

## 5. Fast I/O: three ways to beat the kernel stack

```
 Generic kernel stack     Kernel bypass (DPDK)        In-kernel offload (eBPF/XDP)
 ─────────────────────    ────────────────────        ───────────────────────────
 NIC→kernel→socket→copy   NIC→DMA straight to user    run custom code inside the
 →user (high overhead)    (kernel out of the path)    NIC driver, before sk_buff
```

### (a) Kernel bypass — DPDK (detail: [[30 - DPDK Deep Dive]])
A **poll-mode driver (PMD)** runs in **user space** and **DMAs packets straight into pre-allocated user
buffers** (`mbuf`s in hugepage-backed `mempool`s, delivered via rings). The kernel driver is just a passthrough
(UIO/VFIO). No interrupts, no `sk_buff`, no copy, no kernel stack.
- **Pros:** minimal kernel involvement, hundreds of Gbps possible in software.
- **Cons:** standard kernel tools (`tcpdump`, `netstat`) stop working; hard to coexist with other apps; burns a
  core polling.

### (b) In-kernel offload — eBPF + XDP (detail: [[31 - Network Stack and Kernel Bypass]])
**eBPF** runs safe custom code at kernel **hook points**; **XDP** is the hook **inside the NIC driver**, run on
each packet during NAPI polling — **before** any `sk_buff` or copy. The eBPF program returns a verdict:
`XDP_DROP` (e.g. DDoS filtering), `XDP_TX` (bounce it back), `XDP_PASS` (up to the normal stack),
`XDP_REDIRECT`. Uses: ultra-fast firewalls, load balancers.
- eBPF is made safe/fast by a pipeline: `clang/llvm` compile → `bpf()` syscall loads it → a **verifier** proves
  it's safe (bounded loops, no bad memory) → a **JIT** compiles the bytecode to native code.

### (c) AF_XDP — the hybrid
A special socket type (`AF_XDP`) that receives packets **directly from the XDP hook**, DMA'd into the user
program with no kernel-stack processing/copy — while still letting the kernel keep some control. Higher
throughput than regular sockets, more cooperative than full DPDK bypass.

## 6. The memory wall and DDIO

Even with bypass, **DRAM is the bottleneck**: memory speeds haven't kept up with CPU/NIC speeds, and a few-ns
packet budget can't afford a hundreds-of-ns DRAM access. **Direct Cache Access (Intel DDIO)** has the NIC write
packets **directly into the CPU's last-level cache** instead of DRAM, so software reads them from cache — a big
win for packet processing.

---

## Interview questions

> [!question] Q1. Trace a received packet through the Linux kernel.
> NIC DMAs it into kernel memory (RX ring) and raises an interrupt. The **top half** acks and schedules work; the
> **bottom half** (`ksoftirq`) allocates an `sk_buff` and runs the TCP/IP stack (headers, routing, reliability,
> congestion control), maps the packet to a socket by port, and queues it. When the app calls `read`, the payload
> is **copied** from kernel to user memory.

> [!question] Q2. Why split interrupt handling into top and bottom halves?
> NIC packet processing is heavy (parsing, routing, TCP/congestion control). Doing it all in the hard-interrupt
> context would hog the CPU and badly disrupt the interrupted process. The top half does minimal work (ack,
> schedule) so interrupts stay short; the deferred bottom half (`ksoftirq`) does the real processing when the CPU
> is free.

> [!question] Q3. What is NAPI and what problem does it solve?
> At high load, per-packet interrupts cause an **interrupt storm**. NAPI switches to polling under load: the top
> half disables interrupts and schedules a budgeted poll of the RX ring; when the ring drains, interrupts are
> re-enabled. It's an interrupt/poll hybrid giving low latency at low load and high throughput without interrupt
> overhead at high load.

> [!question] Q4. What are the main overheads of the kernel network stack?
> Per-packet interrupts, user/kernel mode transitions, the context switch to `ksoftirq`, `sk_buff` allocation,
> and the kernel→user **copy**. They cap throughput (~15 Gbps for small packets) well below modern NIC line rate.

> [!question] Q5. How does DPDK achieve much higher throughput?
> Kernel bypass: a user-space **poll-mode driver** DMAs packets straight into pre-allocated user buffers
> (hugepage mempools, rings), with no interrupts, no `sk_buff`, no copy, and no kernel stack. The trade-off is
> losing kernel tooling and dedicating a core to polling.

> [!question] Q6. What is XDP/eBPF and when would you use it instead of DPDK?
> eBPF runs verified, JIT-compiled custom code at kernel hooks; XDP is the hook in the NIC driver, run per packet
> during NAPI before any `sk_buff`/copy, returning DROP/TX/PASS/REDIRECT. Use it for fast in-kernel firewalls/load
> balancers when you want high speed **but** still cooperate with the kernel and its tooling — unlike DPDK's full
> bypass.

> [!question] Q7. What is DDIO / Direct Cache Access?
> A feature where the NIC writes incoming packets directly into the CPU's last-level cache instead of DRAM, so
> software accesses them at cache speed. It addresses the "memory wall" — DRAM latency (hundreds of ns) being far
> larger than the per-packet time budget on fast links.

## Five things to remember
1. RX path: **NIC → DMA to kernel → interrupt → top-half/bottom-half(ksoftirq) → sk_buff + TCP/IP → copy to user socket**.
2. Interrupt handling is **split** (top half = minimal/ack+schedule; bottom half = heavy processing) to keep interrupts short.
3. Overheads = **interrupts + mode switches + context switch + sk_buff alloc + kernel→user copy** → ~15 Gbps cap; **NAPI** = interrupt/poll hybrid to avoid interrupt storms.
4. Beat the stack via **DPDK** (user-space poll-mode driver, full bypass), **eBPF/XDP** (in-kernel custom code pre-sk_buff), or **AF_XDP** (hybrid).
5. The **memory wall** limits even bypass; **DDIO** writes packets straight into LLC to dodge DRAM latency.
