---
type: note
tags: [os, interview, networking, dpdk, advanced]
source: annotated_slides/DPDK_deep_dive_slides.pdf
---
# 30 · DPDK Deep Dive

> **In one line:** DPDK is a user-space framework for line-rate packet processing — a **poll-mode driver** owns
> the NIC, packets land in **hugepage-backed `mbuf` pools** delivered via **lockless rings**, and an app receives/
> sends them in **bursts** (`rte_eth_rx_burst`/`tx_burst`), structured either **run-to-completion** or as a
> **pipeline**.

Prereqs: **[[31 - Network Stack and Kernel Bypass]]** (why bypass; the concepts), [[19 - Network IO in Linux]], [[15 - IO Devices]]. This note is the *implementation* companion to 31.

---

## 1. Where DPDK fits

[[31 - Network Stack and Kernel Bypass]] covers *why* you'd bypass the kernel and the landscape (DPDK, netmap,
mTCP, XDP). This note is the hands-on **how** of DPDK: the two application designs, the core APIs, and the data
structures.

Recall the essence ([[19 - Network IO in Linux]] §5): a **poll-mode driver (PMD)** in user space unbinds the NIC
from the kernel (via UIO/VFIO) and **DMAs packets straight into pre-allocated user buffers** — no interrupts, no
`sk_buff`, no copy, no kernel stack.

## 2. Two application designs

### Run-to-completion (RTC)
Each core does the **whole** pipeline (RX → process → TX) for its own packets. The NIC's **RSS** (Receive Side
Scaling) hashes each packet's 5-tuple to spread flows across per-core RX queues.
```
NIC --RSS--> RXQ_1 → core 1: [RX→process→TX]
        ---> RXQ_n → core n: [RX→process→TX]
```
- **Pros:** easily scalable; **no inter-core communication** → low overhead.
- **Cons:** needs **hardware support** (a good RSS function in the NIC to distribute packets to queues).

### Pipeline
Split the work across cores by **stage**: **master** cores receive and hand packets to **worker** cores (via
rings) for processing.
```
NIC → master core(s) → [ring] → worker core(s): process
```
- **Pros:** scalable; **no NIC hardware distribution needed** (masters distribute in software).
- **Cons:** **more overhead** — inter-core communication through rings.

> RTC = "each core does everything, hardware splits traffic"; Pipeline = "cores specialize by stage, software
> splits traffic." RTC when your NIC has good RSS; pipeline when stages have very different costs or you lack RSS.

## 3. The three core data structures

| Structure | What it is | Why |
|---|---|---|
| **`rte_mempool`** | a pool of fixed-size buffers on **hugepages** | pre-allocation (no per-packet alloc) + hugepages → **fewer TLB misses** ([[13 - Paging]], [[21 - Performance]]) |
| **`rte_mbuf`** | one packet buffer: metadata + packet data | DPDK's **`sk_buff` equivalent** ([[19 - Network IO in Linux]]) but user-space and preallocated |
| **`rte_ring`** | a **lockless** (lock-free) FIFO queue | move `mbuf`s between cores (pipeline) / NIC without lock contention ([[07 - Locks]], [[21 - Performance]]) |

Hugepages matter because at line rate you can't afford TLB misses; lockless rings matter because locks would
serialize the hot path ([[21 - Performance]] on lock contention / cache-line bouncing).

## 4. The L2-forward app skeleton (the canonical example)

An "L2 forwarder" receives Ethernet packets, swaps src/dst MAC, and transmits them back. The DPDK call sequence:

```c
rte_eal_init(argc, argv);              // 1. init Environment Abstraction Layer (EAL) — MUST be first
// ... parse app args ...
rte_eth_dev_configure(port, nrx, ntx); // 2. configure the NIC port (#queues, RSS, ...)
rte_eth_rx_queue_setup(port, q, ...);  //    set up RX queue(s) (# ring descriptors)
rte_eth_tx_queue_setup(port, q, ...);  //    set up TX queue(s)
rte_eth_dev_start(port);               //    start the port (can now RX/TX)

for (;;) {                             // 3. the poll loop (no interrupts!)
    nb_rx = rte_eth_rx_burst(port, q, pkts, MAX_BURST);  // receive up to MAX_BURST packets
    for (i = 0; i < nb_rx; i++) swap_mac(pkts[i]);        // process
    nb_tx = rte_eth_tx_burst(port, q, pkts, nb_rx);      // transmit
}
```
- **`rte_eal_init`** sets up the Environment Abstraction Layer (hugepages, cores, PMD) — always the first call;
  it consumes the EAL args (`-c <core mask>`, `-w <PCI port>`), leaving app args for your own parsing.
- **`rte_eth_rx_burst` / `tx_burst`** move an **array of `mbuf`s** at once (`nb_rx`/`nb_tx` = actual count ≤
  `MAX_BURST`). **Batching** amortizes per-packet overhead — a key reason DPDK is fast ([[15 - IO Devices]]
  polling, [[19 - Network IO in Linux]] NAPI budget).
- Example invocation: `./l2fwd -c 1 -w 81:00.1 -- -p 0x1` (one core, one whitelisted port, port-mask `0x1`).

## 5. NIC offloads

DPDK exposes NIC hardware features so work moves off the CPU:
- **Checksum** verify/compute (L3/L4) in hardware,
- **RSS** — distribute packets to RX queues by a header hash,
- **L3/L4 header parsing** and simple actions (drop/forward) in the NIC.
(Without DPDK you'd query these with `ethtool -k <iface>`.) Offloading is the same theme as DDIO/accelerators in
[[21 - Performance]] — push work to hardware when the CPU is the bottleneck.

## 6. Cooperative multiprocessing / trade-offs

DPDK uses **cooperative** multiprocessing — safe only for **trusted** applications (a buggy poll loop can hog the
core). The general bypass trade-offs ([[19 - Network IO in Linux]] §5): you give up standard kernel tooling
(`tcpdump`, `netstat`), it's hard to coexist with other apps, and a core is **100% busy polling** even when idle
— in exchange for hundreds of Gbps.

---

## Interview questions

> [!question] Q1. What makes DPDK fast compared to the kernel stack?
> A user-space **poll-mode driver** owns the NIC and DMAs packets straight into pre-allocated user buffers — no
> interrupts, no `sk_buff` allocation, no kernel→user copy, no kernel-stack traversal. Add **hugepage mempools**
> (few TLB misses), **lockless rings** (no contention), and **burst** RX/TX (amortized per-packet overhead).

> [!question] Q2. Run-to-completion vs pipeline?
> RTC: each core runs the full RX→process→TX for its packets; the NIC's **RSS** spreads flows across per-core
> queues — scalable, low overhead, but needs NIC RSS support. Pipeline: master cores receive and hand packets to
> worker cores via rings — no NIC distribution needed, but adds inter-core communication overhead. RTC when RSS
> is good; pipeline for stage specialization.

> [!question] Q3. What are `rte_mbuf`, `rte_mempool`, and `rte_ring`?
> `rte_mbuf` = a packet buffer (metadata + data), DPDK's user-space `sk_buff`. `rte_mempool` = a pool of
> fixed-size buffers on hugepages (pre-allocated, few TLB misses). `rte_ring` = a lockless FIFO to pass mbufs
> between cores/NIC without lock contention.

> [!question] Q4. Why does DPDK process packets in bursts and poll instead of using interrupts?
> Polling avoids per-packet interrupt overhead and mode switches at high packet rates (the interrupt-storm problem
> — [[19 - Network IO in Linux]]). Processing a burst (array) of packets per call amortizes fixed per-call costs
> and improves cache/instruction locality. The cost is a core fully consumed by the poll loop.

> [!question] Q5. What are the downsides of DPDK?
> A core is 100% busy polling; standard kernel tools and the kernel stack no longer see the NIC; it's hard to
> coexist with other apps; and cooperative multiprocessing means it's only safe for trusted code. You also
> re-implement any networking you need (TCP via mTCP, etc. — [[31 - Network Stack and Kernel Bypass]]).

## Five things to remember
1. DPDK = **user-space poll-mode driver** owning the NIC; DMA straight to user buffers → no interrupts/`sk_buff`/copy/kernel stack.
2. Data structures: **`rte_mempool`** (hugepage buffer pool), **`rte_mbuf`** (packet buffer = sk_buff analog), **`rte_ring`** (lockless FIFO).
3. Designs: **run-to-completion** (per-core full pipeline, needs NIC **RSS**) vs **pipeline** (master→worker via rings).
4. API flow: **`rte_eal_init`** → configure port + queues → `rte_eth_dev_start` → loop **`rx_burst`/process/`tx_burst`** (batched polling).
5. Trade-offs: line-rate throughput, but a busy polling core, lost kernel tooling, trusted-only; offload work to the NIC (checksum, RSS, parsing).
