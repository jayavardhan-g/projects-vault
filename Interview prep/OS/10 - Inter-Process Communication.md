---
type: note
tags: [os, interview, ipc, concurrency]
source: iitb_slides/ipc.pdf
---
# 10 · Inter-Process Communication (IPC)

> **In one line:** processes share **no memory** by default, so the OS offers IPC channels — **pipes**,
> **sockets**, **message queues**, **shared memory**, and **signals** — that trade off ease, speed, and whether
> the two processes are related or on the same machine.

Prereqs: [[03 - Process System Calls]] (fork, file descriptors), [[06 - Threads]] (why sharing is easy for threads, not processes). Deep dive: [[18 - Network IO Sockets]].

---

## 1. Why IPC exists

A real application is usually **several processes** (different teams, different languages, fault isolation,
security). But after `fork`, parent and child have **separate** memory images — a change in one isn't visible to
the other. Processes can't just share a variable like threads do. So to cooperate they must use **IPC**:
OS-provided mechanisms, reached via syscalls, to move information between address spaces.

The five mechanisms (and when to use each):

| Mechanism | Shape | Same machine only? | Related procs only? | Typical use |
|---|---|---|---|---|
| **Pipe** (anonymous) | unidirectional byte stream | yes | **yes** (parent/child) | shell `a \| b` |
| **Named pipe (FIFO)** | unidirectional byte stream | yes | no (open by pathname) | unrelated local procs |
| **Unix domain socket** | bidirectional messages | yes | no | local client/server |
| **Internet socket** | bidirectional messages | **no** (across network) | no | networked client/server |
| **Message queue** | mailbox of discrete messages | yes | no (open by key) | decoupled producers/consumers |
| **Shared memory** | a shared memory region | yes | no (attach by key) | fastest; large data |
| **Signal** | a one-number notification | yes | no | events, `Ctrl+C`, `kill` |

## 2. The unifying model (and the file descriptor)

Most channels follow the same shape: **sender writes → OS buffers the data in kernel memory → receiver reads later.**

```mermaid
flowchart LR
    S["sender process"] -->|write / send| K["OS kernel buffer"]
    K -->|read / recv| R["receiver process"]
```

And most channels are accessed through a **file descriptor** — a small integer index into the process's
**file-descriptor table** (part of its PCB). Opening a file, socket, or pipe creates an entry and returns its
index; `read`/`write` take that fd.

```
 fd table (per process, in PCB)
 0 → STDIN       (keyboard)
 1 → STDOUT      (screen)
 2 → STDERR      (screen)
 3 → an open file / socket / pipe …
```

This is why "everything is a file" in Unix: pipes, sockets, and files all speak `read`/`write` on an fd.

**Blocking vs non-blocking:** because there's a finite kernel buffer, `send` can **block** when the buffer is
full and `recv`/`read` can **block** when it's empty. You can request non-blocking mode (the call returns an error
instead of blocking) — the basis of event-driven servers ([[19 - Network IO in Linux]]).

## 3. Pipes

A **pipe** is a unidirectional FIFO byte channel: write at one end, read from the other; the kernel buffers
bytes in between. `pipe(fd)` returns **two** fds — `fd[0]` (read end), `fd[1]` (write end).

**Anonymous pipes** work only **between a process and its children**:
```c
int fd[2];
pipe(fd);          // create BEFORE fork, so child inherits the fds
fork();
// parent closes fd[0] (read), writes to fd[1]
// child  closes fd[1] (write), reads from fd[0]
```
Because the pipe is created **before** `fork`, both processes inherit fds pointing at the **same** pipe buffer.
Each side closes the end it doesn't use. Bidirectional communication needs **two** pipes.

**Shell pipelines** `cat foo.txt | grep x` use exactly this: the shell makes a pipe, forks both commands,
connects the **STDOUT of the first** to the pipe's write end and the pipe's read end to the **STDIN of the
second**, and each child closes the unused ends. (This is the fork/exec fd-rewiring trick from
[[03 - Process System Calls]] §6.)

**Named pipes (FIFOs)** let *unrelated* processes talk: `mkfifo("/tmp/myfifo")` creates a pipe with a **pathname**
in the filesystem; one process opens it `O_RDONLY`, another `O_WRONLY`. (Writing with no reader open errors out.)

## 4. Sockets

A **socket** is a bidirectional endpoint connecting two processes — **same machine or across the network**.

- **Unix domain (local) sockets** — same machine, identified by a **pathname**.
- **Internet sockets** — different machines, identified by **IP address + port**.
- **Connection-oriented** (one client ↔ one server, like TCP) vs **connectionless** (one socket talks to many,
  address on each message, like UDP). Type is chosen in the `socket()` args.

**Client/server flow (connection-oriented):**
```
server: socket() → bind(well-known addr) → listen() → accept() → recv()/send()
client: socket() →                         connect(server addr) → send()/recv()
```
- `accept()` returns a **new** socket fd dedicated to that one client; the original **listen** socket keeps
  accepting new connections. So a server has **1 listen socket + 1 connected socket per client.** Full detail in
  [[18 - Network IO Sockets]].

## 5. Message queues

A **mailbox** identified by a **key**. A sender `msgsnd`s a discrete message; it's **buffered in the queue until a
receiver `msgrcv`s it** — so sender and receiver needn't be running at the same time (decoupling). Example: a web
server posts dynamic requests to a queue; an app server pulls and processes them, posting responses back.

## 6. Shared memory — the fastest

The same **physical memory** is mapped into the virtual address spaces of multiple processes:
```c
int shmid = shmget(key, size, ...);   // get/create a segment by key
char *data = shmat(shmid, ...);        // attach it into my address space
// now *data is literally shared — reads/writes are visible to all attachers
```

```
process A virtual AS        physical RAM         process B virtual AS
   [ ... shm ... ] ───────► [ shared frame ] ◄─────── [ ... shm ... ]
```

- **Fastest IPC**: after the one-time mapping, there's **no kernel copy** per message — processes just read/write
  memory directly. Great for large data.
- **But it provides no synchronization.** "How does B know A finished writing?" → you must add your own coordination
  (semaphores, mutexes in shared memory) — exactly the race-condition problem from [[06 - Threads]], now across
  processes.

> [!note] The trade-off in one line
> Pipes/sockets/queues **copy** data through the kernel (slower, but synchronization comes partly for free — a
> `read` blocks until data arrives). Shared memory **avoids the copy** (fastest) but makes you build
> synchronization yourself.

## 7. Signals

A **signal** is a tiny asynchronous notification — just a number — delivered to a process.

- Sent via the **`kill(pid, signum)`** syscall (despite the name, it sends *any* signal, not just SIGKILL). The
  `kill -9 <pid>` command sends SIGKILL (#9). The OS also generates signals from events — e.g. the `Ctrl+C`
  keyboard interrupt handler sends **SIGINT** to the foreground process.
- **Delivery:** the OS queues signals and delivers them when the process next transitions **kernel → user mode**
  (it piggybacks on a trap return — see [[04 - Trap Handling and Context Switching]]).
- **Handling:** each signal has a **default action** (ignore, e.g. SIGCHLD; or terminate, e.g. SIGINT). A process
  can install its **own handler** with `signal(SIGINT, my_handler)` to override the default — it jumps to the
  handler, runs it, then resumes. **SIGKILL and SIGSTOP cannot be caught or overridden** (so you can always kill a
  runaway process).
- **Process groups:** processes belong to groups (a child inherits its parent's group). A signal is delivered to
  **all processes in the group** — which is why `Ctrl+C` kills the whole foreground pipeline, not just one command.
  `setpgid` changes a process's group.

```c
void sigint_handler(int sig) { printf("caught!\n"); exit(0); }
int main() { signal(SIGINT, sigint_handler); /* ... */ }
```

---

## Interview questions

> [!question] Q1. Why do processes need IPC but threads don't?
> Threads of one process share the same address space, so they communicate by reading/writing shared variables
> directly. Processes have **separate** address spaces (after `fork`, memory is copied, not shared), so they need
> OS-mediated channels — pipes, sockets, message queues, shared memory, or signals.

> [!question] Q2. Pipe vs named pipe vs socket?
> Anonymous **pipe**: unidirectional, same machine, only between related (parent/child) processes — created before
> `fork`. **Named pipe (FIFO)**: same but opened by pathname, so unrelated processes can use it. **Socket**:
> bidirectional, works between unrelated processes and across machines (Unix-domain locally, Internet sockets over
> the network).

> [!question] Q3. Why is shared memory the fastest IPC, and what's its catch?
> After the one-time mapping, data isn't copied through the kernel on each exchange — processes read/write the same
> physical frame directly, so there's no per-message copy. The catch: it provides **no synchronization**, so you
> must add your own (semaphores/mutexes) to coordinate who writes/reads when, and you reintroduce race conditions.

> [!question] Q4. How does a shell pipeline `a | b` actually work?
> The shell creates a pipe, forks `a` and `b`, redirects `a`'s STDOUT to the pipe's write end and the pipe's read
> end to `b`'s STDIN (each closing the ends it doesn't use), then execs both. `a`'s output streams through the
> kernel pipe buffer into `b`'s input.

> [!question] Q5. What is a signal? What happens on `Ctrl+C`?
> A signal is a numbered async notification to a process. `Ctrl+C` makes the terminal driver send **SIGINT** to
> the entire **foreground process group**; the default action terminates them, unless a process installed a custom
> handler. It's delivered when the process next returns from kernel to user mode.

> [!question] Q6. Can a process catch every signal?
> No — **SIGKILL (9)** and **SIGSTOP** cannot be caught, blocked, or ignored, guaranteeing the OS/user can always
> kill or stop a process. Others (SIGINT, SIGTERM, …) can be handled or ignored.

> [!question] Q7. When would you choose message queues over sockets or shared memory?
> When you want **decoupling in time** — the sender posts a message that's buffered until the receiver is ready,
> and neither needs to be running simultaneously or maintain a live connection. Good for asynchronous
> producer/consumer pipelines between local processes.

## Five things to remember
1. Processes share **no memory by default** → IPC via syscalls: **pipes, sockets, message queues, shared memory, signals**.
2. Most channels = **sender writes → kernel buffer → receiver reads**, accessed via a **file descriptor**; buffers make `send`/`recv` block.
3. **Pipe** = unidirectional, parent/child (named pipe/FIFO for unrelated); **socket** = bidirectional, even across machines.
4. **Shared memory** = fastest (no per-message copy) but you must add your **own synchronization**.
5. **Signals** = numbered async notifications (`kill`, `Ctrl+C`→SIGINT to the process group); SIGKILL/SIGSTOP can't be caught.
