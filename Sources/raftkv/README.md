# raftkv — a distributed key-value store built on Raft (C++20)

A from-scratch Raft consensus engine with a replicated key-value store, TCP
networking, crash-safe disk storage, a live monitoring dashboard, and an
automated chaos test. No libraries to install — just a C++ compiler and Python 3.

Runs on Linux, macOS and Windows, and nodes on different operating systems can
join the same cluster.

```
build.sh / build.bat      compiles everything into one executable
cluster.conf              which nodes exist, and where they live
include/ + src/           the C++ source
scripts/local_cluster.sh  start a 3-node cluster on this laptop
scripts/client.py         send requests from any other machine
scripts/dashboard.py      live monitor for the whole cluster
scripts/chaos_test.py     automated crash / failover test
```

---

## Build

You need a C++20 compiler and Python 3.

**macOS** — run `xcode-select --install` once, then:

```bash
./build.sh
```

**Linux** — `sudo apt install g++ python3`, then:

```bash
./build.sh
```

**Windows** — install [MSYS2](https://www.msys2.org), open the *MSYS2 MINGW64*
terminal and run `pacman -S mingw-w64-x86_64-gcc` once. Then, from this folder:

```bash
./build.sh          # in Git Bash or MSYS2
build.bat           # in Command Prompt or PowerShell
```

The build produces one executable — `./raftkv`, or `raftkv.exe` on Windows.
That is the whole build; run it again after editing any source file.

> Windows note: use the g++ that comes with MSYS2 / MinGW-w64. The compiler
> shipped inside plain Cygwin, or one built without pthread support, will not
> have `std::thread`. Everything OS-specific lives in `include/platform.h`,
> which maps sockets, disk sync and atomic file replacement onto Winsock on
> Windows and onto POSIX everywhere else — no other file has platform code.

In the commands below, Windows users should read `./raftkv` as `raftkv.exe`.

---

## 1. How to test locally (one laptop)

### The fast way — automated test

```bash
python3 scripts/chaos_test.py
```

This starts its own private 3-node cluster, writes 100 keys, measures
throughput, force-kills the leader, checks that a new leader takes over, re-reads
every key to prove nothing was lost, restarts the dead node and waits for it to
catch up, then confirms all three nodes agree. It prints `RESULT: PASS` or
`RESULT: FAIL` and cleans up after itself. Nothing is left running.

### The hands-on way — drive it yourself

```bash
./scripts/local_cluster.sh
```

Nodes 2 and 3 start in the background; node 1 gives you a prompt. Add `--fresh`
to start from an empty log. It uses its own three-loopback-nodes config
(`data/local.conf`), so your real `cluster.conf` can stay pointed at the other
laptops. To watch it: `python3 scripts/dashboard.py data/local.conf`. Now type:

```
raft> set name ajay
OK
raft> get name
ajay
raft> status
node=1  role=FOLLOWER  term=1  commit=3  applied=3  log=3  ...
raft> delete name
OK
raft> dump
raft> exit
```

You can type `set` on **any** node — if the node you are talking to is not the
leader, it looks up the leader and forwards the request for you.

To watch a failure and recovery, open a second terminal and run the dashboard
(section 3), then in the raft shell type:

```
raft> kill      # this node stops answering — the cluster treats it as crashed
raft> revive    # it rejoins and catches up automatically
```

If you kill a follower, writes keep working (2 of 3 nodes is still a majority).
If you want to kill the *leader*, find which node is the leader in the dashboard
and stop that one. Killing two of three nodes stops writes entirely — that is
correct Raft behaviour, not a bug.

### Running nodes in three separate terminals instead

```bash
mkdir -p data
./raftkv --id 1 --config cluster.conf --data-dir data    # terminal 1
./raftkv --id 2 --config cluster.conf --data-dir data    # terminal 2
./raftkv --id 3 --config cluster.conf --data-dir data    # terminal 3
```

---

## 2. How to run across several laptops

All laptops must be on the same network (same Wi-Fi or same LAN cable).

**Step 1 — find each laptop's IP address.**

```bash
ipconfig getifaddr en0        # macOS
hostname -I                   # Linux
ipconfig                      # Windows — use the IPv4 Address of your Wi-Fi adapter
```

You get something like `192.168.1.10`. Do this on every laptop and write the
addresses down.

**Step 2 — write one cluster.conf and put the same copy on every laptop.**
The format is `<node id> <ip address> <port>`:

```
1 192.168.1.10 5001
2 192.168.1.11 5001
3 192.168.1.12 5001
```

This file must be **identical everywhere**. It is how the nodes find each other.

**Use an odd number of nodes — 3 or 5.** Raft needs a majority to agree, so a
3-node cluster keeps working when 1 machine dies. A **2-node cluster tolerates
zero failures**: the majority is still 2, so if either laptop drops off, writes
stop completely. Two nodes is fine for a first connectivity test, but add a
third before you demo any failover.

**Step 3 — copy the project to each laptop and build it there.** Build on each
machine (`./build.sh`, or `build.bat` on Windows) — you cannot copy a compiled
executable between a Mac, a Linux box and a Windows PC. The laptops may run
different operating systems; they will still form one cluster.

**Step 4 — start one node per laptop, each with its own id.**

```bash
mkdir -p data
./raftkv --id 1 --config cluster.conf --data-dir data    # on 192.168.1.10
./raftkv --id 2 --config cluster.conf --data-dir data    # on 192.168.1.11
./raftkv --id 3 --config cluster.conf --data-dir data    # on 192.168.1.12
```

Within a second or two one node prints `LEADER` in its `status`. Type
`set city mumbai` on **any** laptop and `get city` on **any other** — the value
is there, because it was replicated to a majority before being acknowledged.

**Firewall.** Each laptop must accept incoming TCP on two ports: the raft port
from `cluster.conf` (e.g. 5001) and that port + 1000 for monitoring (6001).

- **macOS** — System Settings → Network → Firewall prompts the first time; click *Allow*.
- **Linux** — `sudo ufw allow 5001/tcp && sudo ufw allow 6001/tcp`
- **Windows** — a "Windows Defender Firewall has blocked some features" popup
  appears on first run; tick **Private networks** and click *Allow access*. If
  you dismissed it, run this once in an **Administrator** PowerShell:

  ```powershell
  New-NetFirewallRule -DisplayName "raftkv" -Direction Inbound -Protocol TCP -LocalPort 5001,6001 -Action Allow
  ```

  Also make sure the Wi-Fi network is set to **Private**, not Public — Windows
  blocks incoming connections on Public networks regardless of the rule.

**Sanity check that the laptops can see each other** before blaming the code:

```bash
ping 192.168.1.11
nc -vz 192.168.1.11 5001                              # macOS / Linux
```

```powershell
Test-NetConnection 192.168.1.11 -Port 5001            # Windows PowerShell
```

**What "sync" means here.** There is nothing to synchronise by hand. A node that
was off, crashed, or disconnected reconnects on its own, the leader detects how
far behind it is, ships it the missing log entries, and it replays them into its
key-value store. State also survives a full power-off: every node keeps its term,
its vote and its whole log in `data/raft_node_<id>.bin`, written with `fsync` and
an atomic file rename, so a crash mid-write can never corrupt it.

---

## 3. How to monitor the whole cluster

### The dashboard

From **any** laptop that can reach the others (it does not have to be running a
node), in a spare terminal:

```bash
python3 scripts/dashboard.py cluster.conf
```

Every node runs a separate status server on its raft port + 1000, on its own
thread, so watching the cluster never slows down or interferes with consensus.
The dashboard polls all of them every 300 ms and shows four panels:

```
RAFT CLUSTER MONITOR   17:56:57  \ polling every 300ms

NODE ADDRESS               ROLE        TERM  COMMIT  APPLIED   LOG  KEYS    RTT   UPTIME  STATE
-----------------------------------------------------------------------------------------------
1    192.168.1.10:5001     FOLLOWER       2      32       32    32    30    1ms       8s  healthy
2    192.168.1.11:5001     LEADER         2      32       32    32    30    4ms       8s  healthy
3    192.168.1.12:5001     --             -       -        -     -     -      -        -  unreachable

REPLICATION  as tracked by leader node 2, log head = 32
  node 1  match     32/32     next 33     [##########]  in sync
  node 3  match      0/32     next 32     [..........]  32 behind (unreachable)
  last applied: SET item29 value29

EVENTS
  17:56:51  node 3 is online (LEADER, term 1)
  17:56:53  node 3 became unreachable
  17:56:54  node 1 term 1 -> 2
  17:56:54  node 2 is now LEADER (term 2)

leader: node 2   reachable: 2/3   quorum: yes   terms agree: yes   logs converged: yes
```

**Node table** — one row per machine. `ROLE` is leader / candidate / follower,
`TERM` is the election round, `COMMIT` is how many entries are agreed and safe,
`APPLIED` is how many have been executed into the key-value store, `LOG` is the
total entries stored, `KEYS` is the size of the key-value store, `RTT` is the
network round trip from your machine, and `STATE` is `healthy`, `killed` (someone
typed `kill`) or `unreachable` (process down, or the network dropped).

**Replication** — the leader's own view of every follower: how many entries each
one has confirmed, how far behind it is, and a progress bar. This is where you
watch a restarted node catch up in real time.

**Events** — a running feed of what actually happened: elections, term changes,
nodes going down, nodes coming back. This is the fastest way to understand a
failure after the fact.

**Bottom line** — is there a leader, how many nodes are reachable, is there a
majority (`quorum`), do all nodes agree on the term, and have all logs converged.

### Reading it: what healthy vs. broken looks like

| What you see | What it means |
|---|---|
| Exactly one `LEADER`, everyone in sync, quorum yes | Normal. |
| Roles flipping to `CANDIDATE`, term climbing fast | Nodes can't reach each other — check the network/firewall. |
| `leader: none`, quorum no | Too many nodes down; writes are refused until a majority returns. This is Raft protecting your data. |
| One node `unreachable`, others fine | That machine is down; the rest carry on. It will catch up on restart. |
| A follower `N behind`, count shrinking | It is catching up right now. |
| RTT jumping to hundreds of ms | Slow Wi-Fi. Restart every node with `--profile wifi` (section 8). |
| Log full of `NOOP` entries, terms in the hundreds or thousands | Constant re-elections. Almost always slow Wi-Fi or a sleeping laptop — use `--profile wifi`. |
| Dashboard says `timings differ between nodes` | One node was started with a different profile. Restart them all the same way. |

### Checking a single node without the dashboard

```bash
curl 192.168.1.11:6001            # on Windows PowerShell use: curl.exe 192.168.1.11:6001
```

```json
{"node_id":2,"role":"LEADER","term":2,"commit_index":32,"last_applied":32,
 "log_size":32,"keys":30,"is_alive":true,"peers":[{"peer_id":1,"match_index":32}]}
```

Or type `status` in any node's shell for the same information locally.

---

## 4. Sending requests from another laptop (the client)

The client machine does **not** need to run a node, and does not even need the
C++ build — only Python 3 and the list of node addresses.

Copy `scripts/client.py` and `cluster.conf` to the client laptop, then:

```bash
python3 client.py set name ajay          # one command and exit
python3 client.py get name
python3 client.py delete name
python3 client.py                        # interactive shell
```

If you did not copy `cluster.conf`, pass the addresses directly:

```bash
python3 client.py --nodes 192.168.1.103:5001,192.168.1.101:5001,192.168.1.104:5001 get name
```

Inside the interactive shell:

```
client> set city mumbai
OK
client> get city
mumbai
client> status
NODE                  ROLE        TERM  COMMIT   LOG  KEYS  STATE
192.168.1.103:5001    FOLLOWER       2       8     8     3  healthy
192.168.1.101:5001    LEADER         2       8     8     3  healthy
192.168.1.104:5001    FOLLOWER       2       8     8     3  healthy
talking to 192.168.1.101:5001
client> bench 500
500 writes in 1.12s -> 446 writes/sec, 0 failed
client> exit
```

**You never have to know which node is the leader.** The client sends to any
node; a follower answers with a redirect naming the leader, the client follows
it and remembers it for later commands. When the leader crashes mid-session the
client retries for a few seconds, discovers the new leader and carries on — try
it by killing the leader while a `bench` is running.

If you already built the C++ binary on the client machine, it can act as a
client too, with no local node and no ports to bind:

```bash
./raftkv --client --config cluster.conf
```

---

## 5. The log file — the replicated log itself

There are two separate outputs, and they hold different things:

| Where | What it holds | Default |
|---|---|---|
| `data/node<id>.log` | The Raft log: every entry as it is appended and again as it commits | always on |
| console (`--debug`) | Everything else: elections, votes, peers going down, redirects | off |

**Every node writes its log file automatically**, one line per change, flushed
immediately, so you can watch it from another terminal while you work:

```bash
tail -f data/node1.log
```

Change the path with `--log <file>` or switch it off with `--no-log`.

```
=== node 2 log opened 22:17:22.013 ===
time          event     index  term  command
22:17:22.706  APPEND        1     1  NOOP -
22:17:22.716  COMMIT        1     1  NOOP -
22:17:25.055  APPEND        2     1  SET city mumbai
22:17:25.060  COMMIT        2     1  SET city mumbai
22:17:25.080  APPEND        3     1  SET lang cpp
22:17:25.083  COMMIT        3     1  SET lang cpp
```

Three events, and nothing else:

| Event | Meaning |
|---|---|
| `APPEND` | The entry is now in this node's log — put there by a client on the leader, or by AppendEntries on a follower. Stored, but not yet safe. |
| `COMMIT` | A majority stores this entry, so it can never be undone. This is the moment the write really counts. |
| `DROP` | This entry disagreed with the leader and was discarded, along with everything after it. Normal after a node rejoins with a stale log. |

The same file on a **follower** shows the one-round-trip lag between the two,
which is exactly what replication looks like:

```
22:17:25.057  APPEND        2     1  SET city mumbai
22:17:25.082  APPEND        3     1  SET lang cpp
22:17:25.082  COMMIT        2     1  SET city mumbai
22:17:25.101  COMMIT        3     1  SET lang cpp
```

A node that rejoins with entries the leader never had:

```
22:18:55.280  DROP          1     1  conflicted with leader, discarded from here
22:18:55.281  APPEND        1     5  NOOP -
22:18:55.281  APPEND        2     5  SET newer yes
22:18:55.385  COMMIT        1     5  NOOP -
```

Comparing these files across machines is the quickest way to check the cluster
agrees: for a given index, the term and command must be identical everywhere.

---

## 6. Debug mode — everything the log file leaves out

Elections, votes, heartbeats and failures go to the console, not the log file.
Start a node with `--debug` (or `--debug-all` to include heartbeats), or type
`debug on` / `debug off` / `debug all` in its shell:

```
21:59:37.460 n1 ELECT t1 silent=835ms>834ms
21:59:37.467 n1 SEND RV->n2 t1 log=0/0
21:59:37.479 n1 RECV RVR<-n2 t1 granted
21:59:37.479 n1 VOTES 2/3 need 2
21:59:37.480 n1 LEADER t1 log=0
21:59:39.649 n1 CLIENT 192.168.1.99:37892 {SET city mumbai}
21:59:39.653 n1 ACK n2 match=2 next=3
21:59:39.653 n1 COMMIT 1->2 t1 on 2/3
21:59:39.653 n1 APPLY i2 SET city mumbai = OK
21:59:39.780 n1 DOWN n2 no-reply x1
21:59:41.181 n1 QUORUM LOST 1/3 need 2
21:59:41.184 n1 STEPDOWN LEADER->F t1->1 lost contact with a majority
```

On a follower:

```
21:59:37.472 n2 RECV RV<-n1 (192.168.1.103:60206) t1 log=0/0
21:59:37.475 n2 VOTE n1 t1 YES log-ok 0/0>=0/0, no prior vote
21:59:37.484 n2 RECV AE<-n1 (192.168.1.103:60206) t1 prev=0/0 e=1 lc=0
21:59:37.485 n2 AE OK log=1
```

| Line | Meaning |
|---|---|
| `ELECT t5 silent=835ms>834ms` | No leader contact for 835 ms, past this node's random limit, so it stands for term 5. |
| `VOTE n3 t5 YES log-ok 4/45>=4/45` | Granted a vote: same term, no earlier vote, and its log is at least as current. |
| `VOTE n3 t5 NO stale-term 3<5` | Refused: the candidate is behind on terms. |
| `VOTE n3 t5 NO already-voted n1 in t5` | Refused: one vote per term, already spent. |
| `VOTE n3 t5 NO log-behind 4/40<4/45` | Refused: the election restriction — its log could be missing committed entries. |
| `AE NO gap prev=40 mine=37 -> conflict=38` | The leader assumed entries this node does not have; resend from 38. |
| `AE NO term-mismatch i40 mine=t3 want=t4 -> conflict=35` | Entry 40 disagrees; rewind past that whole term. |
| `NAK n2 mismatch, next 41->35` | The leader acting on that reply — one term skipped per round trip. |
| `QUORUM LOST 1/3 need 2` | This leader can no longer see a majority; it steps down and stops accepting writes. |

Notation: `t5` is term 5, `i2` is log index 2, `log=4/45` is term 4 / index 45,
`prev=1/1` is prevLogIndex/prevLogTerm, `e=1` entries, `lc=1` leaderCommit.
Every inbound line names the sender by node id **and** real `ip:port`, so on a
multi-laptop run you can see which machine is talking.

---

## 7. What happens when there is no majority

Raft trades availability for consistency: with fewer than a majority of nodes
running, writes **must** be refused, because nothing can be safely committed.
The cluster tells you so immediately instead of leaving you waiting.

```
$ python3 client.py set a 1
error: no leader elected (cluster needs 2 of 3 nodes running)
```

Three checks make this fast:

- A **leader that loses contact with a majority steps down** within about 1.5
  seconds (`QUORUM LOST` in the log). A leader nobody follows is useless, and
  keeping the title would only make clients wait for commits that can never
  happen.
- **Writes are refused before being appended**, so a doomed request never enters
  the log and never has to be cleaned up later.
- **Clients stop retrying.** `client.py` keeps trying for 5 seconds by default,
  long enough to ride out a normal election, then reports the node's own reason.
  Tune it with `--wait <seconds>`.

You will see the term climbing on a node that cannot reach a majority. That is
normal — it is the retry counter for elections it keeps losing, and it stops as
soon as enough nodes are back.

---

## 8. Timings — use `--profile wifi` on wireless

The defaults assume a wired LAN: a heartbeat every 100 ms, and a follower that
hears nothing for 600–1200 ms decides the leader is gone and calls an election.

On Wi-Fi, round trips of 100 ms or more are common. Heartbeats then arrive late,
followers campaign against a perfectly healthy leader, the winner gets deposed a
second later, and the term counter climbs into the thousands while almost no
work gets done. If your log is full of `NOOP` entries with jumping terms, this
is why.

**Start every node with the same profile:**

```bash
./raftkv --id 1 --config cluster.conf --data-dir data --profile wifi
```

| Profile | Heartbeat | Election timeout | Use when |
|---|---|---|---|
| `lan` (default) | 100 ms | 600–1200 ms | Loopback, wired ethernet, RTT under ~20 ms |
| `wifi` | 250 ms | 1500–3000 ms | Normal home or campus Wi-Fi |
| `slow` | 500 ms | 3000–6000 ms | Phone hotspot, congested or flaky Wi-Fi |

Check the `RTT` column in the dashboard to pick one: if it is regularly above
about a fifth of the heartbeat, move to the next profile up. A slower profile
costs you failover speed — with `wifi` a dead leader is replaced in about three
seconds instead of one — and nothing else.

**All nodes must agree.** A fast node will keep unseating a slow one. The
dashboard prints the timings it sees and turns red if they differ, and each node
prints its own settings at startup.

### Tuning individual values

```bash
./raftkv --id 1 --config cluster.conf --profile wifi --election-max 4000
```

| Flag | What it controls |
|---|---|
| `--heartbeat` | How often the leader contacts each follower. |
| `--election-min` / `--election-max` | How long a follower waits before campaigning. The gap between them is what breaks split votes — keep it wide. |
| `--rpc-timeout` | How long to wait for a reply before treating a node as unreachable. |
| `--quorum-window` | How long a leader may go without hearing from a majority before it steps down. |
| `--client-timeout` | How long a client write waits for its commit. |

Bad combinations break the cluster, so unsafe values are raised automatically
and the reason is printed:

```
$ ./raftkv --id 1 --heartbeat 1000 --election-min 300
note: rpc-timeout raised to 1000ms (cannot be shorter than a heartbeat)
note: election-min raised to 3000ms (must allow several heartbeats to be missed before campaigning)
note: election-max raised to 4500ms (needs a wide enough range to break split votes)
```

The rule behind all of them: **heartbeat ≪ election timeout ≪ time you are
willing to wait for failover.** A follower must never time out while heartbeats
are still arriving.

The helper scripts take the same flag: `./scripts/local_cluster.sh --profile wifi`
and `python3 scripts/chaos_test.py --profile wifi`.

---

## Command reference

```bash
./raftkv --id <n> [--config cluster.conf] [--data-dir data] [--no-shell] [--debug]
./raftkv --client [--config cluster.conf]
python3 scripts/client.py [--config cluster.conf | --nodes ip:port,...] [command...]
python3 scripts/dashboard.py [cluster.conf]
python3 scripts/chaos_test.py [--writes 100]
```

| Flag | Meaning |
|---|---|
| `--id` | Which node in `cluster.conf` this process is. Required to run a node. |
| `--config` | Path to the cluster file. Default `cluster.conf`. |
| `--data-dir` | Where to keep `raft_node_<id>.bin`. Default the current folder. |
| `--no-shell` | Run silently as a background service, no prompt. |
| `--debug` | Print elections, votes and failures on the console. |
| `--debug-all` | Same, plus the routine heartbeats. Very noisy. |
| `--log <file>` | Where to write the Raft log. Default `<data-dir>/node<id>.log`. |
| `--no-log` | Do not write the log file. |
| `--profile` | `lan` (default), `wifi` or `slow`. See section 8. |
| `--heartbeat`, `--election-min`, `--election-max`, `--rpc-timeout`, `--quorum-window`, `--client-timeout` | Individual timings in ms, override the profile. |
| `--client` | Do not run a node; just connect to the cluster as a client. |

`client.py` also takes `--wait <seconds>` (how long to keep retrying during an
election, default 5) and `--timeout <seconds>` (per network round trip).

Node shell: `set <key> <value>`, `get <key>`, `delete <key>`, `status`, `dump`,
`kill`, `revive`, `debug on|off|all`, `help`, `exit`.

Client shell: `set`, `get`, `delete`, `status`, `bench [n]`, `help`, `exit`.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `poll.h: No such file`, `'SIGPIPE' was not declared`, `'::fsync' has not been declared` | An older copy of the source that was POSIX-only. Get the current version — `include/platform.h` handles Windows now. |
| `undefined reference to WSAStartup / socket / inet_pton` | Compiled by hand without `-lws2_32`. Use `./build.sh` or `build.bat`. |
| `'thread' is not a member of 'std'` | Your g++ was built without pthread support. Install the MSYS2 MINGW64 toolchain: `pacman -S mingw-w64-x86_64-gcc`. |
| `'raftkv' is not recognized` | On Windows the file is `raftkv.exe`; in PowerShell run `.\raftkv.exe`. |
| `failed to bind raft port 5001` | Another copy is still running, or the firewall blocked it. Close the old process, or pick a different port in `cluster.conf` (on every machine). |
| Nodes never elect a leader across laptops | They cannot reach each other. Check the firewall step, that all IPs in `cluster.conf` are correct and identical everywhere, and that the network is Private, not Public. |
| Dashboard says `unreachable` but the node is running | The telemetry port (raft port + 1000) is blocked. Open it too. |
| Everything works locally but not across laptops | You are probably on a university or café Wi-Fi that isolates clients from each other. Use a phone hotspot or a direct cable to test. |
| `error: no leader elected (cluster needs 2 of 3 nodes running)` | Fewer than a majority are up. Start more nodes — see section 6. |
| `error: no quorum: 1 of 3 nodes reachable` | A leader exists but has lost contact with the others. Check the network on the missing machines. |
| Term keeps climbing on a node | It cannot reach a majority, so it keeps losing elections. Normal; it stops when the others return. |

---

## How it works, briefly

**Storage.** The term, the vote and the whole log are written to a temp file,
`fsync`'d, then renamed over the real file. Nothing is acknowledged to a client
before that completes, so an acknowledged write cannot be lost and the file can
never be found half-written.

**Elections.** Each node picks a fresh random timeout between 600 and 1200 ms
whenever it hears from a leader, which keeps two nodes from campaigning at the
same moment. A candidate only wins with votes from a majority, and a node only
votes for a candidate whose log is at least as up to date as its own.

**Replication.** The leader keeps one thread per follower, heartbeating every
100 ms and sending new entries immediately. When a follower's log disagrees, it
replies with the first index of the conflicting term so the leader can rewind a
whole term per round trip instead of one entry at a time.

**Commit rule.** The leader only counts replicas for entries from its *own* term;
older entries commit indirectly behind them. Each new leader appends a no-op
entry so this rule can never stall. This is what prevents a committed value from
ever being overwritten.

**Reads.** `get` goes through the log like a write, so you can never read a stale
value from a leader that has already been deposed.

**Availability.** A leader checks that a majority has answered it recently; if
not, it steps down and writes are refused with a reason rather than hanging.
This is Raft's deliberate trade — with no majority, the safe answer is "no".

**Known limits.** No log compaction or snapshots yet, so the log grows forever
and each save rewrites it — fine for thousands of entries, not for millions.
Cluster membership is fixed at startup. There is no pre-vote, so an isolated
node's term keeps rising while it is cut off.
