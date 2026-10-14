# Setup

## Environment

| | |
|---|---|
| Conda env name | `bountybrain` |
| Python | 3.11.15 |
| Package manager | `mamba` / `pip` (miniforge3) |
| Env path | `$CONDA_PREFIX/envs/bountybrain` |

Created with:

```bash
mamba create -n bountybrain python=3.11 -y
conda activate bountybrain
pip install vllm                 # pins torch; install first
pip install -r requirements.txt
```

Key pinned versions actually installed:

| package | version |
|---|---|
| torch | 2.13.0+cu130 |
| vllm | 0.27.1 |
| transformers | 5.15.0 |
| sentence-transformers | (see `pip freeze`) |
| sqlalchemy | 2.x |
| networkx | 3.6.1 |
| numpy | 2.3.5 |

## Shared-machine notes

This was developed on a shared multi-user GPU server, which imposed two
constraints worth recording:

**Redirect temp.** The root filesystem was full and `/tmp` lives on it, so
pip's wheel extraction failed with `No space left on device`. Export a temp
directory on a volume you own before installing:

```bash
export TMPDIR=/path/you/own/tmp
export PIP_CACHE_DIR=/path/you/own/.cache/pip
```

**Pick a free GPU explicitly.** Never assume GPU 0 is idle. Check first and
set `CUDA_VISIBLE_DEVICES` to what is actually free:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

The project uses at most two GPUs: one serves the LLM via vLLM, one runs
embeddings and the cross-encoder reranker. `serve_llm.py` picks the freest GPU
automatically and requests only 40% of it, rather than vLLM's 0.9 default —
which would reserve ~130 GB of a 143 GB card for a 14B model and lock out
everyone else for no benefit.

## Known environment issues

Four things bit during setup on this machine. All are environment problems,
not project bugs, but they cost hours to diagnose so they are recorded here.

**vLLM needs `--enforce-eager` on Python 3.11.** `flashinfer` (a vLLM
dependency) annotates with `array.array[int]`, which is only subscriptable on
Python 3.12+. vLLM imports it through its allreduce–RMS-norm fusion compile
pass, so startup dies with `TypeError: type 'array.array' is not subscriptable`
before the engine ever loads. The pass is useless for single-GPU inference, and
`--enforce-eager` skips the compilation that triggers the import. Python 3.12
would also fix it.

### flashinfer patch (Python 3.11 only)

`--enforce-eager` avoids the *compilation* pass that imports flashinfer, but
vLLM's **kernel warmup** imports it too, from an unguarded import at the top of
`vllm/model_executor/warmup/kernel_warmup.py`. Startup therefore still dies —
and it dies *after* ten minutes of weight loading, right at
`compile_or_warm_up_model`.

The actual defect is a single annotation in flashinfer evaluated at import
time:

```python
# flashinfer/comm/fd_exchange.py
def _fd_ancillary(fd: int) -> tuple[tuple[int, int, array.array[int]]]:
```

`array.array` only became subscriptable in Python 3.12. The one-line fix is to
make annotations lazy:

```bash
F=$CONDA_PREFIX/lib/python3.11/site-packages/flashinfer/comm/fd_exchange.py
cp "$F" "$F.orig"
sed -i '/^import array/i from __future__ import annotations\n' "$F"
```

**Using Python 3.12 avoids all of this** and is the better choice for a fresh
environment. The patch is documented because it unblocks an existing 3.11
environment without a 40-minute reinstall over network storage.


**Put compile caches on a local filesystem.** Triton's cache directory cannot
be created on a GlusterFS mount — `makedirs` fails with
`OSError: [Errno 61] No data available` partway through engine init, after ten
minutes of weight loading:

```bash
export TRITON_CACHE_DIR=/local/path/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/local/path/.cache/inductor
export VLLM_CACHE_ROOT=/local/path/.cache/vllm
```

**Log to local disk, not network storage.** NFS/GlusterFS write caching can
hide a process's output for minutes, which makes a healthy-but-slow startup
indistinguishable from a hang. Loading a 14B checkpoint from network storage
runs at roughly 80 seconds per shard — about 11 minutes total. That is normal
here; do not assume it has stalled.

**Keep the env's `bin/` on PATH when launching vLLM.** JIT kernel compilation
shells out to `ninja`, which pip installs as `$CONDA_PREFIX/bin/ninja`. Launch
vLLM from a bare environment (`env -i`, a bare `nohup`, most supervisors) and
it fails with `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`
— again *after* the full weight load, so it reads like a model problem rather
than a PATH one. `conda activate` handles this; explicit launches need
`PATH="$CONDA_PREFIX/bin:$PATH"`.

## Running the pipeline

```bash
python fetch_data.py                    # download all datasets (~1 GB, bulk only)
python build_graph.py --rebuild         # parse -> knowledge graph + metrics
python index_corpus.py                  # chunk + embed -> vector store
python serve_llm.py                     # start vLLM on a free GPU (optional)
python eval.py --all                    # run every evaluation suite
```

Steps 1–3 need no GPU and no LLM. Step 4's classification and
retrieval suites use the LLM if one is reachable and skip those arms
cleanly if not.

## Vector store

Qdrant runs **embedded** by default, writing to `./data/qdrant` — no server,
no port binding, which matters on a shared box. Set `QDRANT_URL` in `.env` to
use a standalone server instead.

## Knowledge graph store

SQLite by default (`./data/bountybrain.sqlite`). Set `DATABASE_URL` to point
at Postgres if you want concurrent writers.
