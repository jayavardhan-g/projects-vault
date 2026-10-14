# Setup

## Environment

| | |
|---|---|
| Conda env name | `vllmmini` |
| Env path | `/mnt/warm_storage/jayavardhan/tools/conda-envs/vllmmini` |
| Python | 3.11 |
| CUDA (driver) | 13.0, driver 580.173.02 |
| PyTorch wheels | `cu128` index |
| GPU | 1× NVIDIA H200 (143 GB) — see "Picking a GPU" |

### Recreating it

```bash
source /mnt/warm_storage/jayavardhan/tools/miniforge3/etc/profile.d/conda.sh

# NOTE: created with -p, not -n. See "Why not the default env dir" below.
mamba create -y -p /mnt/warm_storage/jayavardhan/tools/conda-envs/vllmmini python=3.11
conda activate /mnt/warm_storage/jayavardhan/tools/conda-envs/vllmmini

# See "Machine gotchas" — these two exports are required on this host.
export TMPDIR=/mnt/warm_storage/jayavardhan/tmp
export PIP_CACHE_DIR=/mnt/warm_storage/jayavardhan/tools/pip-cache

pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
```

Triton ships bundled with the PyTorch Linux wheels — it is not installed
separately. Confirmed in Phase 5: **triton 3.6.0** with torch 2.11.0+cu128, no
extra install needed for the fused decode kernel.

## Where things live

The repo stays on `/data` (it is small and belongs in the project tree);
everything bulky or transient goes to warm storage. `scripts/env.sh` sets this
up — source it before installing packages or running benchmarks:

```bash
source scripts/env.sh
```

| | location | size |
|---|---|---|
| repo source, tests, `results/` JSON | `/data/.../vLLMmini` | ~600 KB |
| conda env | warm storage | 7.1 GB |
| pip cache (`PIP_CACHE_DIR`) | warm storage | 13 GB |
| model weights (`HF_HOME`) | warm storage | ~2.5 GB |
| build scratch (`TMPDIR`) | warm storage | transient |
| profiles, large run artifacts (`MINI_VLLM_ARTIFACTS`) | warm storage | Phase 5+ |
| Triton kernel cache | `/data` — **intentionally local** | 13 MB |

The Triton cache is the one deliberate exception: many small files read on
every kernel launch, which suits local disk better than NFS.

## Machine gotchas

Two host-specific constraints that will otherwise bite you, both discovered the
hard way during Phase 0 setup:

1. **`/tmp` is on a full root filesystem.** `/` is at 100% (a few hundred MB
   free of 1.8 TB) and is shared with other users. pip unpacks the ~2.5 GB
   torch wheel into `$TMPDIR`, so a default install fails with
   `OSError: [Errno 28] No space left on device`. Always export `TMPDIR` to
   warm storage before installing. Do not clear `/tmp` — those are other
   users' files.

2. **`/data/jayavardhan` has a 100 GB quota** (~18 GB free). The conda default
   `envs_dirs` points there, and a PyTorch env is 10–20 GB, so the env lives on
   `/mnt/warm_storage` (151 TB free) instead — which is also where `HF_HOME`
   already points. Project *source* stays on `/data`; only the env, caches, and
   large run artifacts go to warm storage.

3. **PyTorch's default thread count is catastrophic for decode here.** This box
   has 224 cores, so torch defaults `num_threads` to 112. Measured on
   Llama-3.2-1B / H200:

   | threads | prefill (387 tok) | decode |
   |---|---|---|
   | 112 (default) | 10.47 ms | **315.27 ms/step** |
   | 16 | 10.49 ms | 9.63 ms/step |
   | 8 | 10.49 ms | 9.65 ms/step |
   | 1 | 10.55 ms | 9.54 ms/step |

   A 33× difference on decode and none on prefill: single-token decode is a
   chain of tiny CPU-side ops, and opening an OpenMP region across 112
   contended threads costs far more than the work. Everything in this project
   calls `mini_vllm.runtime.configure_torch_runtime()` (default 8 threads) so
   the baseline and later phases are measured under identical conditions. Left
   at the default, every speedup this project reports would really be a
   measurement of this misconfiguration.

## transformers 5.x differences worth knowing

This project pins no transformers version, and 5.x moved things that 4.x code
reads directly. Found the hard way in Phase 2:

* **`config.rope_theta` no longer exists.** RoPE settings are consolidated into
  `config.rope_parameters`, a dict that contains `rope_theta` alongside the
  scaling fields. `config.rope_scaling` still works as an alias returning the
  same dict. Reading the old attribute with a `getattr` default silently
  yielded 10000 instead of Llama 3.2's 500000 — a 50x error in the base
  frequency, which leaves short prompts looking correct while long contexts
  degrade. `mini_vllm.model.llama.resolve_rope_config` handles both layouts and
  **raises rather than defaulting**, on purpose.

* **`from_pretrained` takes `dtype=`, not `torch_dtype=`.**

The general lesson: when reading model config fields, prefer raising over
defaulting. A wrong hyperparameter produces plausible output, which is far
harder to notice than a crash.

## Model weights

Phase 0 develops against `meta-llama/Llama-3.2-1B-Instruct` (gated). The host's
`HF_HOME` (`/mnt/warm_storage/jayavardhan/tools/huggingface`) already holds a
token with Meta gated access. If a download 403s, re-accept the license on the
model's HF page.

## Picking a GPU

This is a shared 8×H200 box. Check before launching, and set the device
explicitly rather than defaulting to GPU 0:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
CUDA_VISIBLE_DEVICES=<free-index> python benchmarks/baseline_hf.py
```

## Running

`/mnt/warm_storage` is NFS and conda's shell hook is slow over it. Calling the
interpreter directly is usually faster than activating:

```bash
PY=/mnt/warm_storage/jayavardhan/tools/conda-envs/vllmmini/bin/python

# CPU-only unit tests. Both halves matter: a bare `pytest` collects the
# gpu-marked tests too and seizes whatever CUDA_VISIBLE_DEVICES points at,
# which OOMs if that card is already busy.
CUDA_VISIBLE_DEVICES="" $PY -m pytest -m "not gpu" -q

# GPU tests and benchmarks — pick a free card explicitly, and re-check
# immediately before launching. See "Picking a GPU" above.
CUDA_VISIBLE_DEVICES=7 $PY -m pytest -m gpu -q
CUDA_VISIBLE_DEVICES=7 $PY -u benchmarks/serving.py --repeats 5

# The HTTP server. Expect ~5 minutes before the port opens: importing torch and
# loading weights over NFS dominates, and the startup warmup runs after that.
# Poll /health rather than guessing.
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src $PY -u -m mini_vllm.server --port 8000
```

`PYTHONPATH=src` is only needed if the package is not installed in the env; with
`pip install -e .` the module is importable as it stands. Run one process, not
several workers: `configure_torch_runtime()` is applied in the parent before the
model is built, and a spawned worker would inherit neither it nor the weights
already resident on the card.

Use `python -u` and do not pipe a long run through `tail` or `grep`: both
block-buffer to a non-terminal, so nothing appears until the process exits and
a hang becomes indistinguishable from slow progress.

## Real vLLM, as an upper-bound reference

| | |
|---|---|
| Conda env name | `vllm-ref` |
| Env path | `/mnt/warm_storage/jayavardhan/tools/conda-envs/vllm-ref` |
| Python | 3.11 |
| vLLM | 0.26.0 (pins `torch==2.11.0`, the same torch this project uses) |

```bash
source /mnt/warm_storage/jayavardhan/tools/miniforge3/etc/profile.d/conda.sh
export TMPDIR=/mnt/warm_storage/jayavardhan/tmp
export PIP_CACHE_DIR=/mnt/warm_storage/jayavardhan/tools/pip-cache

mamba create -y -p /mnt/warm_storage/jayavardhan/tools/conda-envs/vllm-ref python=3.11
/mnt/warm_storage/jayavardhan/tools/conda-envs/vllm-ref/bin/pip install "vllm==0.26.0" ninja
```

**`ninja` is not optional and vLLM does not declare it.** flashinfer JIT-compiles
its top-k/top-p sampler during engine start-up; without ninja on PATH that fails
with `FileNotFoundError: [Errno 2] No such file or directory: 'ninja'`, and what
surfaces is `RuntimeError: Engine core initialization failed` several hundred
traceback lines later. Installing it into the env is not enough on its own —
put that env's `bin` on `PATH` when launching, since the JIT shells out by name.

One more failure worth recognising on this box: vLLM's memory profiler asserts
that free GPU memory has not *grown* during init, so a neighbour releasing a
large allocation at the wrong moment aborts start-up with `Error in memory
profiling. Initial free memory 31.53 GiB, current free memory 130.71 GiB`. It is
not a misconfiguration — relaunch.

It is a **separate** env on purpose: `vllmmini` is the environment every number
in the README was measured in, and vLLM brings a large dependency set that would
invalidate those benchmarks silently rather than loudly.

Running the comparison — both servers stay up on one card, and
`benchmarks/vllm_reference.py` alternates rounds between them:

```bash
VLLM_PY=/mnt/warm_storage/jayavardhan/tools/conda-envs/vllm-ref/bin/python

CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src $PY -u -m mini_vllm.server --port 8000
CUDA_VISIBLE_DEVICES=7 $VLLM_PY -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.2-1B-Instruct --port 8001 \
    --gpu-memory-utilization 0.15          # leave room for ours on the same card

$PY -u benchmarks/vllm_reference.py --repeats 5
```

### Which vLLM versions can coexist with this torch

Checked in Phase 6, recorded so nobody has to work it out again. vLLM pins torch
exactly, so only some versions can coexist with this build:

| vLLM | torch pin | |
|---|---|---|
| 0.27.x | `torch==2.13.0` | incompatible |
| **0.26.0** | **`torch==2.11.0`** | matches this env |
| 0.25.1 | `torch==2.11.0` | matches this env |

Even so, install it into a **separate** env rather than `vllmmini`. It brings a
large dependency set of its own, and `vllmmini` is the environment every number
in the README was measured in — an accidental upgrade there invalidates the
benchmarks silently rather than loudly.
