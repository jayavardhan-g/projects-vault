# Setup

## Environment

| | |
|---|---|
| Conda env name | `speculative_decoding` |
| Python | 3.11.15 |
| Env path | `/data/jayavardhan/tools/conda-envs/speculative_decoding` |
| Created with | `mamba create -n speculative_decoding python=3.11 -y` |

```bash
mamba create -n speculative_decoding python=3.11 -y
conda activate speculative_decoding
pip install -e ".[dev]"
```

`pip install -e .` puts `specdec` on the path from `src/`; the tests import it
as an installed package.

## Model weights

The pool is declared in `configs/models.yaml`. Fetch it with:

```bash
python scripts/download_models.py
```

Roughly 50 GB for everything declared: the 14B default target, the 7B
comparison target, and three sub-2B drafts. On this machine `HF_HOME` already
points at `/mnt/warm_storage/jayavardhan/tools/huggingface`, which is where the
weights land — **not** the project directory, and not `/data`.

That mount reads at tens of MB/s, so a cold load of the 14B takes **20–40
minutes** and everything after it is fast (the page cache holds the weights).
Structure GPU work as one long-lived process that sweeps every configuration,
not as many short ones that each reload the pool — which is why
`scripts/demo.py` takes `--all-drafts` and `scripts/profile_overhead.py` does
its whole sweep in a single run.

## GPU

Single GPU, bf16, batch size 1. Target plus all three drafts stay resident on
one H200 simultaneously (~36 GB of weights, plus a preallocated KV cache per
model), which is what Phase 2's runtime switching needs — swapping weights per
request would cost more than speculation saves.

Pick a free GPU before running anything:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=<index>
```

## Running the tests

```bash
pytest -m "not gpu"         # fast: kernel + engine correctness on tiny models
pytest -m slow              # adds the statistical distribution tests
pytest -m hub               # tokenizer compatibility across the pool
pytest -m gpu               # real Qwen2.5 weights, needs a GPU
```

**Use `-m "not gpu"`, not a bare `pytest`.** Bare `pytest` collects the GPU
tests too. On a machine with no visible GPU they skip, which is why they look
harmless — but on this box they will try to load the 14B onto whatever device
is default, and if another user already holds that GPU the run spends sixteen
minutes before failing with `torch.OutOfMemoryError`. Hiding the device
(`CUDA_VISIBLE_DEVICES=""`) makes the CPU run unambiguous.

The CPU run needs neither a GPU nor any downloaded weights, with one exception:
the `hub`-marked tokenizer tests read tokenizer files from the HuggingFace
cache. The heavy correctness tests deliberately use tiny randomly-initialised
models so they can draw enough samples to have real statistical power.

`-m gpu` covers both engines: `test_real_models.py` checks the eager
guarantees, `test_compiled_correctness.py` the compiled ones. They share one
model load via session-scoped fixtures, so run them together rather than
separately.

## Benchmarking

```bash
python scripts/profile_overhead.py --gamma 4 --reps 3   # eager vs compiled, full breakdown
python scripts/demo.py --all-drafts --compile           # side-by-side, eyeballable output
```

Both compile the decode step, which takes a few minutes up front for a 48-layer
target and must happen *before* anything is timed. A cold first pass reports
the baseline at roughly a third of its steady-state throughput, which is more
than enough to invert the conclusion — so every timed path here warms up first,
and repeats round-robin rather than one configuration at a time.
