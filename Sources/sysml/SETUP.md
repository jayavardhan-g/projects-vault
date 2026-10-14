# Setup

## Environment

Conda env **`sysml`**, Python **3.11**, created under the miniforge3 install at
`/mnt/warm_storage/jayavardhan/tools/miniforge3`.

```bash
source /mnt/warm_storage/jayavardhan/tools/miniforge3/etc/profile.d/conda.sh
mamba create -n sysml python=3.11 -y
conda activate sysml

# pip unpacks ~1.5 GB of wheels through TMPDIR. The root filesystem on this
# machine is 100% full, so redirect both TMPDIR and the pip cache onto
# warm_storage or the install fails with "No space left on device".
export TMPDIR=/mnt/warm_storage/jayavardhan/harshith/.cache/pip-tmp
export PIP_CACHE_DIR=/mnt/warm_storage/jayavardhan/harshith/.cache/pip
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

pip install torch numpy matplotlib transformers tiktoken
```

Installed: PyTorch 2.13.0+cu130, matplotlib, transformers, tiktoken.

For PA1 only, keep the HuggingFace cache off the root filesystem too:

```bash
export HF_HOME=/mnt/warm_storage/jayavardhan/harshith/.cache/huggingface
```

## Toolchain notes

* **CUDA 13.0** (`/usr/local/cuda/bin/nvcc`), **gcc 13.3** (12 and 13 available).
* The provided quiz scripts hardcode `-ccbin=/usr/bin/gcc-11`, **which does not
  exist on this machine**. Those are grader-side files and are left untouched;
  `tools/run_quiz.sh` reproduces their steps without the `-ccbin` pin.
* `cudaDeviceProp::clockRate` was removed in CUDA 13 — use
  `cudaDeviceGetAttribute(..., cudaDevAttrClockRate, ...)`.
* **`ncu` does not work here.** It fails with `ERR_NVGPUCTRPERM`; GPU
  performance counters require root. PA4's occupancy figures therefore come
  from the CUDA occupancy API and are theoretical, not measured. Enabling the
  profiler needs an admin to set `NVreg_RestrictProfilingToAdminUsers=0`.

## GPU selection — shared machine

Other people's jobs run on this box. Always check before launching and pin
explicitly; never assume a GPU is free:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
CUDA_VISIBLE_DEVICES=<idle index> ./build/pa2_bench ...
```

## Build and run

```bash
# lab quizzes (all seven must pass; --sanitize adds compute-sanitizer)
bash tools/run_quiz.sh 2 1        # ... 2 2, 2 3, 3 1, 3 2, 3 3, 3 4
bash tools/run_quiz.sh 3 4 --sanitize

# PA2 — matrix multiplication
nvcc -std=c++17 -O3 -arch=sm_90 -Xcompiler "-O3 -march=native" -lcublas \
     -o build/pa2_bench pa2_matmul/matmul_bench.cu
CUDA_VISIBLE_DEVICES=7 ./build/pa2_bench --cpu-max 4096 --out results/pa2_matmul.csv

# PA3 — MLP, streams, checkpointing
nvcc -std=c++17 -O3 -arch=sm_90 -Xcompiler -O3 -o build/pa3_bench pa3_mlp/mlp_bench.cu
CUDA_VISIBLE_DEVICES=7 ./build/pa3_bench --out results/pa3_mlp.csv --p2-iters 100

# PA4 — FlashAttention
nvcc -std=c++17 -O3 -arch=sm_90 --compiler-options -fPIC -shared \
     -o build/libflash.so pa4_flash/flash_lib.cu
CUDA_VISIBLE_DEVICES=5 python pa4_flash/bench_attention.py --iters 10 --budget-gb 16

# figures
python tools/plot_all.py
```

`--cpu-max 4096` caps the single-threaded CPU baseline; N = 32768 on the CPU
would take roughly 34 hours, so larger sizes are extrapolated as N³ and labelled
as such.

`--budget-gb 16` caps how much of the N×N score tensor the explicit PyTorch
attention path materialises at once. At N = 8192 that tensor is 34 GB, so the
run is chunked over the batch; the CSV records the chunk size used.
