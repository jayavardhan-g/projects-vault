---
type: lesson
tags: [lesson, infrastructure]
---
# Infrastructure traps on this server

- **Model loading from NFS (warm storage) through mmap can hang.** bge-m3 had not finished loading after 280 s from the HF cache on
  NFS, and it loaded in 30 s from `/data`. Mirror the weights to local disk first. → [[LongContext RAG - retrieval-head-guided compression]]
  (The thesis hit the same thing with gpt-oss-120b; see `/data/jayavardhan/mtp/vault`.)
- **`ncu` doesn't work here.** It fails with `ERR_NVGPUCTRPERM`, because performance counters need root. Use the CUDA occupancy API for theoretical numbers,
  or ask an admin to set `NVreg_RestrictProfilingToAdminUsers=0`. → [[CUDA kernels - CS794 sysml]]
- **vLLM start-up traps.** BountyBrain's SETUP.md lists four: `ninja` must be on PATH, a flashinfer/Python 3.11 patch, ~15 min to first
  token, and more. mini-vLLM's SETUP.md lists two ways vLLM refuses to start. vLLM 0.26.0 pins torch 2.11.0.
- **The root filesystem is full.** Set `TMPDIR` to a folder you own before builds (BountyBrain PROGRESS).
- **Running the whole pytest suite on a shared GPU** collects the GPU-marked tests and grabs whatever device is visible. Use
  `pytest -m "not gpu"`, or set `CUDA_VISIBLE_DEVICES` explicitly.
- **Co-tenancy skews batch-1 numbers**, because batch-1 decode is CPU-bound. See [[Measurement traps]].
- **Path-based envs** (`-p /mnt/warm_storage/...`) are what LCRAG and mini-vLLM's SETUP docs still say to use. That instruction is stale:
  create envs by name. See [[Environments]].
