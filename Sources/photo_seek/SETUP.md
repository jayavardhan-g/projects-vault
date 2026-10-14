# Setup

## Environment

Python 3.11 in a dedicated conda environment, created and managed with `mamba`:

```bash
mamba create --name photo-seek python=3.11 -y
mamba activate photo-seek
# ... work ...
mamba deactivate
```

The environment is created by name, not by prefix path; where it physically
lands is decided by `envs_dirs` in `~/.condarc`.

| | |
|---|---|
| Environment name | `photo-seek` |
| Path (dev machine) | `/data/jayavardhan/tools/conda-envs/photo-seek` |
| Python | 3.11.16 |
| Created | 2026-08-25 |

## Dependencies

Two tiers. The core tier is all the currently implemented code needs:

```bash
mamba install -y pyyaml pillow pytest
pip install -e .
```

Versions installed on the dev machine, for reproducibility:

| Package | Version | Source |
|---|---|---|
| Python | 3.11.16 | conda-forge |
| PyYAML | 6.0.3 | conda-forge |
| Pillow | 12.3.0 | conda-forge |
| pytest | 9.1.1 | conda-forge |
| photoseek | 0.1.0 | `pip install -e .` |

`pkgs_dirs` in `~/.condarc` already keeps the conda package cache on warm
storage. Point pip's cache there too before any pip install:

```bash
export PIP_CACHE_DIR=/mnt/warm_storage/jayavardhan/.cache/pip
```

The model tier is required for embedding, indexing, and face work. It pulls in
several GB, so install it only when you are ready to run those pipelines:

```bash
pip install -r requirements-models.txt
```

## Model cache

The default cache locations sit in `$HOME` and fill up fast. Point them
somewhere with room before the first model download:

```bash
export HF_HOME=/mnt/warm_storage/jayavardhan/.cache/huggingface
export TORCH_HOME=/mnt/warm_storage/jayavardhan/.cache/torch
export INSIGHTFACE_HOME=/mnt/warm_storage/jayavardhan/.cache/insightface
```

## Local LLM (query router)

The LLM router talks to a local [Ollama](https://ollama.com) server over HTTP.
It is optional — without it, search falls back to the deterministic rule-based
router.

```bash
ollama serve &            # defaults to http://localhost:11434
ollama pull qwen2.5:7b
```

Then point the config at it:

```yaml
search:
  router: ollama
  ollama_model: qwen2.5:7b
  ollama_host: http://localhost:11434
```

## GPU

Embedding and face detection want a GPU but do not require one. On a shared
machine, check what is actually free first and pin the device explicitly:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
export CUDA_VISIBLE_DEVICES=4
```

## Verify

```bash
pytest -q
photoseek --help
photoseek search "photos of mom at the beach" --dry-run
```

The last command exercises the router end to end and needs no models.
