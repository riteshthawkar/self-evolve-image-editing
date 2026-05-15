# Setup

## Bootstrap

Clone the upstream repos into `third_party/`:

```bash
bash scripts/bootstrap.sh
```

This script:

- clones or updates the upstream benchmark and training repositories
- applies the minimal ImgEdit and OneIG patches if needed
- refreshes [third_party/LOCKFILE.md](/Users/ritesh.thawkar/Ritesh/neurips-project/third_party/LOCKFILE.md)

## Dataset and asset preparation

Follow [docs/DATASET_SETUP.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/DATASET_SETUP.md) for:

- supervised finetuning manifest setup
- unlabeled self-evolve image pool setup
- bounded remote source-pool download/filter/split setup
- ImgEdit local assets
- GenEval detector assets

For the Slurm single-GPU workflow, use [docs/REMOTE_DATA_PIPELINE.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/REMOTE_DATA_PIPELINE.md).

## Experiment operations

After environment and datasets are ready, use [docs/EXPERIMENTS.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/EXPERIMENTS.md) for:

- canonical train and evaluation commands
- output path expectations
- resume patterns for LoRA, full finetuning, and self-evolve
- rerun and re-score workflow

## Python package

Install the local package in editable mode:

```bash
pip install -e .
```

## Suggested environments

Training-oriented environment:

```bash
pip install -e ".[vlm]"
pip install -e third_party/diffsynth-studio
pip install accelerate transformers diffusers pillow datasets
```

For AMD GPUs, install a ROCm-enabled PyTorch build first using the official PyTorch selector.

Evaluation-oriented environment:

```bash
pip install -e .
pip install pillow tqdm datasets megfile numpy openai tenacity
```

## Environment variables

Copy from the examples in [configs/env](/Users/ritesh.thawkar/Ritesh/neurips-project/configs/env).

Key variables:

- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `MODELSCOPE_CACHE`
- `ROCR_VISIBLE_DEVICES`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `GEDIT_SECRET_ENV_PATH`

## ROCm notes

- PyTorch ROCm intentionally reuses the `torch.cuda` API surface, so `device="cuda"` remains valid on AMD systems.
- Linux ROCm docs recommend `ROCR_VISIBLE_DEVICES`; `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` are also accepted.
- The local export and validation entry points now default to `--device auto`, which resolves to `cuda` when a ROCm or CUDA PyTorch build is available.
- If your AMD GPU does not behave well with BF16, override the model dtype at runtime, for example: `--set model.torch_dtype=float16`
