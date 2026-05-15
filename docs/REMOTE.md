# Remote Notes

This codebase is path-stable enough to run on another machine, but the implementation does not require a special remote control layer.

The main assumptions are:

- model caches live outside the repo
- benchmark data lives under `data/processed/benchmark/`
- generated artifacts live under `outputs/`
- source-pool data is generated under ignored `data/unlabeled/` paths

For one 128GB GPU on Slurm, use:

```bash
sbatch scripts/slurm/01_prepare_data.sbatch
sbatch scripts/slurm/02_self_evolve_delta_ranker.sbatch
```

The detailed data workflow is in [REMOTE_DATA_PIPELINE.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/REMOTE_DATA_PIPELINE.md).

Useful env vars:

- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `MODELSCOPE_CACHE`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `GEDIT_SECRET_ENV_PATH`
