# Remote Notes

This codebase is path-stable enough to run on another machine, but the implementation does not require a special remote control layer.

The main assumptions are:

- model caches live outside the repo
- benchmark data lives under `data/processed/benchmark/`
- generated artifacts live under `outputs/`

Useful env vars:

- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `MODELSCOPE_CACHE`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `GEDIT_SECRET_ENV_PATH`

