# Paper-Matched Qwen-Image-Edit-2509 Baseline

This baseline is separated from the older DiffSynth smoke-test baseline. Use it when you want numbers that are defensible against the Qwen-Image-Edit-2509 model card and GEdit paper protocol.

## What changed

- Inference backend: official Diffusers `QwenImageEditPlusPipeline`.
- Model: `Qwen/Qwen-Image-Edit-2509`.
- Precision: `bfloat16`.
- Generation settings: `num_inference_steps=40`, `true_cfg_scale=4.0`, `guidance_scale=1.0`, `negative_prompt=" "`, `num_images_per_prompt=1`.
- Resolution handling: do not force benchmark images back to their original raw resolution; let the official pipeline use its own 2509 resize path.
- GEdit evaluator: upstream `--backbone gpt4o`, which the current GEdit `VIEScore` code maps to OpenAI `gpt-4.1`.
- Output name: `qwen_edit_2509_official_diffusers`, so old `qwen_edit_2509_base` DiffSynth outputs are not mixed with paper-matched outputs.

## Run GEdit

```bash
conda activate /share_6/users/ritesh_thawkar/condaenvs/qedit
cd ~/self-evolve-image-editing
git pull

export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$PWD/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

bash scripts/export_gedit.sh --device cuda
```

The export is resumable. If it is interrupted, rerun the same command. The exporter refuses to reuse an existing output directory if its recorded provenance does not match the current paper-matched settings.

```bash
find outputs/benchmark_images/gedit/qwen_edit_2509_official_diffusers/fullset -name "*.png" | wc -l
```

Expected full GEdit count: `1212`.

For scoring, the secret file must contain the raw key only, not `OPENAI_API_KEY=...`.

```bash
cat > gedit_secret.env <<'EOF'
sk-proj-your-key-here
EOF
chmod 600 gedit_secret.env

rm -rf outputs/scores/gedit/qwen_edit_2509_official_diffusers
rm -f outputs/scores/gedit/qwen_edit_2509_official_diffusers_summary.json

bash scripts/score_gedit.sh \
  --set scoring.scorer_secret_env_path=gedit_secret.env
```

## Run ImgEdit

```bash
bash scripts/export_imgedit.sh --device cuda
bash scripts/score_imgedit.sh
```

## Legacy DiffSynth Baseline

The old `qwen_edit_2509_base` outputs were generated through DiffSynth's `QwenImagePipeline` wrapper and are not paper-matched. Keep them only as pipeline debugging artifacts. Do not report those numbers as the Qwen paper baseline.
