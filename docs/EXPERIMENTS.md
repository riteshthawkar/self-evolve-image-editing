# Experiment Operations

This document is the run reference for actual experiments.

Environment bootstrap and dataset staging are intentionally not repeated here. Do those first using [SETUP.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/SETUP.md) and [DATASET_SETUP.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/DATASET_SETUP.md).

For the one-GPU Slurm workflow, use [REMOTE_DATA_PIPELINE.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/REMOTE_DATA_PIPELINE.md). It defines the bounded source download, open-VLM filtering, manifest splits, and Slurm job order.

## Preflight

Before launching a real run, validate the repo-local pipeline:

```bash
bash scripts/run_pipeline_smoke_tests.sh
```

This does not require the real benchmark assets or the full GPU environment. It checks:

- Python compilation
- shell syntax
- dry-run command composition for train and eval suites
- a lightweight `pillow-hybrid` self-evolve run on temporary images
- dry-run command composition for the generic self-reward and delta-ranker self-evolve variants

## Output Contract

Use stable names. The repo assumes:

- each trained checkpoint family has a stable checkpoint directory
- each evaluated model has a stable `model_name`
- each self-evolve run has a stable `output.root_dir`

### Training outputs

LoRA:

```text
outputs/checkpoints/Qwen-Image-Edit-2509_lora/
outputs/logs/train_lora_command.txt
outputs/logs/qwen_edit_2509_lora_<timestamp>.log
outputs/logs/qwen_edit_2509_lora_<timestamp>.json
```

Full finetuning:

```text
outputs/checkpoints/Qwen-Image-Edit-2509_full/
outputs/logs/train_full_command.txt
outputs/logs/qwen_edit_2509_full_<timestamp>.log
outputs/logs/qwen_edit_2509_full_<timestamp>.json
```

### Edit benchmark outputs

GEdit:

```text
outputs/benchmark_images/gedit/<model_name>/fullset/<task_type>/<language>/<key>.png
outputs/scores/gedit/<model_name>_summary.json
```

ImgEdit:

```text
outputs/benchmark_images/imgedit/<model_name>/<key>.png
outputs/scores/imgedit/<model_name>_average_score.json
outputs/scores/imgedit/<model_name>_typescore.json
outputs/scores/imgedit/<model_name>_summary.json
```

### Generation benchmark outputs

GenEval:

```text
outputs/benchmark_images/geneval/<model_name>/<prompt_id>/
outputs/scores/geneval/<model_name>_results.jsonl
outputs/scores/geneval/<model_name>_summary.json
```

DPG-Bench:

```text
outputs/benchmark_images/dpgbench/<model_name>/<item_id>.png
outputs/scores/dpgbench/<model_name>_results.txt
outputs/scores/dpgbench/<model_name>_summary.json
```

OneIG-Bench:

```text
outputs/benchmark_images/oneig/<mode>/<category>/<model_name>/<id>.webp
outputs/scores/oneig/<model_name>/<mode>/<timestamp>/
outputs/scores/oneig/<model_name>/<mode>/<timestamp>/<model_name>_summary.json
```

### Self-evolve outputs

```text
outputs/self_evolve/<run_name>/summary.json
outputs/self_evolve/<run_name>/round_01/proposals.jsonl
outputs/self_evolve/<run_name>/round_01/train_manifest.json
outputs/self_evolve/<run_name>/round_01/train_manifest.jsonl
outputs/self_evolve/<run_name>/round_01/summary.json
outputs/self_evolve/<run_name>/round_01/accepted/images/*.png
```

## Naming Rules

- Reuse the same checkpoint directory when resuming training.
- Reuse the same `model_name` when you want evaluation outputs to overwrite or refresh the same experiment.
- Use a new `model_name` when the checkpoint identity changed and you want separate benchmark trees.
- Use a new `output-prefix` or `output.root_dir` when you want a separate self-evolve run instead of continuing the same experimental branch.

## Baseline Training

### Fresh LoRA run

```bash
bash scripts/run_edit_model_suite.sh \
  --model-type lora \
  --train \
  --skip-export \
  --skip-score \
  --model-name qwen_edit_2509_lora_exp01
```

### Resume LoRA run from the latest checkpoint in the default LoRA directory

```bash
bash scripts/run_edit_model_suite.sh \
  --model-type lora \
  --train \
  --resume \
  --checkpoint-dir outputs/checkpoints/Qwen-Image-Edit-2509_lora \
  --skip-export \
  --skip-score \
  --model-name qwen_edit_2509_lora_exp01
```

### Fresh full finetuning run

```bash
bash scripts/run_edit_model_suite.sh \
  --model-type full \
  --train \
  --skip-export \
  --skip-score \
  --model-name qwen_edit_2509_full_exp01
```

### Resume full finetuning

The wrapper forwards raw upstream resume arguments through `--resume-arg`. This is deliberate because the exact resume CLI belongs to the installed DiffSynth commit.

Example pattern:

```bash
bash scripts/run_edit_model_suite.sh \
  --model-type full \
  --train \
  --resume \
  --checkpoint outputs/checkpoints/Qwen-Image-Edit-2509_full/<resume_ckpt>.safetensors \
  --resume-arg --resume_from_checkpoint \
  --resume-arg outputs/checkpoints/Qwen-Image-Edit-2509_full/<resume_ckpt>.safetensors \
  --skip-export \
  --skip-score \
  --model-name qwen_edit_2509_full_exp01
```

If the upstream training script uses a different resume flag, replace the `--resume-arg` values accordingly.

## Edit Evaluation

### Evaluate a trained LoRA checkpoint on GEdit and ImgEdit

```bash
bash scripts/run_edit_model_suite.sh \
  --model-type lora \
  --checkpoint outputs/checkpoints/Qwen-Image-Edit-2509_lora/<ckpt>.safetensors \
  --model-name qwen_edit_2509_lora_exp01 \
  --limit 64
```

### Evaluate a trained full checkpoint on GEdit and ImgEdit

```bash
bash scripts/run_edit_model_suite.sh \
  --model-type full \
  --checkpoint outputs/checkpoints/Qwen-Image-Edit-2509_full/<ckpt>.safetensors \
  --model-name qwen_edit_2509_full_exp01 \
  --limit 64
```

### Re-score existing edit outputs without exporting images again

```bash
bash scripts/run_edit_model_suite.sh \
  --model-type lora \
  --checkpoint outputs/checkpoints/Qwen-Image-Edit-2509_lora/<ckpt>.safetensors \
  --model-name qwen_edit_2509_lora_exp01 \
  --skip-validate \
  --skip-export
```

## Generation Evaluation

### Base generation model

```bash
bash scripts/run_generation_sanity_suite.sh \
  --model-type base \
  --model-name qwen_image_base_exp01 \
  --limit 32
```

### Finetuned generation-capable checkpoint

```bash
bash scripts/run_generation_sanity_suite.sh \
  --model-type full \
  --checkpoint outputs/checkpoints/Qwen-Image-Edit-2509_full/<ckpt>.safetensors \
  --model-name qwen_image_full_exp01 \
  --limit 32
```

### Re-score existing generation outputs without re-export

```bash
bash scripts/run_generation_sanity_suite.sh \
  --model-type full \
  --checkpoint outputs/checkpoints/Qwen-Image-Edit-2509_full/<ckpt>.safetensors \
  --model-name qwen_image_full_exp01 \
  --skip-export
```

## Self-Evolve Runs

### Select source images before self-evolve

One-command remote preparation:

```bash
bash scripts/prepare_remote_data.sh \
  --stage all \
  --download-limit 20000 \
  --filter-limit 20000 \
  --max-selected 5000 \
  --pilot-count 128 \
  --main-count 1024 \
  --heldout-count 128
```

Open-VLM filtering:

```bash
bash scripts/select_unlabeled_images.sh \
  --set input.images_dir=data/unlabeled/raw \
  --limit 1000
```

Heuristic smoke filter:

```bash
python -m qwen_edit_project.data.select_unlabeled_images \
  --config configs/data/source_image_filter_heuristic.yaml \
  --set input.images_dir=data/unlabeled/raw \
  --limit 128
```

Split selected images into pilot/main/heldout manifests:

```bash
python -m qwen_edit_project.data.split_source_manifest \
  --input data/unlabeled/selected/coco2017/manifest.jsonl \
  --output-dir data/unlabeled/splits/coco2017 \
  --pilot-count 128 \
  --main-count 1024 \
  --heldout-count 128
```

### Fresh hybrid run

```bash
bash scripts/run_self_evolve_matrix.sh \
  --variant hybrid \
  --images-dir data/unlabeled/self_evolve \
  --output-prefix outputs/self_evolve/exp01 \
  --limit 128
```

### Fresh delta-grounded run

This is the stronger research path: it samples multiple candidates per instruction, applies hard
instruction and preservation gates, uses Qwen internal prompt-gain checks for internal-only edits,
ranks feasible candidates, and writes evaluator training records.

```bash
bash scripts/run_self_evolve_matrix.sh \
  --variant delta-grounded \
  --images-dir data/unlabeled/self_evolve \
  --output-prefix outputs/self_evolve/exp01_delta \
  --limit 128
```

### Results-first delta run

Use this when there is no time for pilots. It keeps the same candidate-group ranker, but trains only
on proxy-verifiable edits and requires Qwen internal feature support as an auxiliary check. This is
the highest-precision path for producing benchmarkable LoRA results quickly.

```bash
bash scripts/run_self_evolve_matrix.sh \
  --variant delta-results \
  --set dataset.source=jsonl \
  --set dataset.manifest_jsonl=data/unlabeled/selected/manifest.jsonl \
  --output-prefix outputs/self_evolve/results01 \
  --limit 512
```

### Hybrid run starting from the latest LoRA checkpoint

```bash
bash scripts/run_self_evolve_matrix.sh \
  --variant hybrid \
  --images-dir data/unlabeled/self_evolve \
  --output-prefix outputs/self_evolve/exp01_resume \
  --checkpoint-dir outputs/checkpoints/Qwen-Image-Edit-2509_lora \
  --editor-model-type lora \
  --limit 128
```

### Delta-ranker ablations

Run these to isolate the new method components:

```bash
bash scripts/run_self_evolve_matrix.sh \
  --variant naive-self-train \
  --images-dir data/unlabeled/self_evolve \
  --output-prefix outputs/self_evolve/ablate_naive \
  --limit 128

bash scripts/run_self_evolve_matrix.sh \
  --variant evolmm-style \
  --images-dir data/unlabeled/self_evolve \
  --output-prefix outputs/self_evolve/ablate_evolmm_style \
  --limit 128

bash scripts/run_self_evolve_matrix.sh \
  --variant hybrid-scalar \
  --images-dir data/unlabeled/self_evolve \
  --output-prefix outputs/self_evolve/ablate_hybrid_scalar \
  --limit 128

bash scripts/run_self_evolve_matrix.sh \
  --variant delta-ranker-proxy \
  --images-dir data/unlabeled/self_evolve \
  --output-prefix outputs/self_evolve/ablate_proxy_ranker \
  --limit 128

bash scripts/run_self_evolve_matrix.sh \
  --variant delta-results \
  --images-dir data/unlabeled/self_evolve \
  --output-prefix outputs/self_evolve/ablate_no_counterfactual \
  --set solver.rank_counterfactual_weight=0.0 \
  --limit 128

bash scripts/run_self_evolve_matrix.sh \
  --variant delta-results \
  --images-dir data/unlabeled/self_evolve \
  --output-prefix outputs/self_evolve/ablate_k1 \
  --set candidate_generation.samples_per_proposal=1 \
  --limit 128

bash scripts/run_self_evolve_matrix.sh \
  --variant delta-results \
  --images-dir data/unlabeled/self_evolve \
  --output-prefix outputs/self_evolve/ablate_no_internal \
  --set solver.require_internal_when_weighted=false \
  --set solver.internal_weight=0.0 \
  --set solver.rank_internal_weight=0.0 \
  --limit 128
```

### Launch round-by-round training from inside the self-evolve loop

```bash
bash scripts/run_self_evolve_matrix.sh \
  --variant hybrid \
  --images-dir data/unlabeled/self_evolve \
  --output-prefix outputs/self_evolve/exp02_launch \
  --checkpoint-dir outputs/checkpoints/Qwen-Image-Edit-2509_lora \
  --editor-model-type lora \
  --launch-training \
  --train-config configs/train/lora_2509.yaml \
  --limit 128
```

## Resume Rules

### Training

- LoRA resume is directly supported through `--resume`; the wrapper injects the latest or explicit LoRA checkpoint into `lora.lora_checkpoint`.
- Full-finetuning resume is supported through `--resume` plus one or more `--resume-arg` values that are passed through to DiffSynth.
- If you want evaluation to follow the resumed run automatically, keep the same `--model-name`.

### Evaluation

- Export is rerunnable. Reusing the same `model_name` rewrites the image tree for that experiment.
- Scoring is rerunnable. GEdit, ImgEdit, GenEval, and DPG-Bench rewrite the model summary JSON in place.
- OneIG-Bench creates a new timestamped scoring directory on every score run, so repeated scores are naturally versioned.

### Self-evolve

- Starting from an existing checkpoint is supported through `--checkpoint` or `--checkpoint-dir`.
- Continuing from the exact same `output-prefix` is not treated as a strict transactional resume. Use a new `output-prefix` for a clean continuation branch unless you are deliberately overwriting the prior run.

## Minimal Real-Run Sequence

1. Run one LoRA baseline without scoring:

```bash
bash scripts/run_edit_model_suite.sh \
  --model-type lora \
  --train \
  --skip-export \
  --skip-score \
  --model-name qwen_edit_2509_lora_pilot01
```

2. Evaluate the resulting checkpoint:

```bash
bash scripts/run_edit_model_suite.sh \
  --model-type lora \
  --checkpoint outputs/checkpoints/Qwen-Image-Edit-2509_lora/<ckpt>.safetensors \
  --model-name qwen_edit_2509_lora_pilot01 \
  --limit 64
```

3. Run the hybrid self-evolve branch from that checkpoint:

```bash
bash scripts/run_self_evolve_matrix.sh \
  --variant hybrid \
  --images-dir data/unlabeled/self_evolve \
  --checkpoint outputs/checkpoints/Qwen-Image-Edit-2509_lora/<ckpt>.safetensors \
  --editor-model-type lora \
  --output-prefix outputs/self_evolve/pilot01_hybrid \
  --limit 128
```

4. Re-evaluate the newest checkpoint under a new `model_name`:

```bash
bash scripts/run_edit_model_suite.sh \
  --model-type lora \
  --checkpoint outputs/checkpoints/Qwen-Image-Edit-2509_lora/<new_ckpt>.safetensors \
  --model-name qwen_edit_2509_lora_pilot01_hybrid_r1 \
  --limit 64
```

## Dry-run Commands

Use these before the actual GPU runs:

```bash
bash scripts/run_edit_model_suite.sh --model-type lora --train --dry-run
bash scripts/run_generation_sanity_suite.sh --model-type base --dry-run --limit 8
bash scripts/run_self_evolve_matrix.sh --variant hybrid --dry-run --limit 8 --images-dir data/unlabeled/self_evolve
```
