# Runbook

## Bootstrap

```bash
bash scripts/bootstrap.sh
```

## Build manifest

```bash
python -m qwen_edit_project.data.build_diffsynth_manifest \
  --input path/to/records.json \
  --output data/manifests/train_metadata_qwen_edit.json \
  --prompt-field prompt \
  --image-field image \
  --edit-image-field edit_image \
  --relativize-paths
```

## Validate manifest

```bash
python -m qwen_edit_project.data.validate_manifest \
  --manifest data/manifests/train_metadata_qwen_edit.json
```

## LoRA training

```bash
bash scripts/train_lora_2509.sh
```

Dry run:

```bash
bash scripts/train_lora_2509.sh --dry-run
```

## Full finetuning

```bash
bash scripts/train_full_2509.sh
```

## Smoke validation

LoRA:

```bash
bash scripts/validate_lora_2509.sh outputs/checkpoints/Qwen-Image-Edit-2509_lora/epoch-4.safetensors
```

Full:

```bash
bash scripts/validate_full_2509.sh outputs/checkpoints/Qwen-Image-Edit-2509_full/epoch-1.safetensors
```

## GEdit export

```bash
bash scripts/export_gedit.sh
```

Small pilot run:

```bash
bash scripts/export_gedit.sh --limit 16
```

## GEdit score

```bash
export GEDIT_SECRET_ENV_PATH=/absolute/path/to/secret.env
bash scripts/score_gedit.sh
```

## ImgEdit export

```bash
bash scripts/export_imgedit.sh
```

Small pilot run:

```bash
bash scripts/export_imgedit.sh --limit 16
```

## ImgEdit score

```bash
export OPENAI_API_KEY=sk-...
bash scripts/score_imgedit.sh
```

## GenEval export

```bash
bash scripts/export_geneval.sh
```

Small pilot run:

```bash
bash scripts/export_geneval.sh --limit 8
```

## GenEval score

```bash
bash scripts/score_geneval.sh
```

## DPG-Bench export

```bash
bash scripts/export_dpgbench.sh
```

Small pilot run:

```bash
bash scripts/export_dpgbench.sh --limit 8
```

## DPG-Bench score

```bash
bash scripts/score_dpgbench.sh
```

## OneIG-Bench export

```bash
bash scripts/export_oneig_bench.sh
```

Small pilot run:

```bash
bash scripts/export_oneig_bench.sh --limit 8
```

## OneIG-Bench score

```bash
bash scripts/score_oneig_bench.sh
```

## Self-evolve prototype

Dry prototype without the Qwen editor:

```bash
bash scripts/self_evolve_pillow_demo.sh --limit 8
```

Qwen-backed loop:

```bash
bash scripts/self_evolve_2509.sh --limit 32
```

Combined research variant:

```bash
bash scripts/self_evolve_2509_hybrid.sh --limit 32
```

Delta-ranker path:

```bash
bash scripts/self_evolve_2509_delta_ranker.sh --limit 32
```

Single-method ablations:

```bash
bash scripts/self_evolve_2509_spatial.sh --limit 32
bash scripts/self_evolve_2509_cycle.sh --limit 32
bash scripts/self_evolve_2509_internal.sh --limit 32
```

Candidate-group evaluator ablation:

```bash
bash scripts/run_self_evolve_matrix.sh --variant delta-ranker --limit 32
bash scripts/run_self_evolve_matrix.sh --variant delta-ranker --limit 32 --set solver.rank_counterfactual_weight=0.0
bash scripts/run_self_evolve_matrix.sh --variant delta-ranker --limit 32 --set candidate_generation.samples_per_proposal=1
```

Dry run of the Qwen-backed loop:

```bash
bash scripts/self_evolve_2509.sh --dry-run --limit 32
```

## High-level experiment runners

Edit suite for base, LoRA, or full checkpoints:

```bash
bash scripts/run_edit_model_suite.sh --model-type lora --train --limit 64
bash scripts/run_edit_model_suite.sh --model-type lora --train --resume --checkpoint-dir outputs/checkpoints/Qwen-Image-Edit-2509_lora --model-name qwen_edit_2509_lora_exp01
bash scripts/run_edit_model_suite.sh --model-type full --checkpoint outputs/checkpoints/Qwen-Image-Edit-2509_full/latest.safetensors
bash scripts/run_edit_model_suite.sh --model-type full --train --resume --checkpoint outputs/checkpoints/Qwen-Image-Edit-2509_full/latest.safetensors --resume-arg --resume_from_checkpoint --resume-arg outputs/checkpoints/Qwen-Image-Edit-2509_full/latest.safetensors
bash scripts/run_edit_model_suite.sh --model-type base --dry-run --limit 8
```

Generation sanity suite:

```bash
bash scripts/run_generation_sanity_suite.sh --limit 32
```

Self-evolve matrix:

```bash
bash scripts/run_self_evolve_matrix.sh --variant all --limit 32 --images-dir data/unlabeled/self_evolve
```

Repo-local smoke test without real datasets or GPUs:

```bash
bash scripts/run_pipeline_smoke_tests.sh
```

See [EXPERIMENTS.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/EXPERIMENTS.md) for the canonical command set, output contract, and resume rules.

## Score summarization

GEdit:

```bash
python -m qwen_edit_project.eval.summarize_scores \
  --benchmark gedit \
  --model-name qwen_edit_2509_base \
  --score-root outputs/scores/gedit \
  --backbone gpt4o \
  --output outputs/scores/gedit/qwen_edit_2509_base_summary_compact.json
```

ImgEdit:

```bash
python -m qwen_edit_project.eval.summarize_scores \
  --benchmark imgedit \
  --model-name qwen_edit_2509_base \
  --score-root outputs/scores/imgedit \
  --output outputs/scores/imgedit/qwen_edit_2509_base_summary_compact.json
```

GenEval:

```bash
python -m qwen_edit_project.eval.summarize_scores \
  --benchmark geneval \
  --model-name qwen_image_base \
  --score-root outputs/scores/geneval \
  --output outputs/scores/geneval/qwen_image_base_summary_compact.json
```

DPG-Bench:

```bash
python -m qwen_edit_project.eval.summarize_scores \
  --benchmark dpgbench \
  --model-name qwen_image_base \
  --score-root outputs/scores/dpgbench \
  --output outputs/scores/dpgbench/qwen_image_base_summary_compact.json
```

OneIG-Bench:

```bash
python -m qwen_edit_project.eval.summarize_scores \
  --benchmark oneig \
  --model-name qwen_image_base \
  --score-root outputs/scores/oneig/qwen_image_base/en/<timestamp> \
  --output outputs/scores/oneig/qwen_image_base/en/<timestamp>/summary_compact.json
```
