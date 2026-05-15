# Qwen-Image-Edit Baseline Framework

This repo contains the baseline research codebase for finetuning and evaluating `Qwen/Qwen-Image-Edit-2509`.

The baseline is intentionally limited to:

- manifest creation and validation
- LoRA training
- optional full finetuning
- smoke validation
- benchmark export
- public editing benchmark scoring for GEdit-Bench and ImgEdit
- public generation benchmark scoring for GenEval, DPG-Bench, and OneIG-Bench

The repo now contains both:

- the stable baseline training/evaluation stack
- the first research implementation of the self-evolving loop described in [idea.md](/Users/ritesh.thawkar/Ritesh/neurips-project/idea.md)

Benchmark scope notes:

- editing benchmarks: `GEdit-Bench`, `ImgEdit`
- generation benchmarks: `GenEval`, `DPG-Bench`, `OneIG-Bench`
- `GSO` is not wired yet because the repo does not currently include a clean public scorer path for it

## Layout

- [configs](/Users/ritesh.thawkar/Ritesh/neurips-project/configs): YAML configs and env examples
- [src/qwen_edit_project](/Users/ritesh.thawkar/Ritesh/neurips-project/src/qwen_edit_project): Python package
- [scripts](/Users/ritesh.thawkar/Ritesh/neurips-project/scripts): shell entry points
- [docs](/Users/ritesh.thawkar/Ritesh/neurips-project/docs): setup and runbook docs
- [docs/DATASET_SETUP.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/DATASET_SETUP.md): dataset and asset preparation for training, self-evolve, and benchmarks
- [docs/REMOTE_DATA_PIPELINE.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/REMOTE_DATA_PIPELINE.md): one-GPU Slurm data staging and source-pool splitting
- [docs/SOURCE_IMAGE_SELECTION.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/SOURCE_IMAGE_SELECTION.md): open-VLM filtering for self-evolve source images
- [docs/EXPERIMENTS.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/EXPERIMENTS.md): experiment commands, output contract, and resume patterns
- [docs/SELF_EVOLVE.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/SELF_EVOLVE.md): self-evolving loop details
- [docs/DELTA_GROUNDED_SELF_EVOLVE.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/DELTA_GROUNDED_SELF_EVOLVE.md): stronger candidate-group evaluator path
- [patches](/Users/ritesh.thawkar/Ritesh/neurips-project/patches): minimal upstream patches
- [third_party/LOCKFILE.md](/Users/ritesh.thawkar/Ritesh/neurips-project/third_party/LOCKFILE.md): pinned upstream commit record

## Quick start

1. Bootstrap upstream repos:

```bash
bash scripts/bootstrap.sh
```

2. Build a training manifest:

```bash
python -m qwen_edit_project.data.build_diffsynth_manifest \
  --input path/to/records.json \
  --output data/manifests/train_metadata_qwen_edit.json \
  --relativize-paths
```

3. Validate the manifest:

```bash
python -m qwen_edit_project.data.validate_manifest \
  --manifest data/manifests/train_metadata_qwen_edit.json
```

4. Launch LoRA training:

```bash
bash scripts/train_lora_2509.sh
```

5. Export or score benchmarks:

```bash
bash scripts/export_gedit.sh --limit 16
bash scripts/export_geneval.sh --limit 8
```

6. Run the self-evolving loop:

```bash
bash scripts/select_unlabeled_images.sh --set vlm.backend=heuristic
python -m qwen_edit_project.data.split_source_manifest --input data/unlabeled/selected/manifest.jsonl --output-dir data/unlabeled/splits/local
bash scripts/self_evolve_pillow_demo.sh --limit 8
bash scripts/self_evolve_2509_hybrid.sh --limit 32
```

7. Use the higher-level experiment runners if you want bundled workflows:

```bash
bash scripts/run_edit_model_suite.sh --model-type lora --train --limit 64
bash scripts/run_generation_sanity_suite.sh --limit 32
bash scripts/run_self_evolve_matrix.sh --variant all --limit 32 --images-dir data/unlabeled/self_evolve
bash scripts/run_self_evolve_matrix.sh --variant delta-ranker --limit 32 --images-dir data/unlabeled/self_evolve
bash scripts/run_pipeline_smoke_tests.sh
```

See [docs/SETUP.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/SETUP.md), [docs/DATASET_SETUP.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/DATASET_SETUP.md), [docs/EXPERIMENTS.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/EXPERIMENTS.md), [docs/RUNBOOK.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/RUNBOOK.md), [docs/BENCHMARKS.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/BENCHMARKS.md), and [docs/SELF_EVOLVE.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/SELF_EVOLVE.md) for the full workflow.
