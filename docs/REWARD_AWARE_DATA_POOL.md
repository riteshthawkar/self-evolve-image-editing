# Reward-Aware Training Data Pool

## Goal

The self-evolution loop needs source images that create useful edit candidates, not only images that look clean. The training pool therefore scores each source image per edit operation and builds a balanced manifest with explicit scheduled edit types.

This is not a benchmark-specific curriculum. The pool covers object removal, replacement, addition, attribute, color, material, spatial, background, style, and local enhancement edits so the model is trained toward broad conservative image editing.

## Selection Objective

Each source/edit pair is scored with:

1. Source quality: technical quality, VLM quality, naturalness, editable content, preservation potential, object clarity, low clutter, and low text/watermark risk.
2. Edit opportunity: whether the source image is structurally suitable for the requested edit family. For example, object removal favors clear separable objects, plausible background fill, low clutter, and high preservation potential.
3. Prior feedback utility: if previous self-evolution proposal logs exist, the selector uses their accepted rate, conservative reward quality, no-op rate, drift rate, and artifact rate as weak feedback.
4. Diversity: near-duplicate hashes and repeated use of the same source are limited.
5. Coverage: target fractions ensure the pool does not collapse into easy global/color edits.

The output manifest repeats strong source images only when they are useful for different edit operations. Each repeated entry has a unique key and `metadata.scheduled_edit_type`.

## Why This Helps

Earlier training runs were limited by data utility, not just reward design. The model saw too many easy global/color cases and too few high-quality object-centric cases, while object removal and replacement were exactly where evaluation failures were concentrated. This pool makes the data distribution match the conservative-editing objective:

- enough object/removal/replacement/edit-locality pressure;
- explicit preservation-sensitive source filtering;
- rejection of low-utility source/edit pairs;
- auditability for why each selected row entered the training set.

## Artifacts

The builder writes:

- `manifest.jsonl`: training manifest with scheduled edit type and metadata.
- `profile.jsonl`: all accepted source/edit candidates with utility components.
- `rejected.jsonl`: rejected source/edit candidates and reasons.
- `summary.json`: counts, target quotas, selected edit-type distribution, and mean utility.

## Script

Use `scripts/build_reward_aware_training_pool.py`.

Example:

```bash
python scripts/build_reward_aware_training_pool.py \
  --score-jsonl data/unlabeled/selected/magicbrush_all_images_moe/scores.jsonl \
  --score-jsonl data/unlabeled/selected/coco2017_moe/scores.jsonl \
  --manifest-jsonl data/unlabeled/selected/magicbrush_all_images_moe/manifest.jsonl \
  --manifest-jsonl data/unlabeled/selected/coco2017_moe/manifest.jsonl \
  --feedback-proposals 'outputs/self_evolve/*/round_*/proposals.jsonl' \
  --output data/unlabeled/reward_aware_pool/manifest.jsonl \
  --profile-output data/unlabeled/reward_aware_pool/profile.jsonl \
  --rejected-output data/unlabeled/reward_aware_pool/rejected.jsonl \
  --summary data/unlabeled/reward_aware_pool/summary.json \
  --max-records 4096
```

Run this under a Slurm CPU allocation, not on the login node.

## v1 Pool

Created under `data/unlabeled/reward_aware_pool_v1/` using the MagicBrush-all, MagicBrush-source, and COCO2017 selected source-score files plus prior self-evolution proposal feedback.

Summary:

- selected records: 4096
- source/edit candidate pairs considered: 206453
- prior feedback source/edit pairs used: 1080
- object-family scheduled rows: 2785
- mean selected data utility: 0.8224
- mean selected source quality: 0.8958

Scheduled edit-type counts:

| Edit type | Count |
| --- | ---: |
| object_removal | 655 |
| object_replacement | 573 |
| object_addition | 410 |
| attribute_change | 410 |
| color_change | 410 |
| material_change | 409 |
| background_change | 409 |
| spatial_move | 328 |
| style_transfer | 246 |
| local_enhancement | 246 |

Use with the existing self-evolution config by overriding only the dataset path:

```bash
python -m qwen_edit_project.self_evolve.run_loop \
  --config configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml \
  --override dataset.manifest_jsonl=data/unlabeled/reward_aware_pool_v1/manifest.jsonl
```

The current loop preserves `metadata.scheduled_edit_type` when it already exists, so this pool controls the proposer target edit type without requiring a benchmark-specific curriculum.
