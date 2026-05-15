# Remote Data Pipeline

This pipeline is designed for one Slurm GPU node with one 128GB GPU. The goal is not to download the largest possible dataset. The goal is to build a high-yield source pool where each GPU hour produces useful self-evolve supervision.

## Strategy

Use three stages:

```text
bounded HF image download
-> open-VLM source-image filtering
-> deterministic pilot/main/heldout splits
```

Why this is resource-aware:

- The raw pool is bounded before VLM scoring.
- Open-VLM filtering removes low-editability images before Qwen-Image-Edit generation.
- Pilot and main splits prevent burning the full run budget before the pipeline is validated.
- Heldout source images are kept separate for source-pool sanity checks and reward overfitting checks.

## Default Source Pool

The default config uses `regisss/coco_2017` from Hugging Face because it provides natural images with captions and stable image IDs.

Config:

```text
configs/data/remote_source_pool.yaml
```

Default budget:

```text
downloaded raw images: 20,000
selected source images: 5,000
pilot split: 128
main split: 1,024
heldout split: 128
```

These are intentionally conservative. On one GPU, the first goal is to prove that source filtering improves accepted self-evolve samples per GPU hour.

## Local Dry Run

Use the heuristic filter locally when VLM weights are not available:

```bash
bash scripts/prepare_remote_data.sh \
  --stage filter \
  --raw-dir data/unlabeled/raw/coco2017 \
  --selected-dir data/unlabeled/selected/coco2017 \
  --vlm-backend heuristic \
  --filter-limit 128 \
  --max-selected 64
```

## Remote Data Preparation

On the Slurm machine, after environment setup:

```bash
bash scripts/bootstrap.sh
```

Then submit the data job:

```bash
sbatch scripts/slurm/01_prepare_data.sbatch
```

For a safer pilot:

```bash
sbatch --export=ALL,DOWNLOAD_LIMIT=5000,FILTER_LIMIT=5000,MAX_SELECTED=1000,MAIN_COUNT=512 \
  scripts/slurm/01_prepare_data.sbatch
```

The job writes:

```text
data/unlabeled/raw/coco2017/
data/unlabeled/raw/coco2017_metadata.jsonl
data/unlabeled/selected/coco2017/manifest.jsonl
data/unlabeled/selected/coco2017/rejected.jsonl
data/unlabeled/selected/coco2017/scores.jsonl
data/unlabeled/selected/coco2017/selection_summary.json
data/unlabeled/splits/coco2017/pilot_manifest.jsonl
data/unlabeled/splits/coco2017/main_manifest.jsonl
data/unlabeled/splits/coco2017/heldout_manifest.jsonl
```

The download and metadata writing are resumable. Re-running the job skips already-written keys.

## Self-Evolve Runs

First run the pilot:

```bash
sbatch --export=ALL,MANIFEST=data/unlabeled/splits/coco2017/pilot_manifest.jsonl,LIMIT=128,SAMPLES_PER_PROPOSAL=2,MAX_RECORDS_PER_ROUND=128,RUN_NAME=delta_ranker_pilot \
  scripts/slurm/02_self_evolve_delta_ranker.sbatch
```

Then run the main split:

```bash
sbatch --export=ALL,MANIFEST=data/unlabeled/splits/coco2017/main_manifest.jsonl,LIMIT=512,SAMPLES_PER_PROPOSAL=2,MAX_RECORDS_PER_ROUND=256,RUN_NAME=delta_ranker_main \
  scripts/slurm/02_self_evolve_delta_ranker.sbatch
```

Increase `SAMPLES_PER_PROPOSAL` to `4` only after the pilot shows a reasonable acceptance rate.

## LoRA Training From Self-Evolve Output

Self-evolve writes DiffSynth-compatible training manifests:

```text
outputs/self_evolve/<run>/delta-ranker/round_01/train_manifest.json
outputs/self_evolve/<run>/delta-ranker/round_02/train_manifest.json
outputs/self_evolve/<run>/delta-ranker/round_03/train_manifest.json
```

Train on the latest round:

```bash
sbatch --export=ALL,TRAIN_MANIFEST=outputs/self_evolve/delta_ranker_main/delta-ranker/round_03/train_manifest.json,RUN_NAME=delta_ranker_lora_r03 \
  scripts/slurm/03_train_lora_from_manifest.sbatch
```

If GPU memory is still comfortable, increase data before increasing LoRA rank. For the first paper-grade ablation, keep `LORA_RANK=32` and compare source-pool/reward variants cleanly.

## Evaluation

Evaluate a trained LoRA checkpoint:

```bash
sbatch --export=ALL,CHECKPOINT=outputs/checkpoints/delta_ranker_lora_r03/<ckpt>.safetensors,MODEL_NAME=delta_ranker_lora_r03 \
  scripts/slurm/04_eval_edit_suite.sbatch
```

For a quick sanity check:

```bash
sbatch --export=ALL,CHECKPOINT=outputs/checkpoints/delta_ranker_lora_r03/<ckpt>.safetensors,MODEL_NAME=delta_ranker_lora_r03,LIMIT=64 \
  scripts/slurm/04_eval_edit_suite.sbatch
```

## Recommended Ablations

Run these in order:

```text
baseline Qwen-Image-Edit
raw source pool + delta-ranker
heuristic-filtered source pool + delta-ranker
open-VLM-filtered source pool + delta-ranker
open-VLM-filtered source pool + hybrid reward
```

The most important metric is not only final benchmark score. Track:

- source images selected per raw images scanned
- accepted self-evolve samples per GPU hour
- evaluator rejection reasons
- train manifest size per round
- edit benchmark score after the same LoRA budget

This gives a defensible paper story: the source-image filter is useful if it improves the efficiency and reliability of self-evolving image editing, not just if it makes images look cleaner.
