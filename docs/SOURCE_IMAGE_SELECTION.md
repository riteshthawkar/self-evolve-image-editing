# Source Image Selection

Self-evolving image editing should not start from arbitrary unlabeled images. A source image is useful only if it supports a verifiable edit and still contains enough stable content to test preservation.

This stage filters the raw image pool before the proposer sees it.

## Research Rationale

For image editing, data quality is not just aesthetics. We need **editability**:

- clear content that can be edited
- enough non-edited structure to preserve
- image quality high enough that failures are attributable to the editor, not the source image
- edit families that are neither trivial nor impossible
- diversity across scenes and edit types

This directly supports the delta-grounded self-evolve method. The evaluator can only produce meaningful labels when the source image has a visible edit target and a meaningful preservation region.

## Pipeline

```text
raw image pool
-> technical quality checks
-> open-VLM editability judgment
-> diversity and near-duplicate filtering
-> selected manifest
-> self-evolve loop
```

The selector writes:

```text
data/unlabeled/selected/manifest.jsonl
data/unlabeled/selected/rejected.jsonl
data/unlabeled/selected/scores.jsonl
data/unlabeled/selected/selection_summary.json
```

The selected manifest can be consumed directly by self-evolve:

```bash
bash scripts/run_self_evolve_matrix.sh \
  --variant delta-results \
  --set dataset.source=jsonl \
  --set dataset.manifest_jsonl=data/unlabeled/selected/manifest.jsonl \
  --limit 128
```

## Open-VLM Backend

Default config:

```bash
bash scripts/select_unlabeled_images.sh
```

This uses `Qwen/Qwen3-VL-8B-Instruct` through `transformers` by default.

Override the raw pool:

```bash
bash scripts/select_unlabeled_images.sh \
  --set input.images_dir=/path/to/raw/images
```

Limit for a pilot run:

```bash
bash scripts/select_unlabeled_images.sh \
  --limit 256 \
  --set input.images_dir=/path/to/raw/images
```

On H200, use batched VLM filtering:

```bash
bash scripts/select_unlabeled_images.sh \
  --set selection.batch_size=8 \
  --set vlm.processor_max_pixels=262144 \
  --set vlm.max_new_tokens=192
```

Increase `selection.batch_size` to `16` if GPU memory is comfortable. If you switch to `Qwen/Qwen3-VL-30B-A3B-Instruct`, start with batch size `2` or `4`.

If GPU memory is tight:

```bash
bash scripts/select_unlabeled_images.sh \
  --set vlm.model_id=Qwen/Qwen3-VL-4B-Instruct \
  --set vlm.torch_dtype=float16
```

To reproduce the older filtering baseline, override the model:

```bash
bash scripts/select_unlabeled_images.sh \
  --set vlm.model_id=Qwen/Qwen2.5-VL-7B-Instruct
```

To use the Qwen3-VL MoE judge on H200:

```bash
bash scripts/select_unlabeled_images.sh \
  --set vlm.model_id=Qwen/Qwen3-VL-30B-A3B-Instruct \
  --set selection.batch_size=4
```

## Heuristic Backend

Use this for local smoke tests or when VLM weights are not installed:

```bash
python -m qwen_edit_project.data.select_unlabeled_images \
  --config configs/data/source_image_filter_heuristic.yaml \
  --set input.images_dir=/path/to/raw/images \
  --limit 64
```

The heuristic backend is not a replacement for the open VLM. It exists so the pipeline can be tested without model downloads.

## Scoring Signals

The selector combines:

- technical quality: resolution, aspect ratio, exposure, contrast, sharpness, entropy
- VLM quality: open-VLM judgment of image quality and naturalness
- editable content: whether the image has attributes, objects, or regions that support edits
- preservation potential: whether non-target content can remain stable
- edit family coverage: whether multiple edit families are plausible
- clutter and text penalties: dense text, watermarks, and overly cluttered scenes are downweighted
- diversity filter: near-duplicates are removed using average-hash distance

The score is intentionally not pure top-k aesthetics. It is designed to select images useful for editing self-training. The default gates are intentionally stricter than "is this a natural image": an image must pass editability, preservation, clarity, clutter, and total-score thresholds before final top-k and diversity selection.

The filter streams `scores.jsonl` while running and replays thresholds from existing scores when resumed. This means you can tighten thresholds and rerun without rescoring already processed images.

## Output Format

Each selected manifest line is:

```json
{
  "key": "image_key",
  "image": "path/to/image.png",
  "caption": "short VLM caption",
  "metadata": {
    "source_selection_score": 0.73,
    "edit_families": ["color", "tone", "local"],
    "primary_family": "color"
  }
}
```

Each rejected record includes reject reasons and full scoring details in `rejected.jsonl` and `scores.jsonl`.

## Recommended Experiment Use

Run three source-pool conditions:

```text
raw pool
heuristic-filtered pool
open-VLM-filtered pool
```

Then compare downstream self-evolve results on:

- acceptance rate
- evaluator disagreement
- number of accepted pseudo-labels
- GEdit and ImgEdit scores after training

This ablation tests whether source-image editability filtering is actually helping, rather than only making the data look cleaner.
