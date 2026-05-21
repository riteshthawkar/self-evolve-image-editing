# Benchmarks

## Editing benchmarks

### GEdit-Bench

The public scorer expects this exact layout:

```text
outputs/benchmark_images/gedit/<model_name>/fullset/<task_type>/<language>/<key>.png
```

Examples:

```text
outputs/benchmark_images/gedit/qwen_edit_2509_official_diffusers/fullset/background_change/en/12345.png
outputs/benchmark_images/gedit/qwen_edit_2509_official_diffusers/fullset/text_change/cn/67890.png
```

Scoring entry points:

- `third_party/step1x-edit/GEdit-Bench/run_gedit_score.py`
- `third_party/step1x-edit/GEdit-Bench/calculate_statistics.py`

Important note:

- the Qwen-Image-Edit-2509 paper-matched baseline uses the official Diffusers `QwenImageEditPlusPipeline`, not the legacy DiffSynth edit wrapper
- the official 2509 generation settings are `bfloat16`, `num_inference_steps=40`, `true_cfg_scale=4.0`, `guidance_scale=1.0`, `negative_prompt=" "`, and `num_images_per_prompt=1`
- the upstream scorer option is still named `--backbone gpt4o`, but the current upstream `VIEScore` implementation instantiates `model_name="gpt-4.1"` under that alias
- the upstream scorer currently expects a `secret.env` file via its internal `VIEScore(..., key_path='secret.env')` call
- the local wrapper copies the configured secret file into the repo root for the scoring run
- the wrapper refuses to resume into an existing output directory when the recorded export provenance does not match the current paper-matched settings

### ImgEdit

The public scorer expects a flat output folder:

```text
outputs/benchmark_images/imgedit/<model_name>/<key>.png
```

Examples:

```text
outputs/benchmark_images/imgedit/qwen_edit_2509_official_diffusers/1082.png
outputs/benchmark_images/imgedit/qwen_edit_2509_official_diffusers/1068.png
```

Scoring entry points:

- `third_party/imgedit/Benchmark/Basic/basic_bench.py`
- `third_party/imgedit/Benchmark/Basic/step1_get_avgscore.py`
- `third_party/imgedit/Benchmark/Basic/step2_typescore.py`

Important note:

- the README mentions `--basic_edit`, but the current scorer script actually uses `--edit_json`
- the local wrapper follows the actual script interface
- the local patch changes the scorer to read `OPENAI_API_KEY` and optional `OPENAI_BASE_URL` from the environment

### GSO

`GSO` is intentionally not implemented in this repo yet. I could not find a stable public scorer/reproduction path that is comparable to `GEdit-Bench` and `ImgEdit`, so the framework currently treats the public image-edit evaluation surface as:

- `GEdit-Bench`
- `ImgEdit`

## Generation benchmarks

### GenEval

The exporter writes the official folder structure:

```text
outputs/benchmark_images/geneval/<model_name>/00000/
outputs/benchmark_images/geneval/<model_name>/00000/metadata.jsonl
outputs/benchmark_images/geneval/<model_name>/00000/grid.png
outputs/benchmark_images/geneval/<model_name>/00000/samples/0000.png
```

Scoring entry points:

- `third_party/geneval/evaluation/evaluate_images.py`
- `third_party/geneval/evaluation/summary_scores.py`

Important notes:

- the scorer needs a downloaded Mask2Former checkpoint directory; configure it via `scoring.object_detector_root`
- the wrapper stores raw results in `outputs/scores/geneval/<model_name>_results.jsonl`

### DPG-Bench

The exporter writes one 2x2 grid image per benchmark item:

```text
outputs/benchmark_images/dpgbench/<model_name>/<item_id>.png
```

Scoring entry point:

- `third_party/ella/dpg_bench/compute_dpg_bench.py`

Important notes:

- the scorer is launched through `accelerate`
- `resolution` must match the single-image size inside the grid
- `pic_num` should match `generation.samples_per_prompt`

### OneIG-Bench

The exporter writes category-organized grid images using the upstream folder names:

```text
outputs/benchmark_images/oneig/en/anime/<model_name>/000.webp
outputs/benchmark_images/oneig/en/text/<model_name>/103.webp
outputs/benchmark_images/oneig/en/reasoning/<model_name>/211.webp
```

Scoring entry points:

- `third_party/oneig-bench/scripts/alignment/alignment_score.py`
- `third_party/oneig-bench/scripts/text/text_score.py`
- `third_party/oneig-bench/scripts/diversity/diversity_score.py`
- `third_party/oneig-bench/scripts/style/style_score.py`
- `third_party/oneig-bench/scripts/reasoning/reasoning_score.py`

Important notes:

- the local wrapper calls each scorer module directly instead of using upstream `run_overall.sh`
- bootstrap applies a small patch to the upstream diversity scorer so it does not crash on CSV column construction
- the OneIG scorer stack has extra external model downloads for style, diversity, reasoning, and text evaluation; follow the upstream README after bootstrap
- scored CSVs are copied into `outputs/scores/oneig/<model_name>/<mode>/<timestamp>/`

## Prompt polishing

Prompt polishing is available through [src/qwen_edit_project/utils/prompting.py](/Users/ritesh.thawkar/Ritesh/neurips-project/src/qwen_edit_project/utils/prompting.py), but it is off by default in benchmark configs to preserve apples-to-apples comparisons.
