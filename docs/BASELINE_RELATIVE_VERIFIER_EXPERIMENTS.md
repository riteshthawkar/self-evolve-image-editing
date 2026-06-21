# Baseline-Relative Verifier Experiments

## Current Decision

The next strongest research direction is baseline-relative verifier training.
The goal is not to train on every high-scoring self-generated edit. The goal is
to update the editor only when a candidate clearly beats the base Qwen output
for the same source image and instruction. If the base output is already better
or tied, the base behavior should be preserved through replay or a low-weight
reference-win preference pair.

This directly addresses the observed failure mode:

- broad SFT and self-evolution damaged high-ceiling examples;
- reward-selected DPO/SDPO pairs were not aligned enough with ImgEdit/GEdit;
- the best full benchmark signal came from conservative verifier selection, not
  from a single global LoRA checkpoint;
- object removal/replacement needs grounded or explicit object-contract
  verification rather than only global text-image similarity.

## Research Hypothesis

A strong image editor should be improved by learning only from verified
candidate-over-baseline wins. Training on merely plausible edits is unsafe
because the baseline is already near ceiling on many examples.

The falsifiable claim is:

> Baseline-relative, auto-rubric verified preference pairs can improve hard
> edit categories while preserving the base editor's already-correct behavior.

## Implemented Components

### ARR-Style ImgEdit Selector Prompt

`scripts/build_imgedit_vlm_selector.py` now supports:

```bash
--prompt-mode arr
```

This prompt asks the VLM judge for five scores per candidate:

- instruction completion;
- preservation/locality;
- visual quality;
- object or target-region contract;
- over-edit safety.

This is for verifier/selector experiments and for checking whether a stronger
rubric can beat the previous `rubric` selector.

### Baseline-Relative Manifest Builder

New script:

```bash
scripts/build_baseline_relative_verifier_manifest.py
```

It consumes non-eval JSONL rows with at least:

- `prompt`;
- `edit_image`;
- candidate image, default key `chosen_image`;
- reference/base image, default key `baseline_image`.

For each row, it compares candidate output against the reference output using a
VLM verifier and writes pairwise training rows:

- candidate win: `chosen_image = candidate`, `rejected_image = reference`;
- reference win: optionally `chosen_image = reference`, `rejected_image = candidate`;
- ties or low-confidence rows are skipped.

Important: the real baseline-relative experiment should use actual base Qwen
outputs as `baseline_image`. The script has an emergency flag
`--allow-rejected-as-reference`, but that is a diagnostic fallback, not the main
paper-grade setting.

### Base-Reference Export Preparation

Retired cleanup note, 2026-06-03: the one-off bash launchers for this
baseline-relative diagnostic were removed from `scripts/`. The underlying
builder utilities are still documented for reproducibility, but this is not the
current main pipeline.

The retired reference-export launcher built two files from a non-eval JSONL
manifest:

- an ImgEdit-compatible export JSON for base-Qwen reference generation;
- an indexed JSONL manifest with a `baseline_image` field pointing to the
  expected base output path.

With `RUN_EXPORT=1` it also exports the base-reference images. This export is
GPU work and must be run inside a Slurm allocation.

### Pairwise Training Launcher

The retired training launcher used this objective:

```text
pairwise_linear_sdpo
```

Default low-pressure settings:

- LR `1e-6`;
- LoRA rank `16`;
- 128 steps;
- preference beta `2.0`;
- chosen SFT weight `0.10`;
- reference mode `none` for `pairwise_linear_sdpo`.

For linear SDPO, an initial-LoRA reference only adds a constant to the linear
preference objective and does not change the gradient. Use
`PREFERENCE_REFERENCE_MODE=initial_lora` only for softplus-style
`pairwise_sdpo` ablations.

The script refuses to run outside a Slurm allocation unless
`ALLOW_LOGIN_NODE=1` is explicitly set. Do not use that override for real
training.

### End-To-End Run Wrapper

The removed end-to-end launcher built the verifier manifest and then launched
pairwise SDPO training inside a tmux Slurm resource session.

## Required Data Step Before Main Training

We still need base Qwen reference images for the non-eval training rows. The
clean setup is:

1. Select a non-eval training pool from MagicBrush/self-evolve traces.
2. Build a small export JSON and indexed manifest:

Use `scripts/build_reference_export_manifest.py` directly if this retired
diagnostic is reproduced.

3. Generate base Qwen outputs inside Slurm:

Generate base Qwen outputs with the standard eval export wrapper inside Slurm.

4. Run the verifier manifest builder on the indexed manifest:

Build the verifier manifest with `scripts/build_baseline_relative_verifier_manifest.py`
and launch training through `src/qwen_edit_project/train/launch_train.py`.

For the main experiment, do not set `ALLOW_REJECTED_AS_REFERENCE=1`; the
reference must be actual base Qwen output.

Do not use ImgEdit/GEdit benchmark images for training.

## First Experiment Sequence

### Experiment A: ARR Selector Refresh

Purpose: test whether the stronger auto-rubric prompt improves the current
test-time selector without training.

Run on ImgEdit full or a stride canary first:

```text
Retired one-off VLM selector launcher. Use the standard ImgEdit export/score
wrappers plus selector-builder scripts if this diagnostic is reproduced.
```

Decision rule:

- if it beats the current type-abstaining selector controlled gain, keep ARR as
  the verifier prompt;
- if it is tied or worse, keep ARR only for training-data filtering ablations.

### Experiment B: 128-Pair Baseline-Relative Training Probe

Purpose: test whether verified candidate-over-baseline pairs produce a safe
single checkpoint.

Manifest target:

- 128 high-confidence pairs;
- balanced across object removal, replacement, addition, background, and
  localized color/attribute edits;
- include reference-win pairs at low weight only when the candidate clearly
  damages the image;
- cap reference-win pairs relative to candidate-win pairs so the run does not
  become mostly baseline distillation.

Training:

```text
Retired one-off pairwise training launcher. Reproduce through the generic
training entrypoint and an explicit config.
```

Decision rule:

- ImgEdit broad 64 canary must be non-negative;
- GEdit subject-remove and subject-replace 32-case canaries must not regress;
- if either fails, inspect verifier rows before changing optimizer settings.

### Experiment C: 512-Pair Scale-Up

Only run after Experiment B is positive.

Changes:

- scale manifest to 512 verified rows;
- keep family balance;
- keep LR low;
- checkpoint every 64 steps;
- evaluate checkpoint 64 and final before full benchmark.

Decision rule:

- checkpoint must beat the base model on at least the 128-example ImgEdit
  distributed canary;
- then run full ImgEdit;
- then run full GEdit if ImgEdit is positive and GEdit canaries are safe.

## Evaluation Order

1. ImgEdit broad 64 canary.
2. ImgEdit stride-6 123 canary.
3. GEdit subject-remove 32.
4. GEdit subject-replace 32.
5. GEdit background 32.
6. Full ImgEdit.
7. Full GEdit.
8. GenEval and DPG-Bench only as regression checks for generation behavior.

## Expected Outcome

Do not expect `+0.8` to `+1.0` absolute gain on full ImgEdit. The baseline is
`4.4406 / 5.0`, so the maximum possible full-benchmark gain is about `+0.559`.

Reasonable targets:

- short-term proof: `+0.03` to `+0.08` full ImgEdit with no GEdit regression;
- strong result: `+0.10` to `+0.20` if baseline-relative filtering is clean;
- large improvements only on lower-baseline subsets such as extract, compose,
  object removal, or selected GEdit categories.

## What Not To Do Next

- Do not train longer on the prompt-mix anchor; full ImgEdit is negative.
- Do not scale current generated DPO/SDPO manifests; canaries were negative.
- Do not use rejected self-evolution rows as positive SFT targets.
- Do not use benchmark images for training or data selection.
- Do not run GPU training or benchmark exports on the login node.
