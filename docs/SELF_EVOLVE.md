# Self-Evolving Loop

This module is the first implementation of the project’s main research idea: a proposer-editor-evaluator loop that generates pseudo-labeled editing data from unlabeled images and accepts only high-scoring edits into the next training pool.

Older code and configs may still use the name `solver` for this third component. In the image
editing project, that component is not a solver agent; it is a fixed reward evaluator that scores
whether a candidate image satisfies the edit instruction while preserving non-target content.

## What is implemented

- round-based self-evolving loop orchestration
- difficulty shaping over proposal families
- proposal generation
- editor backends
- reward evaluator backends
- accepted-sample manifest writing for downstream LoRA training
- optional training launch after each round

The code lives under [src/qwen_edit_project/self_evolve](/Users/ritesh.thawkar/Ritesh/neurips-project/src/qwen_edit_project/self_evolve).

## Current backends

### Proposer

- `scripted`
- `internal_qwen` placeholder
- `trainable_qwen_image_edit`
- `trainable_qwen_vl`

### Editor

- `qwen_edit`
- `pillow_demo`

### Reward Evaluator

- `stat`
- `internal_qwen`
- `hybrid`
- `evolmm_style`
- `hard_gated_relative`
- `internal_cepr`

The preferred config name is `evaluator:`. Existing `solver:` configs remain supported as an alias
so old experiment commands keep working.

## Important limitation

The control loop is implemented, but the public `Qwen-Image-Edit` pipeline does not expose the internal understanding branch as a standalone proposer/verifier API. Because of that:

- the current real editor is `qwen_edit`
- the current implemented proposer is `scripted`
- the current baseline evaluator is `stat`
- the current generic self-reward baseline is `evolmm_style`
- the current exploratory research evaluators are `internal_qwen`, `hybrid`, and `hard_gated_relative`

This means the repo now contains the full loop infrastructure and iterative data-generation logic, but the fully closed internal proposer-evaluator path is still behind an adapter boundary rather than fully realized with public upstream APIs.

At the same time, the public DiffSynth stack does expose the Qwen-conditioned hidden states used for edit conditioning. Our utility layer now exposes those features through [qwen_pipeline.py](/Users/ritesh.thawkar/Ritesh/neurips-project/src/qwen_edit_project/utils/qwen_pipeline.py), which gives us a concrete path toward an internal representation-based verifier in the next phase.

## New verifier methods

The repo now includes three research-facing verifier ideas that can be toggled independently:

- `spatial` verification inside the `hybrid` evaluator
  - scores changed-region support and outside-region preservation separately
- `cycle` consistency inside the `hybrid` evaluator
  - applies an inverse edit when available and scores reconstruction back toward the source image
- `internal_qwen` feature verification
  - uses hidden states from Qwen’s public image-plus-instruction understanding path as an additional score

These are intentionally heuristic and exploratory. They are implemented so they can be ablated and tested, not because they are already validated.

## Delta-ranker path

The stronger research path is documented in [DELTA_GROUNDED_SELF_EVOLVE.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/DELTA_GROUNDED_SELF_EVOLVE.md).

It adds:

- multiple editor candidates for the same proposal
- hard instruction and preservation gates
- relative ranking among feasible candidates
- counterfactual instruction scoring
- internal Qwen prompt-gain checks for proposals that cannot be verified by simple image statistics
- evaluator training data export

This path is the intended bridge from heuristic self-training to a learned evaluator LoRA.

## Trainable Proposer Path

The main trainable proposer path uses the Qwen-Image-Edit checkpoint's own VLM/text-encoder
component (`Qwen/Qwen-Image-Edit-2509`, `text_encoder` subfolder) during data generation. The
diffusion editor cannot emit text instructions directly, so the proposer is attached to the edit
model's autoregressive VLM component and trained with a separate proposer LoRA. That proposer LoRA is
not used for final editor evaluation. At the end of each round, proposal outcomes are aggregated
into `proposer_training.jsonl`; proposals receive the highest reward when they create useful
medium-difficulty editor-training samples:

- zero accepted candidates is treated as too hard
- all candidates accepted is treated as probably too easy
- high CEPR++ semantic edit, preservation, and validity scores increase proposer reward

The final editor update now supports CEPR preference learning as the main path. For each proposal,
the editor samples multiple candidates. CEPR accepts the best feasible candidate and the loop writes
same-group preference pairs:

```text
chosen = accepted top CEPR candidate
rejected = lower-ranked candidate from the same source image and instruction
```

When `training.preference.enabled=true`, the editor LoRA trainer consumes
`preference_manifest.jsonl` with `pairwise_linear_sdpo` instead of treating the selected image as a
plain weighted-SFT target. The older CEPR-weighted SFT manifest is still written for auditing and
ablation, but rejected candidates are disabled in the main trainable-proposer configs so low-quality
self-generated images cannot become direct targets.

For hard object-edit groups, CEPR may reject every candidate even when one candidate is visibly
closer to the requested edit than the others. The current main path can write those within-group
comparisons as **near-miss preference pairs**. A near-miss chosen candidate must pass raw internal
quality, semantic-edit, preservation, and validity floors, but it is not treated as an accepted
image target. Its per-record `preference_sft_weight` is `0.0`, so it contributes only a relative
preference signal and does not directly SFT the editor toward a failed output. Accepted pairs retain
a small positive chosen-image SFT anchor.

Object-focused runs also enforce each record's scheduled edit type. The learned proposer gets the
first attempt, but if strict filtering removes all proposals, `template_fallback_on_target_miss`
creates a concrete target-type instruction from the image caption/metadata. This prevents
removal/replacement records from silently disappearing after proposer filtering; skipped records are
now logged in `progress.json` and `summary.json` as `no_proposal_records`.

The next round then loads both the editor LoRA checkpoint and the proposer LoRA checkpoint. This
gives a closed-loop co-evolution schedule:

```text
proposer_r -> editor_r -> K candidates -> internal CEPR ranker
        -> preference-train editor_{r+1}
        -> outcome-train proposer_{r+1}
```

## Current Object-Balanced Preference Direction

The active research direction is `configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml`.
It addresses the main failure observed in earlier canaries: color/global edits were over-sampled,
while object removal and replacement were starved or trained from bad rejected SFT targets.

The current direction changes the loop in four ways:

- inject a round-level edit-type schedule so early rounds contain removal, replacement, attribute,
  material, spatial, color, background, style, and enhancement proposals rather than only easy color
  changes
- cap SFT and preference records by edit type, with rejected SFT targets disabled
- use lower near-miss score margins for removal/replacement only, because CEPR's hard
  forbidden-object gate often rejects every candidate while still ranking which failure is closer
- from round 2 onward, optionally add previous-round opponent candidates to each group; the internal
  ranker then compares the current editor against an older editor under the same source image and
  instruction

This keeps the method self-evolving: the images are generated by the editor family, the edit
instructions are generated by the proposer path, and the supervision is an internal CEPR ranking.
No external VLM reward model is used during training. The versioned opponent branch is meant to make
later rounds closer to self-play: the current editor must beat its previous self, not merely win
among random seeds from the same checkpoint.

Rounds can be micro-batches rather than full-dataset passes. With
`curriculum.record_schedule=sequential_shards`, round 1 uses the first
`curriculum.max_records_per_round` images, round 2 uses the next shard, and so on, wrapping only
after all available records are consumed. This keeps updates frequent without per-image overfitting.
The trainable-proposer CEPR config defaults to 8 source images per round and
`curriculum.num_rounds=auto`, so a run with `--limit 1024` performs 128 update rounds over distinct
8-image shards.

The 8-image micro-round is a compute-control choice, not a benchmark-tuned threshold. With the
default 4 candidates per proposal it gives 32 CEPR-scored candidate edits before each update. This
is the smallest practical batch we use because it gives the editor both positive and borderline
weighted targets, gives the proposer multiple reward observations before a LoRA step, and still
updates far more frequently than a 64- or 256-image round. Single-image updates provide only 4
candidate edits, so a reward outlier can dominate the update and the proposer reward becomes too
noisy.
The default also uses `output.use_cumulative_manifest=false`; past learning is carried by the
continued editor/proposer checkpoints, while each update trains on the current shard to avoid
quadratic retraining cost across many micro-rounds.

Use:

```bash
bash scripts/run_self_evolve_matrix.sh \
  --variant internal-cepr-trainable-proposer \
  --launch-training \
  --launch-proposer-training
```

## Generic Self-Reward Baseline

The `evolmm_style` evaluator is an intentionally limited baseline for the paper story. It keeps the
self-evolving structure and K-candidate relative ranking, but uses a generic continuous scalar
self-reward without preservation gates, counterfactual instruction discrimination, or
delta-grounded feasibility checks. It exists to test whether a reasoning-style self-evolution reward
transfers directly to image editing.

Expected outcome: it may select candidates that satisfy a global edit proxy, but it should be weaker
on non-edit preservation than `delta-results`.

## Round outputs

Each round writes:

```text
outputs/self_evolve/<run_name>/round_01/proposals.jsonl
outputs/self_evolve/<run_name>/round_01/proposal_plan.jsonl
outputs/self_evolve/<run_name>/round_01/progress.json
outputs/self_evolve/<run_name>/round_01/train_manifest.json
outputs/self_evolve/<run_name>/round_01/train_weights.jsonl
outputs/self_evolve/<run_name>/round_01/train_weight_summary.json
outputs/self_evolve/<run_name>/round_01/accepted/images/*.png
outputs/self_evolve/<run_name>/round_01/summary.json
outputs/self_evolve/<run_name>/self_evolve.log
```

The manifest format matches the editor LoRA training flow:

- `prompt`: edit instruction
- `image`: edited image
- `edit_image`: original image
- `sample_weight`: CEPR-derived SFT weight

`proposal_plan.jsonl` is the durable proposal source of truth for a round. `proposals.jsonl` is
appended after each candidate group is evaluated, then rewritten canonically at round end. If a job
is interrupted, restart the same command with the same `output.root_dir`; completed rounds are
skipped, and in-progress rounds skip proposal groups that already have all candidate rows.

`progress.json` and `self_evolve.log` are the main monitoring files. `progress.json` contains the
current round status, candidate rows written, accepted count, acceptance rate, and elapsed time.
`train_weights.jsonl` records the CEPR-derived SFT weight decision for every candidate.

## Running the loop

Prototype run without the Qwen model:

```bash
bash scripts/self_evolve_pillow_demo.sh --limit 8
```

Qwen-backed run:

```bash
bash scripts/self_evolve_2509.sh --limit 32
```

Hybrid NeurIPS-oriented run with all three verifier ideas enabled:

```bash
bash scripts/self_evolve_2509_hybrid.sh --limit 32
```

Results-first delta run:

```bash
bash scripts/self_evolve_2509_delta_results.sh --limit 512
```

EvoLMM-style generic self-reward baseline:

```bash
bash scripts/self_evolve_2509_evolmm_style.sh --limit 512
```

Broader delta-grounded ranker run:

```bash
bash scripts/self_evolve_2509_delta_grounded.sh --limit 32
```

The older `self_evolve_2509_delta_ranker.sh` script is kept as a proxy-ranker ablation. The
`delta-results` config is the default path for producing benchmark numbers quickly because it keeps
training labels high precision. The broader `delta-grounded` config keeps the expanded structured
taxonomy for research data generation.

Single-method ablations:

```bash
bash scripts/self_evolve_2509_spatial.sh --limit 32
bash scripts/self_evolve_2509_cycle.sh --limit 32
bash scripts/self_evolve_2509_internal.sh --limit 32
```

Local verification run without Qwen weights:

```bash
bash scripts/self_evolve_pillow_hybrid.sh --limit 8
```

Local delta-ranker verification without Qwen weights:

```bash
bash scripts/self_evolve_pillow_delta_ranker.sh --limit 8
```

If your GPU does not handle BF16 well:

```bash
bash scripts/self_evolve_2509.sh --set editor.model.torch_dtype=float16
```

## Optional training launch

By default, training is not launched after each round:

```yaml
training:
  trigger: emit_only
```

To launch LoRA after each round, set:

```yaml
training:
  trigger: launch
  base_train_config: configs/train/lora_2509_diffusers.yaml
  resume_from_latest: true
  trained_checkpoint_backend: official_diffusers
```

The loop will write a round-specific training command and then promote the latest produced LoRA
checkpoint into the next round if one is found. For Diffusers-native editor LoRA training, existing
`checkpoint-*` directories inside the round training output automatically add
`--resume_from_checkpoint latest` when `training.resume_from_latest: true`.

For trainable-proposer runs, `proposer.training.resume_from_latest: true` does the same for the
Qwen-Image-Edit VLM proposer LoRA trainer. The final benchmark/evaluation model remains the editor
LoRA, not the proposer LoRA.

## Conservative base-relative preference mode

The current Qwen-Image-Edit research path uses conservative pairwise self-evolution rather than
plain accepted-vs-rejected SFT:

- enable `candidate_generation.reference_candidates` to generate a base/reference output for each proposal;
- enable `training.preference.base_relative` to compare policy candidates against that base output;
- train policy-over-base pairs only when the policy wins by margin;
- train reversed base-over-policy pairs when the base clearly wins, which protects already-good
  base-model behavior;
- keep `output.use_cumulative_preference_manifest=true` so preference replay is not limited to the
  current round's tiny accepted set.

The main patched config is:

```bash
configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml
```

It writes to:

```bash
outputs/self_evolve/qwen_edit_2509_conservative_pairwise_v1
```
