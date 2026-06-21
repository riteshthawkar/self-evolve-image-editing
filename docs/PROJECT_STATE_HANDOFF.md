# Self-Evolving Qwen Image Editing: Project State Handoff

This document summarizes the research idea, implemented system, current experimental status, known gaps, and recommended next technical direction. It is written for a coding or research agent that needs to continue the project without reading the full conversation history.

Machine-use constraints are documented in:

```text
docs/SLURM_RESOURCE_GUIDELINES.md
```

Any training, evaluation, benchmark export, or GPU-heavy experiment must run
inside a Slurm allocation, usually from a tmux session, using the `qedit` conda
environment.

## Current Research Goal

We are building a self-evolving image-editing framework on top of `Qwen/Qwen-Image-Edit-2509`.

The intended research claim is:

> A strong image editor can improve from unlabeled source images by generating its own edit instructions, sampling candidate edits, internally verifying which candidates are useful, and then updating the editor and proposer from those self-generated supervision signals without relying on an external VLM reward model.

The project is constrained to **internal rewards only** for the main method. External GPT/VLM models are allowed for final benchmark evaluation only, not for training-time reward.

## Core Terminology

- **Proposer**: Generates edit instructions for source images.
- **Editor**: Qwen-Image-Edit model being improved with LoRA.
- **Evaluator / reward model**: A fixed internal reward evaluator. Older docs or configs may call this a `solver`, but in the current terminology it is not a trainable third model and not a copy of the editor. It scores candidate edited images.
- **CEPR**: Contrastive Edit-Preservation Reward, the current internal reward evaluator.
- **Round**: A batch of source images. The default current round size is 8 source images. For each image, the proposer creates an edit instruction and the editor generates K candidates.

## Current Implemented Pipeline

The main implemented config is:

```text
configs/self_evolve/qwen_edit_2509_internal_cepr_trainable_proposer.yaml
```

The main launcher variant is:

```text
internal-cepr-trainable-proposer
```

The current high-level loop is:

```text
source image
-> trainable proposer creates structured edit instruction
-> editor generates K candidate edits
-> internal CEPR scores candidates
-> accepted and near-miss same-group preference pairs are written
-> editor LoRA is trained with pairwise preference learning
-> proposer LoRA is trained from proposal-level reward
-> next round continues from the updated checkpoints
```

The current object-balanced diagnostic config is:

```text
configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml
```

This config should be treated as the active next-method direction. It adds scheduled edit-type
coverage, edit-type-balanced training manifests, near-miss preference pairs for hard object edits,
and versioned self-play from round 2 onward. The round-1 schedule deliberately includes
`object_removal` and `object_replacement` before easy color/global edits so the first editor update
cannot be dominated by the old color-heavy shard behavior.

Important current details:

- `weighted_sft.include_rejected=false`; rejected self-generated images are audit data, not direct
  SFT targets.
- `training.preference.enabled=true`; the editor trains on `preference_manifest.jsonl` with
  `pairwise_linear_sdpo`.
- Preference training requires a minimum number of real edit pairs before editor training is trusted;
  preservation anchors are logged separately and do not satisfy this gate.
- Object removal/replacement no longer use the embedding-style forbidden-object absence score as a
  hard training contract. The score remains logged, but `rubric_forbidden_after_absent` and
  `rubric_edit_success` are disabled for object training-contract checks because manual inspection
  confirmed false negatives on valid removals. Object candidates still need high required-after,
  preservation, validity, taxonomy, and raw CEPR.
- The active generalized config does not use object-specific pair margins, pair weights, or
  benchmark-specific curriculum. It uses a uniform low pair margin plus broad CEPR component
  calibration, hard-negative failure-mode diversity, and family-balanced sampling across edit types.
- The active generalized config now enables `evaluator.internal_vlm_judge`: an internal generative
  Qwen-VLM self-judge that reuses the editor pipeline's own Qwen processor and text-encoder/VLM side.
  It scores instruction following, edit success, target correctness, preservation, artifact freedom,
  overall quality, and confidence for same-group candidates. CEPR/rubric gates remain the safety
  mechanism; the judge refines raw reward/ranking and preference confidence, and is configured
  `fail_open: true` so a judge parse/runtime failure falls back to CEPR instead of breaking a round.
- `candidate_generation.self_play.enabled=true`, `start_round=2`, and `opponent=previous_round`.
  After round 1 trains, future rounds can rank current-editor candidates against previous-editor
  candidates for the same proposal.
- Generalization update: the next method should avoid benchmark-specific curriculum. The active
  config now adds broad-support preference calibration over rubric CEPR's internal Qwen-derived
  components plus the internal Qwen-VLM judge, hard-negative failure-mode diversity, and
  preference-mode preservation anchor replay.
  The curriculum cycles uniformly over removal, replacement, addition, attribute, color, material,
  spatial, background, style, and local enhancement edits.
  The failure tags are generic (`under_edit`, `preservation_drift`, `invalid_or_artifact`,
  `taxonomy_mismatch`, `weak_reward`, `hard_near_miss`) and apply across edit classes.
- Preservation anchors are now written into `preference_manifest.jsonl` when preference training is
  active. They use the source image as chosen and a self-generated edited image as rejected under a
  reconstruction prompt. They regularize drift but do not satisfy minimum real edit-pair counts.

Latest active run, 2026-06-02:

- Config: `configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml`.
- Tmux/Slurm session: `uug_balanced_v2`, job `504`.
- Output root: `outputs/self_evolve/balanced_cepr_v5_soft_object_contract_diag96_r4_20260602T130155Z`.
- First round schedule: 24 records, including 4 object-removal and 4 object-replacement groups.
- Early health signal: first object-removal group completed with 1 accepted candidate from 4 scored
  candidates under the corrected soft object contract.
- First object-replacement group also completed with 1 accepted candidate. A local preference
  simulation after patching the pair builder produced 6 pairs from the first two object groups:
  3 removal and 3 replacement.
- Probe evidence: `outputs/self_evolve/object_soft_contract_probe_r1_20260602T125658Z` accepted both
  a removal and replacement object group, wrote zero rejected SFT targets, and completed cleanly
  after fixing disabled proposer-training trigger parsing.

### Editor Training

Implemented editor training uses **Diffusers-native Qwen-Image-Edit LoRA training** through:

```text
src/qwen_edit_project/train/diffusers_qwen_edit_lora.py
configs/train/lora_2509_diffusers.yaml
```

This was added because the final model should be evaluated against the base Qwen-Image-Edit model on a level backend. The self-evolve loop now trains Diffusers-compatible LoRA checkpoints for the official Diffusers Qwen edit backend.

Current editor update style:

- Pairwise preference learning with `pairwise_linear_sdpo`.
- Accepted pairs use the top feasible CEPR candidate as chosen and lower-ranked candidates as rejected.
- Near-miss pairs are allowed for hard object groups when no candidate passes all gates but one candidate is clearly better under raw CEPR components.
- Near-miss chosen images have `preference_sft_weight=0.0`, so failed outputs are never direct SFT targets.
- The weighted SFT manifest is still written for audit/ablation, with rejected SFT targets disabled in the main configs.

### Proposer Training

The proposer is trainable and uses the Qwen-Image-Edit checkpoint's own Qwen2.5-VL/text-encoder side:

```text
proposer.backend: trainable_qwen_image_edit
proposer.model_name_or_path: Qwen/Qwen-Image-Edit-2509
proposer.model_subfolder: text_encoder
```

The proposer is trained by:

```text
src/qwen_edit_project/train/train_proposer_lora.py
```

The proposer reward is band-pass, not "harder is always better." It should generate edit instructions that are useful and learnable for the editor. Very easy edits are weak supervision; overly hard edits produce failed candidates and sparse training.

Object-focused runs enforce each record's scheduled edit type. The learned proposer gets the first
attempt. If strict filtering removes every learned/scripted proposal, the current configs use
`template_fallback_on_target_miss: true` to create a concrete target-type instruction from the
record caption/metadata. This avoids silently starving removal/replacement records while keeping the
fallback auditable through `proposal_plan.jsonl` and `no_proposal_records` in progress/summary files.

### Round Size

Current default:

```text
curriculum.max_records_per_round: 8
candidate_generation.samples_per_proposal: 4
```

Reasoning:

- 8 images per round gives 32 candidate edits per editor update.
- This is enough to reduce single-sample reward noise.
- It is small enough to update frequently and to resume safely on limited GPU allocations.
- Single-image or 3-4-image updates are noisier and can overfit to one bad reward decision.
- Very large rounds are slow and delay feedback.

## Current CEPR Reward Design

The current internal CEPR implementation is in:

```text
src/qwen_edit_project/self_evolve/backends.py
class InternalContrastiveEditPreservationEvaluator
```

The design is intentionally hard-gated and decomposed. It is not a single additive scalar where one strong signal can compensate for a failed constraint.

Conceptual reward:

```text
semantic_edit = sqrt(edit_specificity * taxonomy_score)
raw_reward = sqrt(semantic_edit * preservation)

candidate is feasible only if:
  edit_specificity >= edit_threshold
  taxonomy_score >= taxonomy_threshold
  preservation >= preservation_threshold
  validity >= validity_threshold
  raw_reward >= reward_threshold
```

Important config values in the current main config:

```text
edit_threshold: 0.45
preservation_threshold: 0.20
validity_threshold: 0.50
reward_threshold: 0.30
taxonomy_required: true
top_m: 1
```

### CEPR Components

Current CEPR measures:

- **Edit specificity**: Internal prompt-gain toward the true edit instruction versus distractor edit prompts.
- **Taxonomy score**: Whether the candidate matches the structured edit family and target/replacement prompts better than distractors.
- **Semantic preservation**: Source and edited image similarity in Qwen internal semantic features.
- **Latent locality / preservation**: VAE-latent change should be localized and not globally destructive.
- **Validity**: Candidate should have enough edit-region support without excessive latent drift.

### Why We Used This Design

1. **Internal-only novelty**
   - Using GPT-4V, Qwen-VL, InternVL, or another external VLM at training time would weaken the novelty. The method would look like external reward-model distillation.

2. **Preservation is central to editing**
   - Image editing is not just "generate a good image." The output must satisfy the requested edit while preserving unrelated content.

3. **Benchmark scorers are too expensive for training**
   - GEdit and ImgEdit scoring use external GPT-style evaluation and cannot be called inside the self-evolve loop.

4. **Hard gates reduce reward hacking**
   - A candidate should not be accepted just because it has high prompt similarity if it destroys the source image.

5. **Preference learning avoids direct imitation of failures**
   - Accepted-only SFT starves hard object edits, while rejected-image SFT teaches failures. The current method uses internal same-group preferences and suppresses SFT anchoring for near-miss failures.

## What Is Implemented Beyond Training

### Monitoring

Implemented monitor:

```text
src/qwen_edit_project/self_evolve/monitor.py
```

It generates round-level metrics and a dashboard. The health signal combines:

- group success
- accepted CEPR quality
- weighted training mass

This is a diagnostic signal, not a final benchmark metric.

### Resume and Checkpointing

Implemented:

- Self-evolve round resume.
- Editor LoRA checkpoint saving.
- Proposer checkpoint saving.
- Training command logging.
- Progress JSON per round.
- Evaluation export resume.
- Evaluation scorer retry/resume for ImgEdit.

Important recent evaluation fixes:

- ImgEdit data preparation now supports the current `Benchmark/singleturn/...` layout.
- ImgEdit export creates model output directories correctly.
- ImgEdit export skips already generated samples.
- ImgEdit scorer now retries failed keys and writes detailed diagnostics.
- Config overrides now parse `null` and `none` as actual `None`.

### Benchmarks Wired

Implemented benchmark infrastructure:

- ImgEdit export + scoring
- GEdit export + scoring
- GenEval export + scoring
- DPG-Bench export + scoring
- OneIG-Bench export + scoring

For this phase, the most relevant benchmark has been ImgEdit because it is faster to score than GEdit and directly tests edit quality.

## Current Experimental Status

### Original 1024-Image Main Run

Original main run:

```text
outputs/self_evolve/final_cepr_weighted_magicbrush_1024/internal-cepr-trainable-proposer
```

Observed behavior:

- Early rounds produced useful reward/training signal.
- After roughly round 32, acceptance collapsed.
- Later rounds had almost zero accepted groups.
- Preservation failures dominated.
- Round 122 is not a usable final checkpoint despite being late in training.

Best internal checkpoint from this run:

```text
round_23/training_output/pytorch_lora_weights.safetensors
```

### Stabilized Continuation Run

Continuation run from original round 23:

```text
outputs/self_evolve/continue_r23_stable_magicbrush_256/internal-cepr-trainable-proposer
```

Best internal checkpoint appeared around:

```text
round_24/training_output/pytorch_lora_weights.safetensors
```

The continuation also started to degrade after later rounds. This supports the conclusion that the current CEPR reward is not aligned enough to keep improving over many rounds.

### ImgEdit Result for Current Best Checkpoint

Evaluated checkpoint:

```text
outputs/self_evolve/continue_r23_stable_magicbrush_256/internal-cepr-trainable-proposer/round_24/training_output/pytorch_lora_weights.safetensors
```

Model name:

```text
cepr_stable_continue_r24
```

ImgEdit result:

```text
overall_average: 4.199090909090905
count: 737
unscored_keys: []
```

Type scores:

```text
background: 4.37
adjust:     3.58
style:      4.84
extract:    3.47
remove:     4.10
replace:    4.72
add:        4.43
compose:    3.85
action:     4.50
```

Interpretation:

- This is a complete and valid ImgEdit score.
- It is likely below the expected Qwen baseline around `4.27`, but the baseline must be re-evaluated with the exact same current scorer before making a final claim.
- The current method is not yet strong enough for a convincing ImgEdit improvement claim.

### Baseline Evaluation Status

Baseline evaluation command should use the base Qwen-Image-Edit model with no checkpoint:

```bash
BASE_MODEL=qwen_edit_2509_baseline_imgedit

bash scripts/export_imgedit.sh \
  --batch-size 1 \
  --set model.model_type=base \
  --set model.backend=official_diffusers \
  --set model.model_name="$BASE_MODEL"

bash scripts/score_imgedit.sh \
  --set model.model_name="$BASE_MODEL" \
  --set scoring.num_processes=1 \
  --set scoring.retry_num_processes=1 \
  --set scoring.max_retry_rounds=8 \
  --set scoring.retry_sleep_seconds=20
```

Do not compare against old or reported numbers unless the baseline has been scored through the same current local pipeline and scorer.

## Main Technical Gap

The current CEPR reward is too proxy-like for hard semantic edits.

It can identify:

- simple color/background/style changes
- some preservation failures
- local latent drift
- broad semantic movement toward the edit prompt

It is weaker for:

- object replacement
- object removal
- extraction
- composition
- spatial relation changes
- edits where old-state removal matters
- cases where the model makes a plausible but wrong semantic transformation

Example failure mode:

```text
Instruction: replace the person with a stuffed animal.

Current CEPR may see:
  - image changed
  - source semantics partly preserved
  - prompt similarity improved

But it may not explicitly verify:
  - the source actually had a person
  - the edited image contains a stuffed animal
  - the person is gone
  - the background and unrelated objects remain intact
```

This explains why internal reward health can look useful while ImgEdit does not improve enough.

## Why Not Just Use External VLM Reward

External VLM reward would be easier technically, but it weakens the paper story:

- many existing methods already use external VLM/reward models
- reviewers may see the result as reward-model distillation rather than self-evolving editing
- external calls are expensive and brittle
- the novelty target is internal self-verification from the editor's own model components

External VLMs are acceptable for final evaluation only, not training-time reward.

## Recommended Solution: Internal Rubric CEPR

The next method should be:

```text
internal-cepr-rubric-v1
```

The core idea is to keep rewards internal, but make them explicit and rubric-based.

Instead of relying only on continuous embedding movement, the proposer should emit a structured rubric, and the Qwen-Image-Edit understanding branch should answer atomic verification questions over the source and edited image.

This is motivated by:

- Qwen-Image's architecture: Qwen-Image-Edit uses Qwen2.5-VL semantic representations and VAE reconstructive representations for edit consistency. See https://arxiv.org/abs/2508.02324.
- Rubric-style reward design: explicit criteria are easier to debug and less hackable than opaque scalar rewards. See Auto-Rubric as Reward: https://arxiv.org/abs/2605.08354.

### Proposed Structured Proposal Schema

Extend proposer outputs toward:

```json
{
  "instruction": "Replace the person with a stuffed animal while preserving the snowy slope.",
  "edit_type": "replace",
  "target": "person",
  "replacement": "stuffed animal",
  "required_after": [
    "a stuffed animal is visible"
  ],
  "forbidden_after": [
    "a person remains visible"
  ],
  "preserve": [
    "snowy slope",
    "background",
    "camera viewpoint"
  ],
  "difficulty": "semantic_replacement"
}
```

### Proposed Internal Rubric Questions

For each candidate, ask the internal Qwen understanding path atomic questions:

```text
Source grounding:
  Q1: In the source image, is the target object visible? yes/no

Edit success:
  Q2: In the edited image, is the required new object/attribute/relation visible? yes/no

Old-state removal:
  Q3: In the edited image, is the forbidden old object/attribute still visible? yes/no

Contrastive change:
  Q4: Did the edited image increase the requested concept compared with the source? yes/no

Preservation:
  Q5: Are the listed preserved objects/background/layout still present? yes/no

Validity:
  Q6: Is the edited image realistic, non-corrupt, and visually coherent? yes/no
```

Important: do not ask the model for one free-form score like "rate this edit." That is too noisy and easy to exploit. Ask small, inspectable yes/no or ordinal questions.

### Proposed New Reward

Candidate reward:

```text
source_grounded = source_target_present
edit_success = required_after_present
old_state_removed = 1 - forbidden_after_present
preservation = explicit_preservation_score
validity = visual_validity_score

rubric_reward =
  source_grounded
  * geometric_mean(
      edit_success,
      old_state_removed,
      preservation,
      validity
    )
```

For edit types where old-state removal is not applicable, omit that component.

Then combine with existing CEPR latent preservation:

```text
final_reward =
  hard_gate(source_grounded, validity)
  * geometric_mean(
      rubric_edit_success,
      explicit_preservation,
      latent_preservation
    )
```

The current CEPR embedding and VAE signals should remain as safety checks. The new rubric layer should become the main semantic edit-success signal.

### Training Use

Use rubric reward for candidate ranking and weighted SFT:

```text
For each source image:
  generate 1 proposal + rubric
  sample K=4 candidate edits
  score each candidate with internal rubric CEPR
  choose top feasible candidate
  assign high SFT weight to top candidate
  assign small weight to partial-but-valid candidates
  assign zero weight to invalid or semantically wrong candidates
```

For proposer training:

```text
proposal_reward =
  moderate_difficulty_band
  + best_candidate_rubric_reward
  + proposal_schema_validity
  - too_easy_penalty
  - impossible_edit_penalty
```

This keeps the proposer from generating either trivial edits or impossible edits.

## Implementation Plan for Next Agent

### Step 1: Add Structured Rubric Fields

Files likely involved:

```text
src/qwen_edit_project/self_evolve/edit_schema.py
src/qwen_edit_project/self_evolve/backends.py
src/qwen_edit_project/self_evolve/proposer_training.py
```

Tasks:

- Ensure proposals include `target`, `replacement`, `required_after`, `forbidden_after`, and `preserve`.
- Validate and normalize missing fields.
- Add fallback rubric construction for older/scripted proposals.

### Step 2: Add Internal Qwen Rubric Verifier

Likely location:

```text
src/qwen_edit_project/self_evolve/backends.py
```

Tasks:

- Reuse loaded Qwen-Image-Edit understanding/text encoder path.
- Implement atomic yes/no scoring prompts.
- Return structured signals:

```text
rubric_source_grounded
rubric_required_after
rubric_forbidden_after_absent
rubric_preservation
rubric_validity
rubric_reward
```

Need to be careful about memory. Current CEPR already frees generation-only GPU memory before scoring. Keep that behavior.

### Step 3: Integrate With Existing CEPR

Add new evaluator backend:

```text
internal_cepr_rubric
```

Do not overwrite `internal_cepr` until the new reward is tested.

Start with:

```text
final_reward = sqrt(rubric_reward * cepr_preservation)
```

Gate on:

```text
source_grounded >= threshold
rubric_edit_success >= threshold
cepr_preservation >= threshold
cepr_validity >= threshold
```

### Step 4: Run Small Pilot

Use 8 source images, K=4, 2-4 rounds first.

Check:

- accepted groups per round
- rubric failure reasons
- whether accepted samples visually look correct
- whether reward no longer accepts trivial background/color-only edits for semantic-replacement instructions

### Step 5: Run 24-32 Round Experiment

Only after pilot reward traces look sane:

- run 24-32 rounds
- evaluate checkpoints around round 8, 16, 24, 32
- do not wait for 100+ rounds before evaluating

## Current Risks

1. **Baseline may beat current checkpoint**
   - Current ImgEdit checkpoint scored `4.1991`.
   - If exact baseline is around `4.27`, current method is below baseline.

2. **Reward collapse**
   - Earlier long run collapsed after early rounds. This suggests current CEPR can become too sparse or misaligned.

3. **Training-time reward is not benchmark-aligned enough**
   - Current CEPR does not explicitly verify hard semantic edit success.

4. **Time budget**
   - Full 1024-image self-evolve is slow. Evaluate earlier checkpoints and smaller runs before committing to long jobs.

5. **Evaluation cost and fragility**
   - ImgEdit/GEdit use OpenAI calls. The current scorer has retry/resume support, but scoring still takes time.

## Practical Next Actions

1. Finish exact Qwen baseline ImgEdit evaluation with current scorer.
2. If baseline beats `cepr_stable_continue_r24`, do not report current CEPR as final method.
3. Implement `internal_cepr_rubric`.
4. Run a tiny pilot and inspect accepted candidates.
5. Run a 24-32 round experiment if pilot looks aligned.
6. Evaluate checkpoints early on ImgEdit and, if promising, GEdit.

## Useful Commands

### Score Current Best Checkpoint on ImgEdit

```bash
MODEL=cepr_stable_continue_r24

bash scripts/score_imgedit.sh \
  --set model.model_name="$MODEL" \
  --set scoring.num_processes=1 \
  --set scoring.retry_num_processes=1 \
  --set scoring.max_retry_rounds=8 \
  --set scoring.retry_sleep_seconds=20
```

### Evaluate Base Qwen on ImgEdit

```bash
BASE_MODEL=qwen_edit_2509_baseline_imgedit

bash scripts/export_imgedit.sh \
  --batch-size 1 \
  --set model.model_type=base \
  --set model.backend=official_diffusers \
  --set model.model_name="$BASE_MODEL"

bash scripts/score_imgedit.sh \
  --set model.model_name="$BASE_MODEL" \
  --set scoring.num_processes=1 \
  --set scoring.retry_num_processes=1 \
  --set scoring.max_retry_rounds=8 \
  --set scoring.retry_sleep_seconds=20
```

### Compare ImgEdit Summaries

```bash
python3 - <<'PY'
import json, pathlib

models = [
    "qwen_edit_2509_baseline_imgedit",
    "cepr_stable_continue_r24",
]
for model in models:
    path = pathlib.Path(f"outputs/scores/imgedit/{model}_summary.json")
    summary = json.loads(path.read_text())
    print(model, summary["metrics"]["overall_average"])
PY
```

## Bottom Line

The current implementation is a functioning self-evolve system with trainable proposer, Diffusers-native editor LoRA training, internal CEPR reward, robust checkpoint/resume, monitoring, and benchmark export/scoring.

The current weakness is reward alignment. CEPR is internally principled but not explicit enough for hard semantic edit verification. The next credible solution is **Internal Rubric CEPR**: keep rewards internal, but use Qwen-Image-Edit's own understanding path to answer structured source/edit/preservation questions and combine those answers with existing VAE preservation checks.

This is the most defensible path to improving benchmark performance while preserving the project's internal-reward novelty.

## Latest Implementation Direction: Conservative Pairwise Self-Evolution

The current code/config now implements a stronger no-harm preference pipeline:

- `candidate_generation.reference_candidates` generates one `reference:base` candidate per proposal from the initial Qwen editor state.
- `training.preference.base_relative` compares policy candidates against that base/reference candidate.
- Policy candidates are chosen only when they beat the base by margin; if base wins, the pair is reversed and used as a no-harm preference.
- Ambiguous base-vs-policy comparisons are skipped by default.
- Constraint-aware pair scoring prioritizes CEPR/rubric preservation and validity, with internal VLM only as an agreement/weighting signal.
- `training.preference.vlm_pair_guard` confidence-weights pairs and can reject VLM-disagreeing pairs.
- `output.use_cumulative_preference_manifest=true` replays previous high-confidence preference pairs across rounds.
- Proposer SFT now records policy-over-base margins and can reward useful base-improving proposal distributions.

Primary patched config:

```bash
configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml
```

The config now writes to:

```bash
outputs/self_evolve/qwen_edit_2509_conservative_pairwise_v1
```

Run only inside a Slurm allocation/tmux resource session using the `qedit` environment.

## Latest State: 2026-06-02

Active diagnostic run:

- tmux: `uug_balanced_v2`
- Slurm job: `504`
- output root: `outputs/self_evolve/balanced_cepr_v2_selfplay_margin_diag48_r4_20260602T095235Z`
- method: closed-loop editor/proposer self-play with internal CEPR/rubric reward and pairwise diffusion preference training.

Current findings:

- Round 1 improved ImgEdit 32-canary by `+0.0834`, but regressed GEdit subject-replace/cn 32-canary by `-0.2480` overall.
- Round 2 had only one accepted candidate and mostly near-miss preference pairs.
- Failure audit showed object removal/replacement candidates had decent raw CEPR scores but failed the stricter rubric gates, especially `rubric_forbidden_after_absent` and `rubric_edit_success`.

Code/config fix for future clean runs:

- `src/qwen_edit_project/self_evolve/loop.py` now supports `training.preference.near_miss_contract_filter`.
- `configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml` enables that filter for local edit types so failed object/spatial/local edits cannot become chosen near-miss preference positives.
- `src/qwen_edit_project/self_evolve/backends.py` also has deterministic scheduled-edit fallback templates for non-object edit types; active v2 will not pick this up until a new process starts.

Recommended next run:

- Start a clean v3 diagnostic after current evals finish, using the patched source and a fresh output root.
- Keep `include_rejected=false`; keep pairwise preference learning; use enough records/candidates to compensate for the stricter near-miss filter.

Current replacement run:

- tmux: `uug_balanced_v2`
- output root: `outputs/self_evolve/balanced_cepr_v4_anchor_contract_diag96_r4_20260602T114756Z`
- reason: v2/v3 showed object edits could fail but still create noisy near-miss preferences; v4 uses anchored object target regions and a strict near-miss contract.
- early signal: object removal/replacement are still rejected, but attribute/material groups are accepted, so the run is cleaner and not fully starved.

Current eval:

- tmux: `uug_cepr_nearmiss`
- ImgEdit 32-canary for the same checkpoint was positive: `+0.0934375` over baseline.
- GEdit subject-replace/cn 32-canary was negative: `-0.2681202266795324` overall, so
  v2 round 2 is not a broad final method.
- The same tmux is now running `object_prompt_probe_direct_anchor_r1_20260602T122705Z`,
  an 8-record object-only generation/scoring probe with training disabled.
