# No-Router Checkpoint Recovery Plan

## Verdict

Yes, we can still pursue a single-checkpoint method that improves both ImgEdit and GEdit without test-time routers or candidate selectors. The current evidence says the failure is not that self-evolution is impossible; it says our pseudo-target construction is not yet trustworthy enough for SFT.

The next method should not be "train longer". It should be contract-calibrated self-training: only train on edits that demonstrably satisfy the edit contract, beat or match the base model under a calibrated judge, and preserve the pretrained editor's high-ceiling behavior.

## What Is Actually Going Wrong

### 0. The forbidden-object absence score is not reliable as a hard gate

The strict object-recovery probe exposed a reward-model false negative, not just a
model failure. A visually valid removal of the party hat was rejected because
`rubric_forbidden_after_absent` stayed near `0.10`, even though the required-after
and preservation scores were high and manual inspection showed that the object was
removed. The current absence check is based on text-image support for prompts such
as "party hat remains visible"; this support can remain high because the prompt
contains the salient source object terms and the embedding model is weak at local
absence and exact spatial grounding.

The corrected policy is:

- keep `rubric_forbidden_after_absent` as a logged diagnostic;
- do not use it as the only hard gate for object removal/replacement;
- accept object candidates through an asymmetric contract requiring strong
  required-after support, high preservation, validity, edit specificity, taxonomy,
  and raw reward;
- use strict forbidden-object gating only after replacing/calibrating it with a
  grounded absence check, such as detector/segmentation evidence or a stronger
  pairwise VLM judge.

Update from the corrected v5 self-evolution run:

- The 2-record object soft-contract probe
  `outputs/self_evolve/object_soft_contract_probe_r1_20260602T125658Z` accepted
  both object groups and included zero rejected examples in the SFT audit.
- The accepted removal still had `rubric_forbidden_after_absent` around `0.09`,
  confirming that the old absence score is a false-negative diagnostic on this
  case, not a reliable hard training gate.
- The active v5 diagnostic
  `outputs/self_evolve/balanced_cepr_v5_soft_object_contract_diag96_r4_20260602T130155Z`
  has already accepted its first object-removal group under this asymmetric
  contract and moved to object replacement.
- Code now supports per-edit-type contract overrides for strict forbidden gating,
  disabled component scores, and minimum component thresholds. Disabled
  components stay disabled even if future configs add shorthand `min_*` keys.
- Code also supports edit-type-specific accepted-pair margins and accepted-pair
  weight scales. This was needed because object replacement candidates can all
  pass the object contract while their CEPR rewards differ by only a few
  thousandths. The current config uses those tiny margins only as low-weight
  intra-feasible ranking signal, not as full-confidence preference supervision.

This is consistent with recent reward-modeling work for image editing: successful
reward models use multidimensional criteria for instruction alignment, visual
quality, and preservation rather than a single scalar or one negative prompt.

Current implementation update: the evaluator now has an optional GroundingDINO
object-contract verifier. For object removal/replacement it checks whether the
source object is detectable in the original image, whether the source object
drops below an absence cutoff in the edited image, and whether the replacement
target is detectable for replacement edits. This is the research-grade path for
the old-object absence signal: object edits need localization evidence, not only
global text-image similarity.

Research basis:

- Grounding DINO provides open-set object detection from category names or
  referring expressions, which directly matches our need to verify source and
  target object presence without benchmark-specific labels:
  https://arxiv.org/abs/2303.05499
- SAM/Grounded-SAM show the broader detector-plus-segmentation path for
  grounded editing and automatic annotation; segmentation should be the next
  step if bounding-box presence is still too coarse:
  https://arxiv.org/abs/2304.02643 and
  https://github.com/IDEA-Research/Grounded-Segment-Anything
- Recent VLM missing-object analysis shows that VLMs can fail specifically when
  judging removed or missing object parts, which supports not trusting a plain
  VLM/embedding absence score as the sole hard gate:
  https://openreview.net/forum?id=OuqnHLrjB1
- Diffusion-DPO and Curriculum-DPO motivate the next step after clean candidate
  verification: learn from preference pairs and curricula rather than positive
  SFT alone:
  https://arxiv.org/abs/2311.12908 and https://arxiv.org/abs/2405.13637

### 1. The failed broad checkpoint learned from rejected targets

The completed final-512 run still includes rejected rows in SFT:

- Run: `outputs/self_evolve/final_rubric_cepr_v1_512_k2_scripta30_steps128/internal-cepr-rubric-trainable-proposer/round_04`
- Round-4 summary: `219` included candidates, `82` accepted, `137` rejected.
- Cumulative `train_manifest.jsonl`: `975` rows, including many rejected object rows.
- Object replacement: `45` rows, only `2` accepted and `43` rejected.
- Object removal: `36` rows, only `3` accepted and `33` rejected.

This explains the full ImgEdit regression from `4.4406` to `4.1991`, with the largest damage in adjustment, removal, compose, and action edits.

### 2. The object-focused run removed rejected rows but accepted bad object targets

The later object-primary run fixed `include_rejected=false`, but enabled softened forbidden-object gates and feasible-ranked positives. Its round-2 SFT manifest has:

- Run: `outputs/self_evolve/object_primary_typed_v3_after_bootstrap192_r2_48_s3/round_02`
- `239` total training rows.
- `80` reconstruction replay rows.
- `62` object-addition rows with healthy rubric scores.
- `51` object-removal rows where all `51/51` have `rubric_forbidden_after_absent < 0.30` and `rubric_edit_success < 0.40`.
- `46` object-replacement rows where all `46/46` have `rubric_forbidden_after_absent < 0.30` and `rubric_edit_success < 0.40`.

So the object run was not actually learning clean removal/replacement. It was imitating candidates where the required new object/scene may be present, but the old forbidden object was probably still present or insufficiently removed.

### 3. The reward is absolute, not baseline-relative

The current loop accepts candidates if their internal reward passes thresholds. It does not require the candidate to beat the original Qwen baseline output for the same instruction. This is a problem because Qwen-Image-Edit-2509 is already near the ceiling:

- Full ImgEdit baseline: `4.4406 / 5.0`.
- Replace baseline: `4.833`, style baseline: `4.810`, action baseline: `4.796`.
- Many examples are already scored `5.0`, so SFT can easily damage high-scoring examples while only weakly improving low-scoring ones.

For high-baseline examples, the correct action is usually distillation/replay or skipping, not training on a merely feasible pseudo-target.

### 4. GEdit transfer is negative because the training distribution and contracts do not match GEdit

Full GEdit baseline is already strong:

- Baseline overall: `8.1238`.
- Semantics: `8.8078`.
- Quality: `7.9175`.

Existing GEdit canaries for trained checkpoints are negative:

- `object_v3_r1_s40_gedit_subject_remove_n32`: overall `-0.2148`, semantics `-0.4063`.
- `final_object_recovery_r1_s40_gedit_subject_remove_n32`: overall `-0.4266`, semantics `-0.5938`.

This is consistent with the object-contract audit above. If removal/replacement pseudo-targets do not actually remove or replace the old object, GEdit semantics must drop.

## Method Direction

### A. Contract-calibrated candidate acceptance

For each proposed edit, define hard edit-family contracts. For object removal/replacement:

- `rubric_required_after >= 0.70`.
- `rubric_preservation >= 0.85` for pseudo-target SFT.
- `rubric_validity >= 0.80`.
- `cepr_raw_reward >= 0.55`.
- `cepr_taxonomy >= 0.40`.
- `rubric_forbidden_after_absent` logged, but not used as a hard object gate until
  the absence score is detector/VLM calibrated.

For additions, background, style, material, text, and local adjustments, use family-specific contracts rather than one global reward threshold.

Implementation status:

- Added optional SFT contract filtering in `src/qwen_edit_project/self_evolve/loop.py`.
- Added audit tool: `scripts/audit_self_evolve_training_contract.py`.
- Added per-candidate gate failure signals in `src/qwen_edit_project/self_evolve/backends.py`.
- Added configurable audit thresholds in a retired object-contract launcher so
  strict and soft object contracts were audited with the thresholds they actually used.
- Fixed a continuation-training failure where a rank-32 warm-start LoRA was resumed
  with a rank-16 trainer. Object-recovery launchers now expose the trainer LoRA
  rank/alpha and the clean RR soft-object run uses rank `32`, alpha `32`.
- Fixed a second continuation mismatch where the self-evolve trainer targeted only
  attention LoRA modules while the warm-start also contained `img_mod`,
  `txt_mod`, `img_mlp`, and `txt_mlp` adapters. The clean RR soft-object run now
  resumes with the same full LoRA module list as the warm-start checkpoint.

### B. Baseline-relative pseudo-target selection

Each training candidate should be compared against the base Qwen output for the same input and instruction.

Accept a pseudo-target only if:

- It passes the hard contract.
- It beats the base output by a margin under the calibrated internal/VLM judge, or the base output fails the contract and the candidate passes.
- If the base output is already high-confidence, add base-output distillation or reconstruction replay instead of updating toward a risky candidate.

This is the main missing ingredient for improving a strong base model without routers.

### C. Preference learning, not only positive SFT

SFT on positive pseudo-targets is too weak when the selected positives are noisy. The research-grade objective should include pairs:

- Positive: contract-passing candidate or curated target.
- Negative: failed candidate, base output when it violates the target edit, or same candidate before contract repair.

Use a DPO/IPO/SimPO-style objective or an equivalent pairwise preference loss, plus small positive SFT. This teaches the model the boundary between "object removed" and "object still present", which SFT alone does not.

### D. Balanced benchmark-aligned training data

The next dataset should not be color-heavy or object-only. Build a non-eval training pool balanced over the benchmark families:

- ImgEdit: adjust, style, background, extract, remove, replace, add, compose, action.
- GEdit: background_change, color_alter, material_alter, motion_change, ps_human, style_change, subject-add, subject-remove, subject-replace, text_change, tone_transfer.

Do not train on benchmark images. Use MagicBrush train/source pools, other non-eval images, and generated/translated instructions. Include Chinese instructions for GEdit-like categories because the GEdit canaries tested Chinese subject/background edits and transfer was negative.

### E. Strong preservation and low learning pressure

Because the base model is strong, the checkpoint should be a small corrective update:

- LoRA LR: `1e-6` to `5e-6` for continuation runs.
- High replay/distillation: `0.75` to `1.0` relative to edit samples.
- Disable rejected and feasible-ranked-positive SFT unless they pass the same strict contract.
- Evaluate checkpoints early on fixed canaries, but only trust full ImgEdit/GEdit for final claims.

## Latest Single-Checkpoint Findings

### Prompt-mix object anchor

The current best no-router checkpoint is:

- `outputs/checkpoints/qwen_edit_2509_magicbrush_rr_promptmix_anchor_lr5e6_s256`

It trains on clean MagicBrush removal/replacement pairs with both plain prompts and strict contracts, plus 1:1 replay. Current canary results:

- ImgEdit remove/replace 32-case canary: `+0.0525`.
- ImgEdit broad 64-case canary: `+0.0261`, `17/35/12` wins/ties/losses.
- GEdit subject-remove Chinese 32-case canary: `+0.1520` overall.
- GEdit subject-replace Chinese 32-case canary: `-0.0822` overall.

This is a real improvement over the earlier negative object runs, but it is still weak. Its broad ImgEdit losses are concentrated in remove, extract, add, and style.

### Balanced continuation is rejected

The broad balanced MagicBrush continuation is not a viable main method:

- `outputs/checkpoints/qwen_edit_2509_magicbrush_balanced_promptmix_continue_lr2e6_s192`
- ImgEdit broad 64-case canary: `-0.0681`.
- It mainly damages adjust, extract, and style.
- It slightly improves GEdit subject-replace (`+0.0074`), but the ImgEdit regression makes it unacceptable.

This negative result is important: "balance every edit type with more SFT" is not enough. The training data has to match the benchmark edit contract. MagicBrush does not provide reliable extract/adjust/style supervision for this setting, and the base model is already near ceiling on many of those categories.

### Current corrective run

The next run narrowed the fix instead of broadening it:

- `outputs/checkpoints/qwen_edit_2509_magicbrush_rra_promptmix_continue_lr1e6_s96`
- Manifest: `data/manifests/magicbrush_object_rra_promptmix_train_512_replay125.jsonl`
- Families: object removal, replacement, addition.
- Replay ratio `1.25`, edit weight `0.60`, replay weight `0.55`.
- Warm start: prompt-mix anchor.
- LR `1e-6`, 96 steps.

It did not beat the prompt-mix anchor:

- ImgEdit broad 64-case canary: `+0.0052`.
- GEdit subject-replace Chinese 32-case canary: `-0.0962`.

Additional checkpoint-selection diagnostics also failed:

- Prompt-mix anchor with global LoRA scale `0.5`: ImgEdit broad `-0.0417`, GEdit subject-replace `-0.0731`.
- Balanced continuation checkpoint-64: ImgEdit broad `-0.1358`, GEdit subject-replace `-0.0490`.

Therefore, the best current no-router SFT checkpoint remains the full prompt-mix anchor:

- `outputs/checkpoints/qwen_edit_2509_magicbrush_rr_promptmix_anchor_lr5e6_s256`

The next research-grade step is baseline-relative selection or pairwise preference learning, not another positive-SFT variant.

## Immediate Experiment Plan

### Experiment 1: Calibrated object-contract sanity run

Goal: verify that clean SFT targets do not damage full-checkpoint behavior.

Settings:

- `include_rejected=false`.
- `include_feasible_ranked_positives=false`.
- `evaluator.rubric_soft_forbidden_edit_types=["object_removal","object_replacement"]`.
- Enable `training.weighted_sft.contract_filter`.
- Do not require strict forbidden gate for removal/replacement until absence is recalibrated.
- High reconstruction/base replay.
- Low LR.

Decision rule:

- If the SFT manifest contains rejected rows, stop.
- If accepted removal/replacement rows have weak required-after, preservation, validity,
  taxonomy, or raw reward, stop and tighten the contract.
- If visual spot checks show old objects actually remaining, replace the soft absence
  policy with detector/segmentation or pairwise VLM absence evidence before scaling.
- If canary is neutral/positive on both ImgEdit and GEdit subject-remove/background slices, scale data.

### Experiment 2: Baseline-relative selection run

Goal: avoid training on pseudo-targets that are worse than the base model.

For every generated candidate, also score the base Qwen output. Keep only candidate-vs-base wins with margin. For base wins or ties, use base-output distillation/replay.

Decision rule:

- We should see fewer accepted pseudo-targets but higher precision.
- The checkpoint should not regress high-ceiling ImgEdit families such as replace, style, action.

### Experiment 3: Pairwise preference run

Goal: learn the reward boundary directly.

Construct preference pairs from self-evolution traces:

- Positive: strict accepted candidate.
- Negative: failed candidate with old object still present, over-edited preservation failure, or base output when base fails the edit contract.

Train a small LoRA with preference loss plus replay/SFT.

Decision rule:

- GEdit semantics should improve or at least stop regressing on subject-remove/subject-replace canaries.
- ImgEdit extract/remove/compose should improve without hurting replace/style/action.

## Expected Improvement Range

A `+0.8` to `+1.0` absolute gain on full ImgEdit is mathematically impossible from the current baseline because the baseline is already `4.4406 / 5.0`; the maximum possible average gain is about `+0.5594`.

A realistic no-router target is:

- Short term: neutral to `+0.05` full ImgEdit with no GEdit regression, proving the corrected training is not harmful.
- Medium term: `+0.10` to `+0.25` if baseline-relative filtering and preference learning work.
- Large gains are only plausible on low-baseline subsets such as ImgEdit extract/compose or GEdit style/material/ps_human, not uniformly across all examples.

## Commands For Auditing

Run this audit after every round writes `train_manifest.jsonl`:

```bash
python3 scripts/audit_self_evolve_training_contract.py \
  outputs/self_evolve/<run_name>/round_01/train_manifest.jsonl \
  --min-forbidden 0.30 \
  --min-success 0.40
```

For strict object runs, use stronger thresholds:

```bash
python3 scripts/audit_self_evolve_training_contract.py \
  outputs/self_evolve/<run_name>/round_01/train_manifest.jsonl \
  --min-forbidden 0.65 \
  --min-success 0.55
```

Training and evaluation must still be launched only inside a Slurm allocation, preferably from a tmux resource session, using the `qedit` conda environment.

## 2026-05-30 Object-Grounded Failure Update

The softened object-forbidden run is not trustworthy as a main method:

- Run: `outputs/self_evolve/no_router_rr_clean_soft_forbidden_r1_32_seed123`.
- Training manifest: 18 accepted object pseudo-targets plus 14 replay rows, with zero rejected rows.
- Fast object canary: `outputs/quick_eval/imgedit_object_rr_n32/rr_clean_soft_r1_32_object_rr_n32_vs_baseline_comparison.json`.
- Result on 32 remove/replace ImgEdit examples: baseline `4.7394`, candidate `4.5728`, delta `-0.1666`, wins/ties/losses `7/18/7`.

Detector-gated self-generation also exposed the real failure point:

- Run: `outputs/self_evolve/no_router_rr_grounded_detector_r1_8_seed123_v4_strictprompt`.
- Result: 14 generated object candidates, 0 accepted, empty train manifest.
- Dominant rejection reason: `object_detector_contract`.
- The internal rubric often assigned high required-after/preservation scores while GroundingDINO still detected the source object in the edited image above the absence cutoff.

Conclusion: the current base/strict editor does not reliably self-generate correct object removal/replacement positives. Training longer on self-generated positives is therefore unsafe. The permanent path is:

1. Use human-supervised object pairs as the corrective editor signal, with high replay.
2. Keep detector/SAM/VLM object contracts as acceptance gates for future self-generated data.
3. Treat failed detector-gated generations as negatives for a later preference/DPO stage, not as SFT targets.
4. Re-enter self-evolution only after the supervised corrective checkpoint produces detector-passing object candidates at a useful rate.

Historical corrective run:

- Launcher: retired during scripts cleanup; reproduce with the generic training entrypoint.
- Output: `outputs/checkpoints/qwen_edit_2509_magicbrush_rr_strict_continue_ckpt320_lr3e6_s256`.
- Base LoRA: `outputs/checkpoints/qwen_edit_2509_magicbrush_rr_strict_lora/checkpoint-320`.
- Manifest: `data/manifests/magicbrush_object_rr_strict_train_512_replay100.jsonl`.
- LR `3e-6`, 256 steps, 512 object remove/replace positives plus 512 replay rows.

## 2026-05-30 Prompt-Mix Anchor Update

The strict-continuation canary did not solve the object regression:

- Pre-continuation checkpoint-320 canary: baseline `4.7394`, candidate `4.6766`, delta `-0.0628`.
- Continued checkpoint canary: baseline `4.7394`, candidate `4.6769`, delta `-0.0625`.
- The continued result has fewer losses, but one visually suspicious judge outlier (`562`, tree replacement) dominates the mean. Dropping only that worst key makes the delta positive, so this slice is noisy but still not a trusted win.

Additional failure point found:

- The strict object SFT used appended contract prompts, while ImgEdit/GEdit inference uses plain benchmark instructions.
- This creates a prompt-distribution mismatch: the adapter can learn a useful object edit behavior but still be unstable under the plain benchmark prompt form.

Current fix:

- Manifest builder now supports prompt variants and object weights:
  - script: `scripts/build_magicbrush_object_manifest.py`
  - manifest: `data/manifests/magicbrush_object_rr_promptmix_train_512_replay100.jsonl`
  - 512 clean object remove/replace pairs.
  - Each pair is emitted with both plain and strict-contract prompts, giving 1024 object-variant rows.
  - 1024 reconstruction replay rows keep a 1:1 object/replay balance.
  - object row weight `0.75`, replay weight `0.5`.
- New training launcher:
  - retired during scripts cleanup; reproduce with the generic training entrypoint
  - output: `outputs/checkpoints/qwen_edit_2509_magicbrush_rr_promptmix_anchor_lr5e6_s256`
  - base model start, rank `8`, LR `5e-6`, 256 steps.
- Evaluation utility now supports a global Diffusers LoRA scale through `model.lora_scale`; this is for checking whether regressions are caused by adapter over-strength rather than the learned direction.

Decision rule:

- Evaluate checkpoint-64 and final checkpoint-256 on the 32-example remove/replace canary.
- If checkpoint-64 is better than checkpoint-256, use early stopping or global LoRA scale rather than more training.
- If both remain negative after removing obvious judge noise, the next research-grade fix is not longer SFT; it is pairwise preference training using detector/VLM-verified positives and explicit failed generations as negatives.
