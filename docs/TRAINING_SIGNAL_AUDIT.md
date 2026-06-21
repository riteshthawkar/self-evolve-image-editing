# Training Signal Audit and Next Experiment Plan

Date: 2026-06-20

This note records the current reward/data diagnosis and the changes made before the next GPU run. No training or evaluation was run on the login node.

Reviewer-facing reward framing is documented separately in `docs/REWARD_SYSTEM_REVIEWER_FRAMING.md`. Use that note when writing the method/story section so the reward is presented as one constrained conservative-editing objective rather than an arbitrary collection of reward components.

## Evidence

Saved audits:

- Reward correlation audit: `outputs/analysis/reward_correlation_audit_conservative_pairwise_r1_18/REPORT.md`
- Strict pair-yield audit: `outputs/analysis/strict_preference_pair_audit_conservative_pairwise_r1_18/REPORT.md`

Key measured facts from the conservative pairwise run over rounds 1-18:

- Candidate rows: 5321 across 773 groups.
- Old accepted candidates: 486.
- 41.6% of old accepted candidates had internal VLM score below 0.35, unreliable internal VLM, and no-op flags.
- 43.0% of old accepted candidates had drift flags.
- Strict internal-VLM success rows under the new proposed gate: 639 / 5321 = 12.0%.
- Strict preference pairs surviving the audit: 352 after balancing.
- Strict surviving pairs are available for replacement, addition, material, color, attribute, background, local enhancement, and style transfer.
- Object removal has 0 strict-success candidates. A focused check found 545 object-removal rows, but 543 are no-op and only 2 have judge score >= 0.35.

## Diagnosis

The main failure is not simply low training time. The previous pipeline allowed high-CEPR candidates to become positives even when the internal VLM judge explicitly said the edit was not performed. This is especially damaging for preservation-sensitive edits because CEPR/preservation can stay high when the model changes nothing.

The second failure is selection coupling: candidate positivity was tied to `status=accepted`, which came from the older ranker. If a VLM-good candidate was rejected by the older ranker, it could not become the chosen side of a preference pair.

The third failure is group entropy. All-fail groups provide no reliable self-evolution signal, and all-pass groups are often easy. Training should focus on middle-band groups where at least one candidate succeeds and at least one candidate fails.

The fourth failure is data/proposal quality for object removal. The model is mostly not removing objects at all on the current sources/proposals. Relaxing the reward would mostly train on no-op removals, so the fix must include better source selection and stricter positive gating.

The GEdit quality regression adds a fifth failure: object replacement/removal near-miss pairs can hurt visual realism even when semantics look correct. On the old `balanced_cepr_v2_selfplay_margin_diag48_r4_20260602T095235Z` run, raw-CEPR replay showed the damaging pattern directly:

- Round 1 had 8 object-replacement and 7 object-removal near-miss pairs selected from groups with no strict positive.
- Round 2 had 6 object-removal and 4 object-replacement near-miss pairs selected from groups with no strict positive.
- These are the same edit families that later produced GEdit quality drops on food/object-composition and hard subject-replacement cases.

## Implemented Changes

### Pair Selection

File: `src/qwen_edit_project/self_evolve/loop.py`

- Added `training.preference.score_mode`, used for normal winner ranking and pair margins.
- Normal pairs now rank by constraint-aware conservative score instead of raw effective CEPR reward.
- Added explicit chosen-side VLM fail rejection in `_preference_vlm_pair_guard`.
- Added `accept_strict_vlm_success_as_positive`: candidates that pass the strict internal-VLM success filter can become positives even if the old CEPR ranker marked them rejected.
- Kept unreliable/failed candidates usable as losers; the chosen side must be reliable, but the loser need not be reliable.
- Added `near_miss_positive_anchor_filter`: object removal/replacement groups cannot create near-miss chosen-side pairs unless the group has a strict positive. This blocks rejected-vs-rejected object supervision while still allowing near-miss learning for safer edit types.
- Added structured object-contract sanity checks for training/pair filters. Object edit samples with malformed slots such as `chicken pieces and fill the area naturally` are rejected instead of being used as positives.
- Added `quality` and `artifact_free` aliases to component thresholding so object edits can explicitly require artifact-free local realism.

Compatibility replay on the old self-play run with raw CEPR scoring:

- Round 1: object near-miss pairs removed by the new anchor filter: 8 replacement, 7 removal.
- Round 2: object near-miss pairs removed by the new anchor filter: 4 replacement, 6 removal.

### Active Config

File: `configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml`

- Uses `score_mode: conservative` for preference selection.
- Requires chosen internal-VLM reliability and judge/semantic/preservation/artifact floors of 0.55.
- Filters groups to success-rate band 0.20-0.80.
- Rejects low-confidence/generalization-floor positives.
- Reduces direct SFT pull inside preference training.
- Increases anchor replay to protect source preservation.
- Disables base-reference wins as positives; base/reference candidates remain useful as no-harm comparisons when policy beats base.
- Requires object removal/replacement positives to pass stronger internal-VLM judge, preservation, validity, artifact-free, and reward floors.
- Requires valid structured object contracts in both preference near-miss filtering and weighted-SFT filtering.
- Blocks object removal/replacement near-miss pairs when no strict positive anchor exists in the candidate group.

### Region-Decoupled Conservative Reward

Files:

- `src/qwen_edit_project/self_evolve/image_metrics.py`
- `src/qwen_edit_project/self_evolve/backends.py`
- `src/qwen_edit_project/self_evolve/loop.py`

The reward now explicitly models conservative image editing:

- target-region edit support: the intended editable region must actually change;
- non-target preservation: pixels outside the target region must remain close to the source image;
- locality/minimality: most changed pixels should fall inside the target region;
- hard acceptance gates: outside damage cannot be compensated by a high semantic score.

For object removal and replacement, the target region is derived from GroundingDINO boxes on the original image. Diff masks are only a logging fallback by default, because using observed changed pixels as the target mask can hide outside corruption. For non-object local edits, the same metric can use detector boxes when a source object is available, but missing masks do not fail the sample unless the edit type is configured as mask-required.

New logged scores include:

- `conservative_region_reward`
- `conservative_region_observed_reward`
- `conservative_target_change_score`
- `conservative_outside_preservation`
- `conservative_localization_precision`
- `conservative_region_gate_pass`
- `conservative_region_reject_reason`

The preference-pair scorer now treats `conservative_region_reward` and `conservative_outside_preservation` as preservation constraints. Object-removal and object-replacement positives require these floors in the active config, so a candidate that removes/replaces the right object but corrupts the rest of the image should become a negative rather than a chosen training target.

### Offline Audit Tool

File: `scripts/audit_strict_preference_pairs.py`

This script audits saved `proposals.jsonl` files without running any model. It reports strict-success rows, productive groups, pair yield, edit-type coverage, and skip reasons. Use it before launching long training jobs.

### Source Selection for Removal-Suitable Data

Files:

- `src/qwen_edit_project/data/select_unlabeled_images.py`
- `configs/data/source_image_filter.yaml`

The VLM source selector now asks for and stores:

- `removable_object_score`
- `small_object_separability`
- `removal_background_fill_score`
- `removable_object_description`

The object-family source-selection threshold now prefers images with small, separable removable objects and plausible background fill. This addresses the measured object-removal no-op problem at the data stage.

## Next Slurm-Only Plan

1. When a GPU is available, run source selection with the updated VLM schema under Slurm/qedit to build a cleaner unlabeled manifest.
2. Launch a small smoke self-evolution run on the new manifest: 2-3 rounds, 32-64 records/round, same strict preference settings.
3. After each round, run the strict pair audit on the saved proposals. Do not scale unless object-removal strict successes become nonzero, conservative-region gate failures are interpretable, and total strict pair yield remains healthy.
4. If the smoke passes, launch the full run with the same config.
5. Evaluate only checkpoints with healthy strict-pair telemetry, first on canaries, then full ImgEdit/GEdit.

Research decision: do not relax object-removal rewards just to get more pairs. Current evidence says that would train no-op removal. The correct next step is better source/proposal quality plus hard chosen-side verification.
