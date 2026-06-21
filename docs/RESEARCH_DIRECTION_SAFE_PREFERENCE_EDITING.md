# Research Direction: Safe Preference Learning for Image Editing

## Current Diagnosis

The base Qwen-Image-Edit model is already strong on many ImgEdit examples, so the main failure mode is not lack of editing ability. The main failure mode is harmful drift: training improves some lower-baseline or object-sensitive cases but damages many already-good cases.

## Current Closed-Loop Direction

The active no-router research direction is now the generalized CEPR v2 self-evolution pipeline:

```text
configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml
```

This direction keeps the main method self-evolving:

- The proposer and editor are both updated round by round.
- The editor trains on self-generated same-group preferences rather than reward-filtered SFT.
- Rejected samples are never direct weighted-SFT targets.
- A uniform edit-type coverage cycle prevents sorted manifest order or easy edit families from dominating the round.
- From round 2 onward, current editor candidates compete against previous-editor candidates for the same proposal.
- The reward remains internal CEPR/rubric scoring over Qwen internal semantic features and Qwen VAE/latent locality; external GPT/VLM scoring is reserved for benchmark evaluation.

The main reason for the design change is that object removal/replacement can be visually correct while failing the strict forbidden-object gate. In that case, the method should not starve the round. It should use raw internal CEPR differences as near-miss preferences, with no direct SFT pull toward a failed image. A live round-2 object-removal group showed this exact behavior: all candidates failed the strict gate, but raw rewards were separated enough to create balanced near-miss preference pairs. The previous/base opponent was ranked above the current policy on that example, which is useful recovery signal and also a warning that round-1 policy had not yet improved that hard edit.

This is the current paper-worthy hypothesis:

> For strong image editors, closed-loop self-evolution needs calibrated preference competition rather than accepted-only SFT. Strict internal gates protect against training on false positives, while near-miss ranking and versioned self-play keep hard edit categories from starving.

Immediate validation gates:

- Round summaries must show broad preference coverage across edit families, with no single family dominating pair construction.
- Preference manifests should include zero direct rejected SFT targets.
- Versioned self-play rows should include both `policy` and `opponent:previous_round` candidate roles.
- Early canaries should check whether the update improves or at least preserves ImgEdit and GEdit before scaling.

Latest object-contract update, 2026-06-02:

- The old strict forbidden-object absence score is confirmed unreliable for object removal/replacement. A visually valid party-hat removal had `rubric_forbidden_after_absent` near `0.09`, while required-after, preservation, validity, taxonomy, and raw CEPR were healthy.
- The active config now uses an asymmetric object contract: keep `rubric_forbidden_after_absent` and `rubric_edit_success` for audit, but disable them as hard training-contract components for `object_removal` and `object_replacement`.
- Object pseudo-targets must still satisfy high required-after support, preservation, validity, taxonomy, and raw CEPR thresholds. Non-object edits keep the strict forbidden gate.
- A 2-record object probe at `outputs/self_evolve/object_soft_contract_probe_r1_20260602T125658Z` completed cleanly after the trigger-handling patch: 2/2 object groups accepted, 2/8 candidates accepted, zero rejected rows included in SFT audit.
- The active corrected v5 run is `outputs/self_evolve/balanced_cepr_v5_soft_object_contract_diag96_r4_20260602T130155Z` in tmux `uug_balanced_v2`. Its first object-removal group accepted one candidate under the soft object contract and moved on to object replacement.
- Object replacement CEPR scores are compressed across feasible candidates, so the older diagnostic builder supported edit-type-specific accepted-pair margins and low accepted-pair weights. The current generalized config removes those object-specific pair weights/margins and instead relies on uniform low pair margins, broad component calibration, hard-negative diversity, and family-balanced sampling.

Future runs should use the updated object-edit schema/prompt safeguards: removal and replacement
proposals now require more spatially grounded `target_region` values, and normalization adds an
explicit "separate source object remains visible" forbidden criterion when the proposer only gives a
generic absence phrase. This keeps the reward internal while reducing false rejections caused by
color/object-name collisions with preserved content.

Generalization update, 2026-06-03:

- Do not add benchmark-specific curriculum for the next method. The next run should improve generic
  edit robustness across all edit families, not specialize to GEdit Chinese replacement or ImgEdit
  categories.
- The active config uses a uniform coverage cycle over removal, replacement, addition, attribute,
  color, material, spatial, background, style, and local enhancement edits. Object-specific pair
  margin/weight gates from the diagnostic run have been removed.
- The active rubric CEPR backend already uses Qwen internal edit-understanding, taxonomy,
  preservation, and latent-locality features. The new calibration reads those CEPR components and
  source-grounding signals directly instead of adding benchmark-specific task or language weights.
- Preference pairs are now calibrated by broad support components before they become training signal:
  semantic edit support, preservation, validity, taxonomy, and internal consistency. Low-confidence
  winners are either rejected by component floors or down-weighted by a confidence multiplier.
- Rejected candidates are tagged by generic failure modes: `under_edit`, `preservation_drift`,
  `invalid_or_artifact`, `taxonomy_mismatch`, `weak_reward`, and `hard_near_miss`. Pair construction
  diversifies the first rejected examples across these failure modes instead of only taking the
  nearest reward loser.
- Preference-mode training now receives preservation anchor replay records. These are identity
  preference pairs with the prompt "reconstruct the input image exactly": source image is chosen,
  self-generated edited image is rejected. This finally applies preservation anchoring to
  `pairwise_linear_sdpo`; the older reconstruction replay only affected SFT manifests.
- Anchor replay records are regularizers and do not count toward minimum real edit preference-pair
  requirements.

Internal Qwen-VLM judge update, 2026-06-03:

- The active rubric CEPR evaluator now has an optional internal generative Qwen-VLM self-judge under
  `evaluator.internal_vlm_judge`. This is not an external reward model: it reuses the editor
  pipeline's own Qwen processor and text-encoder/VLM side to compare the original image, candidate
  edits, the instruction, and the structured edit JSON.
- The judge returns auditable 0-1 rubric fields: instruction following, edit success, target
  correctness, preservation, artifact freedom, overall quality, confidence, and a short reason.
- CEPR remains the main safety gate. The judge blends with CEPR raw reward before candidate ranking
  and contributes to preference calibration through the generic `judge` component. With the current
  `fail_open: true` setting, a judge parse/runtime failure logs an error and leaves CEPR behavior
  intact rather than silently discarding a round.
- This specifically targets the earlier reward weakness: embedding-only CEPR can separate near-miss
  candidates but is weak at judging whether a local object/region edit is visually correct. The
  internal self-judge adds structured visual reasoning while preserving the self-evolving/internal
  training story.

Implementation note, 2026-06-02:

- The active diagnostic is running from `uug_balanced_v2` under a Slurm allocation, with output root
  `outputs/self_evolve/balanced_cepr_v2_selfplay_margin_diag48_r4_20260602T095235Z`.
- Round 1 produced accepted SFT audit samples plus 21 internal preference pairs, then trained the
  editor with `pairwise_linear_sdpo` and the proposer with proposal-level rewards.
- Round 2 confirms versioned self-play works: each group has policy candidates and
  `opponent:previous_round` candidates. The first two hard object groups produced 8 near-miss
  preference pairs, balanced across object removal and replacement, with no direct SFT pull toward
  rejected images.
- The round-1 checkpoint scored positively on the 32-example ImgEdit canary:
  `4.6041` vs baseline `4.5206`, mean delta `+0.0834`, with 9 wins, 18 ties, and 5 losses. Per-type
  scores improved `adjust`, `background`, and `extract`, held `replace` and `add`, but regressed
  `remove`, so GEdit/object canaries are required before treating the update as broadly safe.
- The same round-1 checkpoint regressed GEdit subject-replace/cn on 32 examples:
  overall `8.2571` vs baseline `8.5051`, delta `-0.2480`; semantics delta `-0.3125`, quality delta
  `-0.0625`, with 6 wins, 14 ties, and 12 losses. This means round 1 is not a final broad method
  despite the ImgEdit canary gain.
- A remaining coverage weakness was found: strict scheduled edit-type filtering can skip non-object
  records when the learned proposer misses. Future runs should use the patched template fallback for
  color, attribute, material, spatial, background, style, and local-enhancement edits, not only
  object edits.

Observed canary results support this:

- Object-balanced CEPR v2 self-play round 1: ImgEdit 32 canary `+0.0834`
  (`4.6041` vs `4.5206`; 9 wins, 18 ties, 5 losses).
- Object-balanced CEPR v2 self-play round 1: GEdit subject-replace/cn 32 canary `-0.2480`
  (`8.2571` vs `8.5051`; 6 wins, 14 ties, 12 losses).
- Best non-router checkpoint so far: `qwen_edit_2509_magicbrush_rr_promptmix_anchor_lr5e6_s256`, ImgEdit 64 canary `+0.0261`.
- Target-over-source DPO: `-0.0625`.
- Pairmix target/source plus generated DPO: `-0.0519`.
- Reference-subtracted pairmix DPO: `-0.0783`.
- Broad generated DPO: `-0.0788`.
- Pairmix SDPO checkpoint-64: `-0.0209`.
- Pairmix SDPO final: `-0.0264`.
- Linear-SDPO high-confidence pairmix checkpoint-32: `-0.0416`.
- Linear-SDPO high-confidence pairmix final: `-0.0936`.
- Trust-region linear-SDPO pairmix checkpoint-32: `-0.0470`.
- Trust-region linear-SDPO generated-only checkpoint-32: `-0.1098`.
- Prompt-mix anchor remains the best no-router broad canary checkpoint: `+0.0261`.

This means the problem is not solved by adding more reward terms, more preference pairs, or a simple trust region around the prompt-mix adapter. The pairwise labels are not reliably aligned with ImgEdit scoring. The training update can improve background/style-like cases while harming extract/add/remove/replace cases, which dominate the regression.

Full ImgEdit results show that the strongest current signal is not a single LoRA update but conservative expert selection:

- baseline full ImgEdit: `4.4406`;
- conservative type router/rescore: `4.5312` (`+0.0906`);
- VLM selector with conservative abstention: `4.5275` (`+0.0869`);
- oracle score reranker over existing candidates: `4.5190` (`+0.0783`).

This is a useful upper bound. It says the generated expert pool contains useful localized improvements, but a single global adapter still damages too many high-ceiling examples.

## Paper-Grade Story

A stronger story is:

> Self-evolving image editors need preference updates that improve failed edits without degrading already-correct edits. We show that naive reward-filtered SFT and diffusion DPO are brittle for a strong base image editor, then propose safe preference editing: calibrated preference pair construction plus a chosen-preserving diffusion preference update.

This is more defensible than a pure engineering story because it has an analysis-driven method:

- identify ceiling-regression and reward-noise failure modes;
- measure which edit families improve or degrade;
- calibrate generated preferences by score margin and winner quality;
- use a safeguarded pairwise objective that scales the rejected-image gradient when it conflicts with the chosen-image reconstruction gradient;
- evaluate whether improvements transfer across ImgEdit and GEdit without a router.

## Method Direction

### 1. Calibrated Preference Data

Generated preferences should only be trusted when both conditions hold:

- winner quality is high enough;
- winner-loser score margin is large enough.

The current strict manifests use:

- winner score `>= 0.65`;
- score margin `>= 0.08`;
- per-family balancing;
- generated pair repeats to compensate for limited high-confidence data;
- optional low-weight MagicBrush target/source pairs.

Two active ablations:

- `sdpo_highconf_pairmix_ts025_generated_w065_m008_r4`: generated preferences plus low-weight target/source signal.
- `sdpo_highconf_generated_w065_m008_r4`: generated preferences only.

### 2. Safeguarded Diffusion Preference Optimization

Standard pairwise diffusion DPO optimizes a loss difference between chosen and rejected images. The rejected branch can create an update that increases chosen reconstruction error, especially when chosen and rejected gradients are aligned.

The implemented safeguarded objective computes:

- chosen loss gradient;
- rejected loss gradient;
- chosen/rejected gradient dot product;
- a rejected-gradient scale:

```text
scale = min(1, ||grad_chosen||^2 / <grad_chosen, grad_rejected>)
```

when the dot product is positive, otherwise scale is left at 1.

This gives an update direction that keeps the first-order change of the chosen loss non-increasing. In practice, the log key is:

```text
sdpo_rejected_scale
```

Values below 1 mean the safeguard is actively preventing a harmful rejected-branch update.

### 3. Linear Safeguarded Preference Optimization

The current code also includes two new objective ablations:

- `pairwise_linear_dpo`: replaces the softplus DPO classification loss with a linear preference objective.
- `pairwise_linear_sdpo`: uses the same linear preference pressure, but keeps the SDPO rejected-gradient safeguard.

The motivation is that diffusion training is a regression-style denoising problem, and the standard NLP-style sigmoid/softplus DPO utility can be poorly matched to diffusion preference optimization. This should be treated as an ablation, not as a guaranteed fix. The safer first run is `pairwise_linear_sdpo` with a small beta and a chosen SFT term.

### 4. Trust-Region Adapter Updates

The next no-router direction is trust-region preference learning. The warm-start adapter already has the best broad no-router score, so preference optimization should be allowed to make only a bounded update around that adapter.

Implemented trainer controls:

- `--lora_reference_l2_weight`: adds a relative L2 penalty against the initial trainable LoRA state;
- `--lora_reference_max_relative_delta`: projects LoRA weights after each optimizer step so the relative adapter delta cannot exceed a fixed radius.

This is meant to test a falsifiable hypothesis:

> previous preference runs failed because the optimizer moved too far from the high-performing base behavior before gaining enough hard-edit skill.

The first trust-region runs should use:

- warm start: `qwen_edit_2509_magicbrush_rr_promptmix_anchor_lr5e6_s256`;
- objective: `pairwise_linear_sdpo`;
- preference reference: `initial_lora`;
- strict generated preferences: winner score `>=0.70`, margin `>=0.10`;
- low-weight target/source anchors only in the mixed ablation;
- relative adapter radius around `0.05` for the first pass.

Two manifests have been prepared:

- `data/manifests/trust_region_highconf_generated_w070_m010_r8.jsonl`
  - 192 rows after repeat;
  - generated-only high-confidence preferences;
  - main diagnostic for reward-signal usefulness.
- `data/manifests/trust_region_highconf_pairmix_ts010_w070_m010_r8.jsonl`
  - 576 rows;
  - 192 generated rows plus 384 low-weight target/source rows;
  - diagnostic for whether weak supervised object anchors help without causing drift.

Result: this did not solve the broad canary. Pairmix checkpoint-32 scored `-0.0470`; generated-only checkpoint-32 scored `-0.1098`. Per-type deltas show a consistent pattern:

- pairmix checkpoint-32 helps `background` and `remove`, but hurts `extract`, `add`, `action`, and `replace`;
- generated-only checkpoint-32 helps `background` and `adjust`, but severely hurts `remove`, `extract`, `add`, and `replace`.

Conclusion: current reward-selected pairwise data is not sufficiently calibrated for single-adapter training. The next research step should be a better reward/verifier model, not another DPO-style optimizer sweep.

### 5. Verifier-Guided Expert Selection

The full ImgEdit selector results justify a parallel method direction:

> train or collect multiple small self-evolved experts, then use a conservative VLM verifier to select a candidate only when it clearly beats the baseline on instruction compliance, preservation, and quality.

This is not the same story as a brittle hand-written router. It is closer to test-time reward modeling: candidate generation proposes alternatives, and a multidimensional verifier abstains to the base model unless the edit is clearly better. It should be evaluated as:

- VLM selector without access to benchmark scores;
- score-oracle selector only as an upper bound;
- ablations for candidate pool size, abstention margin, and edit-family fallback;
- cost/runtime reporting.

### 6. Evaluation Policy

Use the 64-example ImgEdit canary as the first gate for new checkpoints because the 32-example slices can give false positive signals. If a checkpoint beats the current best no-router checkpoint, run:

- ImgEdit object remove/replace canary;
- GEdit subject-remove;
- GEdit subject-replace;
- then larger/full benchmark evaluation only after the direction is positive.

## Active Runs

- Object-balanced CEPR v2 self-play diagnostic:
  - tmux: `uug_balanced_v2`
  - output root: `outputs/self_evolve/balanced_cepr_v2_selfplay_margin_diag48_r4_20260602T095235Z`
  - config snapshot: `run_config_resolved.json` inside the output root
  - purpose: test whether closed-loop internal preference training plus versioned self-play can
    improve hard object edits without rejected-image SFT poisoning.

- Round-1 ImgEdit canary:
  - tmux: `uug_cepr_nearmiss`
  - checkpoint: `outputs/self_evolve/balanced_cepr_v2_selfplay_margin_diag48_r4_20260602T095235Z/round_01/training_output/pytorch_lora_weights.safetensors`
  - model name: `balanced_cepr_v2_selfplay_r01_imgedit_o0_n32`
  - purpose: quickly decide whether the first preference update is directionally safe before scaling.

## Latest Diagnostic: Near-Miss Reward Noise

Round 2 of `balanced_cepr_v2_selfplay_margin_diag48_r4_20260602T095235Z` exposed a
specific failure mode. The run correctly excluded rejected candidates from direct
SFT, but the near-miss preference path could still choose candidates with
reasonable raw CEPR reward while failing the object/spatial rubric gates.

Observed round-2 pattern:

- `object_removal`: 16 candidates, 0 accepted; average `rubric_forbidden_after_absent`
  about `0.06` and `rubric_edit_success` about `0.22`.
- `object_replacement`: 16 candidates, 0 accepted; average `rubric_forbidden_after_absent`
  about `0.08` and `rubric_edit_success` about `0.25`.
- Preference manifest still contained near-miss pairs from removal/replacement/spatial/local
  edits because near-miss screening used raw reward, semantic edit, preservation, and validity,
  but did not require the stricter edit-specific rubric contract.

Fix implemented for future clean runs:

- `training.preference.near_miss_contract_filter` now applies the rubric contract to local
  near-miss positives before they can become chosen preference examples.
- The filter requires strict forbidden-gate pass plus minimum required-after, forbidden-absence,
  edit-success, preservation, validity, and raw-reward components.
- This should reduce noisy preference learning from failed object edits. It may also reduce
  pair count, so future v3 runs should use enough records/candidates to recover high-quality
  accepted or contract-passing near-miss pairs.

Follow-up implementation update:

- Clean run `balanced_cepr_v4_anchor_contract_diag96_r4_20260602T114756Z` uses the stricter
  near-miss contract plus anchored fallback target regions for common secondary objects.
- Example object-removal proposal now uses `target_region=on the main subject's head` with
  both `party hat remains visible on the main subject's head` and `a separate party hat remains
  visible on the main subject's head`.
- Early v4 signal: object removal/replacement groups still fail, but they fail through the
  rubric gates instead of becoming noisy positives; attribute/material groups produce accepted
  examples. This indicates the remaining problem is hard-object edit generation capacity, not
  direct rejected-data poisoning.

Round-2 old v2 canary:

- ImgEdit 32-canary improved from `4.520625` to `4.6140625`, delta `+0.0934375`, with
  8 wins, 20 ties, and 4 losses.
- GEdit subject-replace/cn 32-canary regressed from `8.505063801521327` to
  `8.236943574841794`, delta `-0.2681202266795324`. This confirms v2 round 2 is
  not a broad main method despite the ImgEdit canary gain.
- A hard-object prompt probe is running with training disabled to test whether shorter direct
  object-removal/replacement prompts improve acceptance before changing the main training run.

## Next Research Steps

If the expanded selector improves:

- run a full VLM selector over baseline, object experts, CEPR, hybrid, prompt-mix anchor, and SDPO checkpoint-64;
- compare against the score-oracle selector upper bound;
- audit choice counts by edit family and per-family score deltas;
- then run GEdit/OneIG checks only for methods that do not degrade ImgEdit.

If linear SDPO improves:

- evaluate checkpoint 32/64/128 to identify the safest update strength;
- test LoRA scales `0.5`, `0.75`, and `1.0`;
- run GEdit subject-remove and subject-replace canaries;
- convert the method into the main paper method.

If linear SDPO still regresses:

- stop treating current generated preferences as reliable training labels;
- keep target/source pairs out of pairwise training unless a separate verifier confirms the target is better than the base output;
- replace scalar reward filtering with listwise ranking or critique-backed preference generation;
- train a lightweight edit-family reward calibrator before using VLM scores as labels;
- prioritize verifier-guided expert selection as the next main method, because existing full ImgEdit results already show `+0.087` to `+0.091` without benchmark-score oracle access.

## Conservative Pairwise Self-Evolution Update

Implemented direction for the next clean run:

- Every proposal can now include a `reference:base` candidate generated by the initial Qwen-Image-Edit policy.
- Preference construction is base-relative when `training.preference.base_relative.enabled=true`.
- A policy output becomes chosen only if its constraint-aware score beats the base/reference output by a configured margin.
- If the base/reference output beats the evolving policy, the preference pair is reversed: `chosen=reference:base`, `rejected=policy`. This is the no-harm path for ceiling cases.
- Ambiguous policy-vs-base pairs are skipped by default, because training on small noisy margins was the observed source of drift.
- The constraint-aware pair score uses CEPR/rubric preservation and validity first; VLM scores are treated as an agreement term, not as a replacement for preservation gates.
- `training.preference.vlm_pair_guard` adds a confidence-weight multiplier and can skip pairs when the internal VLM strongly disagrees with the chosen/rejected ordering.
- `output.use_cumulative_preference_manifest=true` keeps high-confidence preference pairs across rounds so each editor update is not trained only on the tiny current-round accepted set.
- Proposer SFT now records policy-over-base margins and can reward proposals that create clear policy improvements while penalizing proposals where the base wins.

Primary config:

```yaml
configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml
```

The config name/output root now identify the method as `qwen_edit_2509_conservative_pairwise_v1`.

Telemetry to monitor:

- `preference_summary.skipped.base_relative_ambiguous`
- `preference_summary.skipped.base_relative_no_reference_candidate`
- `preference_summary.per_edit_type`
- `preference_summary.round_pairs_without_anchors`
- `preference_summary.pairs_without_anchors`
- `preference_manifest.jsonl` fields `preference_source`, `base_relative`, and `vlm_pair_guard`
- proposer summary metrics `policy_over_reference_margin`, `base_improvement_score`, and `base_harm_score`
- training log fields `preference_loss`, `optimized_preference_delta`, `sdpo_rejected_scale`, and `lora_reference_relative_delta_pre_projection`

Expected behavior:

- The method should reduce ImgEdit ceiling damage because base wins are now explicit training labels.
- Pair count may drop initially because ambiguous pairs are skipped; cumulative preference replay should compensate over rounds.
- If acceptance remains low and `base_relative_no_reference_candidate` is nonzero, check reference candidate generation first.
- If many pairs are skipped by `vlm_judge_margin_too_small`, reduce VLM guard strength; do not raise the VLM weight blindly.

## Research Basis

- Diffusion-DPO shows pairwise preference optimization can align diffusion models, but it relies on clean comparison data and a diffusion-specific likelihood surrogate: https://arxiv.org/abs/2311.12908
- Curriculum-DPO supports ranking candidates and controlling pair difficulty rather than treating all reward gaps equally: https://arxiv.org/abs/2405.13637
- Dense reward alignment argues that preference optimization for diffusion should account for the sequential denoising process, not only a sparse terminal reward: https://arxiv.org/abs/2402.08265
- Diffusion-SDPO identifies the same pathology we observed locally: standard DPO can increase reconstruction error for both preferred and rejected branches; it motivates the safeguarded rejected-gradient scale: https://arxiv.org/abs/2511.03317
- Linear-DPO argues that NLP-style DPO can be mismatched to regression-based diffusion/flow training, motivating the linear objective ablation: https://arxiv.org/abs/2605.21123
- HIVE and EditHF-1M both support multidimensional image-editing feedback over instruction alignment, visual quality, and preservation rather than one scalar reward: https://arxiv.org/abs/2303.09618 and https://arxiv.org/abs/2603.14916
- EditReward directly supports our diagnosis that instruction-guided image editing needs an editing-specific reward model, and that generic VLM-as-judge or ad hoc metrics may be misaligned with human preference: https://arxiv.org/abs/2509.26346
- RPO supports using rich VLM critiques/preferences rather than opaque scalar labels, which is relevant for building better preference pairs after the trust-region diagnostic: https://arxiv.org/abs/2503.11720
- LPO supports a faster latent/step-level reward direction and motivates avoiding expensive pixel-level VLM reward calls inside every denoising step: https://arxiv.org/abs/2502.01051
- RewardHarness supports the verifier-as-context/evaluation-tool direction: improving the reward process itself can be more data-efficient than only optimizing model weights from noisy labels: https://arxiv.org/abs/2605.08703
