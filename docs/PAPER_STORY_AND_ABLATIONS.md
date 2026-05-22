# Paper Story And Ablations

This note is the reviewer-facing method story. It should guide the BMVC-style paper framing,
ablation tables, and experiment order.

## Core Thesis

Self-evolution for image editing is different from self-evolution for multimodal reasoning.

In reasoning tasks, a generated question has a mostly discrete answer and the solver can often be
rewarded by answer correctness or internal consistency. In image editing, every pseudo-label is an
image transformation from `(source image, instruction)` to `edited image`. A valid edit must satisfy
two coupled constraints:

- requested change: the instruction must be visibly realized
- preservation: everything outside the requested edit should remain stable

This makes image editing a delta-evaluation problem. The evaluator must judge not only the final
image, but the difference between source and output.

Our method is therefore:

```text
source image -> structured edit proposal -> K Qwen edit candidates
-> delta-grounded evaluator -> accepted pseudo-labels -> editor LoRA
```

The paper claim should be:

> We adapt self-evolving multimodal training to image editing by replacing generic self-rewarding
> with delta-grounded candidate selection: hard gates for edit success and preservation, relative
> ranking among multiple candidates, counterfactual instruction discrimination, and Qwen
> edit-conditioning features as an auxiliary internal signal.

## Main Contribution

The main contribution should be framed as a **training pipeline plus reward/selection system**:

1. **Unsupervised self-evolution pipeline for direct image editing.**
   Starting from raw unlabeled source images, the system proposes edit instructions, samples multiple
   Qwen-Image-Edit candidates, filters them with an editing-specific evaluator, and converts only
   accepted candidates into LoRA training pairs.

2. **Delta-grounded reward decomposition.**
   The reward is not a generic final-image score. It is decomposed into requested edit success,
   preservation of non-target content, candidate-group relative quality, counterfactual instruction
   discrimination, and optional Qwen internal support.

3. **Reward analysis for image-editing self-evolution.**
   The experiments should show that generic self-evolving rewards can select visually plausible but
   unsafe edits, while the proposed reward improves the model in an unsupervised fashion and keeps
   non-edited content more consistent.

## Why Image Editing Needs A Special Self-Evolve Design

Standard self-evolving methods such as EvoLMM are designed around a proposer-solver loop for
image-grounded reasoning. Borrowing that template directly is insufficient for image editing:

1. **Reasoning has answer correctness; editing has transformation correctness.**
   A reasoning solver can be rewarded by whether an answer is correct. An editor must be rewarded by
   whether the output changed in the right way while preserving unrelated content.

2. **Final-image scoring is under-specified.**
   A generic VLM judge may say the edited image matches the instruction even when the source identity,
   layout, background, or object count changed. For editing, these are failures.

3. **Single-sample self-training is noisy.**
   Image editing is multimodal. One candidate may over-edit, another may under-edit, and another may
   preserve well but miss the instruction. Ranking candidates for the same `(x, e)` is more reliable
   than accepting a single absolute score.

4. **The proposer must respect editability.**
   A reasoning proposer can ask a harder question as the solver improves. An editing proposer must
   ask edits that are visually feasible on the source image and have a meaningful non-edited region
   for preservation checks.

5. **Reward hacking is easier in editing.**
   The editor can increase colorfulness, contrast, or aesthetic appeal and appear better to weak
   global rewards while ignoring the precise instruction.

Because of this, the method should not be framed as "EvoLMM for editing." It is a task-specific
adaptation that changes the reward structure and acceptance rule to match image editing.

## Positioning Against JarvisEvo

JarvisEvo is important related work, but it is not a direct methodological competitor to our setup.
It is best described as a supervised, tool-augmented photo-editing agent. Its training recipe uses
large-scale cold-start SFT, human-annotated evaluator data, Gemini-assisted annotation and filtering,
reflection data generation, and external Adobe Lightroom-style tool execution.

That means JarvisEvo's self-evolution happens in an agent/tool-orchestration space:

```text
image + instruction -> reasoning trace -> Lightroom/tool actions -> edited image
```

Our method studies a different problem:

```text
image + instruction -> Qwen-Image-Edit candidates -> delta-grounded selection -> editor LoRA
```

The distinction is central to the paper story. JarvisEvo does not directly teach a native
instruction-guided image-editing generator to improve from its own candidate image outputs. It
trains an agent to choose and evaluate external editing operations. Therefore, it should be cited as
adjacent self-evolving photo-editing work, not treated as an apples-to-apples baseline.

The clean contrast is:

- **JarvisEvo:** supervised/tool-agent/evaluator-calibrated self-evolution for professional photo
  retouching.
- **Ours:** model-native, delta-grounded self-evolution for a direct generative image-editing model
  from unlabeled source images.

## Why Qwen-Image-Edit Is A Good Host Model

Qwen-Image-Edit is useful because its editing pipeline already conditions generation on both:

- semantic understanding features from Qwen's vision-language path
- visual/appearance information from the source image path

That architecture matches our evaluator design. We can use Qwen edit-conditioning representations
as an internal auxiliary signal while still relying on image-delta checks for preservation. The key
point is not that Qwen magically verifies itself; the point is that Qwen exposes a model-internal
semantic representation that can be combined with explicit edit-delta constraints.

The results-first implementation uses this conservatively:

- train only from proxy-verifiable accepted edits
- require internal Qwen features as an auxiliary support signal
- export broader internal-only/local semantic edits for future evaluator learning, but do not let
  them enter editor SFT by default

This is the safest path for getting BMVC-grade results without overclaiming.

## Generalization Beyond Qwen

The method is model-agnostic at the framework level:

```text
Editor(K candidates) + delta evaluator + relative accept/reject + SFT
```

It can be used with other instruction-guided image editors if they provide:

- multiple stochastic candidates per source/instruction
- a training path from accepted edit pairs
- access to either internal semantic features or a substitute verifier

For other editors:

- **With internal features**: use the same internal-feature auxiliary score.
- **Without internal features**: replace `R_internal` with CLIP/DINO/VLM features or disable it, while
  keeping hard gates, preservation, relative ranking, and counterfactuals.
- **With mask-aware editors**: replace heuristic spatial signals with predicted or generated masks.

So the contribution is not tied only to Qwen. Qwen is the strongest demonstration host because it
offers edit-conditioning features and a public LoRA training path.

## Main Method Variants

Use these names consistently:

| Variant | Purpose | Expected role |
| --- | --- | --- |
| `base_qwen` | no self-evolve | official/local baseline |
| `naive_self_train` | single candidate, weak scalar acceptance | inferior self-training baseline |
| `evolmm_style` | K candidates + generic continuous self-reward, no preservation gates | shows direct reasoning-style reward transfer is under-specified for editing |
| `hybrid_scalar` | weighted proxy + spatial/internal/cycle score | shows weighted rewards are not enough |
| `delta_ranker_proxy` | hard gates + K candidates + relative ranking, no internal requirement | tests candidate ranking |
| `delta_results` | proxy-verifiable edits + hard gates + K ranking + required internal support | main result method |
| `delta_grounded` | broader taxonomy including internal-only local/semantic edits | research extension/export path |

## Reward Analysis Table

The paper should include a reward-system ablation table, not only final benchmark scores:

| Reward/selection rule | What it tests | Expected failure mode |
| --- | --- | --- |
| Naive scalar proxy | single candidate accepted by a weak edit-success score | noisy pseudo-labels; no way to choose among multiple valid-looking edits |
| EvoLMM-style generic self-reward | K candidates ranked by continuous scalar reward only | can reward correct-looking final images while ignoring source preservation |
| Weighted hybrid scalar | proxy + spatial/internal terms collapsed into one score | one strong term can compensate for failed instruction or preservation |
| Hard gates only | separate instruction and preservation constraints | cleaner data, but may not pick the best candidate in multimodal edit groups |
| Delta ranker proxy | hard gates + K relative ranking + counterfactuals | strong non-internal baseline |
| Delta-results | proxy-verifiable hard gates + K ranking + counterfactuals + internal Qwen support | main method |

Report more than one metric for this table:

- benchmark edit score or task score
- preservation score outside changed regions
- edit-success score on the requested operation
- acceptance rate and manifest size
- no-op/generic-edit rejection rate if available
- average source-output identity/structure preservation

## Required Ablations

Minimum table for BMVC completeness:

1. **Base Qwen-Image-Edit**
   Tests whether any self-evolve training helps.

2. **Naive self-training**
   Generate one edit per proposal, accept by scalar proxy score, train on accepted pairs. This should
   be worse because it lacks preservation gates and candidate ranking.

3. **EvoLMM-style generic self-reward**
   Generate multiple candidates and rank by a generic continuous self-reward without edit-specific
   preservation gates or counterfactual instruction discrimination. This should be worse because
   image editing requires transformation-level verification, not final-output confidence alone.

4. **Weighted hybrid reward**
   Uses a single scalar combination. This should be less stable because one strong signal can
   compensate for failure on preservation or instruction satisfaction.

5. **Hard gates only**
   Hard instruction and preservation gates, but no relative ranking. Tests whether constraints alone
   are sufficient.

6. **Hard gates + relative K ranking**
   Tests the benefit of candidate-group comparison.

7. **No counterfactual reward**
   Tests whether distractor instructions reduce generic edits and no-ops.

8. **No internal Qwen support**
   Tests whether Qwen internal features improve filtering beyond image statistics.

9. **K=1 versus K=4**
   Tests whether editing multimodality makes candidate ranking useful.

10. **Train manifest restricted to proxy-verifiable edits versus all accepted edits**
   Tests whether internal-only semantic edits are too noisy for immediate SFT.

Optional if time permits:

- source filtering ablation: raw pool versus VLM-filtered pool
- preservation threshold sweep
- top-1 versus top-2 accepted candidates per group
- one-round versus two-round self-evolve

## Expected Reviewer Questions

### Why not simply use EvoLMM?

Because EvoLMM's reward structure is for solving generated multimodal questions. Image editing
requires evaluating a visual transformation. The proposed method changes the self-evolve loop by
introducing edit-specific hard constraints, preservation-aware delta scoring, candidate-group
ranking, and counterfactual edit discrimination.

### Why does this not collapse to trivial edits?

The proposer has a difficulty ladder and the evaluator tracks acceptance, disagreement, and
relative margins. The results-first path is conservative for benchmark production, while the broader
delta-grounded path retains harder internal/local proposals for future evaluator learning.

### Why not train on every accepted internal edit?

Because uncalibrated internal-only semantic edits can produce noisy SFT targets. The main path uses
internal features as an auxiliary filter and keeps internal-only edits in exported evaluator records
until a learned evaluator is calibrated.

### Is this Qwen-specific?

The internal-feature term is Qwen-specific in this implementation. The overall method is not:
hard edit/preservation gates, counterfactuals, K-candidate ranking, and accepted-pair SFT can be used
with any instruction-guided image editor. Other models can replace Qwen features with their own
conditioning features or an external verifier.

## Recommended Experiment Order

If time is tight:

1. `base_qwen`
2. `naive_self_train`
3. `evolmm_style`
4. `hybrid_scalar`
5. `delta_ranker_proxy`
6. `delta_results`
7. `delta_results --set solver.rank_counterfactual_weight=0.0`
8. `delta_results --set solver.require_internal_when_weighted=false --set solver.internal_weight=0.0 --set solver.rank_internal_weight=0.0`
9. `delta_results --set candidate_generation.samples_per_proposal=1`

The main comparison should be:

```text
naive self-train < EvoLMM-style generic reward < hybrid scalar < proxy delta ranker < delta-results
```

If the final gains are modest, the paper can still argue method soundness through:

- cleaner accepted pseudo-labels
- better preservation metrics
- lower evaluator disagreement
- fewer no-op or generic edits
- comparable generation sanity scores
