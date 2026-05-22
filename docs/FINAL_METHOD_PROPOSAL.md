# Final Method Proposal

This is the concrete method proposal to use going forward unless experiments force a change.

For the paper storyline and ablation map, see
[PAPER_STORY_AND_ABLATIONS.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/PAPER_STORY_AND_ABLATIONS.md).

## Chosen Architecture

We will use:

`Proposer -> Editor(K samples) -> Solver Ensemble -> Relative Ranker -> Accept/Reject -> Train`

This is the final recommended version because it keeps the original self-evolving idea, but changes the scoring path in a way that is much harder to game than a single scalar reward.

## Role Definitions

### 1. Proposer

- Input: raw image `x`
- Output: candidate edit instruction `e`
- Goal: generate edits that are neither trivial nor impossible

The proposer is not rewarded for making the hardest possible edit. It is rewarded for producing **moderate-difficulty** edits that create learning signal.

### 2. Editor

- Input: `(x, e)`
- Output: `K` edited candidates `{y_1, ..., y_K}`
- Base model: `Qwen-Image-Edit`

We use multiple candidates because image editing is multimodal. Single-sample acceptance throws away too much information.

### 3. Solver Ensemble

The solver is not a single scalar head. It is an ensemble of editing-specific checks:

- instruction satisfaction verifier
- preservation verifier
- spatial verifier
- optional cycle verifier
- optional internal-feature verifier

The ensemble produces both:

- per-candidate component scores
- a final relative ranking over the `K` candidates

### 4. Relative Ranker

The best candidate is selected relative to:

- the other generated candidates for the same instruction
- optionally a reference output from a frozen earlier checkpoint

This reduces over-reliance on unstable absolute reward values.

## Chosen Reward Design

## Step 1: Hard-Gated Feasibility

Some rewards should behave like constraints, not soft preferences.

For candidate `y`, require:

```text
R_instruction(x, e, y) >= tau_inst
R_preservation(x, e, y) >= tau_pres
```

If either fails, reject the sample.

This enforces the real task structure:

- the edit must happen
- unchanged content must remain intact

## Step 2: Relative Quality Score

Among candidates that pass the hard gates, compute:

```text
R_quality(y) =
  alpha * R_spatial(x, e, y)
  + beta * R_cycle(x, e, y)
  + gamma * R_internal(x, e, y)
  + delta * R_counterfactual(x, e, y)
  + eta * R_relative(x, e, y, y_ref)
```

Where:

- `R_spatial`: rewards localized and appropriate change
- `R_cycle`: rewards reversibility or consistency under reverse edit
- `R_internal`: rewards semantic movement in internal Qwen features without collapse
- `R_counterfactual`: rewards matching the true instruction more than distractor instructions
- `R_relative`: rewards improvement over a baseline or previous checkpoint

## Step 3: Acceptance Rule

Accept a sample if:

```text
candidate passes hard gates
and
R_quality is in the top-m among K candidates
and
solver disagreement is below a confidence threshold
```

This means acceptance depends on:

- feasibility
- relative superiority
- confidence

instead of just one weighted scalar threshold.

## Proposer Reward

The proposer follows the EvoLMM logic.

Let `d(x, e)` be edit difficulty, estimated from:

- variance across candidate scores
- solver disagreement
- margin between top candidate and next-best candidate
- acceptance probability

Then:

```text
R_proposer = exp(- (d - mu)^2 / (2 * sigma^2)) - lambda_trivial - lambda_invalid
```

Interpretation:

- very easy edits are penalized indirectly because they provide little learning signal
- impossible edits are penalized because they create noisy supervision
- moderate-difficulty edits are preferred because they are most informative

## Most Important New Terms

## 1. Counterfactual Reward

This is one of the highest-value additions.

For the true instruction `e+` and distractor instructions `{e-_1, ..., e-_M}`:

```text
R_counterfactual = score(x, e+, y) - max_j score(x, e-_j, y)
```

This helps suppress:

- no-op edits
- generic beautification edits
- edits that change the wrong attribute

## 2. Relative Reward

This compares the current output to a reference output:

```text
R_relative = score(x, e, y_current) - score(x, e, y_ref)
```

The reference can be:

- a frozen supervised baseline
- a previous checkpoint
- a weaker candidate from the same sample group

This makes training focus on actual improvement rather than raw score inflation.

## 3. Solver Disagreement Penalty

If multiple verifiers disagree strongly, the sample should not be trusted.

Use:

```text
accept only if std([solver scores]) <= delta_conf
```

This directly reduces reward hacking and unstable pseudo-labels.

## Why This Is Better Than The Current Simpler Hybrid

The current weighted hybrid is useful as a baseline, but it still has two weaknesses:

1. It lets a very high score on one component compensate for a failure on a critical component.
2. It treats reward as absolute when image editing is naturally relative and multimodal.

The proposed method fixes this by:

- using hard gates for essential constraints
- ranking among multiple candidates
- adding counterfactual and relative scoring
- using disagreement to filter uncertain pseudo-labels

## Final Paper Framing

The paper should be framed as:

> A self-evolving image editing framework in which a proposer generates edit instructions for raw images, an editor produces multiple candidate edits, and an editing-specific solver ensemble filters those candidates using constraint-based and relative rewards that explicitly separate requested change from preservation.

This is stronger than:

- "RL for image editing"
- "a reward model for image editing"

because the contribution is the **self-evolving editing loop plus reward design**, not one isolated component.

## Minimal Experiment Ladder

### Stage A: Baselines

1. supervised baseline
2. naive self-training
3. current simple hybrid reward

### Stage B: Final Proposed Method

4. hard-gated proposer-editor-solver
5. + multi-candidate ranking
6. + counterfactual reward
7. + relative reward

### Stage C: Optional High-Risk Additions

8. + internal-feature verifier
9. + DPO or groupwise preference optimization on accepted vs rejected candidates

## Recommended Default Choice

If we need one implementation target to commit to immediately, it should be:

```text
Proposer -> Editor(K=4) -> Solver Ensemble -> Relative Ranker -> Accept/Reject -> SFT
```

with:

- hard gates on `R_instruction` and `R_preservation`
- ranking by `R_spatial + R_counterfactual + R_relative`
- proposer trained with moderate-difficulty reward

This is the best balance of novelty, plausibility, and implementation tractability.
