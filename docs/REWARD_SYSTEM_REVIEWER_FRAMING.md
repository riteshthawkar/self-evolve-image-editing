# Reward System Reviewer Framing

Date: 2026-06-20

This note documents how we should explain the reward system in reports and papers. The main risk is that reviewers may see the method as a hand-engineered mixture of CEPR, internal VLM judging, object detection, pixel preservation, locality, thresholds, and training filters. The paper should avoid presenting the method as a list of added reward terms. The stronger framing is that conservative image editing is naturally a constrained objective.

## Core Position

Our reward should be framed as one objective with decomposed necessary conditions:

```text
R_conservative_edit =
    EditSuccess
  x TargetRegionSupport
  x NonTargetPreservation
  x VisualValidity
```

This is not an arbitrary reward bundle. Each factor corresponds to a necessary condition for a successful conservative edit:

- `EditSuccess`: the requested semantic edit happened.
- `TargetRegionSupport`: the intended local target region changed enough to support the edit.
- `NonTargetPreservation`: regions outside the target edit remain close to the source image.
- `VisualValidity`: the edited image remains artifact-free and visually plausible.

The multiplicative and hard-gated structure is important. A candidate should not be accepted if it succeeds on one condition while failing another. For example, a strong semantic edit should not compensate for corrupting the rest of the image, and high preservation should not compensate for a no-op edit.

## Reviewer Risk

A reviewer may object:

> The method appears to combine many reward components and thresholds without a principled reason.

This criticism becomes credible if we describe the method as:

- CEPR score plus internal VLM score;
- plus object detector score;
- plus pixel preservation score;
- plus locality score;
- plus artifact score;
- plus several thresholds.

That presentation sounds like engineering accumulation. We should instead explain that conservative editing has separable failure modes, and a candidate is useful for self-training only if it satisfies all required constraints.

## Principled Explanation

Image editing is different from pure image generation. In generation, global visual quality and prompt alignment may be sufficient. In conservative editing, the model must satisfy two coupled requirements:

1. It must perform the requested target edit.
2. It must avoid damaging non-target content.

These two requirements conflict in practice:

- A no-op image has excellent preservation but zero edit success.
- An over-edited image may satisfy the instruction but damage identity, layout, or background.
- A wrong-region edit changes the image but not the intended target.
- A visually broken edit may satisfy a semantic detector but lower benchmark quality.

Therefore, a scalar reward that does not decompose these conditions is under-specified. Our reward decomposes the conservative editing contract into measurable constraints and uses hard gates so that one term cannot hide another failure.

## Recommended Paper Language

Use language like:

> We formulate self-evolution for image editing as constrained preference learning. A generated candidate is treated as useful supervision only if it satisfies both edit correctness and non-target preservation. Instead of using a single prompt-alignment scalar, we decompose the reward into necessary factors for conservative editing: target edit success, target-region support, non-target preservation, and visual validity. These factors are combined multiplicatively and gated, preventing high semantic alignment from compensating for identity drift, background corruption, or no-op edits.

Another concise version:

> The reward is not a sum of independent heuristics. It is a constraint-aware conservative-editing objective. Each component tests a distinct necessary condition, and failure of any condition makes the candidate unsuitable for self-training.

## Claims To Avoid

Avoid saying:

- We add multiple rewards to improve performance.
- We tune several reward terms for the benchmark.
- We use detector, VLM, pixel, and quality rewards together.
- The full reward is a weighted combination of many signals.

Better alternatives:

- We decompose the conservative editing contract into necessary constraints.
- We use a multiplicative objective so constraints cannot compensate for one another.
- We use region-decoupled scoring to separate target edit success from non-target preservation.
- We select self-training pairs only when the chosen sample satisfies the conservative-editing constraints.

## Relationship To Self-Evolution

The main self-evolution claim should be stated carefully.

The closed-loop part of the method is:

1. The current model proposes edit instructions and edited candidates.
2. The same model family provides internal understanding/judgment signals.
3. The accepted candidates become preference data for the next round.
4. The proposer and editor improve across rounds from model-generated data.

This supports a self-evolution framing when the reward signals come from the model's own internal understanding, latent consistency, and image-difference analysis.

However, GroundingDINO introduces an important caveat. If GroundingDINO is used in the main reward, reviewers may view it as an external verifier. This weakens a strict "no external supervision or verifier" claim.

Recommended positioning:

- Main method: emphasize internal CEPR, internal VLM judge, latent locality, and pixel/region preservation.
- GroundingDINO region masks: present as an optional grounding aid or analysis variant unless we are comfortable claiming a broader "automated verifier-guided self-training" method.
- If GroundingDINO remains in the main experiment, do not claim the reward is purely internal. Instead say the data are self-generated and the reward is automated, with region grounding used to instantiate the conservative-editing constraint.

## How To Defend The Components

Each component should be tied to a measured failure mode:

| Constraint | Failure Prevented | Evidence To Show |
| --- | --- | --- |
| Edit success | No-op edits accepted as positives | Examples where preservation is high but target edit fails |
| Target-region support | Rewarding changes outside the requested target | Target mask/diff visualization |
| Non-target preservation | Background/identity/layout corruption | GEdit quality failures and outside-damage examples |
| Visual validity | Artifacts, broken texture, unnatural inpainting | Internal VLM artifact scores and qualitative failures |
| Object contract | Object removal/replacement semantic failures | Source object remains, replacement missing |

This converts the story from "many rewards" to "one reward with interpretable constraints."

## Required Ablations

To make the reward credible, we should include ablations that show each group of constraints contributes something specific.

Minimum ablation table:

| Variant | Purpose |
| --- | --- |
| CEPR only | Shows baseline internal reward behavior |
| CEPR + internal VLM judge | Tests whether semantic/quality judgment fixes no-op and artifact positives |
| CEPR + region preservation | Tests whether non-target preservation fixes over-editing |
| CEPR + internal VLM + region preservation | Main constrained reward |
| Main without hard gates | Shows why additive/soft combination is insufficient |
| Optional: main with/without GroundingDINO masks | Separates internal-only claim from external region-grounding aid |

The most important ablation is "without hard gates." It directly answers the reviewer concern: a weighted reward can let one score compensate for another, while conservative editing requires all conditions.

## Reward Audit Experiments

We should run or report a reward audit before making strong claims:

- Correlation of each reward variant with ImgEdit/GEdit score.
- Acceptance rate by edit type.
- Distribution of reject reasons:
  - `target_change`
  - `outside_change`
  - `outside_changed_fraction`
  - `outside_preservation`
  - `target_mask_not_supported`
  - internal VLM low semantic/preservation/artifact.
- Qualitative grids showing:
  - accepted positive;
  - rejected no-op;
  - rejected outside-corruption;
  - rejected artifact;
  - rejected wrong-object or source-object-remains.

This audit is essential because our current results showed that raw CEPR can accept candidates that are not useful for training.

## Suggested Method Section Structure

1. Define conservative image editing:
   - target edit success;
   - non-target preservation.
2. Explain why scalar self-reward is insufficient:
   - no-op and over-editing degeneracies.
3. Introduce constraint-aware self-evolution:
   - model generates candidate edits;
   - internal scoring decomposes necessary conditions;
   - preference pairs are formed only from productive groups.
4. Define the reward:
   - edit success;
   - target-region support;
   - non-target preservation;
   - visual validity.
5. Explain hard-gated preference construction:
   - chosen sample must satisfy constraints;
   - failed candidates remain useful as negatives.
6. Discuss optional region grounding:
   - detector masks or internal/fallback masks;
   - note whether the final method is internal-only or uses external grounding.

## Final Recommended Story

Our best story is:

> We study self-evolving conservative image editing. The central challenge is not only to improve instruction following, but to prevent the model from learning from superficially good edits that damage non-target content. We therefore formulate reward modeling as a constraint-aware problem: an edit is useful for self-training only when it changes the intended target, preserves non-target regions, and remains visually valid. This turns self-generated candidates into cleaner preference pairs and avoids the no-op and over-editing failures observed with scalar CEPR rewards.

This story is research-grade because it is about reward modeling and failure analysis, not just adding engineering components.
