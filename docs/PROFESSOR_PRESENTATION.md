# Professor Presentation Outline

This document is designed to help present the project honestly before the full experiment table is ready.

## Settled recommendation

**Project title**
Self-Evolving Image Editing with Preservation-Aware Self-Training on Qwen-Image-Edit

**What to claim**
- we have a real self-evolving edit loop on top of a real Qwen editor
- we have benchmark infrastructure ready for controlled experiments
- the internal-Qwen verifier is the main next research step

**What not to claim**
- that Qwen already exposes a finished proposer-verifier API
- that we already have measured improvement
- that the fully internal closed loop is complete

## Rule for the meeting

Separate three things clearly:

1. **Implemented system status**
2. **Experiments currently running or queued**
3. **Expected outcome if the hypothesis is correct**

Do not present expected gains as measured gains.

## Slide 1: Problem

**Title**
Self-Evolving Image Editing Without an External Reward Model

**Core message**
- Current self-improving image editing methods usually rely on an external VLM or reward model.
- That adds extra model dependency, extra compute, and a mismatch between the editor and the verifier.
- Our hypothesis is that a strong editor with an internal understanding path can be pushed toward self-improvement with iterative pseudo-labeling, and later upgraded to a more internal verifier.

## Slide 2: Key Observation

**Title**
Why Qwen-Image-Edit Is a Good Target

**Core message**
- Qwen-Image-Edit already contains a dual-path structure: an understanding side and an editing side.
- The technical report explicitly states that the original image is separately fed to `Qwen2.5-VL` and the `VAE encoder` to align semantics and reconstruction.
- The public DiffSynth code also exposes hidden states from the Qwen understanding path used to condition editing.

**Safe wording**
- “This is the architectural opening we want to exploit.”
- “The central research question is whether these exposed internal features are enough to build a verifier that can replace proxy scoring.”

## Slide 3: Proposed Method

**Title**
Proposer -> Editor -> Solver -> Train

**Core message**
- **Proposer** samples edit instructions from unlabeled images.
- **Editor** applies the edit.
- **Solver** scores instruction satisfaction and structural preservation.
- Accepted edits become pseudo-labeled training data for the next round.
- Difficulty is increased only when the current editor is solving the current round reliably.

**Professor-safe phrasing**
- phase `1`: proxy proposer and proxy solver with a real Qwen editor
- phase `2`: internal-feature verifier built from Qwen hidden states

## Slide 4: What Is Already Implemented

**Title**
Engineering Progress

**Use these as actual status numbers**
- `2` training modes implemented: LoRA and full finetuning
- `5` benchmark pipelines implemented: `GEdit`, `ImgEdit`, `GenEval`, `DPG-Bench`, `OneIG-Bench`
- `1` self-evolving loop module implemented
- `2` self-evolve editor backends implemented: `qwen_edit`, `pillow_demo`
- `1` public internal-feature extraction path exposed for Qwen edit understanding states
- `2` self-evolve solver/proposer placeholders reserved for future internal-Qwen integration

**Safe wording**
- “The infrastructure is no longer the blocker.”
- “The main remaining work is empirical execution and model-side iteration.”

## Slide 5: Current Status

**Title**
Where We Actually Are Today

**Say this directly**
- The baseline repo is implemented.
- The self-evolving loop control logic is implemented.
- The public benchmark export/score layer is implemented.
- Public Qwen understanding features are accessible in code, but not yet turned into a calibrated verifier.
- The upstream repos are not fully bootstrapped on the experiment machine yet in this workspace snapshot.
- Full result tables are not ready yet.

**Very important**
- “We do not have improvement claims to report yet.”
- “What we have now is a concrete system and an experiment matrix, not a finished empirical story.”

## Slide 6: Experiments Running / Planned

**Title**
Experiment Matrix

**Present this as queued or in progress, not completed**

Recommended matrix:
- Baseline `Qwen-Image-Edit-2509` zero-shot benchmark export
- Baseline LoRA finetune on the supervised edit set
- Self-evolve round `1` data generation only
- Self-evolve round `1` + LoRA update
- Self-evolve rounds `1 -> 2`
- Internal-feature verifier probing on saved Qwen hidden states
- Ablations:
  - no difficulty shaping
  - global reward only
  - local reward only
  - different acceptance thresholds

**Compact number to say**
- “The near-term plan is roughly `8-12` runs, depending on how many ablations we keep.”

## Slide 7: Expected Evaluation Story

**Title**
What We Expect To Test

Main edit benchmarks:
- `GEdit-Bench`
- `ImgEdit`

Secondary generation sanity checks:
- `GenEval`
- `DPG-Bench`
- `OneIG-Bench`

**Why this matters**
- Edit quality must improve without destroying general generation behavior.
- Internal-feature probing must show signal before we invest in a stronger internal verifier head.

## Slide 8: Placeholder Numbers For Discussion Only

These are **not results**. Present them only as expected ranges or target deltas.

**Suggested wording**
- “If the loop is working, I would consider something like the following a promising early signal.”

**Hypothesis table**
- `GEdit overall`: target `+1` to `+3` points over the supervised LoRA baseline
- `ImgEdit overall`: target `+0.5` to `+2` points
- `GenEval / DPG / OneIG`: target `no major regression`, ideally within `-1` point to `+1` point
- acceptance rate in early self-evolve rounds: target `25%` to `50%`

**Why these numbers are safe**
- They are framed as targets or success criteria, not measured outcomes.

## Slide 9: Risks

**Title**
Main Failure Modes

- The solver may reward edits that satisfy low-level metrics but look bad.
- The proposer may collapse to easy global edits.
- The loop may overfit to automatically verifiable edit families before object-level edits.
- The public hidden states may be useful for conditioning but still insufficient for a strong verifier without extra learning.

## Slide 10: Why The Project Should Continue

**Title**
Why It Is Still Worth Backing

- The research question is still novel and technically well grounded.
- The engineering platform is now concrete enough to run controlled experiments.
- Even a negative result is informative:
  - if the internal loop fails, we learn where internal verification breaks
  - if it works on a restricted edit family first, that is still publishable as a strong systems result

## Short Verbal Summary

Use this if the professor asks for a 30-second update:

“The baseline system is implemented, and I have added the first working self-evolving loop: proposer, editor, solver, difficulty control, and pseudo-label generation. The editor is already real Qwen-Image-Edit. Architecturally, the public code also exposes Qwen-conditioned hidden states, so the next research step is to turn those features into a stronger verifier. I’m not claiming gains yet. What I have now is a technically grounded system and an experiment matrix ready for execution.”

## Two-Minute Pitch

Use this if the discussion goes one level deeper:

“The idea I want to settle on is not the strongest possible claim, because that would be premature. The defensible version is: self-evolving image editing on top of Qwen-Image-Edit, using preservation-aware self-training first, and then pushing toward an internal verifier. Qwen is a strong target because the technical report describes a dual path through `Qwen2.5-VL` and the `VAE`, and the public DiffSynth implementation already exposes the hidden states from that understanding path. So the near-term paper can show whether iterative pseudo-labeling improves edit quality on `GEdit` and `ImgEdit` without hurting generation benchmarks, while the architecture research question is whether those exposed internal features are strong enough to replace our proxy solver.” 

## One-Slide Status Version

If you need a single summary slide, use this:

- **Idea**: self-evolving image editing without an external reward model
- **Model target**: `Qwen-Image-Edit`
- **Implemented**: baseline training/eval stack + self-evolve loop prototype + public internal-feature extraction
- **Benchmarks wired**: `GEdit`, `ImgEdit`, `GenEval`, `DPG-Bench`, `OneIG-Bench`
- **Current measured result**: no improvement claim yet
- **Current progress**: system complete enough to begin the experiment phase
- **Immediate next milestone**: first end-to-end self-evolve round with LoRA update and verifier probing
