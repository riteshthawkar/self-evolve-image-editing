# Professor Short Deck

This version is written for a 3-4 slide meeting where the professor already knows the project title and wants to know the actual research idea, why it is still novel enough, and what exactly is being optimized.

## Slide 1

**Title**
What Is Already Known, and What Gap I Am Actually Targeting

**What is already known**
- RL for image editing already exists.
- Edit-specific reward models already exist.
- Proposer-solver or multi-role self-evolving systems already exist for language and multimodal reasoning.

**So my paper cannot simply claim**
- RL for image editing
- a reward model for image editing
- a multi-agent framework

**The gap I want to target**
- Existing self-evolving proposer-solver methods are not designed for image editing, where the hard part is not only making the requested change but also preserving everything that should stay unchanged.
- Existing image-editing RL methods do not cleanly give a self-evolving proposer-editor-solver curriculum over raw unlabeled images.

**One sentence to say**
- "The gap is the combination: self-evolving image editing needs a proposer curriculum plus an editing-specific verifier that separates requested change from collateral damage."

## Slide 2

**Title**
My Proposed Architecture: Proposer -> Editor(K) -> Solver Ensemble

**Architecture**
- `Proposer`: given a raw image, generate candidate edit instructions
- `Editor`: apply the instruction with Qwen-Image-Edit and produce `K` edited candidates
- `Solver Ensemble`: score instruction satisfaction, preservation, spatial correctness, and optional cycle or internal consistency
- `Relative Ranker`: compare candidates against each other, and optionally against a frozen reference model
- `Accept / Reject`: keep only candidates that pass hard constraints and rank highly
- `Train`: use accepted edits as pseudo-labels for the next round

**Why this is not just EvoLMM copied over**
- In EvoLMM, the solver outputs an answer and self-consistency is the reward.
- In image editing, the model outputs an edited image, so the solver must evaluate a structured visual transformation.
- That means we need an extra role, the editor, plus ranking and preservation-aware verification.

**Key design choice**
- The proposer should not maximize difficulty blindly.
- It should target moderate-difficulty edits, meaning edits that are informative but still learnable.

**One sentence to say**
- "I am adapting the proposer-solver idea to image editing by inserting an editor and redefining the solver around edit quality plus preservation."

## Slide 3

**Title**
Reward Function: What I Am Optimizing and Why It Might Work

**Step 1: hard-gated constraints**

```text
require:
  R_instruction >= tau_inst
  R_preservation >= tau_pres
```

**Interpretation**
- `R_instruction`: did the requested edit actually happen?
- `R_preservation`: did unchanged content stay unchanged?

**Step 2: relative quality score**

```text
R_quality =
  alpha * R_spatial
  + beta * R_cycle
  + gamma * R_internal
  + delta * R_counterfactual
  + eta * R_relative
```

**Interpretation**
- `R_spatial`: is the change localized where it should be?
- `R_cycle`: if I reverse the edit, do I recover the source image?
- `R_internal`: do the model's own internal features support the semantic shift without collapse?
- `R_counterfactual`: does the edit match the true instruction more than distractor instructions?
- `R_relative`: is the current edit better than a reference output or previous checkpoint?

**Proposer reward**

```text
R_proposer = exp(- (difficulty - mu)^2 / (2 * sigma^2))
```

**Interpretation**
- very easy edits give weak learning signal
- impossible edits give noisy or misleading signal
- moderate-difficulty edits create a useful curriculum

**Why this could work**
- A single scalar "edit score" is too easy to game.
- Image editing is naturally a two-objective problem: make the requested change and preserve everything else.
- Hard constraints prevent catastrophic compensation between reward terms.
- Relative ranking is more stable than trusting one absolute score.
- Moderate-difficulty proposer curriculum is the closest image-editing analogue of the EvoLMM logic.

**One sentence to say**
- "The main technical idea is hard-gated and relative reward design for the solver, plus learning-frontier curriculum for the proposer."

## Slide 4

**Title**
Novelty Claim, Risks, and Current Status

**Novelty claim I think is defensible**
- A self-evolving proposer-editor-solver framework for instruction-guided image editing from raw unlabeled images, with editing-specific constraint-based and relative rewards that separate edit success from preservation.

**What I will not claim**
- first RL image editor
- first editing reward model
- first multi-agent editing system

**Main risks**
- reward hacking: the model may satisfy low-level metrics without producing genuinely good edits
- proposer collapse: it may drift to trivial or impossible edits
- internal verifier weakness: hidden-state features may not correlate strongly enough with human judgment

**Current progress**
- baseline training and evaluation stack is implemented
- self-evolving loop is implemented
- reward framework is implemented
- final proposed architecture is now narrowed to multi-candidate ranking with constraint-based acceptance
- what remains is running the real experiments on the GPU machine

**What I need approval for**
- proceed with the reward-design-centered version of the paper
- evaluate whether the strongest contribution is decomposed preservation-aware reward
- evaluate whether the strongest contribution is uncertainty-shaped proposer curriculum
- evaluate whether the strongest contribution is internal-feature self-verification

**One sentence to say**
- "The key question is not whether self-training exists, but whether the right reward design makes self-evolving image editing actually reliable."

## 30-second Verbal Version

"The broad ingredients already exist separately: RL for image editing, edit reward models, and proposer-solver self-evolution. My idea is the missing combination. I want a proposer-editor-solver loop for image editing on raw unlabeled images, where the editor generates multiple candidates, the solver uses hard constraints for edit success and preservation, and then ranks surviving candidates with relative rewards such as spatial correctness, counterfactual consistency, and improvement over a reference output. The proposer is trained to generate moderate-difficulty edits rather than arbitrary hard ones. If this works, the model can generate useful new edit data for itself rather than relying only on labeled edit pairs."

## Backup Answer If Asked "Why Is This Still Novel?"

"It is not novel as generic RL or generic multi-agent training. The novelty is the editing-specific self-evolving loop: raw images, proposer-generated edit instructions, preservation-aware continuous reward, and uncertainty-shaped proposer curriculum. I am positioning the contribution around reward design and self-generated data quality, not around claiming to invent RL or reward models for editing."
