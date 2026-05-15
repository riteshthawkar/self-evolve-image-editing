# NeurIPS Methods Shortlist

This note is a practical shortlist of methods that could make the project technically stronger and more publishable.

The standard I am using is simple:

- it should be plausible on top of `Qwen-Image-Edit`
- it should give a real experimental story, not just engineering
- it should still be defensible if some components underperform

## The best main thesis right now

**Recommended main paper direction**

Bootstrapped self-evolving image editing with:

1. a real Qwen editor
2. a preservation-aware verifier
3. iterative self-training on unlabeled images
4. a path from proxy verification to internal-feature verification

In one sentence:

**Can a strong image editor improve itself from unlabeled images by combining pseudo-label self-training, spatial or preservation-aware verification, and progressively more internal model-based rewards?**

This is stronger than plain LoRA finetuning, but safer than claiming a fully internal proposer-verifier loop from day one.

## Tier 1: strongest and most credible methods

These are the methods I would take most seriously for a NeurIPS-quality project.

### 1. Internal-feature verifier on top of Qwen hidden states

**Why it matters**

This is the most project-specific idea. It ties directly to Qwen’s dual-path edit architecture instead of adding a generic external judge.

**Research basis**

- The `Qwen-Image` technical report says edit consistency is improved by aligning `Qwen2.5-VL` and `MMDiT`, and by separately feeding the original image to `Qwen2.5-VL` and the `VAE` encoder.
- The public DiffSynth implementation exposes the hidden states used for edit conditioning.

**How to try it**

- Use the current public hidden states extracted from image plus instruction.
- Train a lightweight verifier head or scorer on top of those features.
- Supervise it with:
  - benchmark pairwise preferences
  - self-generated winner vs loser pairs
  - a small amount of manual labels if needed

**Why this could be publishable**

If it works, the novelty is not just “RL for editing.” It becomes: **internal representation bootstrapping for self-improving image editing**.

**Key risk**

The hidden states may be useful for conditioning but not strong enough by themselves for calibrated verification.

**Primary sources**

- [Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324)
- [DiffSynth Qwen pipeline](https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/pipelines/qwen_image.py)
- [DiffSynth Qwen text encoder](https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/models/qwen_image_text_encoder.py)

### 2. Cycle edit consistency for self-training

**Why it matters**

This is the cleanest way to make self-training less gameable. A model that edits forward but cannot reverse the edit reliably is probably cheating.

**Research basis**

- `UIP2P` shows unsupervised instruction-based editing with cycle edit consistency in both image and attention spaces.
- `Inverse-and-Edit` shows cycle consistency can improve reconstruction and preservation in editing systems.

**How to try it**

- Sample instruction `x -> y`
- Automatically generate reverse instruction `y -> x`
- Enforce:
  - reconstruction consistency
  - attention or localization consistency
  - feature consistency outside the edited region

**Why this could be publishable**

Cycle consistency gives a principled self-supervision signal without requiring ground-truth edited targets.

**Key risk**

Reverse instruction generation can be noisy and may collapse to trivial reversible edits.

**Primary sources**

- [UIP2P: Unsupervised Instruction-based Image Editing via Cycle Edit Consistency](https://arxiv.org/abs/2412.15216)
- [Inverse-and-Edit: Effective and Fast Image Editing by Cycle Consistency Models](https://arxiv.org/abs/2506.19103)

### 3. Spatially grounded verifier instead of global scoring

**Why it matters**

Most self-training loops fail because the reward is too global. The model learns to look “somewhat aligned” while damaging irrelevant regions.

**Research basis**

- `SpatialReward` argues that reward models fail from “attention collapse” and improves performance by grounding judgment in predicted edit regions.
- `InstructRL4Pix` uses attention-map-based rewards to improve edit localization.

**How to try it**

- Predict or estimate the edited region.
- Score instruction satisfaction inside the region.
- Score preservation outside the region.
- Aggregate them separately instead of with a single global score.

**Why this could be publishable**

This directly matches the failure mode of image editing. It is much more convincing than a generic VLM score.

**Key risk**

Region estimation itself may be noisy, especially for attribute edits or diffuse style edits.

**Primary sources**

- [SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning](https://arxiv.org/abs/2602.07458)
- [InstructRL4Pix: Training Diffusion for Image Editing by Reinforcement Learning](https://arxiv.org/abs/2406.09973)
- [Region-Aware Diffusion for Zero-shot Text-driven Image Editing](https://arxiv.org/abs/2302.11797)

### 4. Pairwise preference learning from self-generated winners and losers

**Why it matters**

Once you can score multiple candidates for the same source image and instruction, you have preference data for free.

**Research basis**

- `D3PO` shows diffusion models can be aligned from preferences without learning a separate reward model first.
- `InPO` improves efficiency for diffusion preference optimization.
- `MDPO` is important conceptually because it shows multimodal preference optimization can ignore the image if you are careless.

**How to try it**

- For each source image plus instruction, sample `k` candidates.
- Use your verifier to rank them.
- Train with:
  - DPO-style objective
  - winner-preserving preference loss
  - reward anchor so the chosen sample does not get degraded

**Why this could be publishable**

You can turn self-generated candidates into a scalable preference dataset, which is a strong story if it consistently beats plain SFT.

**Key risk**

Bad verifier means bad preferences. Also, multimodal conditioning can silently collapse if the preference objective over-focuses on text.

**Primary sources**

- [Using Human Feedback to Fine-tune Diffusion Models (D3PO)](https://arxiv.org/abs/2311.13231)
- [InPO: Inversion Preference Optimization with Reparametrized DDIM](https://arxiv.org/abs/2503.18454)
- [MDPO: Conditional Preference Optimization for Multimodal Large Language Models](https://arxiv.org/abs/2406.11839)

## Tier 2: strong additions that can materially improve the project

These are not necessarily the main novelty, but they can make the main method better or easier to sell.

### 5. Adaptive test-time scaling for editing

**Why it matters**

Image editing is constrained by both source image and instruction. Because of that, fixed best-of-`N` sampling wastes compute on easy edits and still misses hard edits.

**Research basis**

- `ADE-CoT` shows adaptive compute allocation, edit-specific verification, and opportunistic stopping improve the performance-efficiency trade-off in editing.

**How to try it**

- For self-training, generate multiple candidates only for hard edits.
- Prune early with a cheap verifier.
- Stop once a candidate exceeds a confidence threshold.

**Why this is useful**

Even if training gains are modest, test-time scaling can improve benchmark numbers quickly and can also feed better pseudo-labels into training.

**Key risk**

If the verifier is weak, adaptive search just amplifies noise.

**Primary source**

- [From Scale to Speed: Adaptive Test-Time Scaling for Image Editing](https://arxiv.org/abs/2603.00141)

### 6. RL or RLAIF once the verifier is good enough

**Why it matters**

If you can build a decent verifier, RL becomes much more credible. Without that, RL usually adds instability faster than it adds gains.

**Research basis**

- `Image-Editing Specialists` shows online RLAIF can improve structural coherence and alignment.
- `Edit-R1` and `Uniworld-V2` show policy optimization can improve editing if the reward signal is stabilized.
- `EditScore` and `EditReward` argue that high-fidelity reward models are the real bottleneck.

**How to try it**

- Start with offline preference optimization.
- Move to online RL only after the verifier correlates well with benchmark outcomes.
- Use group filtering or low-variance batching to stabilize updates.

**Why this could be publishable**

If the verifier is internal or semi-internal, RL becomes more novel than simply reusing an external MLLM judge.

**Key risk**

This is expensive, noisy, and easy to destabilize.

**Primary sources**

- [Image-Editing Specialists: An RLAIF Approach for Diffusion Models](https://arxiv.org/abs/2504.12833)
- [Uniworld-V2 / Edit-R1](https://arxiv.org/abs/2510.16888)
- [EditReward](https://arxiv.org/abs/2509.26346)
- [EditScore](https://arxiv.org/abs/2509.23909)

### 7. Counterfactual or reverse-negative data generation

**Why it matters**

A good verifier needs hard negatives, not only random bad edits.

**How to try it**

- Generate plausible but wrong edits:
  - right object, wrong attribute
  - correct edit region, wrong semantics
  - correct semantics, over-edited background
  - under-edited source with high identity preservation
- Train verifier and preference objective to distinguish these failure modes.

**Why this is useful**

This gives you richer supervision without human labels and makes ablations more convincing.

**Connection to literature**

- `Uniworld-V2` uses negative-aware finetuning.
- `MedEdit` is a useful conceptual reference for counterfactual editing as a way to model plausible changes while preserving fidelity.

**Primary sources**

- [Uniworld-V2 / Edit-R1](https://arxiv.org/abs/2510.16888)
- [MedEdit: Counterfactual Diffusion-based Image Editing](https://arxiv.org/abs/2407.15270)

## Tier 3: unusual but potentially differentiating ideas

These are the ideas that could make the project stand out if they work, but they are not the first thing I would depend on.

### 8. Proposal-family bandit or curriculum scheduling

**Idea**

Treat edit families as arms in a bandit:

- text edits
- color edits
- object insertion
- style transfer
- local geometry

Allocate training or sampling budget to the edit families with the highest expected improvement or uncertainty.

**Why it could work**

Your system already has difficulty shaping. Turning that into an adaptive bandit policy is a natural next step and gives a more principled curriculum.

**Risk**

Can become complicated without giving enough gain.

### 9. Sequential Monte Carlo over edit trajectories

**Idea**

Instead of sampling several edits independently, keep a population of candidate trajectories and repeatedly resample the promising ones using a self-reward.

**Why it could work**

This is a principled alternative to best-of-`N`, especially if you want stronger test-time search with limited compute.

**Inspiration**

- self-rewarded SMC in diffusion-style generation
- test-time scaling for editing

**Primary sources**

- [Self-Rewarding Sequential Monte Carlo for Masked Diffusion Language Models](https://arxiv.org/abs/2602.01849)
- [From Scale to Speed: Adaptive Test-Time Scaling for Image Editing](https://arxiv.org/abs/2603.00141)

### 10. Reason first, render second

**Idea**

Before editing, generate a structured reasoning trace:

- what should change
- what must stay fixed
- where the change should happen
- what failure modes to avoid

Then condition the edit and the verifier on that plan.

**Why it could work**

This is especially promising for reasoning-heavy edits, multi-turn edits, and text edits.

**Primary source**

- [ThinkRL-Edit: Thinking in Reinforcement Learning for Reasoning-Centric Image Editing](https://arxiv.org/abs/2601.03467)

## What I would actually try first

If the goal is a serious paper and not endless exploration, I would stage it like this.

### Stage A: low-risk but publishable baseline extension

- self-evolving loop
- preservation-aware proxy verifier
- cycle edit consistency
- GEdit and ImgEdit as main benchmarks

If this works, you already have a coherent paper.

### Stage B: project-specific novelty

- extract Qwen internal hidden states
- train an internal-feature verifier
- replace or augment the proxy solver

This is the strongest technical differentiator.

### Stage C: scaling and alignment

- generate multiple candidates per edit
- create pairwise winner/loser data
- run preference optimization
- optionally move to online RL

### Stage D: extra upside

- adaptive test-time scaling
- reasoning traces
- bandit scheduling of edit families

## The three best paper shapes

### Paper shape 1: safest

**Title shape**
Self-Evolving Image Editing via Preservation-Aware Self-Training

**Main contribution**

Show that self-training with cycle consistency and locality-aware verification improves editing without heavy RL.

### Paper shape 2: strongest if it works

**Title shape**
Bootstrapping Internal Verifiers for Self-Evolving Image Editing

**Main contribution**

Use Qwen internal hidden states to replace or augment an external or proxy verifier.

### Paper shape 3: most practical and benchmark-friendly

**Title shape**
Adaptive Self-Training and Test-Time Scaling for Image Editing

**Main contribution**

Combine self-training and adaptive inference to improve quality under fixed compute.

## My recommendation

If you want the best balance of novelty, feasibility, and NeurIPS credibility:

1. make the main method **self-training + cycle consistency + spatial or preservation-aware verification**
2. make the main novelty **internal-feature verification on top of Qwen hidden states**
3. keep **preference optimization or RL** as the second-stage extension, not the initial dependency

That gives you:

- a paper that is executable
- a clear technical hypothesis
- a project-specific architectural angle
- a fallback path if RL turns unstable
