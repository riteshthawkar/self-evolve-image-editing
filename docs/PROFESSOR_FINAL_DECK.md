# Professor Final Deck

This is the final 4-slide version centered on the chosen architecture and reward functions.

The slide text is written to be pasted directly into PPT. Each slide includes short paper references so the claims are anchored.

## Slide 1

**Title**
What Is Already Known, and What Gap I Am Targeting

**What is already known**
- Self-evolving proposer-solver training already exists for language and multimodal reasoning.
- RL and reward-based post-training already exist for image editing.
- Specialized scorers and reward models already exist for instruction-guided image editing.

**So I am not claiming**
- RL for image editing
- a reward model for image editing
- a generic multi-agent framework

**The actual gap**
- Existing proposer-solver self-evolution work is not designed for image editing, where the hard part is not only making the requested change but also preserving everything that should stay unchanged.
- Existing image-editing RL methods usually optimize one editor against one reward model, but do not give a proposer-editor-solver curriculum over raw unlabeled images.

**My claim**
- I want a self-evolving image editing loop where the proposer generates edit instructions on raw images, the editor samples multiple candidate edits, and the solver filters those candidates using editing-specific constraint-based and relative rewards.

**One sentence to say**
- "The novelty is not generic RL or generic self-play; it is a self-evolving image editing loop with an editing-specific verifier that explicitly separates requested change from collateral damage."

**Slide references**
- EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards
- Self-Questioning Language Models
- MM-Zero: Self-Evolving Multi-Model Vision Language Models From Zero Data
- InstructRL4Pix: Training Diffusion for Image Editing by Reinforcement Learning
- EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing
- ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation

## Slide 2

**Title**
Final Architecture

**Chosen architecture**

```text
Proposer -> Editor(K=4 samples) -> Solver Ensemble -> Relative Ranker -> Accept/Reject -> Train
```

**Roles**
- `Proposer`: given a raw image, generate a candidate edit instruction
- `Editor`: use Qwen-Image-Edit to sample `K=4` edited candidates for the same instruction
- `Solver Ensemble`: score instruction satisfaction, preservation, spatial correctness, and optional semantic consistency
- `Relative Ranker`: compare surviving candidates against each other, and optionally against a frozen reference output
- `Accept/Reject`: keep only high-confidence, high-quality edits
- `Train`: use accepted edits as pseudo-labels for the next round

**Why K-sample editing matters**
- Image editing is multimodal, so one instruction can have multiple plausible outputs.
- Single-sample acceptance is too noisy.
- Ranking among multiple candidates is more stable than trusting one absolute reward score.

**Why this is the right adaptation of EvoLMM**
- EvoLMM uses proposer and solver because the output is an answer.
- For image editing, the output is an edited image, so we need an extra role: the editor.
- We also need a relative ranker because image quality is comparative, not purely absolute.

**One sentence to say**
- "I keep the proposer-solver logic from EvoLMM, but I insert an editor and a relative ranker because image editing is a multimodal transformation problem, not a single-answer reasoning problem."

**Slide references**
- EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards
- Self-Questioning Language Models
- MM-Zero: Self-Evolving Multi-Model Vision Language Models From Zero Data

## Slide 3

**Title**
Final Reward Functions

**Step 1: hard-gated feasibility**

```text
G(y) = 1[S_inst(x,e,y) >= tau_inst] * 1[S_pres(x,y) >= tau_pres]
```

**Interpretation**
- `S_inst`: did the requested edit actually happen?
- `S_pres`: did unchanged content stay unchanged?

**Why hard gates**
- Edit success and preservation are not soft preferences.
- If either fails, the sample should not be accepted, even if some other score is high.

**Step 2: relative quality score**

```text
Q(y) = alpha * S_spa(x,e,y)
     + beta  * S_cf(x,e+,y,{e-})
     + gamma * S_rel(x,e,y,y_ref)
```

**Interpretation**
- `S_spa`: is the change localized where it should be?
- `S_cf`: does the edit match the true instruction more than distractor instructions?
- `S_rel`: is this candidate better than a reference output or previous checkpoint?

**Step 3: acceptance**

```text
y* = argmax_k Q(yk) over candidates with G(yk)=1

accept y* if:
- G(y*) = 1
- Q(y*) >= tau_q
- std(component_scores(y*)) <= delta_conf
```

**Final proposer reward**

```text
R_prop = exp(- (u - mu)^2 / (2 * sigma^2))
       - lambda_fail * 1[all candidates rejected]
       - lambda_easy * 1[too easy]
```

where `u` is normalized uncertainty or difficulty from candidate disagreement.

**Interpretation**
- very easy edits give weak learning signal
- impossible edits give noisy supervision
- moderate-difficulty edits create the best curriculum

**One sentence to say**
- "The solver reward is hard-gated and relative; the proposer reward is band-pass over edit difficulty."

**Slide references**
- EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards
- Self-Questioning Language Models
- InstructRL4Pix: Training Diffusion for Image Editing by Reinforcement Learning
- SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning
- EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing
- MDPO: Conditional Preference Optimization for Multimodal Large Language Models

## Slide 4

**Title**
Why This Could Work, Main Risks, and What I Will Test

**Why this could work**
- A single scalar edit score is too easy to game.
- Image editing is structurally a two-objective problem: make the requested change and preserve the rest.
- Hard constraints prevent catastrophic compensation between reward terms.
- Relative ranking is more stable than a single absolute reward.
- The proposer curriculum follows the EvoLMM logic of targeting the learning frontier rather than maximum difficulty.

**Main risks**
- reward hacking: the model may satisfy low-level metrics without producing genuinely better edits
- proposer collapse: it may drift to trivial edits or impossible edits
- judge miscalibration: spatial, counterfactual, or reference-based scores may not correlate strongly enough with human judgment

**Main experiment ladder**
1. supervised baseline
2. naive self-training
3. proposer-editor-solver with current simple hybrid reward
4. final method with hard-gated feasibility and K-sample ranking
5. add counterfactual reward
6. add relative reward against frozen reference output

**Current status**
- baseline training and evaluation stack is implemented
- self-evolving loop is implemented
- next step is to implement the final K-sample and relative-ranking reward path and run the experiments on the GPU machine

**One sentence to say**
- "The key technical question is whether better reward design leads to better self-generated edit data, not just whether self-training exists."

**Slide references**
- EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards
- EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing
- ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation
- SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning
- MDPO: Conditional Preference Optimization for Multimodal Large Language Models

## Full Paper List

- EvoLMM: Self-Evolving Large Multimodal Models with Continuous Rewards
  - https://arxiv.org/abs/2511.16672
- Self-Questioning Language Models
  - https://arxiv.org/abs/2508.03682
- MM-Zero: Self-Evolving Multi-Model Vision Language Models From Zero Data
  - https://arxiv.org/abs/2603.09206
- InstructRL4Pix: Training Diffusion for Image Editing by Reinforcement Learning
  - https://arxiv.org/abs/2406.09973
- EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing
  - https://arxiv.org/abs/2509.26346
- ADIEE: Automatic Dataset Creation and Scorer for Instruction-Guided Image Editing Evaluation
  - https://arxiv.org/abs/2507.07317
- SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning
  - https://arxiv.org/abs/2602.07458
- MDPO: Conditional Preference Optimization for Multimodal Large Language Models
  - https://arxiv.org/abs/2406.11839

## 20-Second Version

"My final method is a proposer-editor-solver loop for raw-image self-evolving image editing. The editor samples multiple candidates, the solver first hard-gates instruction success and preservation, then ranks candidates using spatial, counterfactual, and reference-relative rewards, while the proposer is trained to generate moderate-difficulty edits. The paper is about reward design for self-generated edit data quality, not about claiming generic RL or generic self-play."
