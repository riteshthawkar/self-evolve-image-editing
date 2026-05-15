# Novelty Positioning

This note is the defensible version of the novelty claim for the NeurIPS direction.

## Short Answer

The idea is **not novel** if it is framed as any one of the following:

- RL for image editing
- reward models for image editing
- multi-agent or proposer-solver training
- self-improvement for multimodal models

All of those already exist in nearby literature.

The idea is still **plausibly novel** if it is framed as:

- **self-evolving image editing from raw unlabeled images**
- with a **Proposer -> Editor -> Solver** architecture
- where the **solver is editing-specific**, not a generic VLM judge
- and where the reward is explicitly **decomposed into edit success and non-edit preservation**
- with the **proposer trained by uncertainty-shaped curriculum**, following the EvoLMM logic of preferring moderate difficulty rather than maximum difficulty
- and ideally without a separately trained external human-preference reward model

## Exact Claim To Use

Use this as the paper-level claim:

> We study self-evolving image editing from raw images using a proposer-editor-solver loop. The proposer generates candidate edit instructions, the editor applies them, and the solver assigns an editing-specific continuous reward that separates instruction satisfaction from preservation of unchanged content. Accepted edits become pseudo-labels for the next round of training, while the proposer is optimized toward moderate-difficulty edits rather than arbitrarily hard ones.

This is the safest concise version.

## What We Should Not Claim

Do not claim any of the following:

- first RL method for image editing
- first reward model for image editing
- first multi-agent image editing framework
- first self-evolving multimodal framework
- first use of preference optimization for image editing

Those claims are too broad and likely false.

## What We Can More Safely Claim

These are narrower and much more defensible:

- first self-evolving proposer-editor-solver framework for instruction-guided image editing from raw unlabeled images
- first editing-specific self-evolution framework that explicitly separates requested change from preservation outside the edited region
- first attempt to use internal editor features as a self-verifier for image editing within a self-evolving loop
- first uncertainty-shaped proposer curriculum for image editing in the EvoLMM style

The last two are especially strong if the experiments support them.

## Related Work Positioning

| Paper family | What they already cover | What remains open for us |
| --- | --- | --- |
| EvoLMM / SQLM | Self-evolving proposer-solver training, continuous or uncertainty-shaped rewards, learning frontier curriculum | They are not image editing systems and they do not solve the preservation problem specific to editing |
| UniCorn / MM-Zero | Multi-role self-evolving multimodal systems, including proposer, solver, judge, and generated supervision | They are closer to reasoning or generation than instruction-guided image editing with source-image preservation |
| HIVE / InstructRL4Pix / Edit-R1 / ImageEdit-R1 | RL for image editing and editing improvement via learned or implicit rewards | They do not define the same raw-image proposer-editor-solver self-evolving curriculum centered on preservation-aware acceptance |
| EditReward / EditScore / SpatialReward | Strong evidence that image editing needs specialized rewards and spatial reasoning | They motivate our solver design, but they are not themselves proposer-driven self-evolution methods |
| D3PO / InPO / DGPO / mDPO | Preference optimization and groupwise alignment for diffusion models | These are useful optimization layers for accepted vs rejected edits, but not the full self-evolving editing framework |

## Strongest Version Of The Paper

The strongest version is not:

- "We use RL for image editing"

The strongest version is:

> A self-evolving image editing framework in which a proposer generates edit instructions on raw images, an editor produces candidate edits, and an editing-specific solver uses decomposed continuous reward to filter and rank those candidates. The key technical contribution is the reward design: instruction satisfaction, preservation, localization, cycle consistency, and optionally internal-feature verification, together with an uncertainty-shaped proposer curriculum.

This framing is specific enough to be distinct and broad enough to support multiple ablations.

## Why This Can Be NeurIPS-Worthy

The paper has a credible NeurIPS angle if it argues one of these:

1. **Reward decomposition is the main innovation**
- editing success and preservation should be treated as separate objectives
- a single scalar edit score is structurally insufficient

2. **Uncertainty-shaped proposer curriculum is the main innovation**
- self-evolving editing should target the learning frontier, not maximum difficulty
- this adapts the EvoLMM intuition to a continuous visual action space

3. **Internal self-verification is the main innovation**
- the model's own editing features can partially replace external reward models
- this reduces dependence on separate annotators or reward-model training

The safest paper angle is usually a combination of 1 and 2. The highest-risk, highest-upside angle is 3.

## Recommended Final Positioning

If we need one sentence for the professor or for the paper intro, use this:

> Existing work already shows that image editing benefits from reward-based optimization and that multimodal models can self-evolve through proposer-solver training. Our gap is the combination: a self-evolving image editing system needs an editing-specific verifier that rewards both requested change and preservation, plus a proposer curriculum that targets informative edits rather than arbitrary difficulty.

## Best Experimental Story

The cleanest experimental story is:

1. supervised Qwen-Image-Edit baseline
2. naive self-training baseline
3. proposer-editor-solver with plain proxy reward
4. proposer-editor-solver with decomposed preservation-aware reward
5. proposer-editor-solver with uncertainty-shaped proposer reward
6. optional internal-feature verifier or preference-optimization extension

This lets us show that the contribution is not simply "more synthetic data", but **better self-generated data because of better reward design**.
