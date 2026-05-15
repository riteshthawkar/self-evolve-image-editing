# Qwen Architecture Audit

This note answers one narrow question:

Can we honestly present the project as a fully internal proposer-editor-solver loop on top of public `Qwen-Image-Edit` code?

Short answer: not yet.

## What the official sources confirm

- The `Qwen-Image` technical report states that edit consistency is improved by aligning latent representations between `Qwen2.5-VL` and `MMDiT`, and by feeding the original image separately into `Qwen2.5-VL` and the `VAE` encoder.
- The public DiffSynth implementation does load a `Qwen2.5-VL`-based text or understanding module inside the edit pipeline.
- The public prompt embedder for editing already uses image plus instruction prompts to extract hidden states from that understanding module.

Relevant sources:

- Technical report: [arXiv:2508.02324](https://arxiv.org/abs/2508.02324)
- DiffSynth pipeline: [diffsynth/pipelines/qwen_image.py](https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/pipelines/qwen_image.py)
- DiffSynth text encoder: [diffsynth/models/qwen_image_text_encoder.py](https://github.com/modelscope/DiffSynth-Studio/blob/main/diffsynth/models/qwen_image_text_encoder.py)
- DiffSynth trainer: [examples/qwen_image/model_training/train.py](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_training/train.py)

## What is actually exposed in public code

The public code gives us three concrete things:

1. A real edit model we can train and evaluate.
2. A real understanding encoder inside the edit stack.
3. Access to hidden states conditioned on image plus instruction.

This is stronger than a black-box editor. It means we can probe or reuse internal representations.

In our repo, this is now exposed through [qwen_pipeline.py](/Users/ritesh.thawkar/Ritesh/neurips-project/src/qwen_edit_project/utils/qwen_pipeline.py), which includes `extract_qwen_edit_understanding_features(...)`.

## What is not exposed cleanly

The public code does **not** currently give us:

- a standalone proposer API that emits edit instructions from an input image
- a calibrated verifier API that scores instruction satisfaction
- a reward head trained for accept or reject decisions
- a public training recipe for internal verifier learning

So the phrase “fully internal proposer-editor-solver loop” is still a research goal, not a completed capability.

## What this means for project framing

There are two possible pitches.

### Pitch A: too strong right now

“We built a fully closed internal self-improving loop using Qwen’s own verifier and proposer.”

Do not use this. The public codebase does not support that claim yet.

### Pitch B: defensible and still novel

“We are building a self-evolving image editing loop on top of Qwen-Image-Edit. The current system already has a real editor, benchmark stack, and iterative pseudo-labeling loop. The next research step is to replace the proxy solver with a scorer built from Qwen’s publicly exposed internal understanding features.”

Use this.

## Recommended settled idea

**Working title**
Self-Evolving Image Editing with Preservation-Aware Self-Training on Qwen-Image-Edit

**One-sentence thesis**
Can a strong image editor improve itself from unlabeled images using iterative pseudo-labeling and preservation-aware filtering, while moving progressively toward an internal representation-based verifier?

**Why this is the right scope**

- It is executable now.
- It uses the current repo honestly.
- It leaves room for a stronger second phase if the internal feature path works.
- It avoids overclaiming a capability the public upstream code does not directly expose.

## Recommended project phases

### Phase 1: executable now

- `Qwen-Image-Edit-2509` as the editor
- self-evolving loop with difficulty control
- proxy proposer and solver
- LoRA updates between rounds
- evaluation on `GEdit` and `ImgEdit`
- generation sanity checks on `GenEval`, `DPG-Bench`, and `OneIG-Bench`

### Phase 2: architecture research

- use internal hidden states from the public Qwen understanding path
- learn or design a verifier on top of those states
- test whether internal features outperform the proxy solver
- only then revisit the stronger “internal proposer-solver” claim

## Professor-safe wording

Use formulations like these:

- “The editor is real, the loop is real, and the benchmark stack is real.”
- “The internal verifier is the main research question, not a solved component.”
- “We already have public access to Qwen-conditioned hidden states, so this is technically grounded rather than speculative.”

Avoid formulations like these:

- “The model already proposes and verifies edits internally.”
- “We have already shown self-improvement.”
- “The internal closed loop is finished.”

## Bottom line

The project is strong enough to continue, but the right thesis is a staged one:

- first, demonstrate self-evolving edit improvement on top of Qwen with careful filtering and evaluation
- then, investigate whether Qwen’s exposed internal understanding features can replace the proxy solver
