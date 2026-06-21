# BMVC Direction Report: Internal Rubric CEPR

## Submission Position

- The project idea is worth pursuing for BMVC if the final method is framed as internal self-evolving post-training for Qwen-Image-Edit, not as a generic self-training recipe.
- The current codebase is strong enough for a submission pipeline: official Diffusers Qwen-Image-Edit-2509 anchor, resumable self-evolve rounds, Diffusers-native editor LoRA training, trainable proposer LoRA, monitor outputs, and benchmark export/scoring.
- The current CEPR result is not yet enough by itself. The best completed ImgEdit score so far is `4.199090909090905` for `cepr_stable_continue_r24`; if the same-pipeline Qwen baseline is higher, current CEPR should become an ablation and motivation, not the final method.
- The highest-probability direction is **Internal Rubric CEPR**: keep the reward internal, but replace opaque embedding-only semantic acceptance with atomic, structured verification signals.

## External Evidence Checked

- Qwen-Image technical report: Qwen-Image editing consistency is built from both Qwen2.5-VL semantic representations and VAE reconstructive representations, which supports our internal semantic-plus-latent reward design. Source: https://arxiv.org/abs/2508.02324
- Diffusers QwenImage docs: the official `QwenImageEditPlusPipeline` is the correct public anchor for `Qwen/Qwen-Image-Edit-2509`; docs also note `torch.compile` and few-step Lightning LoRAs as speed levers, but those should be used only for pilots unless the baseline is matched. Source: https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage
- EditReward: recent evidence says reliable reward modeling is a central bottleneck for scaling image-editing data; this supports making reward quality the main technical focus. Source: https://arxiv.org/abs/2509.26346
- UniEdit-I: closed-loop understanding-editing-verifying is an active direction, but it is training-free and unified-model oriented; our contribution remains weight-updating post-training for Qwen-Image-Edit using internal reward signals. Source: https://arxiv.org/abs/2508.03142
- Auto-Rubric as Reward: explicit, multi-dimensional rubrics are a stronger and more interpretable reward interface than single scalar VLM judgments; this directly supports moving from scalar CEPR to rubric CEPR. Source: https://huggingface.co/papers/2605.08354

## Main Hypothesis

- Current CEPR can detect broad semantic motion and preservation, but it does not explicitly verify important edit facts:
  - the requested source object exists in the source image
  - the required new object, attribute, or relation appears after editing
  - the old object or old attribute is removed when required
  - preservation constraints remain true
  - the image remains coherent and not globally drifted
- Hard edits fail exactly where these atomic checks matter: replacement, removal, extraction, composition, and spatial relation changes.
- Therefore, the next method should score candidates using structured source/edit/preservation criteria, still computed from Qwen-Image-Edit internal image-text and VAE features.

## Proposed Final Method

- Method name: `internal-cepr-rubric-v1`
- Proposer output should include:
  - `instruction`
  - `edit_type`
  - `source_object` or `target`
  - `target_object` or `replacement`
  - `required_after`
  - `forbidden_after`
  - `preserve`
  - `target_region`
- Evaluator should compute:
  - `rubric_source_grounded`: requested source target is supported in the source image
  - `rubric_required_after`: required after-edit concepts have higher support in the edited image
  - `rubric_forbidden_after_absent`: forbidden old concepts lose support in the edited image
  - `rubric_preservation`: preservation prompts remain semantically consistent between source and edited image
  - `rubric_validity`: reuse existing VAE latent locality and drift validity
  - `rubric_reward`: geometric combination with hard gates
- Existing CEPR remains as a safety layer:
  - internal prompt gain
  - taxonomy distractors
  - semantic preservation
  - VAE latent locality
  - validity gate

## Why This Is The Best Current Direction

- It preserves the novelty: no GPT-4V, Qwen-VL external judge, CLIP, detector, OCR, or external reward model at training time.
- It directly targets the observed weakness instead of tuning thresholds blindly.
- It gives better paper figures: per-rubric failure rates are easier to explain than a single reward scalar.
- It is implementable quickly because the current evaluator already exposes Qwen prompt features and VAE latent checks.
- It creates clean ablations:
  - current CEPR
  - rubric-only semantic gates
  - rubric CEPR without old-state removal
  - rubric CEPR without preservation gates

## Reviewer Risk And Paper Story

- A reviewer may reject the method if it is presented as "we added many reward terms." The paper must instead present CEPR as a reward-modeling argument for image editing.
- The story should be:
  - image editing reward is inherently multi-constraint, unlike pure image generation reward
  - edit success and content preservation are not interchangeable terms in one additive score
  - hard semantic edits require explicit after-state and old-state-removal verification
  - preservation requires both semantic and latent locality constraints
  - hard gates are a modeling choice to prevent reward compensation, not an engineering trick
- The method should be described as a decomposed internal reward model:
  - `source grounding`: is the requested edit meaningful for the source image?
  - `edit realization`: did the requested new concept or relation appear?
  - `old-state removal`: did the replaced/removed concept disappear when required?
  - `preservation`: did unrelated content remain stable?
  - `validity`: did the image avoid corrupt global drift?
- The paper needs analysis figures/tables:
  - acceptance failure breakdown by reward gate
  - per-edit-type benchmark changes, especially replace/remove/add/compose
  - correlation between internal reward components and ImgEdit/GEdit scores on sampled outputs
  - qualitative examples where scalar CEPR accepts a wrong edit and rubric CEPR rejects it
  - ablation showing additive reward is weaker than hard-gated decomposed reward
- Claim carefully: this is not just an internal VLM "judge." It is an internal, decomposed reward model built from Qwen-Image-Edit's own image-text features and VAE latent features.

## Fast Experiment Plan

- First run a tiny pilot:
  - 8 images, K=2 or K=4, 2 rounds
  - no long training
  - inspect accepted candidates and rubric failure reasons
- Then run a short result pilot:
  - 64 to 128 images
  - 8-image rounds
  - capped editor training steps
  - evaluate checkpoints around rounds 4, 8, 12, and 16
- Only run a longer 256-image experiment if the first 16 rounds show stable acceptance and visually plausible accepted samples.

## Training Speed Fixes

- Use capped steps for editor training. The continuation run used `max_train_steps=8` and completed editor training quickly; the older run used full epochs and was much slower.
- Keep `output.use_cumulative_manifest=false` for pilots. Cumulative manifests can silently grow training cost and delay feedback.
- Reduce LoRA cost:
  - use rank 16 or 8 for fast pilots
  - start with attention-heavy target modules before re-expanding to MLP/modulation modules
  - set `checkpointing_steps=0` for pilots because final LoRA weights are saved anyway
- If GPU memory allows, set training `offload=false`; per-step logs show about 13-16 seconds/step, and repeated VAE/text encoder offload can add overhead.
- Use `local_files_only=true` only after model cache is complete to avoid repeated Hugging Face HEAD requests.
- For candidate generation pilots, use fewer steps only for debugging. Final reported experiments must keep the paper-matched 40-step official Diffusers settings unless a separate speed ablation is reported.

## Decision Rule After Baseline

- If exact Qwen baseline ImgEdit is lower than current CEPR:
  - current CEPR may be enough with ablations, but still prefer rubric CEPR if it improves hard edit categories.
- If exact Qwen baseline ImgEdit is higher than current CEPR:
  - do not report current CEPR as the final method.
  - report it as motivation and move final claim to rubric CEPR.
- If rubric CEPR still does not beat baseline:
  - submit only if there is a strong ablation and diagnostic contribution, otherwise continue improving reward alignment before submission.
