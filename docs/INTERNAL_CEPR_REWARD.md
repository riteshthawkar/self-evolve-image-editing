# Internal CEPR Reward

The main method uses **Contrastive Edit-Preservation Reward (CEPR)**. CEPR is a fixed
reward evaluator, not a third trainable agent. The trainable roles are the proposer and the
editor; CEPR only scores candidate edited images and decides which samples enter editor
training.

CEPR is internal-only: it does not use GPT-4V, CLIP, detectors, OCR, or any external reward
model. Instead, it reuses Qwen-Image-Edit internal representations that are already loaded for
the editor:

- prompt-conditioned Qwen understanding features for the original and edited image
- Qwen text or prompt anchor features
- Qwen VAE latents for locality and preservation checks

The current implementation reuses the loaded editor pipeline for these feature reads. CEPR does
not generate images, receive gradients, update LoRA weights, or save checkpoints. In config files
the historical key `solver:` is still accepted, but the correct terminology is **evaluator** or
**reward evaluator**.

For source image `x`, instruction `c`, and candidate edit `y_i`, the reward is:

```text
R(y_i | x, c) = sqrt(E_i * P_i)
```

with hard gates:

```text
E_i >= tau_E, P_i >= tau_P, Q_i >= tau_Q, R_i >= tau_R
```

If any gate fails, the candidate reward is set to zero and it is not used for editor training.

## Edit Specificity

`E_i` measures whether the candidate edit is specifically aligned with the requested instruction:

```text
E_i = sqrt(
  sigmoid((Delta_true_i - max_j Delta_wrong_ij) / T_E)
  *
  sigmoid((Delta_true_i - gamma) / T_abs)
)
```

`Delta_true_i` is the Qwen internal image-text prompt-gain for the true instruction:

```text
Delta_true_i = sim_Q(y_i, c) - sim_Q(x, c)
```

`Delta_wrong_ij` is the same internal prompt-gain for counterfactual distractor instructions. This prevents reward hacking where the model makes a plausible but wrong edit.

## Taxonomy-Aware Internal Rubric

For trainable-proposer runs, CEPR also supports a structured edit rubric. The proposer emits fields
such as:

```json
{
  "edit_type": "object_replacement",
  "source_object": "person",
  "target_object": "stuffed animal",
  "target_region": "main subject",
  "preserve": ["background", "scene layout", "lighting"],
  "instruction": "Replace the person with a stuffed animal while preserving the background."
}
```

CEPR turns those fields into internal contrastive prompts and scores them with Qwen-Image-Edit
features:

- target gain: the edited image should become more similar to the requested target object,
  attribute, material, style, or location
- source drop: for replacement/removal/move edits, the original object or original relation should
  become less supported
- wrong-edit margin: the true structured edit should beat distractor edits such as wrong
  replacement targets or unchanged-object prompts

This is the internal counterpart to external VLM rubrics: the reward is decomposed by edit taxonomy,
but all scoring still comes from Qwen-Image-Edit internal representations.

## Preservation

`P_i` measures whether non-target content is preserved:

```text
P_i = sqrt(P_sem_i * P_lat_i)
```

`P_sem_i` is semantic preservation from Qwen's own image-text understanding features using a blank/preservation prompt. `P_lat_i` is VAE-latent preservation outside the inferred edit region.

The edit region is estimated internally from Qwen VAE latent difference, not from an external segmenter:

```text
M_i = 1[ latent_delta(x, y_i) > mean + alpha * std ]
```

Then preservation is computed outside `M_i`:

```text
P_lat_i = exp(- outside_latent_delta / T_P)
```

## Validity Gate

`Q_i` is a hard internal validity gate. It checks whether the inferred edit region is plausible for the proposed edit and whether total latent drift is not excessive:

```text
Q_i = sqrt(region_plausibility_i * drift_score_i)
```

Quality is not allowed to compensate for wrong editing. `Q_i` only rejects unstable or destructive candidates.

## Candidate Selection

For each source image and instruction, sample `K` candidates from the current
Qwen-Image-Edit checkpoint using the official generation settings. Accept only the top feasible
candidate:

```text
y* = argmax_i R(y_i | x, c)
```

This turns self-evolution into constrained candidate selection: the model trains only on edits that are both instruction-specific and preservation-safe.

## Preference Learning

The main self-evolution path now trains the editor from internal preference pairs rather than direct
weighted SFT targets. For each proposal group:

```text
chosen = highest-ranked accepted CEPR candidate
rejected = lower-ranked candidate from the same group
```

The loop writes these pairs to `preference_manifest.jsonl` and launches the Diffusers LoRA trainer
with `pairwise_linear_sdpo`. This keeps the supervision internal and relative: the model is not told
that a generated image is an absolute target, only that one of its own candidates is preferable to
another candidate under CEPR.

The CEPR-weighted SFT manifest is still written for diagnostics and ablations, but the main
trainable-proposer configs disable rejected-image SFT targets.

When no candidate passes the hard accept gates, the loop can still form a near-miss pair if the best
candidate clears raw internal reward, semantic-edit, preservation, and validity floors. These pairs
use CEPR raw reward for within-group ranking and carry lower sample weight. They set
`chosen_is_near_miss=true` and `preference_sft_weight=0.0`, so the editor learns only the relative
ordering between two self-generated candidates. This is important for object removal/replacement,
where accepted positives are sparse but failed candidates still expose the boundary between "closer
to the requested edit" and "worse failure."

## Main Command

```bash
bash scripts/run_self_evolve_matrix.sh \
  --variant internal-cepr \
  --images-dir data/unlabeled/self_evolve \
  --limit 512 \
  --output-prefix outputs/self_evolve/main_internal_cepr
```

To launch LoRA training after each self-evolve round:

```bash
bash scripts/run_self_evolve_matrix.sh \
  --variant internal-cepr \
  --images-dir data/unlabeled/self_evolve \
  --limit 512 \
  --output-prefix outputs/self_evolve/main_internal_cepr_train \
  --launch-training
```
