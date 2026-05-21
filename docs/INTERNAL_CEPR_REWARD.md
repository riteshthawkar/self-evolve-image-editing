# Internal CEPR Reward

The main method uses **Contrastive Edit-Preservation Reward (CEPR)**. It is an internal-only reward computed from the editor itself, not from GPT-4V, CLIP, detectors, OCR, or any external reward model.

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

For each source image and instruction, sample `K` candidates from Qwen-Image-Edit-2509 using the official generation settings. Accept only the top feasible candidate:

```text
y* = argmax_i R(y_i | x, c)
```

This turns self-evolution into constrained candidate selection: the model trains only on edits that are both instruction-specific and preservation-safe.

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
