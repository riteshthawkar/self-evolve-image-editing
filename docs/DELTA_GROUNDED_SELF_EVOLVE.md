# Delta-Grounded Self-Evolve Method

This is the stronger research method path for the project.

The goal is not just to run self-training. The goal is to make self-training suitable for image editing by evaluating the **edit delta**: what changed, what should have changed, and what should have stayed fixed.

## Architecture

The research target is now:

```text
Structured Proposer -> Qwen Editor(K candidates) -> Delta Evaluator -> Rank/Accept -> Train
```

- `Structured Proposer`: generates edit instructions from raw images using a difficulty ladder and,
  when available, source-selection metadata such as plausible edit families.
- `Editor`: samples multiple candidate edits for the same image and instruction.
- `Delta Evaluator`: checks feasibility, preservation, internal Qwen prompt gain, and candidate rank.
- `Train`: accepted edits become pseudo-labels for the editor.

The intended paper version uses Qwen-Image-Edit as the editor and Qwen edit-conditioning
representations as one evaluator signal:

- editor LoRA trained from accepted pseudo-labels
- evaluator head or LoRA trained from self-generated winner/loser pairs
- optional proposer policy after the evaluator is calibrated

The current implementation adds the candidate-group evaluator path and exports evaluator training data
with source image, candidate image, true instruction, distractor instructions, component scores, and
winner/loser preferences. The learned evaluator is the next stage after the exported labels are
validated.

## Why This Is Different From The Earlier Hybrid Solver

The earlier hybrid solver used one weighted scalar score. That is useful as a baseline, but it allows compensation: a strong score on one component can hide failure on another.

The delta-grounded path uses:

- hard instruction and preservation gates
- multiple candidates for the same instruction
- relative ranking among feasible candidates
- counterfactual instruction discrimination
- Qwen internal prompt-gain scoring for internal-only proposals
- evaluator disagreement filtering

## Evaluator Logic

For each source image and proposed instruction, the editor generates `K` candidates.

The evaluator first applies hard gates:

- instruction gate: did the requested edit happen?
- preservation gate: did non-target content remain stable?

Candidates that fail either gate are rejected before ranking.

For proxy-verifiable edits, the instruction gate uses explicit image statistics such as luminance,
saturation, contrast, or warmth. For internal-only edits such as background blur or subject emphasis,
the instruction gate uses Qwen hidden-state prompt gain; if the internal score is unavailable, the
candidate is rejected rather than accepted by a proxy metric.

Among feasible candidates, the evaluator ranks by:

- spatial quality
- counterfactual specificity
- relative group score
- optional cycle consistency
- optional internal Qwen feature score

Only the top `m` candidates are accepted.

## Counterfactual Signal

For a true instruction, the evaluator creates distractor instructions from the proposal bank.

Example:

```text
true: make the image warmer
distractors: make the image cooler, increase saturation, increase contrast
```

The candidate is rewarded when it matches the true instruction more than the distractors. This discourages generic over-editing and no-op edits.

## Outputs

Each round writes the standard self-evolve outputs plus evaluator training records:

```text
outputs/self_evolve/<run>/round_01/proposals.jsonl
outputs/self_evolve/<run>/round_01/evaluator_training.jsonl
outputs/self_evolve/<run>/round_01/evaluator_preferences.jsonl
outputs/self_evolve/<run>/round_01/train_manifest.json
outputs/self_evolve/<run>/round_01/summary.json
```

`evaluator_training.jsonl` contains per-candidate labels and scores.

`evaluator_preferences.jsonl` contains winner-loser pairs from each candidate group.

These files are the bridge from the current heuristic evaluator to a learned evaluator head or LoRA.

## Commands

Local smoke path:

```bash
bash scripts/self_evolve_pillow_delta_ranker.sh --limit 8
```

Qwen-backed run:

```bash
bash scripts/self_evolve_2509_delta_grounded.sh --limit 32
```

Matrix runner:

```bash
bash scripts/run_self_evolve_matrix.sh \
  --variant delta-grounded \
  --images-dir data/unlabeled/self_evolve \
  --limit 32
```

## Research Ablations

The minimum ablation set is:

- base Qwen editor, no self-evolve
- plain proxy self-training
- weighted hybrid reward
- old proxy-only delta ranker
- delta-grounded ranker with structured proposer and Qwen internal prompt-gain checks
- spatial-only evaluator
- cycle-only evaluator
- internal-only evaluator
- delta-grounded without counterfactual score
- delta-grounded without relative group score
- delta-grounded with `K=1` versus `K=4`

The critical comparison is:

```text
weighted scalar reward vs hard gates + group-relative ranking + internal Qwen delta signals
```

That comparison tests the main research claim.
