# Delta-Grounded Self-Evolve Method

This is the stronger research method path for the project.

The goal is not just to run self-training. The goal is to make self-training suitable for image editing by evaluating the **edit delta**: what changed, what should have changed, and what should have stayed fixed.

## Architecture

The research target is:

```text
Proposer -> Editor(K candidates) -> Evaluator -> Rank/Accept -> Train
```

- `Proposer`: generates edit instructions from raw images.
- `Editor`: samples multiple candidate edits for the same image and instruction.
- `Evaluator`: checks feasibility and ranks candidates.
- `Train`: accepted edits become pseudo-labels for the editor.

The intended paper version uses one shared base model with separate role LoRAs:

- proposer LoRA
- editor LoRA
- evaluator LoRA

The current implementation adds the candidate-group evaluator path and exports evaluator training data. The learned evaluator LoRA is the next stage after the exported labels are validated.

## Why This Is Different From The Earlier Hybrid Solver

The earlier hybrid solver used one weighted scalar score. That is useful as a baseline, but it allows compensation: a strong score on one component can hide failure on another.

The delta-ranker path uses:

- hard instruction and preservation gates
- multiple candidates for the same instruction
- relative ranking among feasible candidates
- counterfactual instruction discrimination
- evaluator disagreement filtering

## Evaluator Logic

For each source image and proposed instruction, the editor generates `K` candidates.

The evaluator first applies hard gates:

- instruction gate: did the requested edit happen?
- preservation gate: did non-target content remain stable?

Candidates that fail either gate are rejected before ranking.

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

These files are the bridge from the current heuristic evaluator to a learned evaluator LoRA.

## Commands

Local smoke path:

```bash
bash scripts/self_evolve_pillow_delta_ranker.sh --limit 8
```

Qwen-backed run:

```bash
bash scripts/self_evolve_2509_delta_ranker.sh --limit 32
```

Matrix runner:

```bash
bash scripts/run_self_evolve_matrix.sh \
  --variant delta-ranker \
  --images-dir data/unlabeled/self_evolve \
  --limit 32
```

## Research Ablations

The minimum ablation set is:

- base Qwen editor, no self-evolve
- plain proxy self-training
- weighted hybrid reward
- spatial-only evaluator
- cycle-only evaluator
- internal-only evaluator
- delta-ranker without counterfactual score
- delta-ranker without relative group score
- delta-ranker with `K=1` versus `K=4`

The critical comparison is:

```text
weighted scalar reward vs hard gates + group-relative ranking
```

That comparison tests the main research claim.
