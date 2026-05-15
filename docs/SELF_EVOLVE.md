# Self-Evolving Loop

This module is the first implementation of the project’s main research idea: a proposer-editor-solver loop that generates pseudo-labeled editing data from unlabeled images and accepts only high-scoring edits into the next training pool.

## What is implemented

- round-based self-evolving loop orchestration
- difficulty shaping over proposal families
- proposal generation
- editor backends
- solver backends
- accepted-sample manifest writing for downstream LoRA training
- optional training launch after each round

The code lives under [src/qwen_edit_project/self_evolve](/Users/ritesh.thawkar/Ritesh/neurips-project/src/qwen_edit_project/self_evolve).

## Current backends

### Proposer

- `scripted`
- `internal_qwen` placeholder

### Editor

- `qwen_edit`
- `pillow_demo`

### Solver

- `stat`
- `internal_qwen`
- `hybrid`

## Important limitation

The control loop is implemented, but the public `Qwen-Image-Edit` pipeline does not expose the internal understanding branch as a standalone proposer/verifier API. Because of that:

- the current real editor is `qwen_edit`
- the current implemented proposer is `scripted`
- the current baseline solver is `stat`
- the current exploratory research solvers are `internal_qwen` and `hybrid`

This means the repo now contains the full loop infrastructure and iterative data-generation logic, but the fully closed internal proposer-solver path is still behind an adapter boundary rather than fully realized with public upstream APIs.

At the same time, the public DiffSynth stack does expose the Qwen-conditioned hidden states used for edit conditioning. Our utility layer now exposes those features through [qwen_pipeline.py](/Users/ritesh.thawkar/Ritesh/neurips-project/src/qwen_edit_project/utils/qwen_pipeline.py), which gives us a concrete path toward an internal representation-based verifier in the next phase.

## New verifier methods

The repo now includes three research-facing verifier ideas that can be toggled independently:

- `spatial` verification inside the `hybrid` solver
  - scores changed-region support and outside-region preservation separately
- `cycle` consistency inside the `hybrid` solver
  - applies an inverse edit when available and scores reconstruction back toward the source image
- `internal_qwen` feature verification
  - uses hidden states from Qwen’s public image-plus-instruction understanding path as an additional score

These are intentionally heuristic and exploratory. They are implemented so they can be ablated and tested, not because they are already validated.

## Delta-ranker path

The stronger research path is documented in [DELTA_GROUNDED_SELF_EVOLVE.md](/Users/ritesh.thawkar/Ritesh/neurips-project/docs/DELTA_GROUNDED_SELF_EVOLVE.md).

It adds:

- multiple editor candidates for the same proposal
- hard instruction and preservation gates
- relative ranking among feasible candidates
- counterfactual instruction scoring
- evaluator training data export

This path is the intended bridge from heuristic self-training to a learned evaluator LoRA.

## Round outputs

Each round writes:

```text
outputs/self_evolve/<run_name>/round_01/proposals.jsonl
outputs/self_evolve/<run_name>/round_01/train_manifest.json
outputs/self_evolve/<run_name>/round_01/accepted/images/*.png
outputs/self_evolve/<run_name>/round_01/summary.json
```

The manifest format matches the existing DiffSynth training flow:

- `prompt`: accepted instruction
- `image`: edited image
- `edit_image`: original image

## Running the loop

Prototype run without the Qwen model:

```bash
bash scripts/self_evolve_pillow_demo.sh --limit 8
```

Qwen-backed run:

```bash
bash scripts/self_evolve_2509.sh --limit 32
```

Hybrid NeurIPS-oriented run with all three verifier ideas enabled:

```bash
bash scripts/self_evolve_2509_hybrid.sh --limit 32
```

Delta-grounded ranker run:

```bash
bash scripts/self_evolve_2509_delta_ranker.sh --limit 32
```

Single-method ablations:

```bash
bash scripts/self_evolve_2509_spatial.sh --limit 32
bash scripts/self_evolve_2509_cycle.sh --limit 32
bash scripts/self_evolve_2509_internal.sh --limit 32
```

Local verification run without Qwen weights:

```bash
bash scripts/self_evolve_pillow_hybrid.sh --limit 8
```

Local delta-ranker verification without Qwen weights:

```bash
bash scripts/self_evolve_pillow_delta_ranker.sh --limit 8
```

If your GPU does not handle BF16 well:

```bash
bash scripts/self_evolve_2509.sh --set editor.model.torch_dtype=float16
```

## Optional training launch

By default, training is not launched after each round:

```yaml
training:
  trigger: emit_only
```

To launch LoRA after each round, set:

```yaml
training:
  trigger: launch
  base_train_config: configs/train/lora_2509.yaml
```

The loop will write a round-specific training command and then promote the latest produced LoRA checkpoint into the next round if one is found.
