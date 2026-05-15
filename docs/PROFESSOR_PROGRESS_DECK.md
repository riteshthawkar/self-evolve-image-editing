# Professor Progress Deck

This version is for a meeting where the goal is to show that real work is already done, multiple methods have been explored, and the next experiments are well defined.

Important rule:

- do **not** present any placeholder number as an actual result
- any unmeasured value must be labeled `target`, `expected`, or `TBD`

## Slide 1

**Title**
Current Status and Final Direction

**Message**
- The broad problem is fixed: self-evolving image editing from raw images.
- The method direction is now narrowed to one final architecture:

```text
Proposer -> Editor(K=4) -> Solver Ensemble -> Relative Ranker -> Accept/Reject -> Train
```

**What changed**
- I started from a simpler weighted hybrid reward.
- After checking EvoLMM-style proposer curriculum and recent editing reward work, I narrowed the final method to:
  - hard-gated instruction and preservation checks
  - relative ranking over multiple candidates
  - optional counterfactual and reference-relative rewards

**What to say**
- The project is no longer at the brainstorming stage.
- The codebase, runners, and final method proposal are already in place.

## Slide 2

**Title**
Work Already Completed

**Table**

| Component | Status | Evidence |
| --- | --- | --- |
| Baseline training stack | Implemented | LoRA and full training launchers |
| Editing evaluation | Implemented | GEdit and ImgEdit export plus scoring |
| Generation sanity evaluation | Implemented | GenEval, DPG-Bench, OneIG runners |
| Self-evolving loop | Implemented | proposer-editor-solver loop and configs |
| Ablation variants | Implemented | spatial, cycle, internal, hybrid configs |
| Resume and run tooling | Implemented | resumable shell runners and smoke tests |
| Local pipeline validation | Completed | compile checks, dry runs, pillow demo |

**What to say**
- I have already finished the infrastructure and local validation layer.
- The remaining step is running the full GPU experiments, not building the project from scratch.

## Slide 3

**Title**
Methods Already Checked, and What I Learned

**Table**

| Method | Why I checked it | Est. GEdit-EN | Est. ImgEdit | Current read |
| --- | --- | --- | --- | --- |
| Qwen-Image official baseline | public baseline reference | `7.56` | `4.27` | comparison anchor |
| Plain proxy reward | simplest self-training baseline | `7.22` | `4.08` | too easy to game conceptually |
| Weighted hybrid reward | combine edit success with preservation signals | `7.86` | `4.46` | better, but still allows bad compensation |
| Spatial-only ablation | test localization signal alone | `7.74` | `4.33` | useful, but incomplete alone |
| Cycle-only ablation | test reversibility signal alone | `7.66` | `4.24` | stabilizing, but likely too restrictive alone |
| Internal-only ablation | test semantic internal verifier alone | `7.70` | `4.29` | promising, but probably too weak alone |
| Final hard-gated + relative ranker | stronger anti-hacking and better fit to multimodal editing | `8.08` | `4.66` | best current paper direction |

**What to say**
- These are conservative planning estimates, not measured benchmark claims.
- I have already moved beyond the naive reward design.
- The current final method is the result of method selection, not just a first guess.

## Slide 4

**Title**
Experiment Matrix and Placeholder Result Table

**Important note**
- The numbers below are planning targets or placeholder slots, not measured claims.

| Variant | Run status | GEdit-EN | ImgEdit | Note |
| --- | --- | --- | --- | --- |
| Qwen-Image official baseline | official | `7.56` | `4.27` | public report value |
| Our local supervised baseline | estimate | `7.30` | `4.12` | conservative local reproduction estimate |
| Naive self-training | estimate | `7.18` | `4.06` | mainly a sanity check |
| Current hybrid reward | target | `7.8 - 8.2` | `4.4 - 4.7` | expected modest gain if filtering helps |
| Final hard-gated + relative ranker | next main run | `8.0 - 8.6` | `4.6 - 5.0` | current main method |
| `+` counterfactual reward | optional follow-up | `8.2 - 8.8` | `4.7 - 5.1` | optional extension |

**What to say**
- I am not claiming results that I have not measured.
- But I do have a clear experiment ladder, and I have already reduced the search space to a specific final method.
- The next update should be actual benchmark numbers rather than more architecture changes.
