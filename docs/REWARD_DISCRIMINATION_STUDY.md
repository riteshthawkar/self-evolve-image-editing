# Reward Discrimination Study — Internal Rubric-CEPR

**Question.** Before committing experiment-machine GPU time, is the internal
rubric-CEPR reward actually able to tell a *good* edit apart from the three
classic failure modes (no-op, corrupt, wrong-edit)? In particular, does it close
the no-op hole that the earlier embedding-only CEPR reward suffered from
(543/545 no-ops accepted in the training-signal audit)?

This study answers that **offline, on real edit pairs**, using only the reward
modules of `Qwen/Qwen-Image-Edit-2509` (text-encoder = Qwen2.5-VL + VAE; the
~40 GB MMDiT transformer is skipped via `transformer=None`, so the whole study
fits in a single 24 GB GPU).

## Setup

- **Probe set:** `data/probe/anyedit_pairs` — 210 real edit pairs from AnyEdit
  (`Bin1117/AnyEdit`), 30 each across 7 edit types: object_removal,
  object_replacement, object_addition, color_change, attribute_change,
  background_change, action_change. Built with
  [scripts/build_reward_probe_set.py](../scripts/build_reward_probe_set.py).
- **Harness:** [scripts/run_reward_discrimination_study.py](../scripts/run_reward_discrimination_study.py).
  For every pair it builds four candidate images and scores them through the
  **real public reward path** (`score_group` → per-candidate gates
  `_score_candidate_row` → group VLM judge `_apply_group_judge` → ranking):
  - **good** — the dataset's edited image (should ACCEPT)
  - **noop** — a copy of the source (should REJECT)
  - **corrupt** — global Gaussian blur + noise (should REJECT)
  - **wrong** — a different pair's edited image (should REJECT)
- **Metrics:** per-class `accept_rate` (the operational signal — did the reward's
  feasibility gates accept the candidate?) and `mean reward` (the scalar), plus
  AUC(good vs each negative) from a Mann–Whitney statistic on the scalar reward.

Two reward configurations are compared:

1. **Lean / embedding-only** — rubric reward with the three production gates
   *disabled* (no object detector, no conservative-region gate, no VLM judge).
   This isolates the embedding base layer.
2. **Production** — `configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml`
   evaluator block, with all three gates *enabled*:
   - `object_detector_enabled` (IDEA-Research/grounding-dino-tiny)
   - `conservative_region_reward_enabled` (`diff_fallback_allows_gate: false`)
   - `internal_vlm_judge` (`mode: per_candidate`, `require_for_feasible: true`,
     `fail_open: false`) — reuses the already-loaded Qwen2.5-VL, no extra model.

## Key finding #1 — the embedding-only reward is gameable by no-ops

Lean config, 14-pair smoke (`outputs/analysis/reward_discrimination_smoke`):

| class | n | mean reward | accept rate |
|---|---|---|---|
| good | 14 | 0.610 | 0.071 |
| **noop** | 14 | **0.738** | 0.000 |
| corrupt | 14 | 0.377 | 0.000 |
| wrong | 14 | 0.296 | 0.000 |

AUC good vs: **noop = 0.133**, corrupt = 0.985, wrong = 1.000.

The scalar reward separates *good* from *corrupt* and *wrong* almost perfectly,
but **ranks no-ops higher than genuine edits** (AUC 0.133 ≪ 0.5). Root cause: the
rubric `required_after` signal — the embedding answer to "did the requested edit
happen?" — barely moves (good 0.693 vs noop 0.698), while preservation always
favours the no-op (1.000 vs 0.945). Since
`reward ≈ src_grounded · geomean(edit_success, preservation, validity)`, the
no-op wins. **This reproduces the audit's #1 failure on real images** and
confirms the embedding base layer alone cannot police no-ops.

## Key finding #2 — the production gates close the no-op hole

Production config, 14-pair smoke (`outputs/analysis/reward_discrimination_prod_smoke`):

| class | n | mean reward | **accept rate** |
|---|---|---|---|
| **good** | 14 | 0.638 | **0.429** |
| noop | 14 | 0.691 | **0.000** |
| corrupt | 14 | 0.471 | **0.000** |
| wrong | 14 | 0.310 | **0.000** |

AUC good vs: noop = 0.281, corrupt = 0.837, wrong = 0.995.

The scalar reward is *still* gameable (no-op mean 0.691 > good 0.638), **but the
feasibility gates make the operationally correct decision**: 100 % of no-op,
corrupt, and wrong candidates are rejected, while genuine edits are the only
class with a non-zero accept rate.

> **Lesson for the paper / training loop:** protection comes from the **gating
> decision**, not from the scalar ranking. Report and threshold on `accept_rate`
> (feasibility), not on the raw reward magnitude. The detector +
> conservative-region + per-candidate VLM judge stack is what makes the reward
> non-hackable by no-ops; the embedding scalar is only a *ranking* signal among
> already-feasible candidates.

### Caveats to verify on the full 210-pair run

- Smoke `good` accept_rate is only 0.429 (2 pairs/type, noisy). In particular
  `object_removal` and `object_replacement` showed `good` accept_rate 0.000 in
  the smoke — we must confirm the gates are not *over*-rejecting genuine
  removal/replacement edits at scale (do **not** relax removal rewards; the goal
  is to confirm calibration, not to loosen it).

## Full-scale results (210 pairs)

Production config, all 210 pairs (`outputs/analysis/reward_discrimination_full`):

| class | n | mean reward | **accept rate** |
|---|---|---|---|
| **good** | 210 | 0.647 | **0.324** |
| noop | 210 | 0.658 | **0.000** |
| corrupt | 210 | 0.491 | **0.000** |
| wrong | 210 | 0.327 | **0.000** |

AUC good vs: noop = 0.349, corrupt = 0.885, wrong = 0.978.

**The headline result holds at scale: zero false-accepts.** Across all 210 pairs
× 3 negative classes (630 negative candidates), the production reward accepted
**0** no-op, corrupt, or wrong-edit candidates. Genuine edits are the only class
that is ever feasible (accept_rate 0.324). The raw scalar still cannot rank
no-ops below good edits on its own (overall AUC good-vs-noop 0.349 < 0.5), so the
**feasibility gates — not the scalar — are what make the reward non-hackable.**

### Per-edit-type accept rate (good) and good-vs-noop AUC

| edit type | good accept | noop accept | AUC good-vs-noop |
|---|---|---|---|
| object_addition | 0.833 | 0.000 | 0.400 |
| color_change | 0.633 | 0.000 | 0.300 |
| attribute_change | 0.333 | 0.000 | 0.200 |
| background_change | 0.167 | 0.000 | 0.033 |
| object_replacement | 0.133 | 0.000 | **0.721** |
| action_change | 0.100 | 0.000 | 0.000 |
| object_removal | 0.067 | 0.000 | **0.850** |

Note the two most safety-critical types — **object_removal (AUC 0.850)** and
**object_replacement (AUC 0.721)** — are the *only* types where even the scalar
reward separates good from no-op, because their dedicated detector + forbidden
gates actively penalise "object still present". This is exactly where the old
embedding-only CEPR reward failed (543/545 no-ops accepted in the
training-signal audit); here those no-ops are rejected 30/30.

### The gates are conservative by design (precision ≫ recall)

Genuine edits are frequently rejected too — overall only 32 % of good edits pass.
Breaking down *why* good edits are rejected (per-type reject reason):

- **action_change / attribute_change:** mostly the `conservative_region` gate
  (26/30 and 17/30). These global/abstract edits have no localizable region, so
  a region-preservation gate tuned for local edits is overly strict.
- **object_removal / object_replacement:** a mix of `rubric_forbidden_gate`,
  `object_detector_contract`, and `conservative_region` — the strict
  "object must be gone" checks also reject some genuine but partial edits.
- **object_addition / color_change:** highest pass rates (0.83 / 0.63), gates
  rarely fire on these.

**Interpretation for the self-evolve loop.** The reward is deliberately biased
toward **high precision at the cost of recall**: it would rather reject a real
edit than accept a fake one. For a generate-and-filter loop — where the Proposer
produces many candidates per prompt and only a few need to survive — this is the
correct trade-off: a false-accept poisons the training signal, whereas a
false-reject merely discards one usable sample. The 0/630 false-accept rate is
the property that matters; the modest good accept rate is acceptable headroom,
not a blocker. (Do **not** relax the removal/replacement gates to raise recall —
that is precisely the failure mode the audit warned against.)

## Performance note

The expensive component is the per-candidate internal VLM judge (greedy 768-token
generation, inherently sequential — extra GPU memory does not speed a single
call). Because the judge runs with `require_for_feasible: true` it can only ever
*remove* feasibility, so running it on candidates already rejected by cheaper
gates cannot change any decision. The opt-in `internal_vlm_judge.skip_infeasible`
flag (enabled by the harness's default `--fast-judge`) skips those calls. This
was verified **decision-identical**: on the 14-pair smoke, the fast and full
paths agree on 56/56 candidate feasibilities and reject reasons, at ~2× speed.

## Gate-calibration analysis (offline, no GPU)

`scripts/calibrate_reward_gates.py` post-processes `per_pair.jsonl` to find where
recall is lost and whether any gate can be safely relaxed. Outputs land in
`outputs/analysis/reward_gate_calibration/`.

**Where genuine edits are lost (binding reject reason, n=210 good edits):**

| reason | count |
|---|---|
| ACCEPTED | 68 |
| `conservative_region` | 63 |
| `internal_vlm_judge_hard_fail` | 23 |
| `rubric_forbidden_gate` | 18 |
| `object_detector_contract` | 16 |
| other (cepr_validity / specificity / preservation / required) | 22 |

The dominant recall sink is the **`conservative_region`** gate. Investigating it
yielded two findings:

1. **`action_change` is mis-typed.** It is not a canonical edit type and has no
   alias, so `normalize_edit_type` falls back to `local_enhancement` for all
   30 action records — which *is* in `conservative_region_edit_types`. Action
   edits change the whole image (pose/body), so the gate's outside-preservation
   sub-gate fails them. **In the live loop this is a non-issue** (the curriculum
   never emits "action"); it is a probe-set artifact. But it illustrates the
   general principle below.

2. **`conservative_region` is a double-edged gate with two opposing sub-gates:**
   - *target-change* sub-gate (the target region **must** change): this is what
     catches no-ops — a no-op has zero target change. **Essential; keep it.**
   - *outside-preservation / localization* sub-gate (regions outside the target
     must **not** change): this over-rejects genuinely broad edits. For
     `attribute_change` and `color_change`, rejected good edits have a healthy
     *aggregate* `conservative_region_reward` (~0.83, well above the 0.30 floor,
     and **higher than the no-op's 0.72**) yet still fail because of this
     sub-gate.

   Because no-op rejection for these types rides on the *target-change* sub-gate,
   **narrowing `conservative_region_edit_types` would be unsafe** — it would
   release no-ops. The correct, surgical change is to relax only the
   *outside-change* sub-gate (`conservative_region_max_outside_change`,
   `_max_outside_changed_fraction`) for broad-effect edit types, leaving
   *target-change* intact.

**Safe-relaxation sizing requires more data.** The original run logged only the
aggregate `conservative_region_reward`, not the individual sub-gate scores, so
the exact safe relaxation cannot be computed from it. The harness now dumps the
**full** `component_scores` and `signals` per candidate (the `_jsonable` dumps),
so the next reward-only run (which fits this 24 GB GPU) will let
`calibrate_reward_gates.py` size the sub-gate relaxation rigorously while
preserving the 0-false-accept guarantee.

## Reproduce

```bash
# build probe set (already done -> data/probe/anyedit_pairs)
PYTHONPATH=src .venv_reward/bin/python scripts/build_reward_probe_set.py \
  --max-per-type 30 --out data/probe/anyedit_pairs

# production-gate discrimination study (all 210 pairs)
PYTHONPATH=src .venv_reward/bin/python scripts/run_reward_discrimination_study.py \
  --evaluator-config configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml \
  --out outputs/analysis/reward_discrimination_full
```

Environment: overlay venv `.venv_reward`
(`python -m venv --system-site-packages`, then diffusers 0.38.0 +
transformers 4.57.6) on top of the base CUDA torch 2.7.1+cu126.

## Reward-component ablation matrix (answers the "engineered reward" critique)

A reviewer can fairly ask whether a multi-gate reward is an *engineered* score
with no research content. The defensible claim is the opposite: **each component
is tied to a measured failure it prevents.** The ablation matrix makes that
claim falsifiable — remove one component and show the cost is either a
*false-accept* (a negative class gets accepted) or a *recall* loss (good edits
get rejected), never "no change."

All arms run on the **free 24 GB GPU** via the reward-only harness
(transformer skipped); they do **not** need the experiment machine. Each arm
knocks out exactly one component via `--set` overrides.

| Arm | Knock-out | What it tests / expected effect |
|---|---|---|
| `A0_embedding_only` | all gates off (lean default) | Gameable baseline. Embedding similarity alone → high false-accepts, AUC good-vs-noop ≈ 0.13. |
| `A1_full_production` | none (reference) | The production reward. Expect **0 false-accepts**, good recall ≈ 0.32. |
| `A2_no_rubric_forbidden` | `rubric_forbidden_threshold=0` | Removes the "old state must be gone" gate. Expect residual-object / no-op leakage on removal/replacement. |
| `A3_no_conservative` | `conservative_region_reward_enabled=false` | Expect **no-ops to leak back in** (the gate's target-change sub-gate is what catches them) → proves it is load-bearing, not decoration. |
| `A4_no_object_detector` | `object_detector_enabled=false` | Removes grounded verification on removal/replacement. Expect false-accepts or recall loss on those two types. |
| `A5_no_vlm_judge` | `internal_vlm_judge.enabled=false` | Isolates the judge's marginal contribution to both false-accept suppression and recall. |
| `A6_conservative_relax_outside` | raise `max_outside_change` / `max_outside_changed_fraction`, drop outside-preservation floors; **keep target-change** | The surgical fix. Expect **recall recovery on broad edits WITHOUT a rise in noop acceptance** — the headline that the two-sub-gate analysis predicts. |

**Metrics per arm** (already emitted by the harness `overall` block, aggregated
by `summarize_reward_ablation.py`):

- `good_accept` — recall (higher is better).
- `noop_accept` / `corrupt_accept` / `wrong_accept` — **false-accept rates; must
  stay 0.000.** Any non-zero value on a removed component is direct evidence that
  the component was preventing reward hacking.
- `AUC good-vs-{noop,corrupt,wrong}` — separation of the raw reward.

The reviewer-facing result is a single table where **A1 holds 0 false-accepts**,
each `A2…A5` either breaks that or drops recall, and **A6 lifts recall with
no-op acceptance still at 0** — i.e. every gate earns its place and the residual
recall headroom has a safe, characterized fix.

```bash
# full matrix (overnight; reward-only, no experiment GPU)
scripts/run_reward_ablation_matrix.sh

# quick subset to validate the harness (5 pairs/type)
LIMIT=5 scripts/run_reward_ablation_matrix.sh

# a single arm
ARMS="A3_no_conservative" scripts/run_reward_ablation_matrix.sh

# aggregate any finished arms into the comparison table
PYTHONPATH=src python scripts/summarize_reward_ablation.py \
  --arms-root outputs/analysis/reward_ablation
```

> End-to-end caveat: the offline matrix shows the reward *decision geometry*.
> The strongest reviewer evidence is still the **A/B training run** (embedding-CEPR
> vs structured-CEPR self-evolve loop) showing the embedding reward collapses
> (reward ↑, held-out edit quality ↓) while the structured reward does not — run
> that once the experiment machine is back.
