# Experiment Launch Checklist — Self-Evolve Balanced CEPR v2

Last updated: 2026-06-24. Owner action items before/while launching on the
experiment machine. Tick each box. Items marked **(BLOCKER)** must be resolved
before the loop will run.

---

## 0. TL;DR — what to launch

Run **two arms** so the result answers the "engineered reward" reviewer critique:

| Arm | Config | Reward | Purpose |
|---|---|---|---|
| **B-structured** (primary) | `configs/self_evolve/qwen_edit_2509_balanced_cepr_v2.yaml` | full gated CEPR (as-is) | the method — should train stably |
| **B-embedding** (control) | same config + gates disabled | embedding-only CEPR | should reward-hack / collapse (no-op acceptance) |

Run **B-structured first**. B-embedding is the paired control that produces the
collapse comparison; run it second (or on a second allocation).

---

## 1. Code / repo readiness

- [x] Working tree clean; all reward-study + ablation work committed.
- [x] Branch `codex/local-vlm-data-audit`, 2 commits ahead of origin → **push** (see §6).
- [x] Test suite green: `PYTHONPATH=src python -m pytest tests/ -q` → 24 passed.
- [x] Offline reward de-risking complete: **0/630 false-accepts** at scale
      (`docs/REWARD_DISCRIMINATION_STUDY.md`).
- [x] Ablation matrix running on the **free 24 GB GPU** (independent of the
      experiment machine); does not block the launch.

## 2. Data dependency — **(BLOCKER)**

The active config points at a MagicBrush-derived manifest:

```text
dataset.manifest_jsonl:
  data/unlabeled/selected/magicbrush_all_images_moe/manifest_balanced_object_color_background_rounds256.jsonl
```

- [ ] **This file is gitignored (`data/unlabeled/*`) and is NOT pushed.** It must
      exist on the experiment machine before launch. Options:
  - Prepare it on the remote: `bash scripts/prepare_magicbrush.sh`
    (and/or `scripts/prepare_remote_data.sh`), then confirm the path resolves; **or**
  - Override at launch to an already-present manifest, e.g.:
    `--set dataset.manifest_jsonl=data/unlabeled/selected/<available>/manifest.jsonl`
- [ ] Confirm the manifest has enough records for the planned rounds
      (`curriculum.max_records_per_round` × `curriculum.num_rounds`).
- [ ] Source images referenced by the manifest are readable on the remote FS.

## 3. Environment / machine — **(BLOCKER if env missing)**

- [ ] Run **inside a Slurm allocation**. The launcher refuses to run on a login
      node unless `ALLOW_LOGIN_NODE=1`. Use `srun`/`sbatch` (see
      `scripts/slurm/02_self_evolve_delta_ranker.sbatch` as the template:
      1 GPU, 8 cpus, ~112 G RAM, up to 24 h).
- [ ] Use a **tmux** session so the run survives SSH disconnect.
- [ ] `qedit` conda env present at the path the launcher expects:
      `/share_6/users/ritesh_thawkar/condaenvs/qedit/bin/python`
      (override with `PYTHON=...` if the path differs on this machine).
- [ ] Base model **`Qwen/Qwen-Image-Edit-2509`** available/downloadable
      (~40 GB+). Set `HF_HOME` to a disk with space; pre-warm the cache if the
      compute node has no internet.
- [ ] GPU has enough memory for the **full editor** (transformer + VAE + text
      encoder) plus LoRA training — this is much larger than the 17 GB
      reward-only footprint used in the offline study. Confirm a 40–48 GB+ GPU.
- [ ] `.venv_reward/` is a **local, gitignored** overlay used ONLY for the
      offline reward study; the training loop does **not** use it. Do not expect
      it on the remote.

## 4. Method / config guardrails

- [ ] **Do NOT apply the A6 `conservative_region` relaxation.** Ship the
      validated gates as-is; the ablation has not yet sized a safe relaxation.
- [ ] Keep `evaluator.internal_vlm_judge.enabled: true` (ablation arm A5 shows
      removing it allows a corrupt candidate through).
- [ ] Object removal/replacement: forbidden-object absence stays **logged but not
      a hard training contract** (already set; confirmed false-negative-prone).
- [ ] Round size defaults: `curriculum.max_records_per_round: 8`,
      `candidate_generation.samples_per_proposal: 4` (32 candidates/update).
      Adjust only deliberately.
- [ ] Self-play turns on at `start_round: 2` (`opponent: previous_round`).

## 5. Launch + monitoring

Inside the Slurm allocation + tmux:

```bash
# Primary (B-structured)
tmux new-session -d -s evolve_structured \
  "bash scripts/self_evolve_2509_balanced_cepr_v2.sh 2>&1 | tee outputs/logs/evolve_structured_$(date +%Y%m%d_%H%M%S).log"

# (optional overrides appended after the script, e.g. a different manifest)
#   ... balanced_cepr_v2.sh --set dataset.manifest_jsonl=...  --set curriculum.num_rounds=4
```

- [ ] **Watch the first 2–3 rounds' accepted-pair counts** — the project's #1
      documented risk is reward collapse from too-sparse acceptance. Offline
      recall ≈ 0.33 with 4 candidates/proposal should yield healthy acceptance,
      but the early rounds are the canary.
- [ ] Monitor with: `bash scripts/monitor_self_evolve.sh <output_root>` and
      `tail -f outputs/logs/evolve_structured_*.log`.
- [ ] Confirm `preference_manifest.jsonl` is being written and the minimum real
      edit-pair gate is met before trusting editor updates.
- [ ] Check `proposal_plan.jsonl` / `no_proposal_records` for silent
      target-type starvation (template fallback is auditable there).
- [ ] (Optional) run `bash scripts/check_experiment_contract.sh` against the run
      output to validate the experiment contract.

## 6. Push to remote (do this now)

```bash
git push origin codex/local-vlm-data-audit
```

Then on the experiment machine:

```bash
git fetch origin
git checkout codex/local-vlm-data-audit
git pull
# then resolve §2 (data) and §3 (env) before launching
```

> Note: data (`data/**`), `.venv_reward/`, and `outputs/analysis/**` are
> gitignored and will NOT transfer via git. Prepare data on the remote per §2.

## 7. The control arm (B-embedding) — strongest reviewer evidence

The most decisive result is the **causal A/B**: the embedding-only reward should
collapse (reward ↑ while held-out edit quality ↓) while the structured reward
does not. Build the control by disabling the structured gates on the same config
(detector + conservative_region + rubric forbidden + VLM judge off, embedding
CEPR only) and run it as a second arm once B-structured is healthy.

- [ ] Create/launch B-embedding control after B-structured shows a healthy first
      few rounds.
- [ ] Compare held-out benchmark edit quality between the two arms — that
      comparison is the headline that converts "engineered reward" into "reward
      that demonstrably prevents collapse."
