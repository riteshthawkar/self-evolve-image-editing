# Source-path diagnostic v1

Test whether controlled attention to the Qwen editor's two source-image representations
improves natural edits under native-only candidate selection. This experiment performs
no training. It is the prerequisite mechanism test for the proposed general self-evolving
framework; a positive diagnostic does not establish novelty or a learned method.

The directory is standalone: copy it alone or use the transfer archive. It imports no
parent project, historical experiment, dataset annotation, or model outside the pinned editor.
`PROTOCOL.md` fixes the experiment, controls and decision rule.

## What you need on the new machine

- One dedicated GPU with at least **44 GiB usable device capacity** (the stated 48GB target).
  The canary prints the actual device name, capacity, peak memory and timing. If the card
  actually has 40GB, it stops before model loading; do not lower the memory gate blindly.
- BF16 support, at least 64 GiB system RAM; **96–128 GiB RAM recommended** for CPU offloading.
- At least 24 GiB free for the run, in addition to the approximately 57.7GB checkpoint.
- The exact checkpoint snapshot in `assets/model_contract.json`.
- At least **2,080 eligible, nonduplicate raw photographs** after protected-source exclusions.
  Bring a larger pool. The 4,096 raw photos used in our earlier training are deliberately
  excluded, together with the declared benchmark/historical source registry. No captions
  or annotations are needed. All input files are copied into the run for portability.
- An OpenAI API key only for the last, CPU-only evaluation command. Never place it in this
  directory, the archive, or source-control files.

## Environment

Python 3.11 is the reference. Restore the existing qedit environment on the new machine,
or create a new environment there. Never change the original shared qedit installation.

```bash
conda create -n source-path python=3.11 -y
conda activate source-path
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py check
```

The PyTorch reference build is 2.11.0+cu128; use a compatible NVIDIA driver. The other
versions are pinned in `requirements.txt`. The environment and implementation hashes
are captured in every GPU run. A passing target-machine canary is mandatory; CPU tests
cannot establish GPU memory sufficiency or compatibility with a newly installed runtime.

If moving the Hugging Face cache, copy **both snapshot symlinks and their blobs**, or
materialize the snapshot with `cp -aL`. Do not copy just broken snapshot symlinks.
Alternatively download the pinned revision on the new machine before running:

```bash
hf download Qwen/Qwen-Image-Edit-2509 --revision d3968ef930e841f4c73640fb8afa3b306a78167e --local-dir /models/qwen-edit-2509
```

Preparation verifies every checkpoint file against its SHA256. Acquisition and generation
disable network connections and load local files only.

## Run on the dedicated machine

Run these commands from this directory in the activated environment. Replace the three
example paths. The shell files contain no conditional launch logic.

```bash
bash prepare.sh --images /data/unlabeled-images --model-dir /models/qwen-edit-2509 --out /runs/source-path-v1
bash canary.sh --model-dir /models/qwen-edit-2509 --out /runs/source-path-v1
bash generate.sh --model-dir /models/qwen-edit-2509 --out /runs/source-path-v1
bash score.sh --out /runs/source-path-v1
```

Set `OPENAI_API_KEY` privately before `score.sh`. GPU selection defaults to device 0;
set `SOURCE_PATH_GPU=1` to select physical device 1. The worker always sees exactly one
GPU. There is no multi-GPU tensor parallelism or distributed launch.

`prepare.sh` uses CPU only. `canary.sh` checks 128 internally known controls, unchanged
baseline pixel parity and three intervention generations at deployment resolution.
It writes a measured generation-time forecast; that forecast excludes acquisition,
native verification and external judging. Review it before the full command.

`generate.sh` completes 256 natural cases in eight equal families and 2,560 candidate
images, then exits. Model components are offloaded between encoder, denoiser and VAE
phases; no quantization or accelerated inference adapter changes the baseline.

`score.sh` hides GPUs and performs 4,104 blinded pairwise judgments with four CPU/API
workers. No scoring job owns GPU memory. The fixed GPT-4.1 snapshot is currently marked
deprecated in the official documentation; the scorer checks access before paid judging
and never substitutes another model silently. Costs are modeled from documented token
rates with a $40 ledger limit, not a provider-enforced billing cap.

## Resume and inspect results

After interruption, resume with the SAME code, model, configuration and GPU/runtime:

```bash
bash canary.sh --model-dir /models/qwen-edit-2509 --out /runs/source-path-v1 --resume
bash generate.sh --model-dir /models/qwen-edit-2509 --out /runs/source-path-v1 --resume
bash score.sh --out /runs/source-path-v1
```

Completed candidate images and local observations are authenticated and reused. An
uncommitted local generation may be recomputed using its original seed. Failed canaries
and exhausted source-family quotas require a separate version; they are not retried or
silently relaxed. API requests are reserved before sending: interrupted/failed requests
block automatic retry to avoid duplicate charges. Preserve receipts for explicit review.

Results live inside the chosen run directory:

```text
plan.json                    Config, portable source cohort and code hashes
calibration.json             Known-control acceptance and false acceptance
canary.json                  Baseline parity, VRAM peaks and timing
progress.json                Last completed stage/case
cases.json                   All 256 native-proposed instructions and contracts
images/                      Every generated candidate
case_receipts/               Native checks, selected candidate and fallbacks
evaluation/receipts/          Evaluation-only external judgments
evaluation/ledger.json       Attempted-call and cost accounting
summary.json                 Macro/category scores and all decision gates
RESULTS.txt                  Readable final decision
```

To recompute the summary from completed receipts without API calls or model execution:

```bash
python run.py summarize --out /runs/source-path-v1
```

The decision is `eligible_for_separate_mechanism_confirmation` or `do_not_train`.
Neither outcome starts training. Do not describe these paired credits as official
ImgEdit/GEdit scores or absolute edit-success rates.

## Shared Slurm machines

The direct scripts intentionally refuse the originating shared host. On a shared cluster,
start and inspect a new user-owned tmux session, submit a finite batch job, activate the
environment inside it, and run the equivalent Python command in a numeric srun step:

```bash
python run.py --runtime slurm canary --model-dir /models/qwen-edit-2509 --out /runs/source-path-v1
```

Use a CPU allocation and `CUDA_VISIBLE_DEVICES=""` for preparation/scoring. Follow that
cluster's allocation policy; this package submits no jobs automatically.
