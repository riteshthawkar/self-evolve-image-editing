# Baseline Codebase Setup Plan for Qwen-Image-Edit

This document defines the baseline codebase setup for the NeurIPS project. It is intentionally limited to the reproducible baseline pipeline:

- dataset manifesting
- LoRA training
- optional full finetuning
- smoke validation
- benchmark export
- public benchmark scoring

It does not include the self-evolving / RL / proposer-editor-solver loop yet. That should be a separate phase after the baseline is stable.

## 1. Baseline scope

The baseline codebase should answer one practical question first:

"Can we reliably fine-tune `Qwen/Qwen-Image-Edit-2509`, export benchmark predictions, and score them on a remote GPU machine with minimal upstream edits and full run reproducibility?"

If the answer is not yes, do not start the self-evolving research branch.

## 2. Key decisions

These decisions should be treated as locked for the baseline:

- Main training codebase: `modelscope/DiffSynth-Studio`
- Official reference repo only: `QwenLM/Qwen-Image`
- Benchmark reference only: `PKU-YuanGroup/Edit-R1`
- GEdit scoring repo: `stepfun-ai/Step1X-Edit`
- ImgEdit scoring repo: `PKU-YuanGroup/ImgEdit`
- Default baseline model: `Qwen/Qwen-Image-Edit-2509`
- Default training mode: LoRA
- Full finetuning: optional, not the default
- DiffSynth validation: smoke test only
- Public evaluation: GEdit-Bench and ImgEdit only

## 3. Remote execution model

Because all real experiments and evaluation will run on a remote GPU machine, the project should be designed around a two-plane workflow.

### 3.1 Local machine role

The local machine is only for:

- writing wrappers, configs, and docs
- reviewing logs and metrics
- small manifest dry-runs
- pushing code changes

The local machine should not be treated as a required runtime target for:

- model downloads
- benchmark downloads
- GPU inference
- training
- benchmark scoring

### 3.2 Remote machine role

The remote machine is the authoritative execution environment for:

- model download and caching
- training
- validation
- benchmark export
- benchmark scoring
- artifact storage

This means the codebase must avoid hardcoded local paths and instead use:

- repo-relative paths where possible
- environment variables for machine-specific paths
- config files for all run settings

### 3.3 Consequence for implementation

Every runnable script must work headlessly over SSH and must:

- print the exact resolved command
- write logs to disk
- fail loudly on missing paths or missing env vars
- avoid interactive prompts

## 4. Upstream facts verified during research

The plan below is based on current upstream repositories, not just the earlier draft.

### 4.1 DiffSynth-Studio has the right Qwen training entry points

Verified upstream paths:

- `examples/qwen_image/model_training/train.py`
- `examples/qwen_image/model_training/lora/Qwen-Image-Edit-2509.sh`
- `examples/qwen_image/model_training/full/Qwen-Image-Edit-2509.sh`
- `examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2509.py`
- `examples/qwen_image/model_training/validate_full/Qwen-Image-Edit-2509.py`

Important observed behavior:

- training uses `--data_file_keys "image,edit_image"`
- editing conditioning is passed via `--extra_inputs "edit_image"`
- LoRA and full training are both already wired in the upstream example commands

### 4.2 Qwen official repo is still reference-only for this baseline

Observed from the official repo:

- `Qwen-Image-Edit-2509` remains a valid public checkpoint
- newer edit checkpoints exist, including `Qwen-Image-Edit-2511`
- the official repo recommends the latest `diffusers` and recommends prompt enhancement for stability

Baseline implication:

- stay on `2509` for the baseline because the public comparison stack and reference scripts are aligned to it
- keep prompt polishing optional and off by default for benchmark runs

### 4.3 Edit-R1 is useful as a reference, but its samplers are not drop-in exporters

Observed from `Edit-R1/reproduction`:

- `sampling_qwen_gedit.py` uses `QwenImageEditPlusPipeline`
- `sampling_qwen_imgedit.py` also uses `QwenImageEditPlusPipeline`

Important mismatches:

- the GEdit sampling script filters to English only and does not emit the scorer's required `.../fullset/<task>/<language>/` directory layout
- the ImgEdit sampling script writes `key.jpg`, while the public basic scorer expects `key.png`

Baseline implication:

- do not call Edit-R1 samplers directly from the baseline pipeline
- use them only to cross-check inference defaults
- implement our own exporters

### 4.4 GEdit-Bench has strict layout expectations

Observed from `Step1X-Edit/GEdit-Bench/EVAL.md` and `run_gedit_score.py`:

- outputs must be organized as:
  - `{edited_images_dir}/{model_name}/fullset/{edit_task}/{language}/{key}.{ext}`
- the scorer loads `stepfun-ai/GEdit-Bench`
- scoring depends on `viescore`
- the script currently initializes `VIEScore(..., key_path='secret.env')`

Baseline implication:

- our wrapper must create the exact directory structure required by the scorer
- scoring setup must document the expected secret/env file handling on the remote machine

### 4.5 ImgEdit basic scoring needs a wrapper patch

Observed from `ImgEdit/Benchmark/Basic/basic_bench.py`:

- the actual current CLI argument is `--edit_json`
- the code hardcodes `api_key="api-key"` and `base_url="url"` in the OpenAI client

Observed from `basic_bench_readme.md` and the scorer code:

- the README still documents `--basic_edit`, but the script actually uses `--edit_json`
- the scorer looks for `result_img_folder/<key>.png`

Baseline implication:

- patch `basic_bench.py` minimally so it reads `OPENAI_API_KEY` and optional `OPENAI_BASE_URL`
- make our wrapper follow the real script interface, not the README
- export `.png` files for ImgEdit

## 5. Recommended repo layout

Use the current repo root directly. Do not create another nested project folder under it.

```text
neurips-project/
  third_party/
    diffsynth-studio/
    qwen-image/
    edit-r1/
    step1x-edit/
    imgedit/
    LOCKFILE.md

  configs/
    train/
      lora_2509.yaml
      full_2509.yaml
    eval/
      gedit.yaml
      imgedit.yaml
    env/
      train.env.example
      eval.env.example

  data/
    raw/
    processed/
      train/
      val/
      benchmark/
        gedit/
        imgedit/
    manifests/
      train_metadata_qwen_edit.json
      val_metadata_qwen_edit.json

  outputs/
    checkpoints/
    validation/
    benchmark_images/
      gedit/
      imgedit/
    scores/
      gedit/
      imgedit/
    logs/

  src/
    qwen_edit_project/
      __init__.py
      data/
        build_diffsynth_manifest.py
        validate_manifest.py
      train/
        launch_train.py
        launch_validate.py
      eval/
        export_gedit.py
        export_imgedit.py
        run_gedit_score.py
        run_imgedit_score.py
        summarize_scores.py
      utils/
        config.py
        image_io.py
        prompting.py
        paths.py
        run_metadata.py

  scripts/
    bootstrap.sh
    train_lora_2509.sh
    train_full_2509.sh
    validate_lora_2509.sh
    validate_full_2509.sh
    export_gedit.sh
    export_imgedit.sh
    score_gedit.sh
    score_imgedit.sh

  patches/
    imgedit_env_key.patch

  docs/
    SETUP.md
    RUNBOOK.md
    BENCHMARKS.md
    DATA_FORMAT.md
    REMOTE.md

  .gitignore
  Makefile
  README.md
```

## 6. Source control and bootstrap strategy

### 6.1 Initialize git first

The current working directory is not a git repository, so the first setup step should be:

1. `git init`
2. add a remote when ready
3. only then add submodules

Without this, the earlier submodule-based layout cannot be applied.

### 6.2 Use submodules unless blocked by workflow friction

Preferred approach:

- keep upstream repos as git submodules under `third_party/`
- pin exact commits in `third_party/LOCKFILE.md`

Fallback approach:

- use regular clones
- still record commit SHAs in the lockfile

### 6.3 Lockfile format

Use one line per upstream dependency:

```text
DiffSynth-Studio: <commit>
Qwen-Image: <commit>
Edit-R1: <commit>
Step1X-Edit: <commit>
ImgEdit: <commit>
```

### 6.4 Patch policy

Do not make ad hoc edits directly inside `third_party/`.

Allowed policy:

1. wrap upstream
2. if wrapping is impossible, patch minimally
3. store the patch under `patches/`
4. document why the patch exists
5. add patch application to `scripts/bootstrap.sh`

## 7. Remote machine prerequisites

This plan assumes the remote machine has:

- Linux
- CUDA-capable GPU(s)
- outbound internet access for model and dataset download, unless everything is pre-cached
- enough disk for model caches, benchmark data, outputs, and logs
- Python 3.10 or 3.11

The codebase should not assume:

- root access
- a GUI
- a specific home directory layout
- that Hugging Face and ModelScope caches live inside the repo

### 7.1 Remote environment variables to standardize

Document these in `docs/REMOTE.md` and surface them in `.env.example` files:

- `PROJECT_ROOT`
- `HF_HOME`
- `HUGGINGFACE_HUB_CACHE`
- `MODELSCOPE_CACHE`
- `TMPDIR`
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional)
- `GEDIT_SECRET_ENV_PATH` or equivalent scorer secret file path

Optional but useful:

- `CUDA_VISIBLE_DEVICES`
- `TOKENIZERS_PARALLELISM=false`
- `PYTORCH_CUDA_ALLOC_CONF`

### 7.2 Cache policy

Model caches should live outside the repo on the remote machine.

Reason:

- repo cleanup should not delete large model artifacts
- repeated runs should hit shared caches
- multiple branches should reuse the same checkpoint downloads

## 8. Python environment strategy

Use two required environments and keep them separate.

### 8.1 Training environment

Purpose:

- DiffSynth installation
- training launchers
- smoke validation
- benchmark export if using DiffSynth-based exporters

Base stack:

- PyTorch with the correct CUDA wheel for the remote machine
- `pip install -e third_party/diffsynth-studio`
- `accelerate`
- `transformers`
- `diffusers`
- `pillow`

Because `DiffSynth-Studio` currently declares dependencies such as `torch`, `torchvision`, `transformers`, `modelscope`, `accelerate`, `peft`, and `datasets`, the remote bootstrap script should install PyTorch first, then install DiffSynth editable.

### 8.2 Evaluation environment

Purpose:

- GEdit scoring
- ImgEdit scoring
- score summarization

Base stack:

- `pillow`
- `tqdm`
- `datasets`
- `megfile`
- `pandas`
- `numpy`
- `openai`
- `tenacity`

### 8.3 Optional third environment only if needed

If you later want local-model GEdit scoring with `--backbone qwen25vl`, treat it as optional and isolated. `Step1X-Edit` ships a large `qwen25vl_environment.yml`, which is much heavier than the GPT-backed scorer path.

Baseline recommendation:

- default to GPT-backed GEdit scoring first
- add Qwen-backed local scoring only after the basic scorer wrapper is stable

## 9. Data contract and dataset handling

### 9.1 Internal data should never be committed

The repo should store:

- manifests
- docs
- configs
- logs and small metadata

The repo should not store:

- raw training images
- benchmark tar files
- model weights

### 9.2 Manifest contract

The manifest builder should output JSON records with at least:

- `prompt`
- `image`
- `edit_image`

Baseline convention:

- `image` = target edited image
- `edit_image` = source or conditioning image(s)

Support:

- `edit_image` as a string
- `edit_image` as a list of strings

### 9.3 Path style in manifests

Write repo-relative paths when possible.

Reason:

- the same manifest should work after cloning onto the remote machine
- absolute local paths would make the manifest non-portable

### 9.4 Manifest validator requirements

The validator should:

- confirm every referenced path exists
- confirm prompts are non-empty
- confirm `image` resolves to exactly one readable image
- confirm `edit_image` resolves to one or more readable images
- print per-split counts
- preview at least five random examples in either:
  - a logged textual summary
  - a contact sheet saved under `outputs/logs/`

## 10. Training wrapper design

The training wrappers are the core baseline engineering deliverable.

### 10.1 Design principle

Do not embed long command lines in shell scripts only.

Instead:

- shell scripts should be thin entry points
- Python launchers should own command rendering
- YAML should own run configuration

### 10.2 LoRA training path

Files:

- `configs/train/lora_2509.yaml`
- `scripts/train_lora_2509.sh`
- `src/qwen_edit_project/train/launch_train.py`

Required behavior:

- load config from YAML
- render the final `accelerate launch ... train.py` command
- write the resolved command to `outputs/logs/train_lora_command.txt`
- stream stdout and stderr to the terminal
- tee stdout and stderr to a timestamped log file
- save a run metadata JSON with:
  - timestamp
  - hostname
  - git commit if available
  - config path
  - full command
  - output path

### 10.3 Full finetuning path

Files:

- `configs/train/full_2509.yaml`
- `scripts/train_full_2509.sh`

Required behavior:

- same run-recording behavior as LoRA
- explicit warning in docs that this is not the default path

### 10.4 Resume behavior

Resume must be designed into the wrapper from day one.

Minimum requirement:

- a config field for resume checkpoint path
- the launcher must append resume-related arguments only when configured
- logs must show whether the run is fresh or resumed

## 11. Smoke validation design

Files:

- `scripts/validate_lora_2509.sh`
- `scripts/validate_full_2509.sh`
- `src/qwen_edit_project/train/launch_validate.py`

Behavior:

- load one checkpoint
- run one deterministic prompt
- save one output image to `outputs/validation/`
- save a sidecar JSON containing:
  - timestamp
  - checkpoint path
  - model mode (`lora` or `full`)
  - prompt
  - seed
  - image path

The validation wrapper should be treated as a post-training gate, not benchmark evaluation.

## 12. Benchmark export design

### 12.1 General rule

Benchmark export should be a first-class codepath, not an afterthought.

The exporters should support:

- upstream base model
- LoRA checkpoint
- optional full finetuned checkpoint

Each exporter must save enough metadata to reproduce exactly which checkpoint produced which image set.

### 12.2 GEdit exporter

Files:

- `src/qwen_edit_project/eval/export_gedit.py`
- `scripts/export_gedit.sh`

Requirements:

- load the benchmark input dataset
- preserve both English and Chinese samples unless explicitly filtered
- write images to:
  - `outputs/benchmark_images/gedit/<model_name>/fullset/<task_type>/<language>/<key>.png`
- save an exporter manifest or summary JSON recording:
  - model name
  - checkpoint or base model path
  - benchmark subset
  - seed
  - inference hyperparameters
  - image count written

Important:

- do not reuse `Edit-R1/reproduction/sampling/sampling_qwen_gedit.py` unchanged because it is not layout-compatible with the scorer and currently filters to English only

### 12.3 ImgEdit exporter

Files:

- `src/qwen_edit_project/eval/export_imgedit.py`
- `scripts/export_imgedit.sh`

Requirements:

- load the ImgEdit benchmark metadata
- save edited images in a flat folder:
  - `outputs/benchmark_images/imgedit/<model_name>/<key>.png`
- emit PNG, not JPG
- save exporter metadata JSON similar to GEdit

Important:

- do not reuse `Edit-R1/reproduction/sampling/sampling_qwen_imgedit.py` unchanged because it writes `.jpg`

## 13. Benchmark scoring wrappers

### 13.1 GEdit scoring wrapper

Files:

- `src/qwen_edit_project/eval/run_gedit_score.py`
- `scripts/score_gedit.sh`

Wrapper responsibilities:

- validate that the expected nested image structure exists
- validate scorer secrets before launch
- call:
  - `run_gedit_score.py`
  - `calculate_statistics.py`
- save stdout and stderr to timestamped logs
- write a small summary JSON under `outputs/scores/gedit/`

### 13.2 ImgEdit scoring wrapper

Files:

- `src/qwen_edit_project/eval/run_imgedit_score.py`
- `scripts/score_imgedit.sh`
- `patches/imgedit_env_key.patch`

Wrapper responsibilities:

- validate that `<key>.png` files exist
- ensure `OPENAI_API_KEY` is present
- optionally pass `OPENAI_BASE_URL`
- call:
  - `basic_bench.py`
  - `step1_get_avgscore.py`
  - `step2_typescore.py`
- write summary JSON under `outputs/scores/imgedit/`

### 13.3 Summary utility

File:

- `src/qwen_edit_project/eval/summarize_scores.py`

Responsibilities:

- normalize GEdit and ImgEdit summary formats
- write one compact machine-readable summary per evaluated model
- make comparison between base / LoRA / full easy

## 14. Secrets handling

No API keys or secret files should be committed.

Required repo behavior:

- ship `.env.example` files only
- load secrets from environment or external files on the remote machine
- fail clearly if a required secret is missing

For ImgEdit:

- patch scorer to use `OPENAI_API_KEY`
- optionally use `OPENAI_BASE_URL`

For GEdit:

- make the expected secret file path configurable in our wrapper
- do not rely on an undocumented file being present in the working directory

## 15. Logging and reproducibility

Every expensive run on the remote machine must leave a paper trail.

Minimum run artifacts:

- resolved command text file
- timestamped stdout/stderr log
- copied config snapshot or config path record
- machine-readable run metadata JSON

Recommended fields for run metadata:

- timestamp
- hostname
- user
- cwd
- commit SHA if available
- dirty-worktree flag if available
- launcher path
- config path
- model identifier
- checkpoint path
- benchmark name
- seed
- output directory

## 16. Makefile targets

The baseline should expose a stable top-level interface:

- `make bootstrap`
- `make train-lora`
- `make train-full`
- `make validate-lora`
- `make validate-full`
- `make export-gedit`
- `make score-gedit`
- `make export-imgedit`
- `make score-imgedit`

Each target should call a script under `scripts/`, not embed complex shell logic directly in the Makefile.

## 17. Docs to produce

### 17.1 `README.md`

Purpose:

- explain the baseline goal
- show the repo structure
- point to setup and runbook docs

### 17.2 `docs/SETUP.md`

Purpose:

- local bootstrap
- remote bootstrap
- environment creation
- submodule initialization

### 17.3 `docs/REMOTE.md`

Purpose:

- recommended remote directory layout
- cache locations
- environment variable setup
- SSH/headless usage assumptions

### 17.4 `docs/RUNBOOK.md`

Purpose:

- exact commands for:
  - bootstrap
  - manifest build
  - manifest validation
  - LoRA train
  - full train
  - validation
  - GEdit export
  - GEdit score
  - ImgEdit export
  - ImgEdit score

### 17.5 `docs/DATA_FORMAT.md`

Purpose:

- internal dataset schema
- manifest schema
- expected image path layout

### 17.6 `docs/BENCHMARKS.md`

Purpose:

- exact GEdit and ImgEdit output layouts
- scorer quirks
- prompt polishing policy
- API-backed scoring notes

## 18. Recommended implementation order

This is the implementation order the coding work should follow.

### Phase 0: repo bootstrap

1. initialize git
2. create repo skeleton
3. add submodules
4. record SHAs in `third_party/LOCKFILE.md`
5. add `.gitignore`

Exit criterion:

- clean clone plus submodule init produces the same upstream tree

### Phase 1: remote-ready environments

1. add `scripts/bootstrap.sh`
2. add `.env.example` files
3. document remote env setup
4. verify imports for training and evaluation envs

Exit criterion:

- remote machine can create both environments without manual patching

### Phase 2: data layer

1. implement manifest builder
2. implement manifest validator
3. test on a tiny internal sample

Exit criterion:

- manifest builder creates valid JSON
- validator passes on a small real subset

### Phase 3: training layer

1. implement LoRA launcher
2. implement full launcher
3. implement validation launcher
4. smoke test on a tiny subset

Exit criterion:

- LoRA launch starts and writes logs
- at least one checkpoint is saved
- validation produces one image

### Phase 4: export layer

1. implement GEdit exporter
2. implement ImgEdit exporter
3. test on a few samples first

Exit criterion:

- GEdit images land in the scorer-compatible nested layout
- ImgEdit images land in the scorer-compatible flat PNG layout

### Phase 5: scoring layer

1. patch ImgEdit scorer for env-based API config
2. implement GEdit scoring wrapper
3. implement ImgEdit scoring wrapper
4. implement score summarizer

Exit criterion:

- both scorers run end to end on a small subset or pilot run

### Phase 6: docs and runbook closure

1. write final docs
2. verify Make targets
3. run one dry end-to-end baseline pass on the remote machine

Exit criterion:

- the remote machine can execute the baseline without undocumented manual steps

## 19. Acceptance criteria for the baseline

The baseline setup should be considered complete only when all of the following are true on the remote machine.

### A. Bootstrap

- the repo can be cloned or pulled cleanly
- submodules initialize correctly
- upstream commits are recorded in `third_party/LOCKFILE.md`

### B. Environments

- training env imports `diffsynth`
- evaluation env imports the public scorer dependencies

### C. Data

- manifest builder produces valid metadata JSON
- manifest validator confirms a real subset is readable

### D. Training

- LoRA training launches on a tiny subset
- at least one checkpoint is written
- validation produces one image and metadata sidecar

### E. Export

- GEdit export matches the public scorer layout exactly
- ImgEdit export writes `<key>.png` files in a flat folder

### F. Scoring

- GEdit scorer produces CSV outputs and aggregate stats
- ImgEdit scorer writes `result.json`, average score JSON, and type score JSON

### G. Documentation

- all required env vars are documented
- all runnable commands are in the runbook
- no step requires unpublished tribal knowledge

## 20. What should not be done yet

Do not mix these into the baseline branch:

- Edit-R1 RL or NFT training
- internal reward modeling
- proposer difficulty shaping
- curriculum generation
- self-evolving loops
- architectural experiments that require changing upstream training internals

Those belong in a separate post-baseline branch once the infrastructure above is working.

## 21. Immediate next coding task after this plan

The next implementation step should be:

1. initialize this repo as git
2. create the skeleton directories
3. add the five third-party repos
4. write `scripts/bootstrap.sh`
5. create the two env examples and the first-pass docs

Only after that should the manifest and launcher code start.

## 22. Research sources

Primary sources checked for this plan:

- DiffSynth-Studio Qwen training script:
  - https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_training/train.py
- DiffSynth-Studio LoRA example:
  - https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_training/lora/Qwen-Image-Edit-2509.sh
- DiffSynth-Studio full finetune example:
  - https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_training/full/Qwen-Image-Edit-2509.sh
- DiffSynth-Studio validation examples:
  - https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2509.py
  - https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_training/validate_full/Qwen-Image-Edit-2509.py
- DiffSynth-Studio package metadata:
  - https://github.com/modelscope/DiffSynth-Studio/blob/main/pyproject.toml
- Qwen official image repo:
  - https://github.com/QwenLM/Qwen-Image/blob/main/README.md
- Edit-R1 reproduction docs:
  - https://github.com/PKU-YuanGroup/Edit-R1/blob/main/reproduction/README.md
- Edit-R1 GEdit sampler:
  - https://github.com/PKU-YuanGroup/Edit-R1/blob/main/reproduction/sampling/sampling_qwen_gedit.py
- Edit-R1 ImgEdit sampler:
  - https://github.com/PKU-YuanGroup/Edit-R1/blob/main/reproduction/sampling/sampling_qwen_imgedit.py
- Step1X-Edit benchmark docs:
  - https://github.com/stepfun-ai/Step1X-Edit/blob/main/GEdit-Bench/EVAL.md
- Step1X-Edit GEdit scorer:
  - https://github.com/stepfun-ai/Step1X-Edit/blob/main/GEdit-Bench/run_gedit_score.py
- Step1X-Edit GEdit statistics:
  - https://github.com/stepfun-ai/Step1X-Edit/blob/main/GEdit-Bench/calculate_statistics.py
- Step1X-Edit qwen25vl env:
  - https://github.com/stepfun-ai/Step1X-Edit/blob/main/GEdit-Bench/qwen25vl_environment.yml
- ImgEdit basic scorer:
  - https://github.com/PKU-YuanGroup/ImgEdit/blob/main/Benchmark/Basic/basic_bench.py
- ImgEdit basic scorer README:
  - https://github.com/PKU-YuanGroup/ImgEdit/blob/main/Benchmark/Basic/basic_bench_readme.md
- ImgEdit score post-processing:
  - https://github.com/PKU-YuanGroup/ImgEdit/blob/main/Benchmark/Basic/step1_get_avgscore.py
  - https://github.com/PKU-YuanGroup/ImgEdit/blob/main/Benchmark/Basic/step2_typescore.py
