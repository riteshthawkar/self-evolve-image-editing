# Qwen-Image-Edit Finetuning Project Plan and Codebase Setup Plan

## 1) Decision summary

Use this stack:

- **Main training codebase:** `modelscope/DiffSynth-Studio`
- **Official reference repo:** `QwenLM/Qwen-Image`
- **Benchmark bridge / reference implementation:** `PKU-YuanGroup/Edit-R1`
- **GEdit evaluation:** `stepfun-ai/Step1X-Edit` (`GEdit-Bench` subtree)
- **ImgEdit evaluation:** `PKU-YuanGroup/ImgEdit`
- **Default base model:** `Qwen/Qwen-Image-Edit-2509`

Do **not** use `QwenLM/Qwen-Image` as the main training repo. Keep it as a reference repo only.

Do **not** start from Edit-R1 RL training unless the baseline LoRA/full finetuning pipeline is already stable. Use Edit-R1 first as a sampling/evaluation reference.

Keep the project generic enough that swapping `Qwen/Qwen-Image-Edit-2509` to `Qwen/Qwen-Image-Edit-2511` later is possible.

## 2) Why this stack

- `DiffSynth-Studio` already has public Qwen image edit training entry points, including:
  - `examples/qwen_image/model_training/train.py`
  - `examples/qwen_image/model_training/lora/Qwen-Image-Edit-2509.sh`
  - `examples/qwen_image/model_training/full/Qwen-Image-Edit-2509.sh`
  - `examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2509.py`
  - `examples/qwen_image/model_training/validate_full/Qwen-Image-Edit-2509.py`
- `Edit-R1` provides the clearest public reference for evaluating **Qwen-Image-Edit-2509** on **GEdit-Bench** and **ImgEdit**.
- `Step1X-Edit` provides the public **GEdit-Bench** scoring code.
- `ImgEdit` provides the public **ImgEdit-Bench** scoring code.
- `QwenLM/Qwen-Image` is the official upstream reference for inference patterns and prompt polishing, but not the main training/eval scaffold.

## 3) Repositories to clone

Prefer **git submodules**. If submodules are inconvenient, clone normally and record commit SHAs in a lockfile.

### Required repos

```bash
git submodule add https://github.com/modelscope/DiffSynth-Studio.git third_party/diffsynth-studio
git submodule add https://github.com/QwenLM/Qwen-Image.git third_party/qwen-image
git submodule add https://github.com/PKU-YuanGroup/Edit-R1.git third_party/edit-r1
git submodule add https://github.com/stepfun-ai/Step1X-Edit.git third_party/step1x-edit
git submodule add https://github.com/PKU-YuanGroup/ImgEdit.git third_party/imgedit
```

### Record current commits

Create:

```text
third_party/LOCKFILE.md
```

and write one line per repo:

```text
DiffSynth-Studio: <commit>
Qwen-Image: <commit>
Edit-R1: <commit>
Step1X-Edit: <commit>
ImgEdit: <commit>
```

## 4) Project layout to create

```text
qwen-edit-project/
  third_party/
    diffsynth-studio/
    qwen-image/
    edit-r1/
    step1x-edit/
    imgedit/

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

  Makefile
  README.md
```

## 5) Environment strategy

Use **two Python environments** to reduce dependency conflicts.

### Training environment

```bash
python -m venv .venv-train
source .venv-train/bin/activate
pip install -U pip setuptools wheel
pip install -e third_party/diffsynth-studio
pip install accelerate transformers diffusers pillow
```

Add any CUDA/PyTorch install commands required for the target machine.

### Evaluation environment

```bash
python -m venv .venv-eval
source .venv-eval/bin/activate
pip install -U pip setuptools wheel
pip install pillow tqdm datasets megfile pandas numpy openai tenacity
```

Notes:
- `Step1X-Edit/GEdit-Bench/run_gedit_score.py` imports local `viescore` plus `datasets`, `megfile`, `PIL`, `tqdm`, etc.
- `ImgEdit/Benchmark/Basic/basic_bench.py` uses `openai`, `tqdm`, and `tenacity`.

## 6) Core engineering rule

Do **not** make large edits inside `third_party/`.

Preferred order:
1. Wrap upstream code.
2. Patch only if required.
3. If patching is required, create a small patch file under `patches/` and document why.

## 7) Data contract to implement

Implement a dataset adapter that converts your internal dataset into the format expected by the DiffSynth Qwen training scripts.

### Required behavior

- Output a metadata JSON file for training and one for validation.
- Each sample must include at least:
  - `prompt`
  - `image`
  - `edit_image`
- Support `edit_image` as either:
  - a single path
  - a list of paths for multi-image editing

### Important mapping

Use this convention:
- `image` = the **target edited image**
- `edit_image` = the **source/reference image or images used as conditioning**

### Validation checks

Create `src/qwen_edit_project/data/validate_manifest.py` that checks:
- all file paths exist
- all prompts are non-empty
- `image` resolves to one image
- `edit_image` resolves to one or more images
- images can be opened
- image counts by split are printed
- at least 5 random samples are previewed in a contact sheet or log

## 8) Training implementation plan

### Phase A — Baseline LoRA training

Create a wrapper around the upstream DiffSynth LoRA command.

#### Upstream reference

```bash
accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata_qwen_imgae_edit_multi.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-Edit-2509_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters
```

#### What to build

Create:

- `configs/train/lora_2509.yaml`
- `scripts/train_lora_2509.sh`
- `src/qwen_edit_project/train/launch_train.py`

Requirements:
- load config from YAML
- render the final `accelerate launch ... train.py` command
- save full command line into `outputs/logs/train_lora_command.txt`
- save stdout/stderr to timestamped log files
- support resume from checkpoint later

### Phase B — Full finetuning path

Create a second wrapper around the upstream full-training command.

#### Upstream reference

```bash
accelerate launch --config_file examples/qwen_image/model_training/full/accelerate_config_zero2offload.yaml examples/qwen_image/model_training/train.py \
  --dataset_base_path data/example_image_dataset \
  --dataset_metadata_path data/example_image_dataset/metadata_qwen_imgae_edit_multi.json \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 50 \
  --model_id_with_origin_paths "Qwen/Qwen-Image-Edit-2509:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Qwen-Image-Edit-2509_full" \
  --trainable_models "dit" \
  --use_gradient_checkpointing \
  --find_unused_parameters
```

Create:
- `configs/train/full_2509.yaml`
- `scripts/train_full_2509.sh`

Do **not** make this the default path. LoRA is the default path.

### Phase C — Smoke validation

Create wrappers for both upstream validation scripts.

Upstream references:
- `examples/qwen_image/model_training/validate_lora/Qwen-Image-Edit-2509.py`
- `examples/qwen_image/model_training/validate_full/Qwen-Image-Edit-2509.py`

What to build:
- `scripts/validate_lora_2509.sh`
- `scripts/validate_full_2509.sh`
- `src/qwen_edit_project/train/launch_validate.py`

Requirements:
- run the smoke validation against one checkpoint
- save generated image to `outputs/validation/`
- write a small JSON sidecar with prompt, seed, checkpoint path, and timestamp

## 9) Evaluation implementation plan

### Guiding rule

Treat DiffSynth validation as a **smoke test only**, not as benchmark evaluation.

For public benchmark evaluation:
- use `Step1X-Edit/GEdit-Bench` for GEdit-Bench
- use `ImgEdit/Benchmark/Basic` for ImgEdit
- use `Edit-R1/reproduction/README.md` as the reference for how Qwen baseline and Qwen+LoRA are sampled before scoring

### 9.1 GEdit-Bench

Create:
- `src/qwen_edit_project/eval/export_gedit.py`
- `scripts/export_gedit.sh`
- `src/qwen_edit_project/eval/run_gedit_score.py`
- `scripts/score_gedit.sh`

#### Output layout to match

The public scorer expects edited images under this pattern:

```text
<edited_images_dir>/<model_name>/fullset/<group_name>/<instruction_language>/<key>.<ext>
```

That means your exporter must write files exactly like:

```text
outputs/benchmark_images/gedit/my_model/fullset/background_change/en/12345.png
outputs/benchmark_images/gedit/my_model/fullset/text_change/cn/67890.png
```

#### Wrapper command

```bash
python third_party/step1x-edit/GEdit-Bench/run_gedit_score.py \
  --model_name my_model \
  --edited_images_dir outputs/benchmark_images/gedit \
  --save_dir outputs/scores/gedit \
  --backbone gpt4o
```

Then run:

```bash
python third_party/step1x-edit/GEdit-Bench/calculate_statistics.py \
  --model_name my_model \
  --backbone gpt4o \
  --save_path outputs/scores/gedit \
  --language all
```

#### Requirements

- support evaluating:
  - upstream baseline model
  - your LoRA checkpoint
  - optional full finetuned checkpoint
- write a summary JSON to:

```text
outputs/scores/gedit/my_model_summary.json
```

### 9.2 ImgEdit

Create:
- `src/qwen_edit_project/eval/export_imgedit.py`
- `scripts/export_imgedit.sh`
- `src/qwen_edit_project/eval/run_imgedit_score.py`
- `scripts/score_imgedit.sh`

#### Output layout to match

Save edited benchmark images in a **flat folder** with filenames keyed by benchmark ID:

```text
outputs/benchmark_images/imgedit/my_model/1082.png
outputs/benchmark_images/imgedit/my_model/1068.png
```

#### Wrapper commands

Use the current script interface, not only the README text.

```bash
python third_party/imgedit/Benchmark/Basic/basic_bench.py \
  --result_img_folder outputs/benchmark_images/imgedit/my_model \
  --edit_json <path_to_imgedit_json> \
  --origin_img_root <path_to_original_images> \
  --num_processes 4 \
  --prompts_json <path_to_prompts_json>
```

Then:

```bash
python third_party/imgedit/Benchmark/Basic/step1_get_avgscore.py \
  --result_json outputs/benchmark_images/imgedit/my_model/result.json \
  --average_score_json outputs/scores/imgedit/my_model_average_score.json
```

Then:

```bash
python third_party/imgedit/Benchmark/Basic/step2_typescore.py \
  --average_score_json outputs/scores/imgedit/my_model_average_score.json \
  --typescore_json outputs/scores/imgedit/my_model_typescore.json \
  --basic_edit <path_to_imgedit_json>
```

#### Important mismatch to handle

The public `basic_bench_readme.md` documents `--basic_edit`, but the current `basic_bench.py` uses `--edit_json`. The wrapper must follow the actual script.

#### Required patch

Patch `third_party/imgedit/Benchmark/Basic/basic_bench.py` usage so the OpenAI key and base URL are loaded from environment variables instead of hardcoded placeholders.

Suggested env vars:
- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`

Store the patch as:

```text
patches/imgedit_env_key.patch
```

## 10) Prompt handling rule

Create `src/qwen_edit_project/utils/prompting.py`.

Requirements:
- add a `--use_prompt_polish` flag for inference/export scripts
- support a no-op mode
- keep benchmark mode **off by default** for apples-to-apples reproducibility
- document in `docs/BENCHMARKS.md` that prompt polishing is available but disabled by default

## 11) Baseline matrix to evaluate

At minimum evaluate:

1. `qwen_edit_2509_base`
   - upstream `Qwen/Qwen-Image-Edit-2509`
2. `qwen_edit_2509_lora`
   - your LoRA finetuned checkpoint
3. `qwen_edit_2509_full` (optional if trained)
   - your full finetuned checkpoint

The project should be able to compare all three using the same exporters and scoring wrappers.

## 12) Deliverables the coding agent must produce

### Repo bootstrap
- all repo clones/submodules in place
- lockfile with commit SHAs
- two environments documented

### Training
- dataset manifest builder
- manifest validator
- LoRA train launcher
- full finetune launcher
- smoke validation launchers

### Evaluation
- GEdit exporter and scorer wrapper
- ImgEdit exporter and scorer wrapper
- score summary utility

### Documentation
- `README.md`
- `docs/SETUP.md`
- `docs/RUNBOOK.md`
- `docs/DATA_FORMAT.md`
- `docs/BENCHMARKS.md`

### Automation
- `Makefile` targets:
  - `bootstrap`
  - `train-lora`
  - `train-full`
  - `validate-lora`
  - `validate-full`
  - `export-gedit`
  - `score-gedit`
  - `export-imgedit`
  - `score-imgedit`

## 13) Acceptance criteria

The setup is complete only when all of the following pass:

### A. Smoke setup
- `make bootstrap` works on a clean machine
- the training environment imports DiffSynth successfully
- the evaluation environment imports GEdit and ImgEdit evaluation dependencies successfully

### B. Data
- manifest builder creates a valid JSON manifest
- manifest validator confirms all referenced files exist

### C. Training
- LoRA training launches successfully on a tiny subset
- at least one checkpoint is saved under `outputs/checkpoints/`
- smoke validation generates one output image from a checkpoint

### D. Evaluation export
- GEdit export produces the exact expected directory structure
- ImgEdit export produces key-named PNGs in a flat folder

### E. Evaluation scoring
- GEdit scorer runs on a small subset or full benchmark and writes CSV outputs
- GEdit statistics script runs and writes/prints averages
- ImgEdit scorer writes `result.json`
- ImgEdit average/type score scripts finish successfully

### F. Documentation
- every runnable command appears in `docs/RUNBOOK.md`
- every required env var appears in `.env.example` files

## 14) Optional Phase 2: RL / post-training branch

Only start this after the SFT/LoRA pipeline is stable.

Use `Edit-R1` for the optional RL/post-training branch.

Relevant upstream paths:
- `config/qwen_image_edit_nft.py`
- `scripts/train_nft_qwen_image_edit.py`
- `reward_server/reward_server.py`
- `reproduction/sampling/sampling_qwen_gedit.py`
- `reproduction/sampling/sampling_qwen_imgedit.py`

Rules for this phase:
- keep it in a separate branch or separate runner
- do not mix RL assumptions into the baseline SFT project
- reuse the same benchmark exporters and scorers

## 15) Common pitfalls the agent should guard against

1. **Using the wrong repo as the base.**
   - Main trainer must be DiffSynth-Studio, not QwenLM/Qwen-Image.

2. **Using DiffSynth validation as benchmark evaluation.**
   - It is only a smoke test.

3. **Wrong benchmark output layout.**
   - GEdit needs nested task/language folders.
   - ImgEdit needs flat key-named outputs.

4. **Hardcoded API credentials in ImgEdit.**
   - Patch to environment variables.

5. **Benchmark wrappers drifting from current upstream script args.**
   - Follow the actual current script signatures.

6. **Not logging exact upstream commits.**
   - Always record SHAs.

7. **Turning prompt polish on by default in benchmark runs.**
   - Keep it off unless explicitly requested.

## 16) Exact task order for the coding agent

1. Create repo skeleton and `third_party/` layout.
2. Clone or add submodules for the five upstream repos.
3. Record commit SHAs in `third_party/LOCKFILE.md`.
4. Create `.venv-train` and `.venv-eval` plus setup docs.
5. Implement dataset manifest builder and validator.
6. Create LoRA train wrapper from the DiffSynth upstream command.
7. Create full finetune wrapper from the DiffSynth upstream command.
8. Create smoke validation wrappers.
9. Implement GEdit exporter.
10. Implement ImgEdit exporter.
11. Implement GEdit score wrapper.
12. Implement ImgEdit score wrapper and env-var patch.
13. Add score summarizer.
14. Add `Makefile` and runbook docs.
15. Run end-to-end smoke test on a tiny sample.
16. Run at least one benchmark export and one benchmark score command.

## 17) Copy-paste instruction block for the coding agent

```text
Set up a Qwen-Image-Edit finetuning project using DiffSynth-Studio as the main training codebase and Qwen/Qwen-Image-Edit-2509 as the default base model. Keep QwenLM/Qwen-Image only as an official reference repo. Use Step1X-Edit/GEdit-Bench and PKU-YuanGroup/ImgEdit as the public benchmark evaluators. Use Edit-R1 only as a reference for Qwen baseline + LoRA sampling and as an optional future RL branch.

Constraints:
- Do not make large changes inside third_party repos.
- Prefer wrappers over edits.
- If patching is required, keep a minimal patch under patches/.
- Build two Python environments: one for training, one for evaluation.
- Create a manifest builder that outputs prompt, image, and edit_image fields for DiffSynth.
- Treat DiffSynth validate scripts as smoke tests only.
- Build exporters for GEdit and ImgEdit with the exact directory/file layouts expected by the public scorers.
- Patch ImgEdit evaluation so API keys come from environment variables.
- Keep prompt polishing optional and OFF by default in benchmark mode.
- Record exact upstream commit SHAs.

Deliverables:
- repo bootstrap
- training wrappers
- evaluation wrappers
- manifest tools
- Makefile
- runbook docs
- smoke-tested end-to-end pipeline
```
