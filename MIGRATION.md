# Run the source-path experiment on the dedicated machine

Branch: `migration/source-path-v1` in `riteshthawkar/self-evolve-image-editing`.
The current runnable experiment is self-contained in
[`revisable-image-editing/experiments/source_path_diagnostic_v1/`](revisable-image-editing/experiments/source_path_diagnostic_v1/README.md).
It requires no installation of the repository's historical package or shared-folder symlinks.

## Get the code

```bash
git clone --single-branch --branch migration/source-path-v1 https://github.com/riteshthawkar/self-evolve-image-editing.git
cd self-evolve-image-editing/revisable-image-editing/experiments/source_path_diagnostic_v1
```

For an existing checkout with local work, preserve that work before switching branches.
Then fetch and create the migration branch:

```bash
git fetch origin migration/source-path-v1
git switch --track origin/migration/source-path-v1
cd revisable-image-editing/experiments/source_path_diagnostic_v1
```

## Install on the new machine

```bash
conda create -n source-path python=3.11 -y
conda activate source-path
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py check
```

One dedicated BF16 GPU with at least 44 GiB device capacity is required. CPU
offloading needs at least 64 GiB system RAM; 96–128 GiB is recommended. Reserve
24 GiB free for run outputs, in addition to the approximately 57.7 GB checkpoint.
The reference CUDA build requires a compatible NVIDIA driver. The target-machine
canary checks actual hardware and runtime compatibility before full generation.

## Bring the model and images

Git contains the implementation, simple configuration, protocol, tests and hash
metadata. Copy model weights and raw photographs separately, or download the
exact model revision on the new machine. Replace the example paths with writable
paths on your machine:

```bash
hf download Qwen/Qwen-Image-Edit-2509 --revision d3968ef930e841f4c73640fb8afa3b306a78167e --local-dir /models/qwen-edit-2509
```

When copying the existing Hugging Face cache, preserve snapshot links together
with their blobs, or materialize a snapshot using `cp -aL`.

Provide a folder with at least 2,080 eligible, nonduplicate raw photographs after
the declared source exclusions. Use a larger pool. Earlier training photographs
and the protected benchmark sources are excluded automatically. Captions and
annotations are not needed.

## Run in order

Run from the experiment directory with the environment activated. These commands
save all results in the research workspace's ignored `outputs/` directory:

```bash
bash prepare.sh --images /data/unlabeled-images --model-dir /models/qwen-edit-2509 --out ../../outputs/source-path-v1
bash canary.sh --model-dir /models/qwen-edit-2509 --out ../../outputs/source-path-v1
bash generate.sh --model-dir /models/qwen-edit-2509 --out ../../outputs/source-path-v1
bash score.sh --out ../../outputs/source-path-v1
```

Preparation and scoring use zero GPUs. Generation uses physical GPU 0 by default;
set `SOURCE_PATH_GPU` to select another dedicated device. Set `OPENAI_API_KEY`
privately for the final scoring command. External judging is evaluation only;
acquisition, native verification and candidate selection operate offline using
the editor's bundled understanding model.

The canary must pass before generation and writes a measured timing forecast.
To resume interrupted generation, repeat its command with `--resume`; completed
candidate images are authenticated and reused. Keep the same code, configuration,
model and runtime when resuming. Local outputs and credentials remain excluded
from Git.

The study covers 256 cases across eight edit families, 2,560 candidate images and
4,104 evaluation requests. Final results are in
`revisable-image-editing/outputs/source-path-v1/summary.json` and `RESULTS.txt`.
The [protocol](revisable-image-editing/experiments/source_path_diagnostic_v1/PROTOCOL.md)
fixes matched controls, source exclusions and continuation criteria. This
diagnostic performs no training and does not automatically launch another study.

## Verification

The standalone experiment passed 32 CPU unit tests, static Python/shell checks
and extraction tests before migration. Checkpoint weights and the target GPU
canary must be verified on the new machine. Historical experiment files already
in this repository remain separate from this new diagnostic.
