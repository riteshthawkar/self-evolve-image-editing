# Slurm Jobs

These scripts are templates for a single 128GB GPU node. Override partition/account/module loading according to the cluster.

Recommended order:

```bash
sbatch scripts/slurm/01_prepare_data.sbatch
sbatch scripts/slurm/02_self_evolve_delta_ranker.sbatch
sbatch --export=ALL,TRAIN_MANIFEST=outputs/self_evolve/<run>/delta-results/round_02/train_manifest.json scripts/slurm/03_train_lora_from_manifest.sbatch
sbatch --export=ALL,CHECKPOINT=outputs/checkpoints/<run>/<ckpt>.safetensors,MODEL_NAME=<run> scripts/slurm/04_eval_edit_suite.sbatch
```

For the first remote pass, keep the budgets conservative:

```bash
sbatch --export=ALL,DOWNLOAD_LIMIT=5000,FILTER_LIMIT=5000,MAX_SELECTED=1000,MAIN_COUNT=512 scripts/slurm/01_prepare_data.sbatch
sbatch --export=ALL,LIMIT=128,SAMPLES_PER_PROPOSAL=4,MAX_RECORDS_PER_ROUND=128 scripts/slurm/02_self_evolve_delta_ranker.sbatch
```

Once the pilot path works, increase `DOWNLOAD_LIMIT`, `MAX_SELECTED`, `LIMIT`, and `SAMPLES_PER_PROPOSAL`.
