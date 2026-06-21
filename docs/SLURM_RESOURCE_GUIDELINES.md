# Slurm Resource Guidelines

This project runs on a shared Slurm GPU machine. Treat these constraints as part
of the project contract for any training, evaluation, benchmark export, or other
GPU-heavy experiment.

## Required User And Environment

- Work only under the `ritesh_thawkar` user.
- Use the `qedit` conda environment:

```bash
source ~/.bashrc
conda activate /share_6/users/ritesh_thawkar/condaenvs/qedit
```

- Set the project source path before running project modules:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

## No Login-Node Experiments

Do not run training, evaluation, benchmark export, or model inference directly on
the login node. Such jobs can be killed and can affect other users.

Use a Slurm allocation inside tmux before running experiments:

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=96G --time=24:00:00 --pty bash
```

Short debug allocation:

```bash
srun --partition=debug --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=02:00:00 --pty bash
```

## Tmux Sessions

- Prefer running resource allocations inside tmux.
- The user commonly uses `uug_1` for an allocated resource shell.
- If `uug_1` is busy, create a separate named tmux session for a new allocation.
- Before launching work in an existing tmux pane, verify that it is inside a
  Slurm allocation, for example with:

```bash
echo "$SLURM_JOB_ID"
squeue -u ritesh_thawkar
hostname
```

## Monitoring

Useful status checks:

```bash
squeue -u ritesh_thawkar -o '%.18i %.9P %.28j %.8T %.10M %.10l %.6D %R'
nvidia-smi
tmux ls
tmux capture-pane -pt <session_name> -S -120
```

Only use idle allocated resources that belong to `ritesh_thawkar`. Do not assume
an empty physical GPU is available unless Slurm has allocated it to this user.
