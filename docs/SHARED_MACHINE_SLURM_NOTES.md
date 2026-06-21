# Shared Machine Slurm Notes

This project is run on a shared Slurm-managed GPU machine. Heavy work must run
only from allocated Slurm resources under the `ritesh_thawkar` user.

Operational rules:

- Use the `qedit` conda environment for scripts and experiments:
  `/share_6/users/ritesh_thawkar/condaenvs/qedit`.
- Do not run training, evaluation export, or scoring jobs directly on an
  unallocated login shell. Those jobs can be killed and can affect other users.
- Prefer an existing allocated tmux session when available. The usual session is
  `uug_1`; additional sessions may be named `uug_2`, `uug_3`, etc.
- For every new experiment, first create or attach to a tmux session, then
  request a single GPU allocation from inside that session, then run the
  experiment from the allocated shell.
- If no allocated session is available, create a tmux session and request one
  GPU with `srun`. The preferred pattern is:

```bash
tmux new-session -d -s uug_N 'bash'
tmux send-keys -t uug_N 'cd /share_6/users/ritesh_thawkar/self-evolve-image-editing && srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=96G --time=1-00:00:00 --pty bash' C-m
```

If a new shell inherits stale Slurm variables, clear them before requesting a
new allocation:

```bash
for v in $(env | awk -F= '/^SLURM_/ {print $1}'); do unset "$v"; done
```

Useful local commands are also recorded in `useful_commands`.
