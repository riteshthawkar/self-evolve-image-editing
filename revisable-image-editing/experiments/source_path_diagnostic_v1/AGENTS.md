# Source-path diagnostic

This standalone directory is a new versioned experiment. Do not modify older
experiments, the parent source tree, installed shared libraries, or frozen data.

The user authorized direct execution on the NEW dedicated GPU machine. Direct
execution on the originating shared cluster remains forbidden. `assets/source_machine.json`
records that host; runtime checks refuse direct workloads there. For any shared
Slurm machine use an inspected user-owned tmux session, an owned RUNNING batch
allocation and numeric srun step. Preparation/API scoring use zero GPUs.

Use only the pinned Qwen-Image-Edit-2509 weights and their bundled understanding
component for acquisition/verification. No external teacher, judge, captioner,
detector or benchmark annotation can influence cases or candidate selection.
External scoring is evaluation only and starts after generation is complete.
No training or automatic job submission is included in this diagnostic.

CPU unit tests use synthetic tensors/images and metadata; they must not load
model weights or initialize CUDA. Target GPU checks belong in the canary.
Do not weaken canary, calibration, cohort, hash, or decision gates to complete a run.
Preserve failures and committed receipts. Never change seeds while resuming.
