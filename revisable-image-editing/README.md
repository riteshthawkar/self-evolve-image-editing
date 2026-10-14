# Revisable image editing

The portable source-path diagnostic is in
[`experiments/source_path_diagnostic_v1/`](experiments/source_path_diagnostic_v1/README.md).
Use [the migration instructions](../MIGRATION.md) to install the environment,
provide local weights and raw images, and run it on a dedicated GPU.

The experiment uses the pinned Qwen editor and its bundled understanding model
for instruction proposals, verification and candidate selection. External
judging is evaluation only. This first diagnostic tests the mechanism before
any training; the protocol defines its controls and decision gates.

Run results belong in `outputs/`, raw photographs in an external data folder and
model weights in an external model folder. Their paths are command-line arguments.
This migrated experiment imports no historical implementation or shared-folder
symlink from the original cluster.
