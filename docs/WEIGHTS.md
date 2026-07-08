# Checkpoints and Model Weights

Large checkpoint files are not included in this lightweight public artifact package.

## Recommended Release Plan

For a complete public release, upload the following files to a stable host such as GitHub Releases, Zenodo, OSF, or institutional storage:

- DIOR Split 1 3-shot final model.
- DIOR main-matrix model checkpoints or the exact trained checkpoint list used for the reported table.
- NWPU VHR-10 Split 1 checkpoint.
- Optional OOC/ablation checkpoints used for diagnostic tables.

For each checkpoint, provide:

```text
file name
dataset / split / shot
configuration name
training seed
SHA256 checksum
download URL
reported metric row
```

## Submission Statement

If the checkpoints are not publicly deposited at submission time, use:

```text
The trained model weights are available from the corresponding author upon reasonable request and will be archived with the public code release upon publication.
```

