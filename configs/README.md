# Configuration Templates

This folder is reserved for portable configuration templates.

The original training configs contained machine-specific paths and were not included in this cleaned public artifact package. Public configs should use environment variables instead of absolute paths:

```text
PROJECT_ROOT
DATA_ROOT
WORK_DIR
CLIP_ROOT
CLIP_MODEL_PATH
```

Before public deposition, add the final runnable configs for:

- DIOR Split 1 3-shot Ours-Robust.
- DIOR 16-cell main matrix.
- NWPU VHR-10 Split 1.
- OOC diagnostic evaluation.
- Generic augmentation baselines.

