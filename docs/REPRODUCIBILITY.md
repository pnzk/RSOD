# Reproducibility Notes

## Environment Variables

Use environment variables instead of absolute paths:

```text
PROJECT_ROOT   root directory of the detection framework
DATA_ROOT      root directory of DIOR and NWPU VHR-10
WORK_DIR       output directory for logs and checkpoints
CLIP_ROOT      optional OpenAI CLIP source directory
CLIP_MODEL_PATH optional CLIP model name or local weight path
```

## Method Integration

The method code is provided under:

```text
src/rsod/
```

When adapting to an MMDetection/MMFewShot-style codebase, register the modules from:

```text
src/rsod/bbox_heads/
src/rsod/roi_heads/
src/rsod/datasets_pipelines/
src/rsod/model_registry/
```

The import paths may need to be adjusted depending on the local framework layout.

## Training and Evaluation

Command templates are provided in:

```text
scripts/
```

Example evaluation template:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/eval_single.ps1 `
  -ProjectRoot $env:PROJECT_ROOT `
  -Config path/to/config.py `
  -Checkpoint path/to/checkpoint.pth
```

## Reporting Metrics

For standard benchmark tables, report Novel AP50, Base AP50, and mAP following the manuscript protocol.

For OOC diagnostics, report:

- IID-Paste Novel AP50
- OOC-Paste Novel AP50
- Delta_ctx
- CRR
- Context-FP per image

