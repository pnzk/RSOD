# RSOD: Context-Aware Remote-Sensing Few-Shot Object Detection

This repository contains the public artifact package for the paper:

**Diagnosing and Mitigating Context Overfitting in Remote-Sensing Few-Shot Object Detection**

The package is intended for review and post-acceptance release. It provides method code, reproducibility notes, compact result summaries, and dataset preparation instructions for the RSOD framework.

## Main Components

- **RSOD augmentation**: context-preserving copy-paste augmentation with mask splitting.
- **RVLP-LCC**: vision-language prototype calibration using CLIP text prototypes and EMA visual prototypes.
- **RSF-Adapter**: RoI-level remote-sensing feature adaptation.
- **OOC protocol**: paired IID-Paste, OOC-Paste, and Context-Only evaluation for context-overfitting diagnosis.

## Repository Layout

```text
src/rsod/       Method implementation files
configs/        Portable configuration notes and templates
scripts/        Train/eval command templates
results/        Compact paper-facing result tables
metadata/       Text-embedding metadata
docs/           Dataset, environment, reproducibility, and weight notes
```

## Environment

The original experiments used an MMDetection 2.x compatible stack:

```text
Python 3.8
PyTorch 1.10.1
TorchVision 0.11.2
MMCV 1.4.0
MMDetection 2.20.0
MMFewShot 0.1.0
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

For CLIP-based components, set:

```bash
export CLIP_ROOT=/path/to/CLIP
export CLIP_MODEL_PATH=ViT-B/32
```

On Windows PowerShell:

```powershell
$env:CLIP_ROOT="<path-to-CLIP>"
$env:CLIP_MODEL_PATH="ViT-B/32"
```

## Data

The experiments use the public DIOR and NWPU VHR-10 benchmarks. Dataset files are not redistributed in this repository. Please download them from their official sources and organize them following:

```text
docs/DATASET_PREPARATION.md
```

## Reproducibility

Set the following environment variables before adapting the scripts:

```text
PROJECT_ROOT   root directory of the detection codebase
DATA_ROOT      root directory containing DIOR and NWPU VHR-10
WORK_DIR       output directory for checkpoints and logs
CLIP_ROOT      optional OpenAI CLIP source directory
```

See:

```text
docs/REPRODUCIBILITY.md
docs/WEIGHTS.md
```

## Results

Compact result summaries are provided under:

```text
results/dior/
results/nwpu/
```

Paper-facing comparisons should use Novel AP50 and the OOC diagnostic metrics described in the manuscript.

## Code Availability Statement

For manuscript submission, the repository can be cited as:

```text
Code and reproducibility materials are available in the RSOD public artifact package. The public URL will be added after repository deposition on GitHub, Zenodo, or an institutional archive.
```

If a public URL is not yet assigned at submission time, use:

```text
The code, configuration templates, result summaries, and reproducibility notes are available from the corresponding author upon reasonable request and will be released in a public repository upon publication.
```

## License

This package is released for academic research and reproducibility. Please also follow the licenses of the upstream detection framework, MMDetection, MMFewShot, and OpenAI CLIP when adapting the code.
