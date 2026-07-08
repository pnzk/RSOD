# Dataset Preparation

This project uses public remote-sensing object detection benchmarks. The raw datasets are not redistributed in this repository.

## Datasets

- DIOR
- NWPU VHR-10

Download each dataset from its official source and follow the license and citation requirements of the dataset provider.

## Expected Directory Variables

Set the dataset root through an environment variable:

```bash
export DATA_ROOT=/path/to/datasets
```

On Windows PowerShell:

```powershell
$env:DATA_ROOT="<path-to-datasets>"
```

Recommended layout:

```text
${DATA_ROOT}/
  DIOR/
    JPEGImages/
    Annotations/
    ImageSets/
  NWPU_VHR10/
    positive image set/
    negative image set/
    ground truth/
```

The exact annotation split files depend on the few-shot split protocol used by the detection framework. Keep the DIOR and NWPU split definitions consistent with the manuscript settings.

## OOC Diagnostic Data

The OOC protocol constructs three diagnostic subsets for DIOR Split 1 3-shot:

- IID-Paste: identical novel instances pasted onto semantically compatible backgrounds.
- OOC-Paste: identical novel instances pasted onto semantically incompatible backgrounds.
- Context-Only: compatible backgrounds without pasted target instances.

The generated diagnostic annotations can be reproduced from the public datasets and the RSOD/OOC generation scripts. The generated images are not redistributed here unless dataset licenses allow redistribution.
