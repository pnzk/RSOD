# Method Code

This folder provides the public method-code view for the RSOD framework.

## Subfolders

- `roi_heads/`: RSF-Adapter RoI heads and LCC-compatible RoI heads.
- `bbox_heads/`: LCC and CLIP/RVLP-LCC-related bbox heads.
- `datasets_pipelines/`: RSOD/context augmentation pipeline implementation.
- `model_registry/`: supporting loss and registry files.

## Integration

The code is designed for an MMDetection/MMFewShot-style detection framework. To reproduce the experiments directly, register or copy these modules into the corresponding framework locations and adapt import paths according to the local project layout.

Some implementation filenames retain `rsf_adapter_v2` because these were the working code paths used during experiments. The paper-visible method name is **RSF-Adapter**.

