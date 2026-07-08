# CLIP Text Embedding Metadata

Archived files:

- `clip_text_embeddings_dior_split1.json`
- `clip_text_embeddings_vhr10_split1.json`
- `clip_text_embeddings_vhr10_split2.json`
- `clip_text_embeddings_vhr10_split3.json`

## DIOR

The DIOR configs currently reference `clip_text_embeddings_dior_split1.json`.

This file is used as a full-class prompt embedding bank. Split-specific DIOR configs construct `all_classes_splitX` and reorder/select embeddings according to the class names in each split. Therefore, separate DIOR split2/split3/split4 JSON files were not used in the current experiments.

For a cleaner public release, either:

- keep one clearly named full-class DIOR embedding file, such as `clip_text_embeddings_dior_all_classes.json`, or
- generate explicit split-specific files and update config paths accordingly.

## NWPU VHR-10

NWPU configs use split-specific embedding files:

- Split 1: `clip_text_embeddings_vhr10_split1.json`
- Split 2: `clip_text_embeddings_vhr10_split2.json`
- Split 3: `clip_text_embeddings_vhr10_split3.json`

These files are included in this package.
