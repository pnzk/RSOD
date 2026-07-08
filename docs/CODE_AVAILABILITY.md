# Code Availability

This folder is prepared as the public artifact package for the RSOD manuscript.

## Submission-Ready Statement

If a repository URL is ready, use:

```text
The source code, configuration templates, and reproducibility notes are publicly available at: [repository URL]. The archived release is available at: [Zenodo/OSF/institutional URL].
```

If the repository has not yet been deposited, use:

```text
The source code, configuration templates, and reproducibility notes are available from the corresponding author upon reasonable request and will be released in a public repository upon publication.
```

## Recommended Public Hosting

- GitHub: source code and issue tracking.
- Zenodo: archived release with DOI.
- OSF or institutional repository: optional mirror for long-term storage.

## Files Included

- `src/rsod/`: implementation of RSOD, RVLP-LCC, and RSF-Adapter components.
- `scripts/`: train/eval command templates.
- `docs/`: environment, data preparation, checkpoints, and reproducibility notes.

## Files Not Included

- Raw DIOR and NWPU VHR-10 images.
- Large model checkpoints.
- Local training logs with machine-specific paths.
- Temporary cache files.
