# Model Zoo Metadata

This directory contains versioned release metadata. Actual model weights are
hosted on Hugging Face and may be cached locally under `pretrained_ckpts/`.

Each dataset directory contains:

- `manifest.json`: model configuration, selection rule, source checkpoint,
  metrics, parameter count, SHA-256, local cache path, log paths, and download
  URL.
- `metrics.csv`: a compact tabular view of the released results.

All file paths stored in manifests are relative to the repository root unless
they are explicit HTTPS download URLs.

```text
model_zoo/
└── rtdetrv2_obb/
    ├── diorr/
    │   ├── manifest.json
    │   └── metrics.csv
    └── dota_1_ss/
        ├── manifest.json
        └── metrics.csv
```
