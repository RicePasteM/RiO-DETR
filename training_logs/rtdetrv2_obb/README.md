# RT-DETRv2-OBB Experiment Records

Experiment records are grouped first by dataset and then by artifact type:

```text
training_logs/rtdetrv2_obb/
├── diorr/
│   └── metrics/   # compact per-epoch JSONL metric histories
├── dota_1_ss/
│   ├── train/     # sanitized console training logs
│   └── eval/      # exported DOTA evaluation HTML reports
└── dota_1_ms/
    ├── train/     # sanitized console training logs
    └── eval/      # exported DOTA evaluation reports and metric tables
```

Published logs must not contain API keys, access tokens, user names, host names,
network addresses, or machine-specific storage paths. Weight files and model
release metadata belong in `pretrained_ckpts/` and `model_zoo/`, respectively.

Historical logs from the original RiO-DETR experiments remain separately
namespaced under `training_logs/original_rio_detr/`.
