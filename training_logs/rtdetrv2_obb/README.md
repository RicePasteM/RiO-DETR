# RT-DETRv2-OBB Experiment Records

Experiment records are grouped first by dataset and then by artifact type:

```text
training_logs/rtdetrv2_obb/
├── diorr/
│   └── metrics/   # compact per-epoch JSONL metric histories
├── dota_1_ss/
│   ├── train/     # sanitized console training logs
│   └── eval/      # DOTA evaluation CSV metric tables
└── dota_1_ms/
    ├── train/     # sanitized console training logs
    └── eval/      # DOTA evaluation CSV metric tables
```

Published logs must not contain API keys, access tokens, user names, host names,
network addresses, or machine-specific storage paths. Weight files and model
release metadata belong in `pretrained_ckpts/` and `model_zoo/`, respectively.

Historical logs from the original RiO-DETR experiments remain separately
namespaced under `training_logs/original_rio_detr/`.

DOTA evaluation CSV files share one schema: task metadata (`epoch`, result ID,
status, and timestamps), aggregate metrics (VOC mAP, AP50, AP75, and COCO mAP),
and the AP for each of the 15 DOTA classes. Submission attempts without a
completed evaluation are retained with empty metric fields.
