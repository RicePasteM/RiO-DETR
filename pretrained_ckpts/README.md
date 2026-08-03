# Local Pretrained Checkpoint Cache

This directory is a local cache for downloaded or converted model weights.
Model files such as `.pth`, `.pt`, `.ckpt`, `.onnx`, and `.safetensors` are
excluded from Git and must not be committed to this repository.

Official RT-DETRv2-OBB release weights are hosted in
[RicePasteM/RT-DETR-OBB](https://huggingface.co/RicePasteM/RT-DETR-OBB).
Versioned metadata, checksums, metrics, and download URLs live under
[`model_zoo/rtdetrv2_obb`](../model_zoo/rtdetrv2_obb/).

Use the following local layout:

```text
pretrained_ckpts/
└── rtdetrv2_obb/
    ├── diorr/
    │   └── rtdetrv2_obb_hgnetv2_{n,s,m,l,x}_diorr.pth
    └── dota_1_ss/
        └── rtdetrv2_obb_hgnetv2_{n,s,m,l,x}_dota_1_ss.pth
```

The presence of a file in this local cache is not evidence that it is part of
the Git release. Verify its SHA-256 value against the corresponding manifest
before use.
