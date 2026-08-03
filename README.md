<h1 align="center">RiO-DETR: DETR for Real-time Oriented Object Detection</h1>
<p align="center">
  <a href="https://arxiv.org/abs/2603.09411">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2603.09411-red">
  </a>
</p>

<p align="center">
    RiO-DETR is a real-time DETR designed for oriented object detection, which solves foundamental issues in this area. It serves as a robust foundation for future research and applications in the field of real-time end-to-end oriented object detection.
</p>

<div align="center">
  Zhangchi Hu<sup>1</sup>,
  Yifan Zhao<sup>1</sup>,
  Yansong Peng<sup>1</sup>,
  Wenzhang Sun<sup>3</sup>,
  Xiangchen Yin<sup>1</sup>,
  Jie Chen<sup>1</sup>,
  Peixi Wu<sup>1</sup>,
  Hebei Li<sup>1†</sup>,
  Xinghao Wang<sup>2</sup>,
  Dongsheng Jiang<sup>2</sup>,
  and Xiaoyan Sun<sup>1,4</sup>
</div>

<p></p>

<div align="center">
<i>
1. University of Science and Technology of China
</i>
</div>

<div align="center">
<i>
2. Huawei Technologies Co., Ltd.
</i>
</div>

<div align="center">
<i>
3. Tsinghua University
</i>
</div>

<div align="center">
<i>
4. Institute of Artificial Intelligence, Hefei Comprehensive National Science Center
</i>
</div>

<p></p>

<p align="center">
  <b>📧 Corresponding author:</b>
  <a href="mailto:lihebei@mail.ustc.edu.cn">lihebei@mail.ustc.edu.cn</a>
</p>

<p align="center">
<strong>If you like our work, please give us a ⭐!</strong>
</p>

<img width="1820" height="470" alt="screenshot-20260327-181717" src="https://github.com/user-attachments/assets/1ab0da78-748f-4f85-81e8-f124160eedae" />


📢 A Note on the Release of RiO-DETR
---
Unfortunately, I am currently unable to release the complete research code.

I have always hoped to make the full implementation of RiO-DETR publicly available, and I began preparing the codebase for an open-source release at an early stage. However, due to **intellectual property restrictions involving one of the collaborating institutions**, I was not granted permission to release the parts of the codebase containing the specific technical improvements introduced in the paper.

This restriction does not reflect any change in my personal commitment to open research and reproducibility. I sincerely appreciate your understanding.

### What Will Be Released

I will release a clean implementation of the core framework built upon RT-DETRv2. This release will include:

- A baseline RT-DETR-OBB model;

- A code architecture and supporting infrastructure that I have independently reorganized and optimized;

- Pretrained weights, which will be uploaded progressively.

However, please note that this version will not include the paper-specific technical improvements affected by the intellectual property restrictions. As a result, there remains a performance gap between this foundational release and the full RiO-DETR implementation. I hope this foundational release can still provide the community with a useful starting point. I also warmly welcome the community to build upon this framework, independently reproduce the missing components, and share open implementations.

### Good News: RiO-DETRv2 Is on the Way

We are excited to share that **RiO-DETRv2 is already under development** and will be published soon. This new version includes reworked architectural improvements and optimizations for bounding-box localization, and it will not be subject to the intellectual property restrictions mentioned above. Please stay tuned. 🔥


🚀 Updates
---
- [x] **\[2026.07.31\]** We have released a clean implementation of the core **RT-DETR-OBB** framework, along with a selection of pretrained weights. More pretrained weights will be uploaded in the next few days.

- [x] **\[2026.06.20\]** 🎉 RiO-DETR has been accepted to ECCV 2026!

- [x] **\[2026.03.10\]** Release paper on [arxiv](https://arxiv.org/abs/2603.09411).


## Model Zoo

### DIOR-R

| Model | mAP | Params | Config | Log | Checkpoint |
| --- | ---: | ---: | --- | --- | --- |
| RT-DETRv2-OBB-N | 62.74 | 3.97M | [yml](configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_n_diorr.yml) | [metrics](training_logs/rtdetrv2_obb/diorr/metrics/rtdetrv2_obb_hgnetv2_n_diorr.jsonl) | [ckpt](https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main/diorr/rtdetrv2_obb_hgnetv2_n_diorr.pth) |
| RT-DETRv2-OBB-S | 73.75 | 8.15M | [yml](configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_diorr.yml) | [metrics](training_logs/rtdetrv2_obb/diorr/metrics/rtdetrv2_obb_hgnetv2_s_diorr.jsonl) | [ckpt](https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main/diorr/rtdetrv2_obb_hgnetv2_s_diorr.pth) |
| RT-DETRv2-OBB-M | 75.61 | 19.07M | [yml](configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_m_diorr.yml) | [metrics](training_logs/rtdetrv2_obb/diorr/metrics/rtdetrv2_obb_hgnetv2_m_diorr.jsonl) | [ckpt](https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main/diorr/rtdetrv2_obb_hgnetv2_m_diorr.pth) |
| RT-DETRv2-OBB-L | 75.69 | 27.96M | [yml](configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_l_diorr.yml) | [metrics](training_logs/rtdetrv2_obb/diorr/metrics/rtdetrv2_obb_hgnetv2_l_diorr.jsonl) | [ckpt](https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main/diorr/rtdetrv2_obb_hgnetv2_l_diorr.pth) |
| RT-DETRv2-OBB-X | 76.52 | 63.59M | [yml](configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_x_diorr.yml) | [metrics](training_logs/rtdetrv2_obb/diorr/metrics/rtdetrv2_obb_hgnetv2_x_diorr.jsonl) | [ckpt](https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main/diorr/rtdetrv2_obb_hgnetv2_x_diorr.pth) |

Release metadata: [manifest](model_zoo/rtdetrv2_obb/diorr/manifest.json) · [metrics](model_zoo/rtdetrv2_obb/diorr/metrics.csv).

### DOTA-v1.0 Single-Scale

The released checkpoints are selected by the highest completed VOC mAP
(AP50) among the odd-epoch evaluations available through 2026-08-03. AP75
and COCO mAP below are reported for the same selected epoch.

| Model | Epoch | AP50 | AP75 | COCO mAP | Params | Config | Logs | Checkpoint |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| RT-DETRv2-OBB-N | 159 | 69.82 | 35.27 | 38.08 | 3.97M | [yml](configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_n_dota_1_ss.yml) | [train](training_logs/rtdetrv2_obb/dota_1_ss/train/rtdetrv2_obb_hgnetv2_n_dota_1_ss.log) · [eval](training_logs/rtdetrv2_obb/dota_1_ss/eval/rtdetrv2_obb_hgnetv2_n_dota_1_ss_eval.html) | [ckpt](https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main/dota_1_ss/rtdetrv2_obb_hgnetv2_n_dota_1_ss.pth) |
| RT-DETRv2-OBB-S | 139 | 78.12 | 51.36 | 48.63 | 8.15M | [yml](configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_dota_1_ss.yml) | [train](training_logs/rtdetrv2_obb/dota_1_ss/train/rtdetrv2_obb_hgnetv2_s_dota_1_ss.log) · [eval](training_logs/rtdetrv2_obb/dota_1_ss/eval/rtdetrv2_obb_hgnetv2_s_dota_1_ss_eval.html) | [ckpt](https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main/dota_1_ss/rtdetrv2_obb_hgnetv2_s_dota_1_ss.pth) |
| RT-DETRv2-OBB-M | 101 | 80.12 | 54.21 | 50.57 | 19.06M | [yml](configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_m_dota_1_ss.yml) | [train](training_logs/rtdetrv2_obb/dota_1_ss/train/rtdetrv2_obb_hgnetv2_m_dota_1_ss.log) · [eval](training_logs/rtdetrv2_obb/dota_1_ss/eval/rtdetrv2_obb_hgnetv2_m_dota_1_ss_eval.html) | [ckpt](https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main/dota_1_ss/rtdetrv2_obb_hgnetv2_m_dota_1_ss.pth) |
| RT-DETRv2-OBB-L | 93 | 80.48 | 54.97 | 51.18 | 27.95M | [yml](configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_l_dota_1_ss.yml) | [train](training_logs/rtdetrv2_obb/dota_1_ss/train/rtdetrv2_obb_hgnetv2_l_dota_1_ss.log) · [eval](training_logs/rtdetrv2_obb/dota_1_ss/eval/rtdetrv2_obb_hgnetv2_l_dota_1_ss_eval.html) | [ckpt](https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main/dota_1_ss/rtdetrv2_obb_hgnetv2_l_dota_1_ss.pth) |
| RT-DETRv2-OBB-X | 65 | 80.63 | 56.73 | 51.85 | 63.57M | [yml](configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_x_dota_1_ss.yml) | [train](training_logs/rtdetrv2_obb/dota_1_ss/train/rtdetrv2_obb_hgnetv2_x_dota_1_ss.log) · [eval](training_logs/rtdetrv2_obb/dota_1_ss/eval/rtdetrv2_obb_hgnetv2_x_dota_1_ss_eval.html) | [ckpt](https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main/dota_1_ss/rtdetrv2_obb_hgnetv2_x_dota_1_ss.pth) |

Release metadata: [manifest](model_zoo/rtdetrv2_obb/dota_1_ss/manifest.json) · [metrics](model_zoo/rtdetrv2_obb/dota_1_ss/metrics.csv).

### Original RiO-DETR Training and Evaluation Logs

For reference, we provide the console logs from the original RiO-DETR
experiments. DOTA-v1.0 SS entries also include HTML evaluation reports.
Machine-specific storage paths, user names, host names, and network addresses
have been redacted from the published logs.

| Dataset | N | S | M | L | X |
| --- | --- | --- | --- | --- | --- |
| DIOR-R | [log](training_logs/original_rio_detr/diorr/rio_hgnetv2_n_diorr.log) | [log](training_logs/original_rio_detr/diorr/rio_hgnetv2_s_diorr.log) | [log](training_logs/original_rio_detr/diorr/rio_hgnetv2_m_diorr.log) | [log](training_logs/original_rio_detr/diorr/rio_hgnetv2_l_diorr.log) | [log](training_logs/original_rio_detr/diorr/rio_hgnetv2_x_diorr.log) |
| DOTA-v1.0 SS | [train](training_logs/original_rio_detr/dota_1_ss/rio_hgnetv2_n_dota_1_ss.log) · [eval](training_logs/original_rio_detr/dota_1_ss/rio_n_dota_ss.html) | [train](training_logs/original_rio_detr/dota_1_ss/rio_hgnetv2_s_dota_1_ss.log) · [eval](training_logs/original_rio_detr/dota_1_ss/rio_s_dota_ss.html) | [train](training_logs/original_rio_detr/dota_1_ss/rio_hgnetv2_m_dota_1_ss.log) · [eval](training_logs/original_rio_detr/dota_1_ss/rio_m_dota_ss.html) | [train](training_logs/original_rio_detr/dota_1_ss/rio_hgnetv2_l_dota_1_ss.log) · [eval](training_logs/original_rio_detr/dota_1_ss/rio_l_dota_ss.html) | [train](training_logs/original_rio_detr/dota_1_ss/rio_hgnetv2_x_dota_1_ss.log) · [eval](training_logs/original_rio_detr/dota_1_ss/rio_x_dota_ss.html) |
| FAIR1M-2.0 MS | — | — | [log](training_logs/original_rio_detr/fair1m_2_ms/rio_hgnetv2_m_fair1m_2_ms.log) | — | [log](training_logs/original_rio_detr/fair1m_2_ms/rio_hgnetv2_x_fair1m_2_ms.log) |


## Getting Started

### Installation

Create a Python environment, install a CUDA-compatible build of PyTorch and
torchvision, and then install the remaining dependencies:

```bash
conda create -n rio-detr python=3.10 -y
conda activate rio-detr

# Install PyTorch and torchvision for your CUDA version first.
pip install -r requirements.txt
```

The HGNetV2 ImageNet-pretrained backbone weights are downloaded automatically
on first use and cached under `pretrain/hgnetv2/`. If automatic downloading is
unavailable, follow the URL printed by the program and place the downloaded
file in that directory.

## Data Preparation


### DOTA-v1.0

Split the original DOTA images and polygon annotations into 1024 × 1024
patches using the standard DOTA patch naming convention. Both single-scale
(SS) and multi-scale (MS) configurations are provided. Arrange the processed
data as follows:

```text
DOTA-v1.0/
├── split_ss_dota/
│   ├── trainval/
│   │   ├── images/
│   │   └── annfiles/
│   └── test/
│       ├── images/
│       └── annfiles/
└── split_ms_dota/
    ├── trainval/
    │   ├── images/
    │   └── annfiles/
    └── test/
        ├── images/
        └── annfiles/
```

Each annotation is a DOTA-format text file with one object per line:

```text
x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
```

Keep an empty matching annotation file for each unlabeled test image. Update
the `/path/to/DOTA-v1.0/...` placeholders in:

- `configs/dataset/dota_1_ss_detection_offline.yml`
- `configs/dataset/dota_1_ms_detection_offline.yml`

The released training configurations use the offline evaluator. It merges
patch predictions and writes a DOTA submission archive.

For automated checkpoint submission and evaluation on DOTA, we recommend
[DOTA-Auto-Eval](https://github.com/RicePasteM/DOTA-Auto-Eval.git). This
companion repository streamlines submitting checkpoints during training.


### DIOR-R

The released DIOR-R configurations expect the original images, oriented XML
annotations, and train/validation/test split files:

```text
DIOR/
├── JPEGImages-trainval/
├── JPEGImages-test/
├── Annotations/
│   └── Oriented Bounding Boxes/
└── ImageSets/
    └── Main/
        ├── train.txt
        ├── val.txt
        └── test.txt
```

Update the `img_folder`, `ann_folder`, and `ann_file` entries in
`configs/dataset/dior_detection.yml` to match your local dataset location.
The checked-in paths use `/path/to/DIOR/...` placeholders and must be changed
before training.


### FAIR1M-2.0

Convert the FAIR1M oriented annotations to DOTA-style text files, split the
large images into 1024 × 1024 patches, and use the FAIR1M-2.0 class names
defined in `engine/data/dataset/fair1m_dataset.py`:

```text
FAIR1M-2.0/
└── fair1m_split/
    ├── train/
    │   ├── images/
    │   └── annfiles/
    └── validation/
        ├── images/
        └── annfiles/
```

Update the `/path/to/FAIR1M-2.0/...` placeholders in
`configs/dataset/fair1m_2_ms_detection.yml`. The FAIR1M configurations use
the labeled validation split for local mAP evaluation.

## Usage

### Available Configurations

| Dataset | Variants | Configuration |
| --- | --- | --- |
| DIOR-R | N, S, M, L, X | `configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_{n,s,m,l,x}_diorr.yml` |
| DOTA-v1.0 SS | N, S, M, L, X | `configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_{n,s,m,l,x}_dota_1_ss.yml` |
| DOTA-v1.0 MS | M, X | `configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_{m,x}_dota_1_ms.yml` |
| FAIR1M-2.0 MS | M, X | `configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_{m,x}_fair1m_2_ms.yml` |

### Training

To train
it on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
  --master_port=7001 \
  --nproc_per_node=1 \
  train.py \
  -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_{}.yml \
  --use-amp \
  --seed=0
```

For multi-GPU training, expose the desired devices and set
`--nproc_per_node` to the number of GPUs. The `total_batch_size` in the YAML
configuration is divided across all processes:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --master_port=7001 \
  --nproc_per_node=4 \
  train.py \
  -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_{}.yml \
  --use-amp \
  --seed=0
```

To resume an interrupted run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --master_port=7001 \
  --nproc_per_node=4 \
  train.py \
  -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_{}.yml \
  --use-amp \
  --seed=0 \
  -r outputs/rtdetrv2_obb_hgnetv2_{}/last.pth
```

Checkpoints and training logs are written to the `output_dir` declared in the
selected configuration. You can override it with `--output-dir`.

### Evaluation

Evaluate a checkpoint on the validation/test split with:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun \
  --master_port=7001 \
  --nproc_per_node=1 \
  train.py \
  -c configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_{}.yml \
  --test-only \
  -r /path/to/checkpoint.pth
```

The equivalent convenience command is:

```bash
bash test.sh /path/to/checkpoint.pth \
  configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_{}.yml
```

For DOTA-v1.0, the offline evaluator stores merged submission archives under
`<output_dir>/dota_results/`. DIOR-R and FAIR1M-2.0 report local mAP when
ground-truth annotations are available.

### TensorRT FP16 Latency Benchmark

`tools/deployment/benchmark_tensorrt_latency.py` reproduces the latency
measurement used for RiO-DETR on Tesla T4. It uses GPU-resident input/output
buffers and CUDA events, so the reported latency covers TensorRT execution
only; preprocessing, host/device copies, NMS, and host-side postprocessing are
excluded.

Install the deployment dependencies and benchmark one or more TensorRT 10
engines. Repeat `--engine` to compare multiple model variants:

```bash
pip install -r tools/benchmark/requirements.txt

CUDA_VISIBLE_DEVICES=0 python tools/deployment/benchmark_tensorrt_latency.py \
  --engine RT-DETRv2-OBB-S=/path/to/model_s_fp16.engine \
  --engine RT-DETRv2-OBB-M=/path/to/model_m_fp16.engine \
  --require-fp16-io \
  --warmup 100 \
  --runs 300 \
  --trials 5 \
  --json-out outputs/tensorrt_latency.json
```

Use `--cuda-graph` to benchmark CUDA Graph replay. The optional
`--require-fp16-io` check rejects engines whose floating-point bindings are
not FP16.

## Citation
If you use `RiO-DETR` or its methods in your work, please cite the following BibTeX entries:
<details open>
<summary> bibtex </summary>

```latex
@article{hu2026rio,
  title={RiO-DETR: DETR for Real-time Oriented Object Detection},
  author={Hu, Zhangchi and Zhao, Yifan and Peng, Yansong and Sun, Wenzhang and Yin, Xiangchen and Chen, Jie and Wu, Peixi and Li, Hebei and Wang, Xinghao and Jiang, Dongsheng and others},
  journal={arXiv preprint arXiv:2603.09411},
  year={2026}
}
```
</details>

## Acknowledgement
Our work is built upon [RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4).
Thanks to the inspirations from [RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4), [D-FINE](https://github.com/Peterande/D-FINE), [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2), and [RHINO](https://github.com/SIAnalytics/RHINO).

✨ Feel free to contribute and reach out if you have any questions! ✨
