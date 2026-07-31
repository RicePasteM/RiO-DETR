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
- [x] **\[2025.07.31\]** We have released a clean implementation of the core **RT-DETR-OBB** framework, along with a selection of pretrained weights. More pretrained weights will be uploaded in the next few days.

- [x] **\[2025.06.20\]** 🎉 RiO-DETR has been accepted to ECCV 2026!

- [x] **\[2025.03.10\]** Release paper on [arxiv](https://arxiv.org/abs/2603.09411).


## Model Zoo

### DIOR-R

| Model | mAP | Params | Config | Log | Checkpoint |
| --- | ---: | ---: | --- | --- | --- |
| RT-DETRv2-OBB-S | 73.75 | 8.15M | [yml](configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_s_diorr.yml) | [log](pretrained_ckpts/diorr/logs/rtdetrv2_obb_hgnetv2_s_diorr.jsonl) | [ckpt](https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main/diorr/rtdetrv2_obb_hgnetv2_s_diorr.pth) |
| RT-DETRv2-OBB-M | 75.61 | 19.07M | [yml](configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_m_diorr.yml) | [log](pretrained_ckpts/diorr/logs/rtdetrv2_obb_hgnetv2_m_diorr.jsonl) | [ckpt](https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main/diorr/rtdetrv2_obb_hgnetv2_m_diorr.pth) |
| RT-DETRv2-OBB-L | 75.69 | 27.96M | [yml](configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_l_diorr.yml) | [log](pretrained_ckpts/diorr/logs/rtdetrv2_obb_hgnetv2_l_diorr.jsonl) | [ckpt](https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main/diorr/rtdetrv2_obb_hgnetv2_l_diorr.pth) |
| RT-DETRv2-OBB-X | 76.52 | 63.59M | [yml](configs/rtdetrv2_obb/rtdetrv2_obb_hgnetv2_x_diorr.yml) | [log](pretrained_ckpts/diorr/logs/rtdetrv2_obb_hgnetv2_x_diorr.jsonl) | [ckpt](https://huggingface.co/RicePasteM/RT-DETR-OBB/resolve/main/diorr/rtdetrv2_obb_hgnetv2_x_diorr.pth) |


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

## Model Zoo




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
