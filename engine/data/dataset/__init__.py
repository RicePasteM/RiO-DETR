"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# from ._dataset import DetDataset
from .coco_dataset import CocoDetection
from .coco_dataset import (
    mscoco_category2name,
    mscoco_category2label,
    mscoco_label2category,
)
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
from .voc_detection import VOCDetection
from .voc_eval import VOCEvaluator
from .dior_dataset import DiorDataset
from .dota_dataset import DOTADataset
from .dota_evaluator import DotaEvaluator
from .dota_auto_evaluator import DotaAutoEvaluator, DotaAutoEvaluatorOffline
from .hrsc_dataset import HRSC2016Dataset
from .hrsc_evaluator import HRSCEvaluator
from .fair1m_dataset import FAIR1MDataset
