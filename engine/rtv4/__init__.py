"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

from .rtv4 import RTv4

from .matcher import HungarianMatcher
from .obb_matcher import HungarianMatcherOBB
from .hybrid_encoder import HybridEncoder
from .dfine_decoder import DFINETransformer
from .rtdetrv2_decoder import RTDETRTransformerv2
from .rtdetrv2_obb_decoder import RTDETRTransformerv2OBB

from .postprocessor import PostProcessor
from .rtv4_criterion import RTv4Criterion
from .rtv4_obb_criterion import RTv4OBBCriterion

from .dinov3_teacher import DINOv3TeacherModel