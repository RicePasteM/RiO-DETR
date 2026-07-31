"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# for register purpose
from . import optim
from . import data
from . import rtv4

from .backbone import *

from .backbone import (
    get_activation,
    FrozenBatchNorm2d,
    freeze_batch_norm2d,
)