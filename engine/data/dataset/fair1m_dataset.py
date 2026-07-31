import os
import torch
import numpy as np
import cv2

from .dota_dataset import DOTADataset
from ...core import register


# FAIR1M Classes (keep order consistent with tools/submit_fair1m.py)
FAIR1M_CLASSES = (
    'Boeing737', 'Boeing777', 'Boeing747', 'Boeing787', 'A321',
    'A220', 'A330', 'A350', 'C919', 'ARJ21', 'other-airplane',
    'Passenger_Ship', 'Motorboat', 'Fishing_Boat', 'Tugboat', 'Engineering_Ship',
    'Liquid_Cargo_Ship', 'Dry_Cargo_Ship', 'Warship', 'other-ship', 'Small_Car', 'Bus', 'Cargo_Truck',
    'Dump_Truck', 'Van', 'Trailer', 'Tractor', 'Truck_Tractor', 'Excavator', 'other-vehicle',
    'Baseball_Field', 'Basketball_Court', 'Football_Field', 'Tennis_Court', 'Roundabout', 'Intersection', 'Bridge'
)


@register()
class FAIR1MDataset(DOTADataset):
    """
    FAIR1M OBB Dataset.

    Reuses DOTADataset pipeline (txt format, OBB conversion, filtering),
    but with FAIR1M-specific category names and 37 classes.
    """

    CLASSES = FAIR1M_CLASSES

    def __init__(self, img_folder, ann_folder, transforms=None, return_masks=False):
        super().__init__(img_folder, ann_folder, transforms=transforms, return_masks=return_masks)
