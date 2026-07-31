"""
Transforms for OBB (Oriented Bounding Box)
"""

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision import tv_tensors
from torchvision.transforms.v2 import Transform
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import math

from ...core import register

def get_spatial_size(inpt: Any) -> List[int]:
    if isinstance(inpt, torch.Tensor):
        return inpt.shape[-2:]
    elif isinstance(inpt, (tv_tensors.Image, tv_tensors.Video)):
        return inpt.shape[-2:]
    elif isinstance(inpt, tv_tensors.Mask):
        return inpt.shape[-2:]
    else:
        # PIL Image
        if hasattr(inpt, 'size'):
            w, h = inpt.size
            return [h, w]
        raise TypeError(f"Got unexpected type {type(inpt)}")

def obb2poly(obbs):
    """
    Convert OBBs (cx, cy, w, h, angle_norm) to polygon (4, 2)
    angle_norm in [0, 1] -> [0, 180] degrees
    """
    if isinstance(obbs, torch.Tensor):
        obbs = obbs.cpu().numpy()

    polys = []
    for obb in obbs:
        cx, cy, w, h, angle_norm = obb
        # [0, 1] -> [0, 180]
        angle = angle_norm * 180.0
        rect = ((cx, cy), (w, h), angle)
        poly = cv2.boxPoints(rect) # (4, 2) float32
        polys.append(poly)
    return np.array(polys)

def poly2obb(polys, device):
    """
    Convert polygons (N, 4, 2) to OBBs (N, 5) (0-pi)
    """
    obbs = []
    for poly in polys:
        rect = cv2.minAreaRect(poly.astype(np.float32))
        (cx, cy), (w, h), angle = rect

        # Long Edge Definition
        if w < h:
            w, h = h, w
            angle += 90.0

        # Normalize to [0, 180)
        angle = angle % 180.0

        angle_norm = angle / 180.0
        obbs.append([cx, cy, w, h, angle_norm])

    return torch.tensor(obbs, dtype=torch.float32, device=device)

@register()
class PadToSizeOBB(T.Pad):
    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        # Assume first input is image
        h, w = get_spatial_size(flat_inputs[0])
        ph, pw = self.size[1] - h, self.size[0] - w
        self.padding = [0, 0, pw, ph]
        return dict(padding=self.padding)

    def __call__(self, *inputs: Any) -> Any:
        # We need to handle 5-dim boxes specially if they are not wrapped in BoundingBoxes
        # But here inputs might be (img, target) where target is dict

        # Flatten inputs for parameter generation
        flat_inputs = []
        if len(inputs) == 1:
            if isinstance(inputs[0], (tuple, list)):
                flat_inputs.extend(inputs[0])
            else:
                flat_inputs.append(inputs[0])
        else:
            flat_inputs.extend(inputs)

        params = self._get_params(flat_inputs)

        outputs = []
        # Process actual inputs
        actual_inputs = inputs
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            actual_inputs = inputs[0]

        for inp in actual_inputs:
            if isinstance(inp, dict) and 'boxes' in inp:
                # Handle target dict
                target = inp.copy()
                # Pad image/mask in target if any? usually target doesn't contain image
                # Just pad boxes? No, boxes coordinates don't change with padding (if padding is bottom-right)
                # But we might need to record padding for collation
                target['padding'] = torch.tensor(self.padding)
                outputs.append(target)
            elif isinstance(inp, (torch.Tensor, tv_tensors.Image, tv_tensors.Video)):
                # Pad image tensor
                outputs.append(F.pad(inp, padding=params['padding'], fill=self.fill, padding_mode=self.padding_mode))
            else:
                outputs.append(inp)

        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
             return tuple(outputs)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

@register()
class RandomIoUCropOBB(Transform):
    """
    RandomIoUCrop implementation that supports OBB (5-dim boxes).
    Simplified version that delegates to standard crop but handles OBB boxes manually.
    """
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.sampler_options = sampler_options or [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        self.trials = trials
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        # Unpack inputs
        if len(inputs) == 1:
            if isinstance(inputs[0], (tuple, list)) and len(inputs[0]) >= 1:
                img = inputs[0][0]
                target = inputs[0][1] if len(inputs[0]) > 1 else None
            else:
                img = inputs[0]
                target = None
        else:
            img = inputs[0]
            target = inputs[1]

        if target is None or 'boxes' not in target:
            return inputs if len(inputs) > 1 else inputs[0]

        # Image size
        h, w = get_spatial_size(img)

        # Try to find a valid crop
        for _ in range(self.trials):
            # 1. Sample scale and aspect ratio
            scale = self.min_scale + torch.rand(1) * (self.max_scale - self.min_scale)
            aspect_ratio = self.min_aspect_ratio + torch.rand(1) * (self.max_aspect_ratio - self.min_aspect_ratio)

            # Calculate crop size
            crop_h = int(h * scale / aspect_ratio.sqrt())
            crop_w = int(w * scale * aspect_ratio.sqrt())

            if crop_h > h or crop_w > w:
                continue

            # Random position
            top = torch.randint(0, h - crop_h + 1, (1,)).item()
            left = torch.randint(0, w - crop_w + 1, (1,)).item()

            region = (top, left, crop_h, crop_w)

            # Check IoU constraint if needed (skip for simplicity in this OBB version or approximate)
            # Standard RandomIoUCrop checks if centers of boxes are within crop

            # Crop Image
            img_cropped = F.crop(img, top, left, crop_h, crop_w)

            # Crop Boxes
            boxes = target['boxes'] # (N, 5)
            if boxes.shape[0] > 0:
                # Filter boxes whose center is in crop
                cx = boxes[:, 0]
                cy = boxes[:, 1]

                keep = (cx >= left) & (cx < left + crop_w) & (cy >= top) & (cy < top + crop_h)

                if not keep.any():
                    continue

                new_boxes = boxes[keep].clone()
                # Shift coordinates
                new_boxes[:, 0] -= left
                new_boxes[:, 1] -= top

                new_labels = target['labels'][keep]

                new_target = target.copy()
                new_target['boxes'] = new_boxes
                new_target['labels'] = new_labels

                if len(inputs) > 1:
                    # Return all original inputs, with modified img and target
                    # inputs is (img, target, arg2, arg3...)
                    # We return (img_cropped, new_target, arg2, arg3...)
                    return (img_cropped, new_target) + inputs[2:]
                elif len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
                    # inputs[0] is (img, target, arg2, arg3...)
                    return (img_cropped, new_target) + tuple(inputs[0][2:])
                return img_cropped
            else:
                # No boxes, just crop image
                if len(inputs) > 1:
                    return (img_cropped, target) + inputs[2:]
                elif len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
                    return (img_cropped, target) + tuple(inputs[0][2:])
                return img_cropped

        # If no valid crop found, return original
        return inputs if len(inputs) > 1 else inputs[0]

@register()
class RandomZoomOutOBB(T.RandomZoomOut):
    """
    RandomZoomOut for OBB.
    Pads image with random scale factor and random position.
    """
    def __init__(self, fill=0, side_range=(1.0, 4.0), p=0.5):
        super().__init__(fill, side_range, p)

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        # Unpack
        if len(inputs) == 1:
            if isinstance(inputs[0], (tuple, list)) and len(inputs[0]) >= 1:
                img = inputs[0][0]
                target = inputs[0][1] if len(inputs[0]) > 1 else None
            else:
                img = inputs[0]
                target = None
        else:
            img = inputs[0]
            target = inputs[1]

        h, w = get_spatial_size(img)
        scale = torch.empty(1).uniform_(self.side_range[0], self.side_range[1]).item()

        new_h = int(h * scale)
        new_w = int(w * scale)

        # Random position
        left = torch.randint(0, new_w - w + 1, (1,)).item()
        top = torch.randint(0, new_h - h + 1, (1,)).item()

        # Pad image
        # Create canvas
        # Since we handle tensor/PIL, use F.pad if possible or F.canvas
        # F.pad only pads sides.
        # Left padding = left
        # Top padding = top
        # Right padding = new_w - w - left
        # Bottom padding = new_h - h - top

        padding = [left, top, new_w - w - left, new_h - h - top]
        img_padded = F.pad(img, padding, fill=self.fill)

        # Update boxes
        if target is not None and 'boxes' in target:
            boxes = target['boxes'].clone()
            if boxes.shape[0] > 0:
                boxes[:, 0] += left
                boxes[:, 1] += top

            new_target = target.copy()
            new_target['boxes'] = boxes

            if len(inputs) > 1:
                return (img_padded, new_target) + inputs[2:]
            elif len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
                return (img_padded, new_target) + tuple(inputs[0][2:])
            return img_padded

        return (img_padded, target) + inputs[2:] if len(inputs) > 1 else ((img_padded, target) + tuple(inputs[0][2:]) if (len(inputs) == 1 and isinstance(inputs[0], (tuple, list)) and len(inputs[0]) > 1) else img_padded)

@register()
class RandomHorizontalFlipOBB(T.RandomHorizontalFlip):
    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) < self.p:
            # Flip
            # Unpack
            if len(inputs) == 1:
                if isinstance(inputs[0], (tuple, list)) and len(inputs[0]) >= 1:
                    img = inputs[0][0]
                    target = inputs[0][1] if len(inputs[0]) > 1 else None
                else:
                    img = inputs[0]
                    target = None
            else:
                img = inputs[0]
                target = inputs[1]

            h, w = get_spatial_size(img)

            # Flip image
            img_flipped = F.hflip(img)

            if target is not None and 'boxes' in target:
                boxes = target['boxes'].clone() # (N, 5)
                if boxes.shape[0] > 0:
                    # Robust flip using corners
                    polys = obb2poly(boxes) # (N, 4, 2)

                    # Flip x coordinate
                    polys[:, :, 0] = w - polys[:, :, 0]

                    # Recompute OBB
                    new_boxes = poly2obb(polys, device=boxes.device)

                    # Preserve gradients? No, data loading.

                else:
                    new_boxes = boxes

                new_target = target.copy()
                new_target['boxes'] = new_boxes

                if len(inputs) > 1:
                    return (img_flipped, new_target) + inputs[2:]
                elif len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
                    return (img_flipped, new_target) + tuple(inputs[0][2:])
                return img_flipped

            return (img_flipped, target) + inputs[2:] if len(inputs) > 1 else ((img_flipped, target) + tuple(inputs[0][2:]) if (len(inputs) == 1 and isinstance(inputs[0], (tuple, list)) and len(inputs[0]) > 1) else img_flipped)

        return inputs if len(inputs) > 1 else inputs[0]

@register()
class RandomFlipOBB(Transform):
    def __init__(self, prob: float = 0.5, direction: Union[str, List[str]] = 'horizontal', p: Optional[float] = None) -> None:
        super().__init__()
        if p is not None:
            prob = p
        self.prob = float(prob)
        if isinstance(direction, str):
            direction = [direction]
        self.direction = list(direction)

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.prob:
            return inputs if len(inputs) > 1 else inputs[0]

        if len(inputs) == 1:
            if isinstance(inputs[0], (tuple, list)) and len(inputs[0]) >= 1:
                img = inputs[0][0]
                target = inputs[0][1] if len(inputs[0]) > 1 else None
            else:
                img = inputs[0]
                target = None
        else:
            img = inputs[0]
            target = inputs[1]

        if len(self.direction) == 0:
            return inputs if len(inputs) > 1 else inputs[0]

        h, w = get_spatial_size(img)
        direction = self.direction[torch.randint(low=0, high=len(self.direction), size=(1,)).item()]

        if direction == 'horizontal':
            img_flipped = F.hflip(img)
        elif direction == 'vertical':
            img_flipped = F.vflip(img)
        elif direction == 'diagonal':
            img_flipped = F.vflip(F.hflip(img))
        else:
            return inputs if len(inputs) > 1 else inputs[0]

        if target is not None and 'boxes' in target:
            boxes = target['boxes'].clone()
            if boxes.shape[0] > 0:
                polys = obb2poly(boxes)
                if direction in ('horizontal', 'diagonal'):
                    polys[:, :, 0] = w - polys[:, :, 0]
                if direction in ('vertical', 'diagonal'):
                    polys[:, :, 1] = h - polys[:, :, 1]
                new_boxes = poly2obb(polys, device=boxes.device)
            else:
                new_boxes = boxes

            new_target = target.copy()
            new_target['boxes'] = new_boxes

            if len(inputs) > 1:
                return (img_flipped, new_target) + inputs[2:]
            elif len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
                return (img_flipped, new_target) + tuple(inputs[0][2:])
            return img_flipped

        return (img_flipped, target) + inputs[2:] if len(inputs) > 1 else ((img_flipped, target) + tuple(inputs[0][2:]) if (len(inputs) == 1 and isinstance(inputs[0], (tuple, list)) and len(inputs[0]) > 1) else img_flipped)

@register()
class RandomAffineOBB(Transform):
    def __init__(self, degrees=0, translate=None, scale=None, shear=None, fill=0):
        super().__init__()
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.fill = fill

    def get_params(self, img_size):
        h, w = img_size

        # Rotation
        angle = float(torch.empty(1).uniform_(self.degrees[0], self.degrees[1]).item()) if self.degrees else 0.0

        # Translation
        if self.translate:
            max_dx = float(self.translate[0] * w)
            max_dy = float(self.translate[1] * h)
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        # Scale
        if self.scale:
            scale = float(torch.empty(1).uniform_(self.scale[0], self.scale[1]).item())
        else:
            scale = 1.0

        # Shear (not implemented for simplicity as Mosaic usually doesn't use it, but keeping param)
        shear_x, shear_y = 0.0, 0.0

        return angle, translations, scale, (shear_x, shear_y)

    def __call__(self, *inputs: Any) -> Any:
        # Unpack
        if len(inputs) == 1:
            if isinstance(inputs[0], (tuple, list)) and len(inputs[0]) >= 1:
                img = inputs[0][0]
                target = inputs[0][1] if len(inputs[0]) > 1 else None
            else:
                img = inputs[0]
                target = None
        else:
            img = inputs[0]
            target = inputs[1]

        h, w = get_spatial_size(img)
        angle, translations, scale, shear = self.get_params((h, w))

        # Affine matrix
        center = (w * 0.5, h * 0.5)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += translations[0]
        matrix[1, 2] += translations[1]

        # Apply to image
        if isinstance(img, torch.Tensor):
            # Use torchvision for tensor image
            # F.affine expects angle in degrees, translate=(tx, ty), scale, shear
            # Note: F.affine might differ slightly from cv2.warpAffine
            # But let's try to stick to one method. Since we need to transform OBBs accurately,
            # and we computed matrix using cv2, maybe use cv2 for image too if it was numpy.
            # But img is tensor.
            # Let's use F.affine but be careful with parameters.
            # F.affine: angle is counter-clockwise. cv2.getRotationMatrix2D: angle is positive for counter-clockwise?
            # cv2: positive angle is counter-clockwise.
            # So angle matches.
            # F.affine center is center of image by default. Matches.
            # Translation: (tx, ty). Matches.

            img_transformed = F.affine(img, angle=angle, translate=translations, scale=scale, shear=shear, interpolation=T.InterpolationMode.BILINEAR, fill=self.fill)
        else:
            # PIL
            img_transformed = F.affine(img, angle=angle, translate=translations, scale=scale, shear=shear, interpolation=T.InterpolationMode.BILINEAR, fill=self.fill)

        # Apply to OBB
        if target is not None and 'boxes' in target:
            boxes = target['boxes'].clone()
            if boxes.shape[0] > 0:
                polys = obb2poly(boxes) # (N, 4, 2)

                # Transform polys
                # poly is (x, y). matrix is 2x3.
                # [x', y']^T = M * [x, y, 1]^T

                N = polys.shape[0]
                polys_flat = polys.reshape(-1, 2) # (N*4, 2)
                ones = np.ones((polys_flat.shape[0], 1), dtype=polys_flat.dtype)
                polys_homo = np.hstack([polys_flat, ones]) # (N*4, 3)

                polys_trans = (matrix @ polys_homo.T).T # (N*4, 2)
                polys_trans = polys_trans.reshape(N, 4, 2)
                polys_trans = np.nan_to_num(polys_trans, nan=0.0, posinf=1e7, neginf=-1e7)
                polys_trans = np.clip(polys_trans, -1e7, 1e7)

                # Check boundaries (optional, or rely on Sanitize)
                # Filter out boxes that are completely outside?
                # Or just keep them and let Sanitize handle it?
                # Usually RandomAffine might push boxes outside.

                new_boxes = poly2obb(polys_trans, device=boxes.device)
            else:
                new_boxes = boxes

            new_target = target.copy()
            new_target['boxes'] = new_boxes

            if len(inputs) > 1:
                return (img_transformed, new_target) + inputs[2:]
            elif len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
                return (img_transformed, new_target) + tuple(inputs[0][2:])
            return img_transformed

        return (img_transformed, target) + inputs[2:] if len(inputs) > 1 else ((img_transformed, target) + tuple(inputs[0][2:]) if (len(inputs) == 1 and isinstance(inputs[0], (tuple, list)) and len(inputs[0]) > 1) else img_transformed)


@register()
class ResizeOBB(T.Resize):
    def __init__(
        self,
        size,
        interpolation=T.InterpolationMode.BILINEAR,
        max_size=None,
        antialias=True,
        keep_ratio: bool = False,
        pad_to_size: bool = False,
        fill=0,
    ) -> None:
        super().__init__(size=size, interpolation=interpolation, max_size=max_size, antialias=antialias)
        self.keep_ratio = bool(keep_ratio)
        self.pad_to_size = bool(pad_to_size)
        self.fill = fill
        if isinstance(size, int):
            self.target_size = (size, size)
        else:
            self.target_size = tuple(size)

    def __call__(self, *inputs: Any) -> Any:
        # Resize image, scale boxes
        # Unpack
        if len(inputs) == 1:
            if isinstance(inputs[0], (tuple, list)) and len(inputs[0]) >= 1:
                img = inputs[0][0]
                target = inputs[0][1] if len(inputs[0]) > 1 else None
            else:
                img = inputs[0]
                target = None
        else:
            img = inputs[0]
            target = inputs[1]

        h, w = get_spatial_size(img)

        if self.keep_ratio:
            target_h, target_w = int(self.target_size[0]), int(self.target_size[1])
            scale = min(target_w / w, target_h / h)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img_resized = F.resize(img, [new_h, new_w], interpolation=self.interpolation, antialias=self.antialias)
        else:
            img_resized = super().forward(img)
            new_h, new_w = get_spatial_size(img_resized)

        if self.keep_ratio and self.pad_to_size:
            target_h, target_w = int(self.target_size[0]), int(self.target_size[1])
            pad_right = max(0, target_w - new_w)
            pad_bottom = max(0, target_h - new_h)
            if pad_right > 0 or pad_bottom > 0:
                img_resized = F.pad(img_resized, [0, 0, pad_right, pad_bottom], fill=self.fill)

        if target is not None and 'boxes' in target:
            boxes = target['boxes'].clone()
            if boxes.shape[0] > 0:
                scale_x = new_w / w
                scale_y = new_h / h

                if abs(scale_x - scale_y) < 1e-5:
                     # Uniform scaling, safe to just scale w, h, cx, cy
                     boxes[:, 0] *= scale_x
                     boxes[:, 1] *= scale_y
                     boxes[:, 2] *= scale_x
                     boxes[:, 3] *= scale_y
                else:
                    # Non-uniform scaling, use corners
                    polys = obb2poly(boxes) # (N, 4, 2)
                    polys[:, :, 0] *= scale_x
                    polys[:, :, 1] *= scale_y
                    boxes = poly2obb(polys, device=boxes.device)

            new_target = target.copy()
            new_target['boxes'] = boxes

            if len(inputs) > 1:
                return (img_resized, new_target) + inputs[2:]
            elif len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
                return (img_resized, new_target) + tuple(inputs[0][2:])
            return img_resized

        return (img_resized, target) + inputs[2:] if len(inputs) > 1 else ((img_resized, target) + tuple(inputs[0][2:]) if (len(inputs) == 1 and isinstance(inputs[0], (tuple, list)) and len(inputs[0]) > 1) else img_resized)

@register()
class SanitizeBoundingBoxesOBB(Transform):
    def __init__(self, min_size: float = 1.0) -> None:
        super().__init__()
        self.min_size = float(min_size)

    def __call__(self, *inputs: Any) -> Any:
        if len(inputs) == 1:
            if isinstance(inputs[0], (tuple, list)) and len(inputs[0]) >= 1:
                img = inputs[0][0]
                target = inputs[0][1] if len(inputs[0]) > 1 else None
            else:
                img = inputs[0]
                target = None
        else:
            img = inputs[0]
            target = inputs[1]

        if target is None or 'boxes' not in target:
            return inputs if len(inputs) > 1 else inputs[0]

        h, w = get_spatial_size(img)
        boxes = target['boxes']

        if not isinstance(boxes, torch.Tensor):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

        if boxes.numel() == 0:
            new_target = target.copy()
            new_target['boxes'] = boxes.reshape(0, 5)
            if 'labels' in new_target:
                new_target['labels'] = new_target['labels'].reshape(0)
        else:
            boxes = boxes.to(dtype=torch.float32)

            keep = torch.isfinite(boxes).all(dim=-1)
            keep &= boxes[:, 2] >= self.min_size
            keep &= boxes[:, 3] >= self.min_size
            keep &= (boxes[:, 0] >= 0.0) & (boxes[:, 0] <= float(w))
            keep &= (boxes[:, 1] >= 0.0) & (boxes[:, 1] <= float(h))
            keep &= (boxes[:, 4] >= 0.0) & (boxes[:, 4] <= 1.0)

            boxes = boxes[keep]
            boxes[:, 0] = boxes[:, 0].clamp(min=0.0, max=float(w))
            boxes[:, 1] = boxes[:, 1].clamp(min=0.0, max=float(h))
            boxes[:, 2] = boxes[:, 2].clamp(min=0.0, max=float(w))
            boxes[:, 3] = boxes[:, 3].clamp(min=0.0, max=float(h))
            boxes[:, 4] = boxes[:, 4].clamp(min=0.0, max=1.0)

            new_target = target.copy()
            new_target['boxes'] = boxes
            if 'labels' in new_target:
                new_target['labels'] = new_target['labels'][keep]

        if len(inputs) > 1:
            return (img, new_target) + inputs[2:]
        elif len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            out_list = list(inputs[0])
            out_list[0] = img
            if len(out_list) > 1:
                out_list[1] = new_target
            return tuple(out_list)
        return img

@register()
class ConvertOBB(Transform):
    def __init__(self, normalize=False) -> None:
        super().__init__()
        self.normalize = normalize

    def __call__(self, *inputs: Any) -> Any:
        # Unpack
        if len(inputs) == 1:
            if isinstance(inputs[0], (tuple, list)) and len(inputs[0]) >= 1:
                img = inputs[0][0]
                target = inputs[0][1] if len(inputs[0]) > 1 else None
            else:
                img = inputs[0]
                target = None
        else:
            img = inputs[0]
            target = inputs[1]

        if self.normalize and target is not None and 'boxes' in target:
            h, w = get_spatial_size(img)
            boxes = target['boxes'].clone() # (N, 5)
            if boxes.shape[0] > 0:
                # Normalize cx, cy, w, h
                scale = torch.tensor([w, h, w, h], device=boxes.device, dtype=boxes.dtype)
                boxes[:, :4] = boxes[:, :4] / scale
                # angle is already normalized or handled separately
                target['boxes'] = boxes

        # Repack
        if len(inputs) > 1:
            return (img, target) + inputs[2:]
        elif len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
             # Reconstruct tuple
             out_list = list(inputs[0])
             out_list[0] = img
             if len(out_list) > 1:
                 out_list[1] = target
             return tuple(out_list)
        return img
