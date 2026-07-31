import os
import torch
import torch.utils.data
import numpy as np
import cv2
from PIL import Image

from ._dataset import DetDataset
from ...core import register

@register()
class DOTADataset(DetDataset):
    __inject__ = ['transforms']

    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle',
               'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',
               'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter')

    def __init__(self, img_folder, ann_folder, transforms=None, return_masks=False):
        super().__init__()
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self._transforms = transforms
        self.return_masks = return_masks

        self.category2label = {k: i for i, k in enumerate(self.CLASSES)}
        self.img_ids = self._load_img_ids()

    def _load_img_ids(self):
        img_ids = []
        if not os.path.exists(self.img_folder):
            return img_ids

        for filename in os.listdir(self.img_folder):
            if filename.lower().endswith(('.jpg', '.png', '.bmp', '.tif', '.tiff')):
                img_ids.append(os.path.splitext(filename)[0])
        return sorted(img_ids)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        img_id = self.img_ids[idx]
        # Try different extensions
        img_path = None
        for ext in ['.png', '.jpg', '.bmp', '.tif', '.tiff']:
            p = os.path.join(self.img_folder, f'{img_id}{ext}')
            if os.path.exists(p):
                img_path = p
                break

        if img_path is None:
             raise FileNotFoundError(f"Image not found for id {img_id}")

        txt_path = os.path.join(self.ann_folder, f'{img_id}.txt')

        # Load image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Parse Text Annotation
        target = self._parse_txt(txt_path, w, h)
        target['image_id'] = torch.tensor([idx])
        target['idx'] = torch.tensor([idx])

        return img, target

    def get_ann_info(self, idx):
        img_id = self.img_ids[idx]
        txt_path = os.path.join(self.ann_folder, f'{img_id}.txt')
        target = self._parse_txt(txt_path, 0, 0)
        target['image_id'] = torch.tensor([idx])
        target['idx'] = torch.tensor([idx])
        return target

    def _parse_txt(self, txt_path, w, h):
        boxes = []
        labels = []

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    # DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
                    if len(parts) < 9:
                        continue

                    category = parts[8]
                    if category not in self.category2label:
                        continue

                    poly = list(map(float, parts[:8]))

                    # Convert poly to rotated rect (cx, cy, w, h, angle)
                    poly_np = np.array(poly, dtype=np.float32).reshape(4, 2)
                    rect = cv2.minAreaRect(poly_np)
                    (cx, cy), (w_box, h_box), angle = rect

                    # Filter invalid boxes
                    if w_box <= 1 or h_box <= 1:
                        continue

                    if w_box < h_box:
                        w_box, h_box = h_box, w_box
                        angle += 90.0
                    angle = angle % 180.0
                    angle_norm = angle / 180.0
                    boxes.append([cx, cy, w_box, h_box, angle_norm])
                    labels.append(self.category2label[category])

        if len(boxes) > 0:
            target = {
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'labels': torch.tensor(labels, dtype=torch.int64),
                'orig_size': torch.tensor([h, w]),
                'size': torch.tensor([h, w])
            }
        else:
             target = {
                'boxes': torch.zeros((0, 5), dtype=torch.float32),
                'labels': torch.zeros((0,), dtype=torch.int64),
                'orig_size': torch.tensor([h, w]),
                'size': torch.tensor([h, w])
            }

        return target
