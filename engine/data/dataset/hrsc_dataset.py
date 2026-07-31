import os
import torch
import torch.utils.data
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from PIL import Image

from ._dataset import DetDataset
from ...core import register


@register()
class HRSC2016Dataset(DetDataset):
    __inject__ = ['transforms']

    CLASSES = ('ship',)

    def __init__(self, img_folder, ann_folder, ann_file, transforms=None, return_masks=False):
        super().__init__()
        self.img_folder = img_folder
        self.ann_folder = ann_folder
        self.ann_file = ann_file
        self._transforms = transforms
        self.return_masks = return_masks
        self.img_ids = self._load_img_ids()

    def _load_img_ids(self):
        if isinstance(self.ann_file, (list, tuple)):
            img_ids = []
            for file in self.ann_file:
                with open(file, 'r') as f:
                    img_ids.extend([line.strip() for line in f])
        else:
            with open(self.ann_file, 'r') as f:
                img_ids = [line.strip() for line in f]
        return img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img, target = self.load_item(idx)
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        img_id = self.img_ids[idx]
        img_path = self._find_image_path(img_id)
        xml_path = os.path.join(self.ann_folder, f'{os.path.splitext(img_id)[0]}.xml')

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        target = self._parse_xml(xml_path, w, h)
        target['image_id'] = torch.tensor([idx])
        target['idx'] = torch.tensor([idx])
        return img, target

    def get_ann_info(self, idx):
        img_id = self.img_ids[idx]
        xml_path = os.path.join(self.ann_folder, f'{os.path.splitext(img_id)[0]}.xml')
        target = self._parse_xml(xml_path, 0, 0)
        target['image_id'] = torch.tensor([idx])
        target['idx'] = torch.tensor([idx])
        return target

    def _find_image_path(self, img_id):
        candidate = os.path.join(self.img_folder, img_id)
        if os.path.exists(candidate):
            return candidate
        for ext in ['.jpg', '.png', '.bmp', '.tif', '.tiff']:
            p = os.path.join(self.img_folder, f'{img_id}{ext}')
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Image not found for id {img_id}")

    def _parse_xml(self, xml_path, w, h):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        cat2label = self.category2label

        objects = root.findall('.//HRSC_Object')
        if not objects:
            objects = root.findall('object')

        for obj in objects:
            name = obj.findtext('Class') or obj.findtext('name') or 'ship'
            name = name.lower()
            if name in cat2label:
                label = cat2label[name]
            elif len(cat2label) == 1:
                label = 0
            else:
                continue

            mbox_cx = obj.findtext('mbox_cx')
            mbox_cy = obj.findtext('mbox_cy')
            mbox_w = obj.findtext('mbox_w')
            mbox_h = obj.findtext('mbox_h')
            mbox_ang = obj.findtext('mbox_ang')

            if all(v is not None for v in [mbox_cx, mbox_cy, mbox_w, mbox_h, mbox_ang]):
                cx = float(mbox_cx)
                cy = float(mbox_cy)
                w_box = float(mbox_w)
                h_box = float(mbox_h)
                angle = float(mbox_ang)

                if w_box <= 1e-6 or h_box <= 1e-6:
                    continue

                if abs(angle) <= 3.2:
                    angle_deg = angle * 180.0 / np.pi
                else:
                    angle_deg = angle

                if w_box < h_box:
                    w_box, h_box = h_box, w_box
                    angle_deg += 90.0

                angle_deg = angle_deg % 180.0
                angle_norm = angle_deg / 180.0

                boxes.append([cx, cy, w_box, h_box, angle_norm])
                labels.append(label)
                continue

            robndbox = obj.find('robndbox')
            if robndbox is not None:
                x_lt = float(robndbox.find('x_left_top').text)
                y_lt = float(robndbox.find('y_left_top').text)
                x_rt = float(robndbox.find('x_right_top').text)
                y_rt = float(robndbox.find('y_right_top').text)
                x_rb = float(robndbox.find('x_right_bottom').text)
                y_rb = float(robndbox.find('y_right_bottom').text)
                x_lb = float(robndbox.find('x_left_bottom').text)
                y_lb = float(robndbox.find('y_left_bottom').text)

                poly = np.array([[x_lt, y_lt], [x_rt, y_rt], [x_rb, y_rb], [x_lb, y_lb]], dtype=np.float32)
                rect = cv2.minAreaRect(poly)
                (cx, cy), (w_box, h_box), angle_deg = rect

                if w_box <= 1e-6 or h_box <= 1e-6:
                    continue

                if w_box < h_box:
                    w_box, h_box = h_box, w_box
                    angle_deg += 90.0

                angle_deg = angle_deg % 180.0
                angle_norm = angle_deg / 180.0

                boxes.append([cx, cy, w_box, h_box, angle_norm])
                labels.append(label)
                continue

            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                cx = (xmin + xmax) / 2
                cy = (ymin + ymax) / 2
                w_box = xmax - xmin
                h_box = ymax - ymin

                angle_deg = 0.0
                if w_box < h_box:
                    w_box, h_box = h_box, w_box
                    angle_deg = 90.0

                angle_norm = angle_deg / 180.0

                boxes.append([cx, cy, w_box, h_box, angle_norm])
                labels.append(label)

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 5), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        return {'boxes': boxes, 'labels': labels, 'orig_size': torch.tensor([h, w])}

    @property
    def category2name(self):
        return {i: cat for i, cat in enumerate(self.CLASSES)}

    @property
    def category2label(self):
        return {cat: i for i, cat in enumerate(self.CLASSES)}

    @property
    def label2category(self):
        return {i: cat for i, cat in enumerate(self.CLASSES)}
