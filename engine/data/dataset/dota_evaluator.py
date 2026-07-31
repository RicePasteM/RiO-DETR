"""
DOTA Evaluator for OBB (Oriented Bounding Box)
"""
import copy
import os
import re
import torch
import numpy as np
import os.path as osp
from collections import defaultdict, OrderedDict
from .mmeval import eval_rbbox_map, warmup_numba
from ...core import register
from ...misc import dist_utils

def rbox2qbox(rbox):
    """Convert rbox to qbox.
    Args:
        rbox (torch.Tensor): (N, 5) [cx, cy, w, h, angle]
    Returns:
        qbox (torch.Tensor): (N, 8) [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    if rbox.numel() == 0:
        return torch.zeros((0, 8), device=rbox.device)

    cx, cy, w, h, a = rbox.unbind(dim=-1)

    # angle in radian
    cos = torch.cos(a)
    sin = torch.sin(a)

    wx, wy = w / 2 * cos, w / 2 * sin
    hx, hy = -h / 2 * sin, h / 2 * cos

    p1x, p1y = cx - wx - hx, cy - wy - hy
    p2x, p2y = cx + wx - hx, cy + wy - hy
    p3x, p3y = cx + wx + hx, cy + wy + hy
    p4x, p4y = cx - wx + hx, cy - wy + hy

    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1)

@register()
class DotaEvaluator(object):
    def __init__(self, dataset, iou_types=None, iou_thrs=None):
        self.dataset = dataset
        self.iou_types = iou_types or ['mAP']
        self.iou_thrs = [float(v) for v in (iou_thrs or [0.5])]
        if not self.iou_thrs:
            raise ValueError("iou_thrs must contain at least one IoU threshold")
        if any(v <= 0.0 or v > 1.0 for v in self.iou_thrs):
            raise ValueError(f"Invalid IoU thresholds: {self.iou_thrs}")
        self.predictions = {}
        self.img_ids = []
        self.results = []

        # Meta info from dataset
        if hasattr(dataset, 'CLASSES'):
            self.classes = dataset.CLASSES
        else:
            self.classes = dataset.categories if hasattr(dataset, 'categories') else []
        warmup_numba()

    def cleanup(self):
        self.predictions = {}
        self.img_ids = []
        self.results = []

    def update(self, predictions):
        """
        predictions: dict of {img_id: {'boxes': tensor, 'scores': tensor, 'labels': tensor}}
        boxes should be (cx, cy, w, h, angle_norm)
        """
        for img_id, pred in predictions.items():
            self.img_ids.append(img_id)

            # Convert tensors to numpy
            boxes = pred['boxes'].cpu()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()

            # Boxes angle are already in radian (from PostProcessorOBB)
            # angle_norm = boxes[:, 4]
            # angle_rad = (angle_norm * 90.0 - 90.0) * np.pi / 180.0
            # boxes[:, 4] = angle_rad

            self.predictions[img_id] = {
                'bboxes': boxes.numpy(), # (N, 5)
                'scores': scores,
                'labels': labels
            }

    def synchronize_between_processes(self):
        if not dist_utils.is_dist_available_and_initialized():
            return

        # Gather predictions from all processes
        # self.predictions is a dict, we can gather a list of dicts and merge them
        all_predictions = dist_utils.all_gather(self.predictions)
        merged_predictions = {}
        for p in all_predictions:
            merged_predictions.update(p)
        self.predictions = merged_predictions

        # Also gather img_ids
        all_img_ids = dist_utils.all_gather(self.img_ids)
        merged_img_ids = []
        for ids in all_img_ids:
            merged_img_ids.extend(ids)

        # Use set to remove potential duplicates if any (though usually shouldn't be with proper sampler)
        # But img_ids need to be hashable (they are strings or ints usually)
        self.img_ids = merged_img_ids

    def accumulate(self):
        pass

    def _get_target(self, idx):
        if hasattr(self.dataset, 'get_ann_info'):
            return self.dataset.get_ann_info(idx)
        _, target = self.dataset.load_item(idx)
        return target

    def _to_numpy(self, value):
        if hasattr(value, 'cpu'):
            return value.cpu().numpy()
        return np.asarray(value)

    def summarize(self):
        print("Evaluating DOTA metrics...")

        print(f"Collected predictions for {len(self.predictions)} images.")

        dets = []
        gts = []

        for i, img_id in enumerate(self.img_ids):
            if isinstance(img_id, torch.Tensor):
                img_id = img_id.item()

            idx = int(img_id)
            target = self._get_target(idx)

            gt_bboxes = self._to_numpy(target['boxes']).copy()
            gt_labels = self._to_numpy(target['labels'])
            gt_bboxes_ignore = self._to_numpy(
                target.get('boxes_ignore', np.zeros((0, 5), dtype=np.float32))
            ).copy()
            gt_labels_ignore = self._to_numpy(
                target.get('labels_ignore', np.zeros((0,), dtype=np.int64))
            )

            # Convert GT angle to radian (0~pi)
            if gt_bboxes.size > 0:
                gt_bboxes[:, 4] = gt_bboxes[:, 4] * np.pi
            if gt_bboxes_ignore.size > 0:
                gt_bboxes_ignore[:, 4] = gt_bboxes_ignore[:, 4] * np.pi

            gts.append({
                'bboxes': gt_bboxes,
                'labels': gt_labels,
                'bboxes_ignore': gt_bboxes_ignore,
                'labels_ignore': gt_labels_ignore
            })

            pred = self.predictions[img_id]
            pred_bboxes = pred['bboxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']

            img_dets = []
            for cls_id in range(len(self.classes)):
                cls_mask = pred_labels == cls_id
                if np.any(cls_mask):
                    cls_bbox = pred_bboxes[cls_mask]
                    cls_score = pred_scores[cls_mask]
                    res = np.concatenate([cls_bbox, cls_score[:, None]], axis=1)
                    img_dets.append(res)
                else:
                    img_dets.append(np.zeros((0, 6)))
            dets.append(img_dets)

        per_iou_ap = OrderedDict()
        per_iou_results = OrderedDict()
        multi_iou = len(self.iou_thrs) > 1

        for iou_thr in self.iou_thrs:
            mean_ap, eval_results = eval_rbbox_map(
                dets,
                gts,
                scale_ranges=None,
                iou_thr=iou_thr,
                use_07_metric=True,
                # Preserve the original per-class table for the default AP50
                # path, but keep multi-IoU evaluation output compact.
                dataset=None if multi_iou else self.classes
            )
            per_iou_ap[iou_thr] = float(mean_ap)
            per_iou_results[iou_thr] = eval_results
            print(f"mAP@{iou_thr:.2f}: {mean_ap:.4f}")

        if not multi_iou:
            mean_ap = next(iter(per_iou_ap.values()))
            self.eval_results = next(iter(per_iou_results.values()))
            self.stats = {'mAP': mean_ap}
            return

        def metric_at(target):
            for iou_thr, value in per_iou_ap.items():
                if np.isclose(iou_thr, target):
                    return value
            raise ValueError(
                f"Required IoU threshold {target:.2f} is missing from {self.iou_thrs}")

        ap50 = metric_at(0.50)
        ap75 = metric_at(0.75)
        ap50_95 = float(np.mean(list(per_iou_ap.values())))

        self.eval_results = per_iou_results
        self.stats = {
            'AP50': ap50,
            'AP75': ap75,
            'AP50_95': ap50_95,
        }
        print(
            "OBB_METRICS "
            f"AP50={ap50:.6f} AP75={ap75:.6f} AP50_95={ap50_95:.6f}"
        )
