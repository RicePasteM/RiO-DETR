import torch
import numpy as np

from .mmeval import warmup_numba
from ...core import register
from ...misc import dist_utils


@register()
class HRSCEvaluator(object):
    def __init__(self, dataset, iou_types=None):
        self.dataset = dataset
        self.iou_types = iou_types or ['mAP']
        self.predictions = {}
        self.img_ids = []
        self.stats = {}

        if hasattr(dataset, 'CLASSES'):
            self.classes = dataset.CLASSES
        else:
            self.classes = dataset.categories if hasattr(dataset, 'categories') else []
        warmup_numba()

    def cleanup(self):
        self.predictions = {}
        self.img_ids = []
        self.stats = {}

    def update(self, predictions):
        for img_id, pred in predictions.items():
            self.img_ids.append(img_id)
            boxes = pred['boxes'].cpu()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()

            self.predictions[img_id] = {
                'bboxes': boxes.numpy(),
                'scores': scores,
                'labels': labels
            }

    def synchronize_between_processes(self):
        if not dist_utils.is_dist_available_and_initialized():
            return

        all_predictions = dist_utils.all_gather(self.predictions)
        merged_predictions = {}
        for p in all_predictions:
            merged_predictions.update(p)
        self.predictions = merged_predictions

        all_img_ids = dist_utils.all_gather(self.img_ids)
        merged_img_ids = []
        for ids in all_img_ids:
            merged_img_ids.extend(ids)
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
        if not dist_utils.is_main_process():
            return

        from .mmeval import eval_rbbox_map

        dets = []
        gts = []

        for img_id in self.predictions.keys():
            idx = self._resolve_idx(img_id)
            if idx is None:
                continue

            target = self._get_target(idx)
            gt_bboxes = self._to_numpy(target['boxes']).copy()
            gt_labels = self._to_numpy(target['labels'])
            gt_bboxes_ignore = self._to_numpy(
                target.get('boxes_ignore', np.zeros((0, 5), dtype=np.float32))
            ).copy()
            gt_labels_ignore = self._to_numpy(
                target.get('labels_ignore', np.zeros((0,), dtype=np.int64))
            )
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
                else:
                    res = np.zeros((0, 6), dtype=np.float32)
                img_dets.append(res)
            dets.append(img_dets)

        mean_ap_07, self.eval_results_07 = eval_rbbox_map(
            dets,
            gts,
            scale_ranges=None,
            iou_thr=0.5,
            use_07_metric=True,
            dataset=self.classes
        )
        mean_ap_12, self.eval_results_12 = eval_rbbox_map(
            dets,
            gts,
            scale_ranges=None,
            iou_thr=0.5,
            use_07_metric=False,
            dataset=self.classes
        )

        self.stats = {'mAP_07': float(mean_ap_07), 'mAP_12': float(mean_ap_12)}
        print(f"mAP(07): {mean_ap_07:.4f}")
        print(f"mAP(12): {mean_ap_12:.4f}")

    def _resolve_idx(self, img_id):
        if isinstance(img_id, torch.Tensor):
            img_id = img_id.item()
        if isinstance(img_id, (int, np.integer)):
            return int(img_id)
        if hasattr(self.dataset, 'img_ids'):
            try:
                return self.dataset.img_ids.index(str(img_id))
            except ValueError:
                return None
        return None
