"""
DOTA Auto Evaluator for OBB (Oriented Bounding Box) with Server Submission
"""
import copy
import os
import re
import torch
import numpy as np
import os.path as osp
import shutil
import zipfile
import time
from collections import defaultdict, OrderedDict

try:
    from dota_auto_eval import DOTAEvaluator as RemoteEvaluator
except ImportError:
    RemoteEvaluator = None

from .mmeval import nms_rotated, nms_quadri, warmup_numba
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
class DotaAutoEvaluator(object):
    def __init__(self, dataset, iou_types=None,
                 backend_url='http://dota-auto-eval.codesocean.top',
                 api_key='',
                 server_id=1,
                 outfile_prefix=None,
                 merge_patches=True,
                 iou_thr=0.1,
                 predict_box_type='rbox'):
        self.dataset = dataset
        self.iou_types = iou_types or ['mAP']
        self.backend_url = backend_url
        self.api_key = api_key
        self.server_id = server_id
        self.outfile_prefix = outfile_prefix
        self.merge_patches = merge_patches
        self.iou_thr = iou_thr
        self.predict_box_type = predict_box_type

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

            self.predictions[img_id] = {
                'bboxes': boxes.numpy(), # (N, 5)
                'scores': scores,
                'labels': labels,
                'img_id': img_id
            }

    def synchronize_between_processes(self):
        if not dist_utils.is_dist_available_and_initialized():
            return

        # Gather predictions from all processes
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
        self.img_ids = merged_img_ids

    def merge_results(self, results, outfile_prefix):
        """Merge patches' predictions into full image's results and generate a
        zip file for DOTA online evaluation.
        """
        collector = defaultdict(list)

        for result in results:
            img_id = result['img_id']
            if isinstance(img_id, (int, float)):
                if hasattr(self.dataset, 'img_ids'):
                    try:
                        img_id = self.dataset.img_ids[int(img_id)]
                    except IndexError:
                        img_id = str(img_id)
                else:
                    img_id = str(img_id)
            else:
                img_id = str(img_id)

            # Parse img_id to get original name and coordinates
            # Format usually: P0000__1__0___0 or similar
            if '__' in img_id:
                splitname = img_id.split('__')
                oriname = splitname[0]
                pattern1 = re.compile(r'__\d+___\d+')
                x_y = re.findall(pattern1, img_id)
                if x_y:
                    x_y_2 = re.findall(r'\d+', x_y[0])
                    x, y = int(x_y_2[0]), int(x_y_2[1])
                else:
                    # Fallback or different format
                    x, y = 0, 0
            else:
                oriname = img_id
                x, y = 0, 0

            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            ori_bboxes = bboxes.copy()

            if self.predict_box_type == 'rbox':
                # bboxes are (cx, cy, w, h, angle)
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
            elif self.predict_box_type == 'qbox':
                ori_bboxes[..., :] = ori_bboxes[..., :] + np.array(
                    [x, y, x, y, x, y, x, y], dtype=np.float32)
            else:
                raise NotImplementedError

            label_dets = np.concatenate(
                [labels[:, np.newaxis], ori_bboxes, scores[:, np.newaxis]],
                axis=1)
            collector[oriname].append(label_dets)

        id_list, dets_list = [], []
        for oriname, label_dets_list in collector.items():
            big_img_results = []
            label_dets = np.concatenate(label_dets_list, axis=0)
            labels, dets = label_dets[:, 0], label_dets[:, 1:]
            for i in range(len(self.classes)):
                if len(dets[labels == i]) == 0:
                    big_img_results.append(dets[labels == i])
                else:
                    cls_dets = dets[labels == i].astype(np.float32, copy=False)

                    if self.predict_box_type == 'rbox':
                        # to prevent overflow
                        if len(cls_dets) > 200000:
                            b_1 = len(cls_dets) // 2
                            nms_dets_1, _ = nms_rotated(
                                cls_dets[:b_1, :5], cls_dets[:b_1, -1],
                                self.iou_thr)
                            nms_dets_2, _ = nms_rotated(
                                cls_dets[b_1:, :5], cls_dets[b_1:, -1],
                                self.iou_thr)
                            nms_dets = np.concatenate([nms_dets_1, nms_dets_2], 0)
                            nms_dets, _ = nms_rotated(nms_dets[:, :5],
                                                      nms_dets[:, -1],
                                                      self.iou_thr)

                        else:
                            nms_dets, _ = nms_rotated(cls_dets[:, :5],
                                                      cls_dets[:, -1],
                                                      self.iou_thr)
                    elif self.predict_box_type == 'qbox':
                        nms_dets, _ = nms_quadri(cls_dets[:, :8],
                                                 cls_dets[:, -1], self.iou_thr)
                    else:
                        raise NotImplementedError
                    big_img_results.append(np.asarray(nms_dets))
            id_list.append(oriname)
            dets_list.append(big_img_results)

        # 使用传入的 outfile_prefix 参数，避免多任务并行时共用同一目录导致冲突
        work_dir = outfile_prefix
        if work_dir is None:
            work_dir = self.outfile_prefix if self.outfile_prefix is not None else 'dota_results'

        if osp.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)

        files = [
            osp.join(work_dir, 'Task1_' + cls + '.txt')
            for cls in self.classes
        ]
        file_objs = [open(f, 'w') for f in files]
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for f, dets in zip(file_objs, dets_per_cls):
                if dets.size == 0:
                    continue
                th_dets = torch.from_numpy(dets)
                if self.predict_box_type == 'rbox':
                    rboxes, scores = torch.split(th_dets, (5, 1), dim=-1)
                    qboxes = rbox2qbox(rboxes)
                elif self.predict_box_type == 'qbox':
                    qboxes, scores = torch.split(th_dets, (8, 1), dim=-1)
                else:
                    raise NotImplementedError
                for qbox, score in zip(qboxes, scores):
                    txt_element = [img_id, str(round(float(score), 2))
                                   ] + [f'{p:.2f}' for p in qbox]
                    f.writelines(' '.join(txt_element) + '\n')

        for f in file_objs:
            f.close()

        target_name = osp.split(work_dir)[-1]
        zip_path = osp.join(work_dir, target_name + '.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as t:
            for f in files:
                t.write(f, osp.split(f)[-1])

        return zip_path

    def accumulate(self):
        pass

    def summarize(self):
        if not dist_utils.is_main_process():
            return

        print("Submitting DOTA results to auto evaluation server...")

        preds = list(self.predictions.values())

        if self.merge_patches:
            zip_path = self.merge_results(preds, self.outfile_prefix)
            print(f'The submission file save at {zip_path}')
        else:
            print("Merge patches is required for auto evaluation.")
            return

        if RemoteEvaluator is None:
            print("dota_auto_eval not installed. Skipping submission.")
            self.stats = {'mAP': 0.0}
            return

        print(f"Auto eval is enabled, starting to submit task...")
        evaluator = RemoteEvaluator(
            base_url=self.backend_url,
            api_key=self.api_key,
            print_func=print
        )

        retry_times = 0
        max_retry_times = 3

        current_epoch = getattr(self, 'current_epoch', None)
        if current_epoch is None:
            current_epoch = int(os.environ.get('CURRENT_EPOCH', 0))
        else:
            current_epoch = int(current_epoch)
        task_id = os.environ.get('DOTA_AUTO_EVAL_TASK_ID', str(self.server_id))

        while retry_times < max_retry_times:
            try:
                # Submit evaluation task
                result = evaluator.submit_training_result(
                    task_id=task_id,
                    epoch=current_epoch,
                    eval_file=zip_path
                )

                print(f"\n=== 提交结果 ===")
                print(result)

                # Return epoch as mAP as requested
                self.stats = {'mAP': float(current_epoch)}
                return

            except Exception as e:
                print(f"Error happend when auto eval at {retry_times} times: {e}")
                retry_times += 1
                retry_interval = retry_times * 30
                print(f"Retry in {retry_interval} seconds")
                time.sleep(retry_interval)
                if retry_times >= max_retry_times:
                    print("*"*50)
                    print(f"Submmision took too much time than expected, you can retry manually later.")
                    print("*"*50)
                    self.stats = {'mAP': float(current_epoch)}
                    return


@register()
class DotaAutoEvaluatorOffline(DotaAutoEvaluator):
    __share__ = ['output_dir']

    def __init__(self, dataset, iou_types=None, output_dir=None, outfile_prefix=None,
                 merge_patches=True, iou_thr=0.1, predict_box_type='rbox'):
        super().__init__(
            dataset,
            iou_types=iou_types,
            outfile_prefix=outfile_prefix,
            merge_patches=merge_patches,
            iou_thr=iou_thr,
            predict_box_type=predict_box_type
        )
        self.output_dir = output_dir

    def summarize(self):
        if not dist_utils.is_main_process():
            return

        preds = list(self.predictions.values())

        if not self.merge_patches:
            print("Merge patches is required for auto evaluation.")
            return

        current_epoch = getattr(self, 'current_epoch', None)
        if current_epoch is None:
            current_epoch = int(os.environ.get('CURRENT_EPOCH', 0))
        else:
            current_epoch = int(current_epoch)
        output_dir = self.output_dir or os.environ.get('work_dir') or os.environ.get('OUTPUT_DIR')
        if output_dir:
            output_dir = os.fspath(output_dir)
            base_dir = osp.join(output_dir, 'dota_results')
        else:
            base_dir = 'dota_results'

        epoch_dir = osp.join(base_dir, f'epoch_{current_epoch}')

        zip_path = self.merge_results(preds, epoch_dir)
        final_zip = osp.join(base_dir, f'epoch_{current_epoch}.zip')
        if zip_path != final_zip:
            os.makedirs(base_dir, exist_ok=True)
            shutil.copyfile(zip_path, final_zip)
        print(f'Offline DOTA results saved at {final_zip}')
        self.stats = {'mAP': float(current_epoch)}
