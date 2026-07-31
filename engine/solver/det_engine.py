"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import sys
import math
import os
import numpy as np
import cv2
from typing import Iterable

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils
from ..misc.vis_obb import visualize_obb_training

def _compute_encoder_transformer_grad_percentage(model: torch.nn.Module) -> float:
    """Compute percentage of gradients attributed to encoder transformer only.
    This avoids collecting/printing any other stats for speed.
    """
    total_l1 = 0.0
    enc_l1 = 0.0
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        val = grad.detach().abs().sum().item()
        total_l1 += val
        # Support both DDP ('module.') and non-DDP naming
        if name.startswith('module.encoder.encoder'):
            enc_l1 += val
    if total_l1 <= 0.0 or not math.isfinite(total_l1):
        return 0.0
    return 100.0 * enc_l1 / total_l1


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    # Gradient Analysis
    encoder_grad_percentages = []
    cur_iters = epoch * len(data_loader)

    teacher_model = kwargs.get('teacher_model', None)

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        teacher_encoder_output_for_distillation = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_encoder_output_for_distillation = teacher_model(samples).detach()

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets,
                                teacher_encoder_output=teacher_encoder_output_for_distillation)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(f"\n[NaN Detected] Epoch {epoch}, Step {i}")

            if dist_utils.is_main_process() and i <= 8:
                 output_dir = kwargs.get('output_dir', '.')
                 if output_dir:
                     visualize_obb_training(samples, targets, outputs, epoch, i, output_dir)

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # Collect gradient
            if dist_utils.is_main_process() and hasattr(criterion, 'distill_adaptive_params') and \
               getattr(criterion, 'distill_adaptive_params') and \
               criterion.distill_adaptive_params.get('enabled', False):
                pct = _compute_encoder_transformer_grad_percentage(model)
                encoder_grad_percentages.append(pct)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets,
                            teacher_encoder_output=teacher_encoder_output_for_distillation) # NEW kwarg

            if dist_utils.is_main_process() and epoch < 3 and i == 0:
                 output_dir = kwargs.get('output_dir', '.')
                 if output_dir:
                     visualize_obb_training(samples, targets, outputs, epoch, i, output_dir)

            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            # Collect gradient
            if dist_utils.is_main_process() and hasattr(criterion, 'distill_adaptive_params') and \
               getattr(criterion, 'distill_adaptive_params') and \
               criterion.distill_adaptive_params.get('enabled', False):
                pct = _compute_encoder_transformer_grad_percentage(model)
                encoder_grad_percentages.append(pct)

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, encoder_grad_percentages


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, coco_evaluator: CocoEvaluator, device, output_dir=None, current_epoch=None):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessor.keys())
    iou_types = coco_evaluator.iou_types
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    # Debug visualization setup
    if output_dir:
        vis_dir = os.path.join(output_dir, "val_vis_results")
    else:
        vis_dir = "val_vis_results"

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir, exist_ok=True)
    vis_count = 0

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        # --- Visualization Start ---
        if vis_count < 10:
            try:
                if isinstance(samples, torch.Tensor):
                    imgs_tensor = samples
                else:
                    imgs_tensor = samples.tensors

                for i in range(len(targets)):
                    if vis_count >= 10: break

                    # Prepare image
                    img_tensor = imgs_tensor[i].detach().cpu()
                    img_np = img_tensor.numpy().transpose(1, 2, 0)
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6) * 255
                    img_cv = img_np.astype(np.uint8).copy()
                    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                    h, w = img_cv.shape[:2]

                    # Draw Predictions (Green)
                    pred_logits = outputs['pred_logits'][i].detach().cpu()
                    pred_boxes = outputs['pred_boxes'][i].detach().cpu()

                    scores, labels = pred_logits.sigmoid().max(-1)
                    keep = scores > 0.3

                    for box, score, label in zip(pred_boxes[keep], scores[keep], labels[keep]):
                        box_np = box.numpy()
                        if box_np.shape[0] == 5:
                            cx, cy, bw, bh, angle = box_np
                            rect = ((cx * w, cy * h), (bw * w, bh * h), angle * 180)
                            box_points = cv2.boxPoints(rect)
                            box_points = np.int32(box_points)
                            cv2.drawContours(img_cv, [box_points], 0, (0, 255, 0), 2)
                            x_label = int(np.min(box_points[:, 0]))
                            y_label = int(np.min(box_points[:, 1]))
                            cv2.putText(img_cv, f"{label}:{score:.2f}", (x_label, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        else:
                            cx, cy, bw, bh = box_np[:4]
                            x1, y1 = int((cx - bw/2) * w), int((cy - bh/2) * h)
                            x2, y2 = int((cx + bw/2) * w), int((cy + bh/2) * h)
                            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img_cv, f"{label}:{score:.2f}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Draw GT (Red)
                    if 'boxes' in targets[i]:
                        gt_boxes = targets[i]['boxes'].detach().cpu()
                        for box in gt_boxes:
                            box_np = box.numpy()
                            if box_np.shape[0] == 5:
                                cx, cy, bw, bh, angle = box_np
                                rect = ((cx * w, cy * h), (bw * w, bh * h), angle * 180)
                                box_points = cv2.boxPoints(rect)
                                box_points = np.int32(box_points)
                                cv2.drawContours(img_cv, [box_points], 0, (0, 0, 255), 2)
                            else:
                                cx, cy, bw, bh = box_np[:4]
                                x1, y1 = int((cx - bw/2) * w), int((cy - bh/2) * h)
                                x2, y2 = int((cx + bw/2) * w), int((cy + bh/2) * h)
                                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)

                    save_path = os.path.join(vis_dir, f"vis_{vis_count}.jpg")
                    cv2.imwrite(save_path, img_cv)
                    vis_count += 1
            except Exception as e:
                print(f"Visualization failed: {e}")
        # --- Visualization End ---

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        # if 'segm' in postprocessor.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessor['segm'](results, outputs, orig_target_sizes, target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        if current_epoch is not None:
            coco_evaluator.current_epoch = current_epoch

        # DOTA/HRSC-style evaluators rebuild GT and compute mAP on CPU, so run once.
        summarize_on_main_only = not hasattr(coco_evaluator, 'coco_eval')
        if summarize_on_main_only:
            if dist_utils.is_main_process():
                coco_evaluator.summarize()
                evaluator_stats = getattr(coco_evaluator, 'stats', {})
            else:
                evaluator_stats = None

            if dist_utils.is_dist_available_and_initialized():
                stats_list = dist_utils.all_gather(evaluator_stats)
                coco_evaluator.stats = stats_list[0]
        else:
            coco_evaluator.summarize()

    stats = {}
    # stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if hasattr(coco_evaluator, 'coco_eval'):
            if 'bbox' in iou_types:
                stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
            if 'segm' in iou_types:
                stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
        elif hasattr(coco_evaluator, 'stats'):
            stats.update(coco_evaluator.stats)

    return stats, coco_evaluator
