"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from ..core import register


__all__ = ['PostProcessor', 'PostProcessorOBB']


def mod(a, b):
    out = a - a // b * b
    return out


@register()
class PostProcessor(nn.Module):
    __share__ = [
        'num_classes',
        'use_focal_loss',
        'num_top_queries',
        'remap_mscoco_category'
    ]

    def __init__(
        self,
        num_classes=80,
        use_focal_loss=True,
        num_top_queries=300,
        remap_mscoco_category=False
    ) -> None:
        super().__init__()
        self.use_focal_loss = use_focal_loss
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.remap_mscoco_category = remap_mscoco_category
        self.deploy_mode = False

    def extra_repr(self) -> str:
        return f'use_focal_loss={self.use_focal_loss}, num_classes={self.num_classes}, num_top_queries={self.num_top_queries}'

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']

        bbox_pred = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')
        bbox_pred *= orig_target_sizes.repeat(1, 2).unsqueeze(1)

        if self.use_focal_loss:
            scores = F.sigmoid(logits)
            scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)
            labels = mod(index, self.num_classes)
            index = index // self.num_classes
            boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, bbox_pred.shape[-1]))

        else:
            scores = F.softmax(logits)[:, :, :-1]
            scores, labels = scores.max(dim=-1)
            if scores.shape[1] > self.num_top_queries:
                scores, index = torch.topk(scores, self.num_top_queries, dim=-1)
                labels = torch.gather(labels, dim=1, index=index)
                boxes = torch.gather(boxes, dim=1, index=index.unsqueeze(-1).tile(1, 1, boxes.shape[-1]))

        if self.deploy_mode:
            return labels, boxes, scores

        if self.remap_mscoco_category:
            from ..data.dataset import mscoco_label2category
            labels = torch.tensor([mscoco_label2category[int(x.item())] for x in labels.flatten()])\
                .to(boxes.device).reshape(labels.shape)

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results


    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self


@register()
class PostProcessorOBB(nn.Module):
    __share__ = ['num_classes', 'num_top_queries']

    def __init__(self, num_classes=80, num_top_queries=300, input_shape=None) -> None:
        super().__init__()
        self.num_top_queries = num_top_queries
        self.num_classes = int(num_classes)
        self.deploy_mode = False
        self.input_shape = input_shape

    def forward(self, outputs, orig_target_sizes: torch.Tensor):
        logits, boxes = outputs['pred_logits'], outputs['pred_boxes']
        if self.input_shape is not None:
            target_h, target_w = self.input_shape
            h = orig_target_sizes[:, 0]
            w = orig_target_sizes[:, 1]
            scale_w = target_w / w
            scale_h = target_h / h
            scale = torch.min(scale_w, scale_h).unsqueeze(1).unsqueeze(1)
            target_wh = torch.tensor([target_w, target_h, target_w, target_h], device=boxes.device).reshape(1, 1, 4)

            boxes_xywh = boxes[..., :4] * target_wh / scale
            boxes_angle = boxes[..., 4:] * 3.141592653589793
            bbox_pred = torch.cat([boxes_xywh, boxes_angle], dim=-1)

        else:
            h = orig_target_sizes[:, 0]
            w = orig_target_sizes[:, 1]
            scale_fct = torch.stack([w, h, w, h], dim=1).unsqueeze(1)
            boxes_xywh = boxes[..., :4] * scale_fct
            boxes_angle = boxes[..., 4:] * 3.141592653589793
            bbox_pred = torch.cat([boxes_xywh, boxes_angle], dim=-1)

        scores = F.sigmoid(logits)
        scores, index = torch.topk(scores.flatten(1), self.num_top_queries, dim=-1)

        labels = mod(index, self.num_classes)
        index = index // self.num_classes
        boxes = bbox_pred.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, 5))

        if self.deploy_mode:
            return labels, boxes, scores

        results = []
        for lab, box, sco in zip(labels, boxes, scores):
            result = dict(labels=lab, boxes=box, scores=sco)
            results.append(result)

        return results

    def deploy(self, ):
        self.eval()
        self.deploy_mode = True
        return self
