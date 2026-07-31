"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Modules to compute the matching cost and solve the corresponding LSAP.

Copyright (c) 2024 The D-FINE Authors All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from typing import Dict

from .utils_obb import kld_loss, hausdorff_loss

from ..core import register
import numpy as np
import os
import matplotlib.pyplot as plt


@register()
class HungarianMatcherOBB(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    __share__ = ['use_focal_loss', ]

    def __init__(self, weight_dict, use_focal_loss=False, alpha=0.25, gamma=2.0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = weight_dict['cost_class']
        self.cost_bbox = weight_dict['cost_bbox']
        self.cost_giou = weight_dict['cost_giou']
        self.cost_hausdorff = weight_dict.get('cost_hausdorff', 0)

        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma

        self.vis_count = 0
        self.vis_limit = 10
        self.vis_output_dir = None # To be set dynamically

        assert self.cost_class != 0 or self.cost_bbox != 0 or self.cost_giou != 0 or self.cost_hausdorff != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets, return_topk=False):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        if self.use_focal_loss:
            out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
        else:
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if self.use_focal_loss:
            out_prob = out_prob[:, tgt_ids]
            neg_cost_class = (1 - self.alpha) * (out_prob ** self.gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - out_prob) ** self.gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -out_prob[:, tgt_ids]

        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        out_bbox_ex = out_bbox.unsqueeze(1).expand(-1, len(tgt_bbox), -1)
        tgt_bbox_ex = tgt_bbox.unsqueeze(0).expand(len(out_bbox), -1, -1)

        cost_gd = kld_loss(out_bbox_ex, tgt_bbox_ex, fun='log1p', tau=1.0, sqrt=False)

        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_gd

        if self.cost_hausdorff > 0:
            cost_hausdorff = hausdorff_loss(out_bbox, tgt_bbox)
            C += self.cost_hausdorff * cost_hausdorff

        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        C = torch.nan_to_num(C, nan=1.0)
        indices_pre = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices_pre]
        if return_topk:
            return {'indices_o2m': self.get_top_k_matches(C, sizes=sizes, k=return_topk, initial_indices=indices_pre)}

        return {'indices': indices}

    def get_top_k_matches(self, C, sizes, k=1, initial_indices=None):
        indices_list = []
        for i in range(k):
            indices_k = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))] if i > 0 else initial_indices
            indices_list.append([
                (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices_k
            ])
            for c, idx_k in zip(C.split(sizes, -1), indices_k):
                idx_k = np.stack(idx_k)
                c[:, idx_k] = 1e6
        indices_list = [(torch.cat([indices_list[i][j][0] for i in range(k)], dim=0),
                        torch.cat([indices_list[i][j][1] for i in range(k)], dim=0)) for j in range(len(sizes))]
        return indices_list

    def visualize_assignment(self, outputs, targets, indices):
        if self.vis_count >= self.vis_limit:
            return

        if self.vis_output_dir is None:
            return

        if not os.path.exists(self.vis_output_dir):
            os.makedirs(self.vis_output_dir, exist_ok=True)

        pred_boxes = outputs["pred_boxes"]
        bs = len(targets)

        for b in range(bs):
            if self.vis_count >= self.vis_limit:
                break
            src_idx, tgt_idx = indices[b]
            matched_preds = pred_boxes[b][src_idx].detach().cpu().numpy()
            all_tgt_boxes = targets[b]["boxes"].detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(10, 10))
            if "orig_size" in targets[b]:
                if isinstance(targets[b]["orig_size"], torch.Tensor):
                    h, w = targets[b]["orig_size"].detach().cpu().numpy()
                else:
                    h, w = targets[b]["orig_size"]
            else:
                h, w = 640, 640

            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)

            def get_poly(cx, cy, w_b, h_b, angle_norm):
                cx *= w
                cy *= h
                w_b *= w
                h_b *= h
                angle_rad = angle_norm * np.pi

                c, s = np.cos(angle_rad), np.sin(angle_rad)
                R = np.array([[c, -s], [s, c]])
                pts = np.array([
                    [-w_b/2, -h_b/2],
                    [w_b/2, -h_b/2],
                    [w_b/2, h_b/2],
                    [-w_b/2, h_b/2]
                ])
                return (pts @ R.T) + np.array([cx, cy])

            for box in all_tgt_boxes:
                try:
                    pts = get_poly(*box)
                    poly = plt.Polygon(pts, fill=False, edgecolor='green', linewidth=2, label='GT')
                    ax.add_patch(poly)
                except Exception as e:
                    print(f"Error drawing GT box: {e}")

            for box in matched_preds:
                try:
                    pts = get_poly(*box)
                    poly = plt.Polygon(pts, fill=False, edgecolor='red', linewidth=2, label='Pred')
                    ax.add_patch(poly)
                except Exception as e:
                    print(f"Error drawing Pred box: {e}")

            handles, labels = plt.gca().get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())

            plt.title(f'Batch {b} - Match Count {len(src_idx)}')
            save_path = os.path.join(self.vis_output_dir, f'vis_{self.vis_count}.png')
            plt.savefig(save_path)
            plt.close()
            self.vis_count += 1
