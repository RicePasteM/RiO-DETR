
import torch
import cv2
import numpy as np
import math
import os

def norm_angle_to_radian(norm_angle):
    # [0, 1] -> [0, 180] degrees
    angle_deg = norm_angle * 180.0
    # degree to radian
    return angle_deg * math.pi / 180.0

def get_rotated_rect(cx, cy, w, h, angle_rad):
    # Center (cx, cy)
    # Width w, Height h
    # Angle theta (radians)

    # Corners relative to center:
    # (-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    corners = []
    # Order: top-left, top-right, bottom-right, bottom-left (relative to unrotated)
    # But with rotation, just 4 points.
    for dx, dy in [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]:
        # Rotate
        # x' = x cos - y sin
        # y' = x sin + y cos
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        corners.append([cx + rx, cy + ry])

    return np.array(corners, dtype=np.int32)

def visualize_obb_training(samples, targets, outputs, epoch, batch_idx, output_dir, threshold=0):
    """
    Visualize OBB training samples.

    Args:
        samples: (B, C, H, W) normalized [0, 1]
        targets: list of dicts with 'boxes' (N, 5) normalized
        outputs: dict with 'pred_boxes' (B, Q, 5) normalized, 'pred_logits' (B, Q, K)
        epoch: current epoch
        batch_idx: current batch index
        output_dir: directory to save images
        threshold: score threshold for predictions
    """
    vis_dir = os.path.join(output_dir, "vis_train")
    os.makedirs(vis_dir, exist_ok=True)

    B, C, H, W = samples.shape

    # Limit to first 4 images in batch to save time/space
    limit = min(B, 4)

    for i in range(limit):
        img_tensor = samples[i].cpu()
        # Convert to numpy (H, W, C)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        # Scale to 255
        img_np = (img_np * 255).astype(np.uint8).copy() # copy to make it contiguous/writable
        # OpenCV uses BGR
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Draw GT
        if i < len(targets) and 'boxes' in targets[i]:
            tgt_boxes = targets[i]['boxes'].cpu()
            for box in tgt_boxes:
                cx, cy, w, h, angle_norm = box.tolist()
                cx *= W
                cy *= H
                w *= W
                h *= H
                angle = norm_angle_to_radian(angle_norm)

                rect = get_rotated_rect(cx, cy, w, h, angle)
                cv2.polylines(img_cv, [rect], isClosed=True, color=(0, 255, 0), thickness=2) # Green for GT

        # Draw Pred
        if 'pred_boxes' in outputs and 'pred_logits' in outputs:
            pred_boxes = outputs['pred_boxes'][i].detach().cpu()
            pred_logits = outputs['pred_logits'][i].detach().cpu()

            prob = pred_logits.sigmoid()
            scores, labels = prob.max(dim=-1)

            keep = scores >= threshold
            pred_boxes = pred_boxes[keep]
            scores = scores[keep]

            for box, score in zip(pred_boxes, scores):
                cx, cy, w, h, angle_norm = box.tolist()
                cx *= W
                cy *= H
                w *= W
                h *= H
                angle = norm_angle_to_radian(angle_norm)

                rect = get_rotated_rect(cx, cy, w, h, angle)
                cv2.polylines(img_cv, [rect], isClosed=True, color=(0, 0, 255), thickness=2) # Red for Pred
                # cv2.putText(img_cv, f"{score:.2f}", (rect[0][0], rect[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw DN queries (Orange)
        if 'dn_outputs' in outputs and outputs['dn_outputs'] is not None:
            # Take the last layer output
            # dn_outputs is a list of dicts, we take the last one
            if len(outputs['dn_outputs']) > 0 and 'pred_boxes' in outputs['dn_outputs'][-1]:
                dn_boxes = outputs['dn_outputs'][-1]['pred_boxes'][i].detach().cpu()

                for box in dn_boxes:
                    cx, cy, w, h, angle_norm = box.tolist()
                    cx *= W
                    cy *= H
                    w *= W
                    h *= H
                    angle = norm_angle_to_radian(angle_norm)

                    rect = get_rotated_rect(cx, cy, w, h, angle)
                    # Orange color: BGR (0, 165, 255)
                    cv2.polylines(img_cv, [rect], isClosed=True, color=(0, 165, 255), thickness=2)

        save_path = os.path.join(vis_dir, f"epoch_{epoch}_batch_{batch_idx}_img_{i}.jpg")
        cv2.imwrite(save_path, img_cv)
