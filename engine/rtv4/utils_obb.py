import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def norm_angle_to_radian(norm_angle):
    """
    Convert normalized angle [0, 1] to radian.
    Now supports: [0, 180] degrees (0 to pi radians).
    """
    # [0, 1] -> [0, 180] degrees
    angle_deg = norm_angle * 180.0
    # degree to radian
    return angle_deg * math.pi / 180.0

def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.
    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).
                              xywh are absolute or normalized?
                              Formula depends on relative scale.
                              r is in radian.
    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]

    wh = xywhr[..., 2:4].clamp(min=1e-8, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4]

    if xywhr.dtype == torch.float16 or xywhr.dtype == torch.bfloat16:
         xy = xy.float()
         wh = wh.float()
         r = r.float()

    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2, 1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma

def postprocess(distance, fun='log1p', tau=1.0):
    """Convert distance to loss."""
    if fun == 'log1p':
        distance = torch.log1p(distance.clamp(min=-1.0 + 1e-6))
    elif fun == 'sqrt':
        distance = torch.sqrt(distance.clamp(min=1e-7))
    elif fun == 'none':
        pass
    else:
        raise ValueError(f'Invalid non-linear function {fun}')

    if tau >= 1.0:
        denom = tau + distance
        denom = torch.where(denom >= 0, denom.clamp(min=1e-6), denom.clamp(max=-1e-6))
        return 1 - 1 / denom
    else:
        return distance


def kld_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, sqrt=True):
    pred = pred.float()
    target = target.float()

    epsilon = 1e-8

    pred_rad = pred.clone()
    pred_rad[..., 4] = norm_angle_to_radian(pred[..., 4])

    target_rad = target.clone()
    target_rad[..., 4] = norm_angle_to_radian(target[..., 4])

    xy_p, Sigma_p = xy_wh_r_2_xy_sigma(pred_rad)
    xy_t, Sigma_t = xy_wh_r_2_xy_sigma(target_rad)

    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2).float()
    xy_t = xy_t.reshape(-1, 2).float()
    Sigma_p = Sigma_p.reshape(-1, 2, 2).float()
    Sigma_t = Sigma_t.reshape(-1, 2, 2).float()

    eye = torch.eye(2, device=Sigma_p.device, dtype=Sigma_p.dtype).unsqueeze(0)
    Sigma_p = Sigma_p + epsilon * eye
    Sigma_t = Sigma_t + epsilon * eye

    det_p = Sigma_p.det()
    Sigma_p_inv = torch.stack(
        (
            Sigma_p[..., 1, 1],
            -Sigma_p[..., 0, 1],
            -Sigma_p[..., 1, 0],
            Sigma_p[..., 0, 0],
        ),
        dim=-1,
    ).reshape(-1, 2, 2)
    Sigma_p_inv = Sigma_p_inv / det_p.unsqueeze(-1).unsqueeze(-1)

    dxy = (xy_p - xy_t).unsqueeze(-1)
    xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(dxy).view(-1)

    whr_distance = 0.5 * Sigma_p_inv.bmm(Sigma_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    Sigma_p_det_log = det_p.log()
    det_t = Sigma_t.det()
    Sigma_t_det_log = det_t.log()
    whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
    whr_distance = whr_distance - 1

    alpha_squared = max(alpha * alpha, epsilon)
    distance = (xy_distance / alpha_squared + whr_distance).clamp(min=0.0)
    if sqrt:
        distance = distance.clamp(min=1e-7).sqrt()

    distance = distance.reshape(_shape[:-1])
    return postprocess(distance.to(pred.dtype), fun=fun, tau=tau)

def box2corners_th(box):
    """
    box: (N, 5) or (B, N, 5) -> (cx, cy, w, h, angle_rad)
    Returns: (..., 4, 2)
    """
    shape = box.shape
    box = box.reshape(-1, 5)
    x_ctr, y_ctr, w, h, alpha = box.unbind(dim=-1)

    b = 0.5 * w
    # h = 0.5 * h # Typo in my thought, w and h are full width/height
    h_ = 0.5 * h

    # Corners in canonical frame
    # TL: (-b, h), TR: (b, h), BR: (b, -h), BL: (-b, -h)
    # Order: (-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)
    # (-b, -h_), (b, -h_), (b, h_), (-b, h_)

    x_corners = torch.stack([-b, b, b, -b], dim=1)
    y_corners = torch.stack([-h_, -h_, h_, h_], dim=1)
    corners = torch.stack([x_corners, y_corners], dim=-1) # (N, 4, 2)

    sin, cos = torch.sin(alpha), torch.cos(alpha)
    rot_matrix = torch.stack([
        torch.stack([cos, sin], dim=-1),
        torch.stack([-sin, cos], dim=-1)
    ], dim=-2) # (N, 2, 2)

    # Rotate
    # (N, 4, 2) @ (N, 2, 2) -> (N, 4, 2)
    rotated = torch.matmul(corners, rot_matrix)

    # Translate
    rotated[..., 0] += x_ctr.unsqueeze(1)
    rotated[..., 1] += y_ctr.unsqueeze(1)

    return rotated.reshape(shape[:-1] + (4, 2))

def hausdorff_loss(pred, target):
    """
    pred: (N, 5) (cx, cy, w, h, angle_norm)
    target: (M, 5) (cx, cy, w, h, angle_norm)
    """
    pred_rad = pred.clone()
    pred_rad[..., 4] = norm_angle_to_radian(pred[..., 4])

    target_rad = target.clone()
    target_rad[..., 4] = norm_angle_to_radian(target[..., 4])

    corners1 = box2corners_th(pred_rad) # (N, 4, 2)
    corners2 = box2corners_th(target_rad) # (M, 4, 2)

    N, P, _ = corners1.shape
    M, _, _ = corners2.shape

    c1 = corners1.reshape(N * P, 2)
    c2 = corners2.reshape(M * P, 2)

    c1_sq = torch.sum(c1 ** 2, dim=1, keepdim=True)
    c2_sq = torch.sum(c2 ** 2, dim=1, keepdim=True).t()

    # (N*P, M*P)
    dist_sq = torch.addmm(c1_sq + c2_sq, c1, c2.t(), alpha=-2.0, beta=1.0)
    dist_sq = torch.clamp(dist_sq, min=0.0)

    dist_view = dist_sq.view(N, P, M, P)

    # Direction 1: Pred -> GT
    min_dist_p2g, _ = dist_view.min(dim=3) # (N, P, M)
    d1_sq, _ = min_dist_p2g.max(dim=1)     # (N, M)

    # Direction 2: GT -> Pred
    min_dist_g2p, _ = dist_view.min(dim=1) # (N, M, P)
    d2_sq, _ = min_dist_g2p.max(dim=2)     # (N, M)

    bbox_cost_sq = torch.maximum(d1_sq, d2_sq)
    bbox_cost = torch.sqrt(bbox_cost_sq)

    return bbox_cost
