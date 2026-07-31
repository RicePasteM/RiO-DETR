"""
Pure Python replacements for mmrotate/mmcv evaluation utilities.
No dependency on mmcv, mmdet, mmengine, or mmrotate.

Acceleration: numba JIT (optional, auto-detected) + multiprocessing.
"""
import numpy as np
from collections import defaultdict
import os as _os
import multiprocessing as _mp

# Try to import numba for JIT acceleration
_numba = None
try:
    import numba as _numba
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

# Try to import torch for SAT pre-filter acceleration
_torch = None
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _has_numba():
    return _NUMBA_AVAILABLE


def _has_torch():
    return _TORCH_AVAILABLE


# ============================================================
#  1. Rotated Box → Polygon conversion
# ============================================================

def rbox_to_poly(rboxes):
    """Convert rotated boxes [cx, cy, w, h, angle] to polygons [x1,y1,...,x4,y4].
    angle is in radians, defined as the angle from the x-axis to the width (w) direction.
    (consistent with OpenCV / mmrotate convention)

    Args:
        rboxes: np.ndarray of shape (N, 5), each row = [cx, cy, w, h, angle]

    Returns:
        np.ndarray of shape (N, 8)
    """
    if rboxes.size == 0:
        return np.zeros((0, 8), dtype=rboxes.dtype)

    cx, cy, w, h, a = rboxes[:, 0], rboxes[:, 1], rboxes[:, 2], rboxes[:, 3], rboxes[:, 4]
    cos_a = np.cos(a)
    sin_a = np.sin(a)

    # half-dimensions along width and height directions
    w2 = w / 2.0
    h2 = h / 2.0

    # four corners in the local (rotated) frame, then rotate
    # corners: (w/2, h/2), (-w/2, h/2), (-w/2, -h/2), (w/2, -h/2)
    c1x = cx + w2 * cos_a - h2 * sin_a
    c1y = cy + w2 * sin_a + h2 * cos_a
    c2x = cx - w2 * cos_a - h2 * sin_a
    c2y = cy - w2 * sin_a + h2 * cos_a
    c3x = cx - w2 * cos_a + h2 * sin_a
    c3y = cy - w2 * sin_a - h2 * cos_a
    c4x = cx + w2 * cos_a + h2 * sin_a
    c4y = cy + w2 * sin_a - h2 * cos_a

    return np.stack([c1x, c1y, c2x, c2y, c3x, c3y, c4x, c4y], axis=-1)


def rbox_to_aabb(rboxes):
    """Compute axis-aligned enclosing boxes for rotated boxes.

    Returns:
        np.ndarray of shape (N, 4), each row = [xmin, ymin, xmax, ymax].
    """
    if rboxes.size == 0:
        return np.zeros((rboxes.shape[0], 4), dtype=rboxes.dtype)

    cx, cy, w, h, a = rboxes[:, 0], rboxes[:, 1], rboxes[:, 2], rboxes[:, 3], rboxes[:, 4]
    cos_a = np.abs(np.cos(a))
    sin_a = np.abs(np.sin(a))
    half_w = 0.5 * (w * cos_a + h * sin_a)
    half_h = 0.5 * (w * sin_a + h * cos_a)
    return np.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=-1)


def _poly_aabb(polys):
    if polys.size == 0:
        return np.zeros((polys.shape[0], 4), dtype=polys.dtype)
    xs = polys[:, 0::2]
    ys = polys[:, 1::2]
    return np.stack([xs.min(axis=1), ys.min(axis=1), xs.max(axis=1), ys.max(axis=1)], axis=-1)


def _aabb_overlap_mask(aabb, aabbs):
    return (
        (aabbs[:, 2] >= aabb[0]) &
        (aabbs[:, 0] <= aabb[2]) &
        (aabbs[:, 3] >= aabb[1]) &
        (aabbs[:, 1] <= aabb[3])
    )


# ============================================================
#  2. Polygon Intersection (Sutherland-Hodgman clipping)
# ============================================================

def _polygon_area(points):
    """Compute signed area of a polygon using the shoelace formula.
    points: list of (x, y) tuples. Counter-clockwise → positive area.
    """
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _line_intersection(p1, p2, edge_start, edge_end):
    """Compute intersection of line segment p1→p2 with the line through edge_start→edge_end.
    Returns the intersection point (x, y) or None if parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = edge_start
    x4, y4 = edge_end

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)
    return None


def _clip_polygon_by_halfplane(subject, edge_start, edge_end):
    """Clip polygon 'subject' by the half-plane defined by directed edge edge_start→edge_end.
    Keeps only the portion to the left of (or on) the edge.
    """
    if len(subject) == 0:
        return []

    output = []
    for i in range(len(subject)):
        current = subject[i]
        prev = subject[i - 1]

        # Check which side of the edge each point is on
        # Using cross product: (edge_dx, edge_dy) × (p - edge_start)
        edx = edge_end[0] - edge_start[0]
        edy = edge_end[1] - edge_start[1]

        def _inside(p):
            return (p[0] - edge_start[0]) * edy - (p[1] - edge_start[1]) * edx <= 1e-12

        curr_inside = _inside(current)
        prev_inside = _inside(prev)

        if curr_inside:
            if not prev_inside:
                inter = _line_intersection(prev, current, edge_start, edge_end)
                if inter is not None:
                    output.append(inter)
            output.append(current)
        elif prev_inside:
            inter = _line_intersection(prev, current, edge_start, edge_end)
            if inter is not None:
                output.append(inter)

    return output


def _point_in_convex_polygon(point, polygon, eps=1e-9):
    """Return whether a point lies inside a convex polygon.

    The check is orientation-agnostic, which is important because rotated box
    corners may be listed clockwise in image coordinates.
    """
    pos = False
    neg = False
    x, y = point
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross > eps:
            pos = True
        elif cross < -eps:
            neg = True
        if pos and neg:
            return False
    return True


# ============================================================
#  3b. PyTorch-based SAT pre-filter for rotated box IoU
# ============================================================

if _TORCH_AVAILABLE:

    def _rbox2corners_torch(boxes):
        """PyTorch: convert rotated boxes to 4 corner points.
        Args:
            boxes: Tensor (N, 5) [cx, cy, w, h, angle_rad]
        Returns:
            Tensor (N, 4, 2) corner coordinates
        """
        cx = boxes[:, 0]
        cy = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        a = boxes[:, 4]
        cos_a = _torch.cos(a)
        sin_a = _torch.sin(a)
        w2 = w * 0.5
        h2 = h * 0.5
        c0x = cx + w2 * cos_a - h2 * sin_a
        c0y = cy + w2 * sin_a + h2 * cos_a
        c1x = cx - w2 * cos_a - h2 * sin_a
        c1y = cy - w2 * sin_a + h2 * cos_a
        c2x = cx - w2 * cos_a + h2 * sin_a
        c2y = cy - w2 * sin_a - h2 * cos_a
        c3x = cx + w2 * cos_a + h2 * sin_a
        c3y = cy + w2 * sin_a - h2 * cos_a
        return _torch.stack([c0x, c0y, c1x, c1y, c2x, c2y, c3x, c3y], dim=-1).reshape(-1, 4, 2)


def _torch_sat_overlap_mask(boxes1_np, boxes2_np, device=None):
    """PyTorch SAT overlap test: returns (N, M) bool mask.

    Uses Separating Axis Theorem to quickly reject non-overlapping rotated boxes.
    For OBBs it is sufficient to check 4 axes: 2 edge directions from each box.
    This is a mathematically conservative pre-filter -- it NEVER marks truly
    non-overlapping pairs as overlapping, so correctness is preserved.

    Args:
        boxes1_np: np.ndarray (N, 5) [cx, cy, w, h, angle_rad]
        boxes2_np: np.ndarray (M, 5)
        device: torch device or None (uses CPU)
    Returns:
        np.ndarray (N, M) boolean mask, True where boxes might overlap
    """
    if not _TORCH_AVAILABLE:
        return np.ones((boxes1_np.shape[0], boxes2_np.shape[0]), dtype=bool)

    N, M = boxes1_np.shape[0], boxes2_np.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=bool)

    if device is None:
        device = _torch.device('cpu')

    b1 = _torch.from_numpy(boxes1_np).to(device).float()
    b2 = _torch.from_numpy(boxes2_np).to(device).float()

    corners1 = _rbox2corners_torch(b1)
    corners2 = _rbox2corners_torch(b2)

    overlap = _torch.ones(N, M, dtype=_torch.bool, device=device)
    eps = _torch.tensor(1e-9, device=device, dtype=_torch.float32)

    cos1, sin1 = _torch.cos(b1[:, 4]), _torch.sin(b1[:, 4])
    cos2, sin2 = _torch.cos(b2[:, 4]), _torch.sin(b2[:, 4])

    axes1_w = _torch.stack([cos1, sin1], dim=1)
    axes1_h = _torch.stack([-sin1, cos1], dim=1)
    axes2_w = _torch.stack([cos2, sin2], dim=1)
    axes2_h = _torch.stack([-sin2, cos2], dim=1)

    for ax in [axes1_w, axes1_h]:
        proj1 = (corners1 * ax.unsqueeze(1)).sum(-1)
        proj2 = _torch.einsum('mvk,nk->nmv', corners2, ax)
        min1 = proj1.min(1).values - eps
        max1 = proj1.max(1).values + eps
        min2 = proj2.min(2).values
        max2 = proj2.max(2).values
        overlap &= (max1.unsqueeze(1) >= min2) & (max2 >= min1.unsqueeze(1))

    for ax in [axes2_w, axes2_h]:
        proj1 = _torch.einsum('nvk,mk->nmv', corners1, ax)
        proj2 = (corners2 * ax.unsqueeze(1)).sum(-1)
        min1 = proj1.min(2).values
        max1 = proj1.max(2).values
        min2 = proj2.min(1).values - eps
        max2 = proj2.max(1).values + eps
        overlap &= (max1 >= min2.unsqueeze(0)) & (max2.unsqueeze(0) >= min1)

    return overlap.cpu().numpy()


def _segment_intersection(p1, p2, q1, q2, eps=1e-9):
    """Return the intersection point of two closed line segments, if any."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    r_x, r_y = x2 - x1, y2 - y1
    s_x, s_y = x4 - x3, y4 - y3
    denom = r_x * s_y - r_y * s_x
    if abs(denom) <= eps:
        return None
    qpx, qpy = x3 - x1, y3 - y1
    t = (qpx * s_y - qpy * s_x) / denom
    u = (qpx * r_y - qpy * r_x) / denom
    if -eps <= t <= 1.0 + eps and -eps <= u <= 1.0 + eps:
        return (x1 + t * r_x, y1 + t * r_y)
    return None


def _append_unique_point(points, point, eps=1e-7):
    for px, py in points:
        if abs(px - point[0]) <= eps and abs(py - point[1]) <= eps:
            return
    points.append(point)


def _polygon_intersection_area(poly1_pts, poly2_pts):
    """Compute intersection area of two convex quadrilaterals.

    This mirrors the geometry used by rotated IoU kernels: collect all polygon
    corners that lie inside the other polygon, plus all edge intersections, then
    sort the resulting convex polygon by angle around its centroid.
    """
    points = []
    for p in poly1_pts:
        if _point_in_convex_polygon(p, poly2_pts):
            _append_unique_point(points, p)
    for p in poly2_pts:
        if _point_in_convex_polygon(p, poly1_pts):
            _append_unique_point(points, p)
    for i in range(len(poly1_pts)):
        p1 = poly1_pts[i]
        p2 = poly1_pts[(i + 1) % len(poly1_pts)]
        for j in range(len(poly2_pts)):
            q1 = poly2_pts[j]
            q2 = poly2_pts[(j + 1) % len(poly2_pts)]
            inter = _segment_intersection(p1, p2, q1, q2)
            if inter is not None:
                _append_unique_point(points, inter)

    if len(points) < 3:
        return 0.0

    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    points.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))
    return _polygon_area(points)


# ============================================================
#  3. Rotated Box IoU
# ============================================================

if _NUMBA_AVAILABLE:

    @_numba.njit(cache=False, fastmath=True, nogil=True)
    def _intersection_area_numba(b1, b2):
        """Numba JIT: compute intersection area of two rotated boxes.
        b1, b2: (5,) arrays [cx, cy, w, h, angle]
        Returns: float (intersection area)
        """
        cx1, cy1, w1, h1, a1 = b1[0], b1[1], b1[2], b1[3], b1[4]
        cx2, cy2, w2, h2, a2 = b2[0], b2[1], b2[2], b2[3], b2[4]

        if w1 <= 0.0 or h1 <= 0.0 or w2 <= 0.0 or h2 <= 0.0:
            return 0.0

        cos1, sin1 = np.cos(a1), np.sin(a1)
        cos2, sin2 = np.cos(a2), np.sin(a2)

        w1h, h1h = w1 * 0.5, h1 * 0.5
        w2h, h2h = w2 * 0.5, h2 * 0.5

        c1 = np.empty((4, 2), dtype=np.float64)
        c1[0, 0] = cx1 + w1h * cos1 - h1h * sin1
        c1[0, 1] = cy1 + w1h * sin1 + h1h * cos1
        c1[1, 0] = cx1 - w1h * cos1 - h1h * sin1
        c1[1, 1] = cy1 - w1h * sin1 + h1h * cos1
        c1[2, 0] = cx1 - w1h * cos1 + h1h * sin1
        c1[2, 1] = cy1 - w1h * sin1 - h1h * cos1
        c1[3, 0] = cx1 + w1h * cos1 + h1h * sin1
        c1[3, 1] = cy1 + w1h * sin1 - h1h * cos1

        c2 = np.empty((4, 2), dtype=np.float64)
        c2[0, 0] = cx2 + w2h * cos2 - h2h * sin2
        c2[0, 1] = cy2 + w2h * sin2 + h2h * cos2
        c2[1, 0] = cx2 - w2h * cos2 - h2h * sin2
        c2[1, 1] = cy2 - w2h * sin2 + h2h * cos2
        c2[2, 0] = cx2 - w2h * cos2 + h2h * sin2
        c2[2, 1] = cy2 - w2h * sin2 - h2h * cos2
        c2[3, 0] = cx2 + w2h * cos2 + h2h * sin2
        c2[3, 1] = cy2 + w2h * sin2 - h2h * cos2

        # SAT early-exit: check 4 separating axes using already-computed corners
        sat_axes = np.empty((4, 2), dtype=np.float64)
        sat_axes[0, 0] = cos1
        sat_axes[0, 1] = sin1
        sat_axes[1, 0] = -sin1
        sat_axes[1, 1] = cos1
        sat_axes[2, 0] = cos2
        sat_axes[2, 1] = sin2
        sat_axes[3, 0] = -sin2
        sat_axes[3, 1] = cos2

        for ai in range(4):
            dx = sat_axes[ai, 0]
            dy = sat_axes[ai, 1]
            p1_0 = c1[0, 0] * dx + c1[0, 1] * dy
            p1_1 = c1[1, 0] * dx + c1[1, 1] * dy
            p1_2 = c1[2, 0] * dx + c1[2, 1] * dy
            p1_3 = c1[3, 0] * dx + c1[3, 1] * dy
            min1 = min(min(p1_0, p1_1), min(p1_2, p1_3))
            max1 = max(max(p1_0, p1_1), max(p1_2, p1_3))
            p2_0 = c2[0, 0] * dx + c2[0, 1] * dy
            p2_1 = c2[1, 0] * dx + c2[1, 1] * dy
            p2_2 = c2[2, 0] * dx + c2[2, 1] * dy
            p2_3 = c2[3, 0] * dx + c2[3, 1] * dy
            min2 = min(min(p2_0, p2_1), min(p2_2, p2_3))
            max2 = max(max(p2_0, p2_1), max(p2_2, p2_3))
            if max1 < min2 - 1e-12 or max2 < min1 - 1e-12:
                return 0.0

        output = np.empty((16, 2), dtype=np.float64)
        output_size = 0

        # Corners from c1 inside c2, and corners from c2 inside c1.
        for src in range(2):
            poly_a = c1 if src == 0 else c2
            poly_b = c2 if src == 0 else c1
            for pi in range(4):
                px = poly_a[pi, 0]
                py = poly_a[pi, 1]
                pos = False
                neg = False
                for ei in range(4):
                    ej = (ei + 1) % 4
                    ex = poly_b[ej, 0] - poly_b[ei, 0]
                    ey = poly_b[ej, 1] - poly_b[ei, 1]
                    cross = ex * (py - poly_b[ei, 1]) - ey * (px - poly_b[ei, 0])
                    if cross > 1e-9:
                        pos = True
                    elif cross < -1e-9:
                        neg = True
                    if pos and neg:
                        break
                if not (pos and neg):
                    duplicate = False
                    for k in range(output_size):
                        if abs(output[k, 0] - px) <= 1e-7 and abs(output[k, 1] - py) <= 1e-7:
                            duplicate = True
                            break
                    if not duplicate:
                        output[output_size, 0] = px
                        output[output_size, 1] = py
                        output_size += 1

        # Edge intersections.
        for i in range(4):
            i2 = (i + 1) % 4
            p1x = c1[i, 0]
            p1y = c1[i, 1]
            r_x = c1[i2, 0] - p1x
            r_y = c1[i2, 1] - p1y
            for j in range(4):
                j2 = (j + 1) % 4
                q1x = c2[j, 0]
                q1y = c2[j, 1]
                s_x = c2[j2, 0] - q1x
                s_y = c2[j2, 1] - q1y
                denom = r_x * s_y - r_y * s_x
                if abs(denom) <= 1e-12:
                    continue
                qpx = q1x - p1x
                qpy = q1y - p1y
                t = (qpx * s_y - qpy * s_x) / denom
                u = (qpx * r_y - qpy * r_x) / denom
                if t >= -1e-9 and t <= 1.0 + 1e-9 and u >= -1e-9 and u <= 1.0 + 1e-9:
                    ix = p1x + t * r_x
                    iy = p1y + t * r_y
                    duplicate = False
                    for k in range(output_size):
                        if abs(output[k, 0] - ix) <= 1e-7 and abs(output[k, 1] - iy) <= 1e-7:
                            duplicate = True
                            break
                    if not duplicate:
                        output[output_size, 0] = ix
                        output[output_size, 1] = iy
                        output_size += 1

        if output_size < 3:
            return 0.0

        center_x = 0.0
        center_y = 0.0
        for i in range(output_size):
            center_x += output[i, 0]
            center_y += output[i, 1]
        center_x /= output_size
        center_y /= output_size

        angles = np.empty(output_size, dtype=np.float64)
        for i in range(output_size):
            angles[i] = np.arctan2(output[i, 1] - center_y, output[i, 0] - center_x)

        for i in range(1, output_size):
            key_x = output[i, 0]
            key_y = output[i, 1]
            key_a = angles[i]
            j = i - 1
            while j >= 0 and angles[j] > key_a:
                output[j + 1, 0] = output[j, 0]
                output[j + 1, 1] = output[j, 1]
                angles[j + 1] = angles[j]
                j -= 1
            output[j + 1, 0] = key_x
            output[j + 1, 1] = key_y
            angles[j + 1] = key_a

        area = 0.0
        for i in range(output_size):
            j = (i + 1) % output_size
            area += output[i, 0] * output[j, 1] - output[j, 0] * output[i, 1]
        return abs(area) * 0.5


    @_numba.njit(cache=False, fastmath=True, nogil=True)
    def _rbox_iou_numba(boxes1, boxes2):
        """Numba JIT: compute pairwise IoU for two sets of rotated boxes."""
        N, M = boxes1.shape[0], boxes2.shape[0]
        ious = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            w1, h1 = boxes1[i, 2], boxes1[i, 3]
            area1 = w1 * h1
            if area1 <= 0.0:
                continue
            cos1_abs = abs(np.cos(boxes1[i, 4]))
            sin1_abs = abs(np.sin(boxes1[i, 4]))
            ext_x1 = 0.5 * (w1 * cos1_abs + h1 * sin1_abs)
            ext_y1 = 0.5 * (w1 * sin1_abs + h1 * cos1_abs)
            min_x1 = boxes1[i, 0] - ext_x1
            min_y1 = boxes1[i, 1] - ext_y1
            max_x1 = boxes1[i, 0] + ext_x1
            max_y1 = boxes1[i, 1] + ext_y1
            for j in range(M):
                w2, h2 = boxes2[j, 2], boxes2[j, 3]
                area2 = w2 * h2
                if area2 <= 0.0:
                    continue
                cos2_abs = abs(np.cos(boxes2[j, 4]))
                sin2_abs = abs(np.sin(boxes2[j, 4]))
                ext_x2 = 0.5 * (w2 * cos2_abs + h2 * sin2_abs)
                ext_y2 = 0.5 * (w2 * sin2_abs + h2 * cos2_abs)
                if (boxes2[j, 0] + ext_x2 < min_x1 or
                        boxes2[j, 0] - ext_x2 > max_x1 or
                        boxes2[j, 1] + ext_y2 < min_y1 or
                        boxes2[j, 1] - ext_y2 > max_y1):
                    continue
                inter = _intersection_area_numba(boxes1[i], boxes2[j])
                union = area1 + area2 - inter
                if union > 0.0:
                    ious[i, j] = inter / union
        return ious


    @_numba.njit(cache=False, fastmath=True, nogil=True)
    def _nms_rotated_indices_numba(boxes, scores, iou_thr):
        """Numba JIT: rotated NMS over all boxes in one kernel."""
        num_boxes = boxes.shape[0]
        order = np.argsort(scores)
        suppressed = np.zeros(num_boxes, dtype=np.uint8)
        keep = np.empty(num_boxes, dtype=np.int64)
        num_keep = 0

        areas = np.empty(num_boxes, dtype=np.float64)
        min_x = np.empty(num_boxes, dtype=np.float64)
        min_y = np.empty(num_boxes, dtype=np.float64)
        max_x = np.empty(num_boxes, dtype=np.float64)
        max_y = np.empty(num_boxes, dtype=np.float64)

        for i in range(num_boxes):
            w, h = boxes[i, 2], boxes[i, 3]
            areas[i] = w * h
            cos_abs = abs(np.cos(boxes[i, 4]))
            sin_abs = abs(np.sin(boxes[i, 4]))
            ext_x = 0.5 * (w * cos_abs + h * sin_abs)
            ext_y = 0.5 * (w * sin_abs + h * cos_abs)
            min_x[i] = boxes[i, 0] - ext_x
            min_y[i] = boxes[i, 1] - ext_y
            max_x[i] = boxes[i, 0] + ext_x
            max_y[i] = boxes[i, 1] + ext_y

        for oi in range(num_boxes - 1, -1, -1):
            i = order[oi]
            if suppressed[i] != 0:
                continue
            keep[num_keep] = i
            num_keep += 1

            if areas[i] <= 0.0:
                continue

            for oj in range(oi - 1, -1, -1):
                j = order[oj]
                if suppressed[j] != 0 or areas[j] <= 0.0:
                    continue
                if (max_x[j] < min_x[i] or min_x[j] > max_x[i] or
                        max_y[j] < min_y[i] or min_y[j] > max_y[i]):
                    continue
                inter = _intersection_area_numba(boxes[i], boxes[j])
                union = areas[i] + areas[j] - inter
                if union > 0.0 and inter / union >= iou_thr:
                    suppressed[j] = 1

        return keep[:num_keep]


    @_numba.njit(cache=False, fastmath=True, nogil=True)
    def _rbox_iou_numba_sat(boxes1, boxes2, sat_mask):
        """Numba JIT: compute pairwise IoU for rotated boxes, only for pairs
        where sat_mask[i,j] is True (SAT pre-filter pass).
        Single-threaded -- class-level parallelism is handled externally.
        """
        N, M = boxes1.shape[0], boxes2.shape[0]
        ious = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            w1, h1 = boxes1[i, 2], boxes1[i, 3]
            area1 = w1 * h1
            if area1 <= 0.0:
                continue
            for j in range(M):
                if not sat_mask[i, j]:
                    continue
                w2, h2 = boxes2[j, 2], boxes2[j, 3]
                area2 = w2 * h2
                if area2 <= 0.0:
                    continue
                inter = _intersection_area_numba(boxes1[i], boxes2[j])
                union = area1 + area2 - inter
                if union > 0.0:
                    ious[i, j] = inter / union
        return ious


_NUMBA_WARMED_UP = False


def warmup_numba():
    """Compile numba kernels on tiny inputs before the first real evaluation."""
    global _NUMBA_AVAILABLE, _NUMBA_WARMED_UP
    if not _NUMBA_AVAILABLE or _NUMBA_WARMED_UP:
        return _NUMBA_AVAILABLE
    try:
        boxes = np.array([[0, 0, 10, 4, 0], [1, 0, 10, 4, 0]], dtype=np.float64)
        scores = np.array([0.9, 0.8], dtype=np.float64)
        _rbox_iou_numba(boxes, boxes)
        _nms_rotated_indices_numba(boxes, scores, 0.5)
        if _TORCH_AVAILABLE:
            mask = _torch_sat_overlap_mask(boxes, boxes)
            _rbox_iou_numba_sat(boxes, boxes, mask)
        _NUMBA_WARMED_UP = True
        return True
    except Exception as exc:
        print(f"Warning: numba acceleration disabled: {exc}")
        _NUMBA_AVAILABLE = False
        _NUMBA_WARMED_UP = False
        return False


def rbox_iou(boxes1, boxes2, use_torch_sat=False):
    """Compute pairwise IoU between two sets of rotated boxes.
    Uses numba acceleration (with embedded SAT early-exit) if available,
    otherwise pure Python.

    Args:
        boxes1: np.ndarray (N, 5) [cx, cy, w, h, angle_in_radians]
        boxes2: np.ndarray (M, 5)
        use_torch_sat: if True and numba+torch available and GPU available,
            use PyTorch SAT pre-filter (beneficial for large N*M on GPU)
    Returns:
        np.ndarray (N, M) of IoU values
    """
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    N, M = boxes1.shape[0], boxes2.shape[0]

    # --- Torch SAT pre-filter (GPU only, for large matrices) ---
    if (use_torch_sat and _TORCH_AVAILABLE and _NUMBA_AVAILABLE
            and _torch.cuda.is_available() and N * M > 1000):
        if not _NUMBA_WARMED_UP:
            warmup_numba()
        if _NUMBA_AVAILABLE:
            boxes1_f64 = boxes1.astype(np.float64, copy=False)
            boxes2_f64 = boxes2.astype(np.float64, copy=False)
            sat_mask = _torch_sat_overlap_mask(
                boxes1_f64, boxes2_f64, device=_torch.device('cuda'))
            if not sat_mask.any():
                return np.zeros((N, M), dtype=np.float32)
            return _rbox_iou_numba_sat(boxes1_f64, boxes2_f64, sat_mask)

    # --- Optimized numba path (with embedded SAT early-exit) ---
    if _NUMBA_AVAILABLE and N * M > 100:
        if not _NUMBA_WARMED_UP:
            warmup_numba()
        if _NUMBA_AVAILABLE:
            boxes1 = boxes1.astype(np.float64, copy=False)
            boxes2 = boxes2.astype(np.float64, copy=False)
            return _rbox_iou_numba(boxes1, boxes2)

    # Pure Python fallback
    polys1 = rbox_to_poly(boxes1)
    polys2 = rbox_to_poly(boxes2)
    aabbs1 = _poly_aabb(polys1)
    aabbs2 = _poly_aabb(polys2)
    ious = np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)
    areas1 = boxes1[:, 2] * boxes1[:, 3]
    areas2 = boxes2[:, 2] * boxes2[:, 3]
    pts2_list = [
        [(polys2[j, 2*k], polys2[j, 2*k+1]) for k in range(4)]
        for j in range(boxes2.shape[0])
    ]

    for i in range(boxes1.shape[0]):
        pts1 = [(polys1[i, 2*j], polys1[i, 2*j+1]) for j in range(4)]
        if areas1[i] <= 0:
            continue
        candidate_inds = np.nonzero(_aabb_overlap_mask(aabbs1[i], aabbs2))[0]
        for j in candidate_inds:
            if areas2[j] <= 0:
                continue
            inter_area = _polygon_intersection_area(pts1, pts2_list[j])
            union = areas1[i] + areas2[j] - inter_area
            if union > 0:
                ious[i, j] = inter_area / union
    return ious


# ============================================================
#  4. Rotated NMS
# ============================================================

def nms_rotated(boxes, scores, iou_thr=0.1):
    """Pure Python rotated NMS. Accepts numpy or torch tensors.
    Args:
        boxes: np.ndarray or torch.Tensor (N, 5) [cx, cy, w, h, angle_in_radians]
        scores: np.ndarray or torch.Tensor (N,)
        iou_thr: float, IoU threshold for suppression

    Returns:
        keep_dets: same type as input, shape (K, 6), [cx, cy, w, h, angle, score]
        keep_indices: np.ndarray (K,) if numpy input, torch.Tensor if torch input
    """
    # Accept torch tensors by converting to numpy
    is_torch = hasattr(boxes, 'detach') and hasattr(boxes, 'cpu')
    if is_torch:
        orig_device = boxes.device
        boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
    else:
        orig_device = None

    if boxes.size == 0:
        if is_torch:
            import torch as _torch
            return _torch.empty((0, 6), device=orig_device, dtype=_torch.float32), _torch.empty(0, device=orig_device, dtype=_torch.int64)
        return np.zeros((0, 6), dtype=boxes.dtype), np.array([], dtype=np.int64)

    if _NUMBA_AVAILABLE and not _NUMBA_WARMED_UP:
        warmup_numba()

    if _NUMBA_AVAILABLE:
        keep_indices = _nms_rotated_indices_numba(
            boxes.astype(np.float64, copy=False),
            scores.astype(np.float64, copy=False),
            float(iou_thr))
    else:
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            if order.size == 1:
                break

            # Compute IoU of the highest-scoring box with the rest
            ious = rbox_iou(boxes[i:i+1], boxes[order[1:]])[0]
            inds = np.where(ious < iou_thr)[0]
            order = order[inds + 1]

        keep_indices = np.array(keep, dtype=np.int64)
    keep_boxes = boxes[keep_indices]
    keep_scores = scores[keep_indices].reshape(-1, 1)
    keep_dets = np.concatenate([keep_boxes, keep_scores], axis=1)
    if is_torch:
        import torch as _torch
        keep_dets = _torch.from_numpy(keep_dets).to(orig_device)
        keep_indices = _torch.from_numpy(keep_indices).to(orig_device)
    return keep_dets, keep_indices


# ============================================================
#  4b. Quadrilateral IoU and NMS
# ============================================================

def _qbox_to_poly_list(qboxes):
    """Convert quad boxes (N, 8) to list of polygon point lists.
    Each qbox: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    polys = []
    for i in range(qboxes.shape[0]):
        pts = [(qboxes[i, 2*j], qboxes[i, 2*j+1]) for j in range(4)]
        polys.append(pts)
    return polys


def _polygon_union_area(poly_pts):
    """Compute area of polygon from list of (x, y) tuples."""
    n = len(poly_pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = poly_pts[i]
        x2, y2 = poly_pts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def qbox_iou(qboxes1, qboxes2):
    """Compute pairwise IoU between two sets of quadrilateral boxes.
    Args:
        qboxes1: np.ndarray (N, 8)
        qboxes2: np.ndarray (M, 8)
    Returns:
        np.ndarray (N, M) of IoU values
    """
    if qboxes1.size == 0 or qboxes2.size == 0:
        return np.zeros((qboxes1.shape[0], qboxes2.shape[0]), dtype=np.float32)

    polys1 = _qbox_to_poly_list(qboxes1)
    polys2 = _qbox_to_poly_list(qboxes2)
    aabbs1 = _poly_aabb(qboxes1)
    aabbs2 = _poly_aabb(qboxes2)

    areas1 = np.array([_polygon_union_area(p) for p in polys1], dtype=np.float32)
    areas2 = np.array([_polygon_union_area(p) for p in polys2], dtype=np.float32)

    ious = np.zeros((len(polys1), len(polys2)), dtype=np.float32)

    for i in range(len(polys1)):
        if areas1[i] <= 0:
            continue
        candidate_inds = np.nonzero(_aabb_overlap_mask(aabbs1[i], aabbs2))[0]
        for j in candidate_inds:
            if areas2[j] <= 0:
                continue
            inter_area = _polygon_intersection_area(polys1[i], polys2[j])
            union = areas1[i] + areas2[j] - inter_area
            if union > 0:
                ious[i, j] = inter_area / union

    return ious


def nms_quadri(qboxes, scores, iou_thr=0.1):
    """Pure Python quadrilateral NMS. Accepts numpy or torch tensors.
    Args:
        qboxes: np.ndarray or torch.Tensor (N, 8) [x1,y1,...,x4,y4]
        scores: np.ndarray or torch.Tensor (N,)
        iou_thr: float
    Returns:
        keep_dets: same type as input, shape (K, 9), [x1, y1, ..., x4, y4, score]
        keep_indices: same type as input
    """
    is_torch = hasattr(qboxes, 'detach') and hasattr(qboxes, 'cpu')
    if is_torch:
        orig_device = qboxes.device
        qboxes = qboxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
    else:
        orig_device = None

    if qboxes.size == 0:
        if is_torch:
            import torch as _torch
            return _torch.empty((0, 9), device=orig_device, dtype=_torch.float32), _torch.empty(0, device=orig_device, dtype=_torch.int64)
        return np.zeros((0, 9), dtype=qboxes.dtype), np.array([], dtype=np.int64)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = qbox_iou(qboxes[i:i+1], qboxes[order[1:]])[0]
        inds = np.where(ious < iou_thr)[0]
        order = order[inds + 1]

    keep_indices = np.array(keep, dtype=np.int64)
    keep_qboxes = qboxes[keep_indices]
    keep_scores = scores[keep_indices].reshape(-1, 1)
    keep_dets = np.concatenate([keep_qboxes, keep_scores], axis=1)
    if is_torch:
        import torch as _torch
        keep_dets = _torch.from_numpy(keep_dets).to(orig_device)
        keep_indices = _torch.from_numpy(keep_indices).to(orig_device)
    return keep_dets, keep_indices


# ============================================================
#  5. Average Precision calculation
# ============================================================

def _average_precision(recalls, precisions, use_07_metric=False):
    """Compute Average Precision given recall and precision values.
    Args:
        recalls: np.ndarray, cumulative recall values (monotonically increasing)
        precisions: np.ndarray, corresponding precision values
        use_07_metric: bool, if True use VOC2007 11-point interpolation,
                       otherwise compute area under PR curve

    Returns:
        float: Average Precision
    """
    if use_07_metric:
        # 11-point interpolation
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            mask = recalls >= t
            if np.any(mask):
                p = np.max(precisions[mask])
            else:
                p = 0.0
            ap += p / 11.0
        return float(np.clip(ap, 0.0, 1.0))
    else:
        # Area under PR curve (AUC)
        # Insert sentinel values
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        # Make precision monotonically decreasing
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])

        # Compute AUC by summing rectangular areas
        ap = 0.0
        for i in range(len(mrec) - 1):
            if mrec[i + 1] > mrec[i]:
                ap += (mrec[i + 1] - mrec[i]) * mpre[i + 1]
        return float(np.clip(ap, 0.0, 1.0))


# ============================================================
#  6. TP/FP computation
# ============================================================

def _tpfp_default(det_bboxes, det_scores, gt_bboxes, gt_bboxes_ignore, iou_thr):
    """Compute true positives and false positives for a single class.
    Args:
        det_bboxes: np.ndarray (N, 5) detected rotated boxes
        det_scores: np.ndarray (N,) detection confidence scores
        gt_bboxes: np.ndarray (M, 5) ground truth rotated boxes
        gt_bboxes_ignore: np.ndarray (M2, 5) ignored ground truth rotated boxes
        iou_thr: IoU threshold for matching

    Returns:
        tp: np.ndarray (N,) 1 if TP, 0 if FP
        fp: np.ndarray (N,) 1 if FP, 0 if TP
    """
    if det_bboxes.size == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    N = len(det_bboxes)

    order = det_scores.argsort()[::-1]
    tp = np.zeros(N, dtype=np.float32)
    fp = np.zeros(N, dtype=np.float32)

    num_gts = gt_bboxes.shape[0]
    if gt_bboxes_ignore is None:
        gt_bboxes_ignore = np.zeros((0, 5), dtype=np.float32)
    num_gts_ignore = gt_bboxes_ignore.shape[0]

    if num_gts == 0 and num_gts_ignore == 0:
        fp[:] = 1.0
        return tp, fp

    all_gt_bboxes = gt_bboxes
    if num_gts_ignore > 0:
        all_gt_bboxes = np.concatenate([gt_bboxes, gt_bboxes_ignore], axis=0)

    ious = rbox_iou(det_bboxes, all_gt_bboxes)
    ious_max = ious.max(axis=1)
    ious_argmax = ious.argmax(axis=1)

    gt_matched = set()

    for det_idx in order:
        if ious_max[det_idx] < iou_thr:
            fp[det_idx] = 1.0
            continue

        gt_idx = ious_argmax[det_idx]
        if gt_idx < num_gts:
            if gt_idx in gt_matched:
                # GT already matched by a higher-scoring detection
                fp[det_idx] = 1.0
            else:
                tp[det_idx] = 1.0
                gt_matched.add(gt_idx)
        # Matches to ignored GTs are neither TP nor FP.

    return tp, fp


# ============================================================
#  7. eval_rbbox_map -- main evaluation function
# ============================================================

def _compute_cls_ap(args):
    """Compute AP for a single class (picklable for multiprocessing)."""
    cls_idx, class_dets_cls, class_gts_cls, class_gts_ignore_cls, iou_thr, use_07_metric, num_images = args
    warmup_numba()
    all_dets = np.concatenate(class_dets_cls, axis=0)
    num_gts = int(sum(len(g) for g in class_gts_cls))
    num_dets = int(all_dets.shape[0])
    empty_curve = np.zeros((0,), dtype=np.float32)

    if num_gts == 0:
        return cls_idx, {
            'num_gts': 0,
            'num_dets': num_dets,
            'recall': empty_curve,
            'precision': empty_curve,
            'ap': np.nan
        }
    if all_dets.size == 0:
        return cls_idx, {
            'num_gts': num_gts,
            'num_dets': 0,
            'recall': empty_curve,
            'precision': empty_curve,
            'ap': 0.0
        }

    all_tp, all_fp, all_scores_list = [], [], []
    for img_idx in range(num_images):
        img_dets = class_dets_cls[img_idx]
        img_gts = class_gts_cls[img_idx]
        img_gts_ignore = class_gts_ignore_cls[img_idx]
        if img_dets.shape[0] > 0:
            tp, fp = _tpfp_default(
                img_dets[:, :5].astype(np.float32),
                img_dets[:, 5].astype(np.float32),
                img_gts.astype(np.float32),
                img_gts_ignore.astype(np.float32),
                iou_thr)
            all_tp.append(tp)
            all_fp.append(fp)
            all_scores_list.append(img_dets[:, 5])
    if not all_tp:
        return cls_idx, {
            'num_gts': num_gts,
            'num_dets': num_dets,
            'recall': empty_curve,
            'precision': empty_curve,
            'ap': 0.0
        }

    all_tp = np.concatenate(all_tp)
    all_fp = np.concatenate(all_fp)
    all_scores = np.concatenate(all_scores_list)
    order = all_scores.argsort()[::-1]
    all_tp, all_fp = all_tp[order], all_fp[order]
    eps = 1e-16
    precisions = np.cumsum(all_tp) / (np.cumsum(all_tp) + np.cumsum(all_fp) + eps)
    recalls = np.cumsum(all_tp) / (num_gts + eps)
    ap = _average_precision(recalls, precisions, use_07_metric)
    return cls_idx, {
        'num_gts': num_gts,
        'num_dets': num_dets,
        'recall': recalls,
        'precision': precisions,
        'ap': ap
    }


def eval_rbbox_map(dets, gts, scale_ranges=None, iou_thr=0.5,
                   use_07_metric=True, dataset=None, logger=None, nproc=4,
                   use_torch_sat=False):
    """Evaluate mAP for rotated bounding boxes.

    Acceleration: numba JIT (with embedded SAT + optimized sort) + optional multiprocessing.

    The default nproc=4 uses ProcessPoolExecutor across classes. It first
    uses Python's default fork context for the fast copy-on-write path, while
    delaying numba warmup until worker processes so the parent does not
    initialize OpenMP immediately before forking. If the default context still
    fails, it retries once with forkserver. Set MMEVAL_MP_CONTEXT to force a
    specific start method.

    Args:
        dets: list[list[np.ndarray]] -- per-image list, each element is a list of
              per-class detections. Each detection array is (K, 6): [cx,cy,w,h,angle,score]
        gts: list[dict] -- per-image ground truth dicts
        scale_ranges: optional list of (min_size, max_size) tuples per class
        iou_thr: IoU threshold for positive match
        use_07_metric: if True, use VOC2007 11-point AP; else AUC
        dataset: list of class names or None
        nproc: number of parallel processes (0 or 1 = serial, >1 = ProcessPoolExecutor).
               Default 4 enables numba+mp4 when numba is available.
        use_torch_sat: if True and GPU available, use PyTorch SAT pre-filter
    """
    assert len(dets) == len(gts), f"Length mismatch: dets {len(dets)} vs gts {len(gts)}"

    num_classes = len(dets[0])
    num_images = len(dets)
    use_mp = nproc > 1 and num_classes > 1 and _os.cpu_count() > 1
    if not use_mp:
        warmup_numba()

    # Collect all detections and ground truths per class
    class_dets = [[] for _ in range(num_classes)]
    class_gts = [[] for _ in range(num_classes)]
    class_gts_ignore = [[] for _ in range(num_classes)]

    for img_idx in range(num_images):
        img_dets, img_gt = dets[img_idx], gts[img_idx]
        gt_bboxes = img_gt.get('bboxes', np.zeros((0, 5), dtype=np.float32))
        gt_labels = img_gt.get('labels', np.zeros((0,), dtype=np.int64))
        gt_bboxes_ignore = img_gt.get('bboxes_ignore', np.zeros((0, 5), dtype=np.float32))
        gt_labels_ignore = img_gt.get('labels_ignore', np.zeros((0,), dtype=np.int64))
        for cls_idx in range(num_classes):
            cls_det = img_dets[cls_idx]
            class_dets[cls_idx].append(cls_det if cls_det.size > 0 else np.zeros((0, 6), dtype=np.float32))
            cls_gt_mask = gt_labels == cls_idx
            class_gts[cls_idx].append(gt_bboxes[cls_gt_mask] if np.any(cls_gt_mask) else np.zeros((0, 5), dtype=np.float32))
            cls_gt_ignore_mask = gt_labels_ignore == cls_idx
            class_gts_ignore[cls_idx].append(
                gt_bboxes_ignore[cls_gt_ignore_mask]
                if np.any(cls_gt_ignore_mask) else np.zeros((0, 5), dtype=np.float32))

    # Choose serial or parallel
    # ProcessPoolExecutor parallelizes across classes. Leaving the context unset
    # uses fork on Linux, keeping large det/gt arrays on the copy-on-write path
    # and avoiding the forkserver/spawn serialization penalty.
    if use_mp:
        from concurrent.futures import ProcessPoolExecutor as _Executor
        try:
            from concurrent.futures.process import BrokenProcessPool as _BrokenProcessPool
        except Exception:
            _BrokenProcessPool = RuntimeError

        real_nproc = min(nproc, num_classes, _os.cpu_count())
        args_list = [(i, class_dets[i], class_gts[i], class_gts_ignore[i], iou_thr, use_07_metric, num_images)
                     for i in range(num_classes)]

        def _run_pool(context_name):
            pool_kwargs = {'max_workers': real_nproc}
            if context_name:
                try:
                    pool_kwargs['mp_context'] = _mp.get_context(context_name)
                except ValueError:
                    pool_kwargs['mp_context'] = _mp.get_context('spawn')
            results = [None] * num_classes
            with _Executor(**pool_kwargs) as pool:
                for cls_idx, cls_result in pool.map(_compute_cls_ap, args_list):
                    results[cls_idx] = cls_result
            return results

        mp_context_name = _os.environ.get('MMEVAL_MP_CONTEXT')
        try:
            eval_results = _run_pool(mp_context_name)
        except _BrokenProcessPool:
            if mp_context_name:
                raise
            print('mmeval: default process context failed; retrying with forkserver.')
            eval_results = _run_pool('forkserver')
    else:
        eval_results = [None] * num_classes
        for cls_idx in range(num_classes):
            name, cls_result = _compute_cls_ap((cls_idx, class_dets[cls_idx], class_gts[cls_idx], class_gts_ignore[cls_idx],
                                        iou_thr, use_07_metric, num_images))
            eval_results[name] = cls_result

    # Compute mAP (exclude NaN classes)
    valid_aps = [res['ap'] for res in eval_results if res is not None and not np.isnan(res['ap'])]
    mean_ap = float(np.mean(valid_aps)) if valid_aps else 0.0

    # Print results
    if dataset:
        _hybrid = _TORCH_AVAILABLE and _NUMBA_AVAILABLE and use_torch_sat
        print("\n" + "=" * 70)
        print(f"{'class':<20s} {'gts':>8s} {'dets':>8s} {'recall':>8s} {'ap':>8s}")
        print("-" * 70)
        for cls_idx, cls_name in enumerate(dataset):
            cls_result = eval_results[cls_idx]
            ap_val = cls_result['ap']
            recall_curve = cls_result['recall']
            recall_val = recall_curve[-1] if recall_curve.size > 0 else 0.0
            ap_str = 'N/A' if np.isnan(ap_val) else f'{ap_val:.4f}'
            print(
                f"{cls_name:<20s} {cls_result['num_gts']:>8d} "
                f"{cls_result['num_dets']:>8d} {recall_val:>8.4f} {ap_str:>8s}")
        print("-" * 70)
        print(f"{'mAP':<20s} {'':>8s} {'':>8s} {'':>8s} {mean_ap:>8.4f}")
        if _hybrid and nproc > 1:
            speed_note = f"torch-sat+numba+mp{nproc}"
        elif _hybrid:
            speed_note = "torch-sat+numba"
        elif _NUMBA_AVAILABLE:
            speed_note = "numba" + (f"+mp{nproc}" if nproc > 1 else "")
        elif nproc > 1:
            speed_note = f"mp{nproc}"
        else:
            speed_note = "serial"
        print(f"(mmeval: {speed_note})")
        print("=" * 70 + "\n")

    return mean_ap, eval_results
