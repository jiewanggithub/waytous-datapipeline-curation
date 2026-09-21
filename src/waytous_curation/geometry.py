from __future__ import annotations

import math

from .domain import Box3D, Features, Sample


def point_inside_box(point: tuple[float, float, float], box: Box3D) -> bool:
    """Return whether a point is inside a yaw-rotated 3D box."""
    dx = point[0] - box.center[0]
    dy = point[1] - box.center[1]
    dz = point[2] - box.center[2]
    cos_yaw = math.cos(-box.yaw)
    sin_yaw = math.sin(-box.yaw)
    local_x = dx * cos_yaw - dy * sin_yaw
    local_y = dx * sin_yaw + dy * cos_yaw
    return (
        abs(local_x) <= box.size[0] / 2.0
        and abs(local_y) <= box.size[1] / 2.0
        and abs(dz) <= box.size[2] / 2.0
    )


def axis_aligned_iou_3d(first: Box3D, second: Box3D) -> float:
    """Compute an axis-aligned approximation to 3D IoU (yaw is ignored)."""
    intersections = []
    for axis in range(3):
        first_min = first.center[axis] - first.size[axis] / 2.0
        first_max = first.center[axis] + first.size[axis] / 2.0
        second_min = second.center[axis] - second.size[axis] / 2.0
        second_max = second.center[axis] + second.size[axis] / 2.0
        intersections.append(max(0.0, min(first_max, second_max) - max(first_min, second_min)))
    intersection = intersections[0] * intersections[1] * intersections[2]
    union = first.volume + second.volume - intersection
    return 0.0 if union <= 0 else intersection / union


def extract_features(sample: Sample) -> Features:
    point_count = sum(point_inside_box(point, sample.bbox) for point in sample.points)
    density = point_count / sample.bbox.volume
    max_iou = max(
        (axis_aligned_iou_3d(sample.bbox, neighbor) for neighbor in sample.neighbor_boxes),
        default=0.0,
    )
    return Features(
        length=sample.bbox.size[0],
        width=sample.bbox.size[1],
        height=sample.bbox.size[2],
        volume=sample.bbox.volume,
        point_count=point_count,
        point_density=density,
        ground_offset=sample.bbox.bottom_z - sample.ground_z,
        max_iou_3d=max_iou,
    )
