"""Skeleton graph splitting and segment merging for wire detection."""

import cv2
import numpy as np

from wire_types import WireSegment


def skeletonize(mask):
    image = (mask > 0).astype(np.uint8)
    if np.count_nonzero(image) == 0:
        return image * 255
    changed = True
    while changed:
        before = image.copy()
        image = _zhang_suen_step(image, first=True)
        image = _zhang_suen_step(image, first=False)
        changed = not np.array_equal(before, image)
    return image.astype(np.uint8) * 255


def segments_from_skeleton(skeleton, distance, depth_roi, depth_scale_mm, config):
    skeleton_bool = skeleton > 0
    branch = skeleton_bool & (_transition_count(skeleton_bool) >= 3)
    split = skeleton_bool.copy()
    if np.any(branch):
        branch_u8 = branch.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        branch_u8 = cv2.dilate(branch_u8, kernel, iterations=1)
        split[branch_u8 > 0] = False

    count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(
        split.astype(np.uint8) * 255
    )
    min_segment_length = max(8.0, float(config.min_length_px) * 0.12)
    segments = []
    for label in range(1, count):
        ys, xs = np.nonzero(labels == label)
        if len(xs) < 2:
            continue
        points = np.column_stack([xs, ys]).astype(np.float32)
        ordered = order_points_roughly(points)
        length = polyline_length(ordered)
        if length < min_segment_length:
            continue
        thickness = float(np.median(distance[ys, xs]) * 2.0)
        if thickness <= 0.0:
            continue
        endpoints = (tuple(ordered[0]), tuple(ordered[-1]))
        directions = (
            _endpoint_direction(ordered, start=True),
            _endpoint_direction(ordered, start=False),
        )
        depth_mm = _segment_depth(depth_roi, ordered, depth_scale_mm)
        segments.append(WireSegment(
            id=len(segments) + 1,
            points=ordered,
            endpoints=endpoints,
            directions=directions,
            length_px=length,
            thickness_px=thickness,
            depth_mm=depth_mm,
        ))
    return segments


def merge_segments(segments, config):
    if not segments:
        return []
    parent = list(range(len(segments)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(segments)):
        for right in range(left + 1, len(segments)):
            if _segments_should_merge(segments[left], segments[right], config):
                union(left, right)

    groups = {}
    for index, segment in enumerate(segments):
        groups.setdefault(find(index), []).append(segment)
    return list(groups.values())


def polyline_length(points):
    if len(points) < 2:
        return 0.0
    ordered = order_points_roughly(points)
    diffs = np.diff(ordered, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def order_points_roughly(points):
    if len(points) < 3:
        return points
    centered = points - np.mean(points, axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    order = np.argsort(centered @ axis)
    return points[order]


def _segments_should_merge(left, right, config):
    max_distance = float(config.merge_distance_px)
    max_depth_gap = max(8.0, float(config.plane_threshold_mm) * 3.0)
    min_opposite_dot = -float(np.cos(np.deg2rad(float(config.merge_angle_deg))))
    best_score = None

    for left_index, left_endpoint in enumerate(left.endpoints):
        left_direction = left.directions[left_index]
        for right_index, right_endpoint in enumerate(right.endpoints):
            right_direction = right.directions[right_index]
            distance = float(np.linalg.norm(
                np.asarray(left_endpoint, dtype=np.float32)
                - np.asarray(right_endpoint, dtype=np.float32)
            ))
            if distance > max_distance:
                continue
            direction_dot = float(np.dot(left_direction, right_direction))
            if direction_dot > min_opposite_dot:
                continue
            if not _thickness_compatible(left.thickness_px, right.thickness_px):
                continue
            if (
                left.depth_mm is not None
                and right.depth_mm is not None
                and abs(left.depth_mm - right.depth_mm) > max_depth_gap
            ):
                continue
            score = distance + (direction_dot + 1.0) * max_distance
            if best_score is None or score < best_score:
                best_score = score
    return best_score is not None


def _zhang_suen_step(image, first):
    padded = np.pad(image, 1, mode="constant")
    p2 = padded[:-2, 1:-1]
    p3 = padded[:-2, 2:]
    p4 = padded[1:-1, 2:]
    p5 = padded[2:, 2:]
    p6 = padded[2:, 1:-1]
    p7 = padded[2:, :-2]
    p8 = padded[1:-1, :-2]
    p9 = padded[:-2, :-2]
    neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
    transitions = np.zeros_like(image, dtype=np.uint8)
    cycle = neighbors + (p2,)
    for left, right in zip(cycle, cycle[1:]):
        transitions += ((left == 0) & (right == 1)).astype(np.uint8)
    neighbor_sum = sum(neighbors)
    if first:
        condition_a = (p2 * p4 * p6) == 0
        condition_b = (p4 * p6 * p8) == 0
    else:
        condition_a = (p2 * p4 * p8) == 0
        condition_b = (p2 * p6 * p8) == 0
    remove = (
        (image == 1)
        & (neighbor_sum >= 2)
        & (neighbor_sum <= 6)
        & (transitions == 1)
        & condition_a
        & condition_b
    )
    result = image.copy()
    result[remove] = 0
    return result


def _transition_count(skeleton_bool):
    padded = np.pad(skeleton_bool.astype(np.uint8), 1, mode="constant")
    p2 = padded[:-2, 1:-1]
    p3 = padded[:-2, 2:]
    p4 = padded[1:-1, 2:]
    p5 = padded[2:, 2:]
    p6 = padded[2:, 1:-1]
    p7 = padded[2:, :-2]
    p8 = padded[1:-1, :-2]
    p9 = padded[:-2, :-2]
    neighbors = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
    transitions = np.zeros(skeleton_bool.shape, dtype=np.uint8)
    for left, right in zip(neighbors, neighbors[1:]):
        transitions += ((left == 0) & (right == 1)).astype(np.uint8)
    return transitions


def _endpoint_direction(ordered, start):
    if len(ordered) < 2:
        return np.array([1.0, 0.0], dtype=np.float32)
    if start:
        anchor = ordered[0]
        inner = ordered[min(len(ordered) - 1, 5)]
    else:
        anchor = ordered[-1]
        inner = ordered[max(0, len(ordered) - 6)]
    vector = np.asarray(inner - anchor, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return np.array([1.0, 0.0], dtype=np.float32)
    return vector / norm


def _segment_depth(depth_roi, points, depth_scale_mm):
    if depth_roi is None or len(points) == 0:
        return None
    depth = np.asarray(depth_roi)
    h, w = depth.shape[:2]
    rounded = np.rint(points).astype(np.int32)
    xs = np.clip(rounded[:, 0], 0, w - 1)
    ys = np.clip(rounded[:, 1], 0, h - 1)
    values = depth[ys, xs]
    values = values[values > 0]
    if values.size == 0:
        return None
    return float(np.median(values.astype(np.float64))) * float(depth_scale_mm)


def _thickness_compatible(left, right):
    if left <= 0.0 or right <= 0.0:
        return False
    average = 0.5 * (left + right)
    return abs(left - right) <= max(6.0, average * 0.75)
