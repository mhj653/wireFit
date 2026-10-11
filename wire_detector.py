"""Wire centerline detection pipeline."""

import cv2
import numpy as np

from detection_debug import DebugFrame, normalize_gray
from wire_debug_tools import draw_segments, draw_wires, save_debug_artifacts
from wire_graph import (
    merge_segments,
    order_points_roughly,
    segments_from_skeleton,
    skeletonize,
)
from wire_tracker import WireTracker
from wire_types import (
    REP_BOTTOM,
    REP_CENTER,
    REP_CHOICES,
    REP_FIRST_SORT,
    REP_LEFT,
    REP_RIGHT,
    REP_TOP,
    SORT_CHOICES,
    SORT_CUSTOM,
    SORT_X,
    SORT_Y,
    SORT_Z,
    WIRE_MODE_BIN_TOP,
    WIRE_MODE_CHOICES,
    WIRE_MODE_EDGE,
    WIRE_MODE_PLANE,
    WireCandidate,
    WireDetectionConfig,
    WireDetectionResult,
)


class WireDetector:
    def __init__(self, config=None, tracker=None):
        self.config = config or WireDetectionConfig()
        self.tracker = tracker or WireTracker()

    def analyze(
        self,
        bgr_image,
        depth=None,
        intrinsics=None,
        depth_scale_mm=1.0,
        roi=None,
    ):
        image, offset_x, offset_y, roi_applied = _crop_with_roi(bgr_image, roi)
        depth_roi = _crop_depth_to_color(
            depth, bgr_image.shape[:2], image.shape[:2], offset_x, offset_y
        )

        foreground, foreground_debug, foreground_label = self._foreground(
            image, depth_roi, depth_scale_mm
        )
        mask = self._wire_mask(foreground)
        skeleton = skeletonize(mask) if self.config.skeleton_enabled else mask.copy()
        wires, segments = self._extract_wires(
            skeleton, mask, depth_roi, depth_scale_mm, offset_x, offset_y
        )
        if self.config.foreground_mode == WIRE_MODE_BIN_TOP:
            _mark_top_layer_wires(wires, self.config)

        camera_matrix, dist_coeffs = intrinsics or (None, None)
        _attach_xyz(
            wires,
            depth,
            camera_matrix,
            dist_coeffs,
            bgr_image.shape[:2],
            depth_scale_mm,
        )
        self._score_wires(wires)
        self._sort_wires(wires)
        wires = wires[:max(0, int(self.config.max_wires))]
        for index, wire in enumerate(wires, 1):
            wire.id = index
        if self.config.tracking_enabled:
            self.tracker.update(wires, self.config.max_tracking_distance_px)
        pick_wires = [
            wire for wire in wires if getattr(wire, "is_pick_candidate", True)
        ]

        binary_full = np.zeros(bgr_image.shape[:2], dtype=np.uint8)
        h, w = mask.shape[:2]
        binary_full[offset_y:offset_y + h, offset_x:offset_x + w] = mask

        holes = [
            (int(round(wire.representative_uv[0])),
             int(round(wire.representative_uv[1])),
             max(3.0, float(wire.thickness_px) * 0.5))
            for wire in pick_wires
        ]

        frames = []
        if self.config.debug_enabled:
            frames = self._debug_frames(
                bgr_image, foreground_debug, foreground_label, mask, skeleton, wires, segments,
                offset_x, offset_y, roi_applied
            )

        debug_path = None
        if self.config.save_debug_images and frames:
            debug_path = save_debug_artifacts(frames, wires)

        note = f"wire {len(wires)} detected"
        if self.config.foreground_mode == WIRE_MODE_BIN_TOP:
            note += f" / pick {len(pick_wires)}"
        uncertain = [
            wire for wire in wires if wire.confidence < self.config.confidence_minimum
        ]
        if uncertain:
            note += f" / uncertain {len(uncertain)}"
        if depth is None:
            note += " (no depth: RGB fallback)"
        return WireDetectionResult(
            wires=wires,
            holes=holes,
            binary=binary_full,
            debug_frames=frames,
            status="ok" if wires else "empty",
            note=note,
            debug_path=debug_path,
        )

    def _foreground(self, image, depth_roi, depth_scale_mm):
        mode = self.config.foreground_mode
        if (
            self.config.use_depth_foreground
            and mode != WIRE_MODE_EDGE
            and depth_roi is not None
            and np.count_nonzero(depth_roi) > 100
        ):
            depth_mm = depth_roi.astype(np.float32) * float(depth_scale_mm)
            valid = _valid_depth_mask(depth_mm, self.config)
            if mode == WIRE_MODE_BIN_TOP:
                plane = _plane_foreground(depth_mm, valid, self.config)
                if plane is not None:
                    return plane[0], plane[1], "Plane candidates + top filter"
                top = _top_layer_foreground(depth_mm, valid, self.config)
                if top is not None:
                    return top[0], top[1], "Bin top fallback"
            elif mode == WIRE_MODE_PLANE:
                plane = _plane_foreground(depth_mm, valid, self.config)
                if plane is not None:
                    return plane[0], plane[1], "Plane height"

        if not self.config.use_edge_fallback:
            empty = np.zeros(image.shape[:2], dtype=np.uint8)
            return empty, empty, "No foreground"

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        foreground = cv2.dilate(edges, kernel, iterations=1)
        return foreground, foreground, "RGB edge fallback"

    def _wire_mask(self, foreground):
        mask = foreground.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if self.config.apply_morph_open:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        if self.config.apply_morph_close:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        if not self.config.filter_components:
            return mask
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)
        clean = np.zeros_like(mask)
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= int(self.config.min_area_px):
                clean[labels == label] = 255
        return clean

    def _extract_wires(self, skeleton, mask, depth_roi, depth_scale_mm, offset_x, offset_y):
        if skeleton is None or np.count_nonzero(skeleton) == 0:
            return [], []
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        segments = segments_from_skeleton(
            skeleton, distance, depth_roi, depth_scale_mm, self.config
        )
        groups = merge_segments(segments, self.config)
        candidates = []
        for group in groups:
            if not group:
                continue
            points_local = np.vstack([segment.points for segment in group])
            length = float(sum(segment.length_px for segment in group))
            if length < float(self.config.min_length_px):
                continue
            thickness_values = [
                segment.thickness_px for segment in group if segment.thickness_px > 0
            ]
            thickness = float(np.median(thickness_values)) if thickness_values else 0.0
            if not (self.config.min_thickness_px <= thickness <= self.config.max_thickness_px):
                continue
            points = order_points_roughly(points_local)
            depth_mm = _candidate_depth(depth_roi, points, depth_scale_mm)
            points = points + np.array([offset_x, offset_y], dtype=np.float32)
            representative = _representative_point(
                points, self.config.representative_mode, self.config
            )
            candidates.append(WireCandidate(
                id=0,
                points=points,
                representative_uv=representative,
                depth_mm=depth_mm,
                length_px=length,
                thickness_px=thickness,
                segment_count=len(group),
            ))
        return candidates, segments

    def _score_wires(self, wires):
        for wire in wires:
            parts = _confidence_parts(wire, self.config)
            wire.confidence_parts = parts
            wire.confidence = float(sum(parts.values()) / max(1, len(parts)))
            wire.pick_score = _pick_score(wire)

    def _sort_wires(self, wires):
        axis = self.config.sort_axis if self.config.sort_axis in SORT_CHOICES else SORT_X
        for wire in wires:
            wire.sort_key = float(_sort_value(wire, axis, self.config))
        wires.sort(key=lambda item: item.sort_key, reverse=bool(self.config.sort_desc))

    def _debug_frames(
        self, full_image, foreground_debug, foreground_label, mask, skeleton, wires, segments,
        offset_x, offset_y, roi_applied
    ):
        overlay = draw_wires(full_image, wires, label_prefix="W")
        pick_overlay = draw_wires(full_image, wires, label_prefix="P")
        cluster = full_image.copy()
        draw_wires(cluster, wires, label_prefix="W")
        segment_image = full_image.copy()
        draw_segments(segment_image, segments, offset_x, offset_y)
        roi_note = "ROI" if roi_applied else "full"
        frames = [
            DebugFrame("Original", full_image, roi_note),
            DebugFrame(f"{foreground_label} / foreground", foreground_debug),
            DebugFrame("Wire mask", mask),
            DebugFrame("Skeleton", skeleton),
            DebugFrame("Skeleton segments", segment_image),
            DebugFrame("Clusters", cluster),
        ]
        if any(getattr(wire, "layer_index", None) is not None for wire in wires):
            frames.append(DebugFrame("Depth layers / pick candidates", pick_overlay))
        frames.append(DebugFrame("Final sorted points", overlay))
        return frames


def _crop_with_roi(image, roi):
    if isinstance(roi, (list, tuple)) and len(roi) == 4:
        try:
            x, y, w, h = (int(value) for value in roi)
        except (TypeError, ValueError):
            pass
        else:
            height, width = image.shape[:2]
            x = max(0, min(x, width))
            y = max(0, min(y, height))
            w = max(0, min(w, width - x))
            h = max(0, min(h, height - y))
            if w > 0 and h > 0:
                return image[y:y + h, x:x + w], x, y, True
    return image, 0, 0, False


def _crop_depth_to_color(depth, color_shape, crop_shape, offset_x, offset_y):
    if depth is None:
        return None
    array = np.asarray(depth)
    color_h, color_w = color_shape
    if array.shape[:2] != (color_h, color_w):
        array = cv2.resize(array, (color_w, color_h), interpolation=cv2.INTER_NEAREST)
    crop_h, crop_w = crop_shape
    return array[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]


def _valid_depth_mask(depth_mm, config):
    valid = depth_mm > 0
    min_mm = float(config.depth_min_mm)
    max_mm = float(config.depth_max_mm)
    if min_mm > 0:
        valid &= depth_mm >= min_mm
    if max_mm > 0:
        valid &= depth_mm <= max_mm
    return valid


def _top_layer_foreground(depth_mm, valid, config):
    values = depth_mm[valid]
    if values.size < 100:
        return None
    percentile = float(np.clip(config.top_percentile, 0.1, 50.0))
    near_depth = float(np.percentile(values, percentile))
    band = max(1.0, float(config.top_band_mm))
    top_limit = near_depth + band
    foreground = np.zeros(depth_mm.shape, dtype=np.uint8)
    foreground[valid & (depth_mm <= top_limit)] = 255
    top_score = np.zeros(depth_mm.shape, dtype=np.float32)
    top_score[valid] = np.maximum(0.0, top_limit - depth_mm[valid])
    return foreground, normalize_gray(top_score)


def _plane_foreground(depth_mm, valid, config):
    plane = _fit_depth_plane(depth_mm, valid)
    if plane is None:
        return None
    yy, xx = np.indices(depth_mm.shape)
    plane_z = plane[0] * xx + plane[1] * yy + plane[2]
    residual = plane_z - depth_mm
    foreground = np.zeros(depth_mm.shape, dtype=np.uint8)
    foreground[valid & (residual > float(config.plane_threshold_mm))] = 255
    return foreground, normalize_gray(np.where(valid, residual, 0.0))


def _candidate_depth(depth_roi, points, depth_scale_mm):
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
    values_mm = values.astype(np.float64) * float(depth_scale_mm)
    return float(np.percentile(values_mm, 20.0))


def _mark_top_layer_wires(wires, config):
    depth_wires = [
        wire for wire in wires if wire.depth_mm is not None
    ]
    if not depth_wires:
        for wire in wires:
            wire.layer_index = None
            wire.is_pick_candidate = True
            wire.pick_score = _pick_score(wire)
        return

    layer_gap = max(4.0, float(config.top_band_mm))
    ordered = sorted(depth_wires, key=lambda wire: float(wire.depth_mm))
    layers = []
    for wire in ordered:
        depth = float(wire.depth_mm)
        if not layers or depth - layers[-1]["reference"] > layer_gap:
            layers.append({"reference": depth, "wires": [wire]})
        else:
            layers[-1]["wires"].append(wire)
            layer_depths = [float(item.depth_mm) for item in layers[-1]["wires"]]
            layers[-1]["reference"] = float(np.median(layer_depths))

    for layer_index, layer in enumerate(layers):
        for wire in layer["wires"]:
            wire.layer_index = layer_index

    percentile = float(np.clip(config.top_percentile, 0.1, 50.0))
    depths = np.array([float(wire.depth_mm) for wire in depth_wires], dtype=np.float64)
    top_reference = float(np.percentile(depths, percentile))
    top_limit = top_reference + max(1.0, float(config.top_band_mm))
    for wire in wires:
        if wire.depth_mm is None:
            wire.layer_index = None
            wire.is_pick_candidate = True
        else:
            wire.is_pick_candidate = float(wire.depth_mm) <= top_limit
        wire.pick_score = _pick_score(wire)


def _pick_score(wire):
    depth_score = 1.0
    if wire.layer_index is not None:
        depth_score = max(0.0, 1.0 - 0.2 * float(wire.layer_index))
    length_score = min(1.0, float(wire.length_px) / 250.0)
    confidence = float(getattr(wire, "confidence", 0.0) or 0.0)
    return float(0.45 * depth_score + 0.35 * length_score + 0.20 * confidence)


def _fit_depth_plane(depth_mm, valid):
    ys, xs = np.nonzero(valid)
    if len(xs) < 200:
        return None
    step = max(1, len(xs) // 8000)
    xs = xs[::step].astype(np.float32)
    ys = ys[::step].astype(np.float32)
    zs = depth_mm[ys.astype(np.int32), xs.astype(np.int32)].astype(np.float32)
    keep = np.ones_like(zs, dtype=bool)
    coeff = None
    for _ in range(4):
        if np.count_nonzero(keep) < 100:
            break
        a = np.column_stack([xs[keep], ys[keep], np.ones(np.count_nonzero(keep))])
        coeff, *_ = np.linalg.lstsq(a, zs[keep], rcond=None)
        pred = coeff[0] * xs + coeff[1] * ys + coeff[2]
        residual = zs - pred
        median = float(np.median(residual[keep]))
        spread = max(3.0, float(np.median(np.abs(residual[keep] - median))) * 2.5)
        keep = np.abs(residual - median) <= spread
    return coeff


def _representative_point(points, mode, config):
    ordered = order_points_roughly(np.asarray(points, dtype=np.float32))
    if mode == REP_TOP:
        picked = ordered[np.argmin(ordered[:, 1])]
    elif mode == REP_BOTTOM:
        picked = ordered[np.argmax(ordered[:, 1])]
    elif mode == REP_LEFT:
        picked = ordered[np.argmin(ordered[:, 0])]
    elif mode == REP_RIGHT:
        picked = ordered[np.argmax(ordered[:, 0])]
    elif mode == REP_FIRST_SORT:
        vector = np.asarray(config.sort_vector_uv, dtype=np.float32)
        if np.linalg.norm(vector) <= 1e-6:
            vector = np.array([1.0, 0.0], dtype=np.float32)
        scores = ordered @ vector
        index = np.argmax(scores) if config.sort_desc else np.argmin(scores)
        picked = ordered[index]
    else:
        picked = ordered[len(ordered) // 2]
    return (float(picked[0]), float(picked[1]))


def _attach_xyz(wires, depth, camera_matrix, dist_coeffs, color_shape, depth_scale_mm):
    if depth is None or camera_matrix is None:
        return
    depth_array = np.asarray(depth)
    color_h, color_w = color_shape
    if depth_array.shape[:2] != (color_h, color_w):
        depth_array = cv2.resize(
            depth_array, (color_w, color_h), interpolation=cv2.INTER_NEAREST
        )
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    coeffs = (
        np.zeros(5)
        if dist_coeffs is None
        else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    )
    for wire in wires:
        u, v = wire.representative_uv
        z = _local_depth(depth_array, int(round(u)), int(round(v)), depth_scale_mm)
        if z is None:
            continue
        point = np.array([[[float(u), float(v)]]], dtype=np.float64)
        norm = cv2.undistortPoints(point, matrix, coeffs).reshape(2)
        wire.representative_xyz = (
            float(norm[0] * z),
            float(norm[1] * z),
            float(z),
        )


def _local_depth(depth, u, v, depth_scale_mm):
    h, w = depth.shape[:2]
    left, right = max(0, u - 3), min(w, u + 4)
    top, bottom = max(0, v - 3), min(h, v + 4)
    patch = depth[top:bottom, left:right]
    values = patch[patch > 0]
    if values.size == 0:
        return None
    return float(np.median(values.astype(np.float64))) * float(depth_scale_mm)


def _sort_value(wire, axis, config):
    if axis == SORT_CUSTOM:
        if wire.representative_xyz is not None:
            vector = np.asarray(config.sort_vector_xyz, dtype=np.float64)
            point = np.asarray(wire.representative_xyz, dtype=np.float64)
            return float(point @ vector)
        vector = np.asarray(config.sort_vector_uv, dtype=np.float64)
        point = np.asarray(wire.representative_uv, dtype=np.float64)
        return float(point @ vector)
    if wire.representative_xyz is not None:
        return {"x": wire.representative_xyz[0],
                "y": wire.representative_xyz[1],
                "z": wire.representative_xyz[2]}[axis]
    uv = wire.representative_uv
    return uv[0] if axis in (SORT_X, SORT_Z) else uv[1]


def _confidence_parts(wire, config):
    length = min(1.0, wire.length_px / max(1.0, float(config.min_length_px) * 2.0))
    if wire.thickness_px <= 0:
        thickness = 0.0
    else:
        mid = 0.5 * (config.min_thickness_px + config.max_thickness_px)
        span = max(1.0, 0.5 * (config.max_thickness_px - config.min_thickness_px))
        thickness = max(0.0, 1.0 - abs(wire.thickness_px - mid) / span)
    continuity = min(1.0, 1.0 / max(1.0, float(wire.segment_count)) + 0.5)
    depth = 1.0 if wire.representative_xyz is not None else 0.45
    return {
        "length": float(length),
        "thickness": float(thickness),
        "continuity": float(continuity),
        "depth": float(depth),
    }
