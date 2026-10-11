"""Wire overlay and debug artifact helpers."""

import json
import os
from datetime import datetime

import cv2
import numpy as np

from app_paths import user_data_path
from detection_debug import make_contact_sheet

DEBUG_DIR_NAME = "WireDebug"


def draw_wires(image, wires, label_prefix="W"):
    result = image.copy()
    for wire in wires or []:
        if getattr(wire, "is_pick_candidate", True):
            color = (40, 230, 70)
            thickness = 3
        else:
            color = (170, 170, 170)
            thickness = 1
        _draw_wire(result, wire, color, label_prefix=label_prefix, thickness=thickness)
    return result


def draw_segments(image, segments, offset_x=0, offset_y=0):
    for segment in segments or []:
        color = _color_for_id(segment.id)
        points = segment.points + np.array([offset_x, offset_y], dtype=np.float32)
        points = np.asarray(points, dtype=np.int32)
        if len(points) >= 2:
            cv2.polylines(image, [points.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)
        for endpoint in segment.endpoints:
            x, y = (int(round(value)) for value in endpoint)
            cv2.circle(image, (x + offset_x, y + offset_y), 3, color, -1)
    return image


def save_debug_artifacts(frames, wires, root=None):
    sheet = make_contact_sheet(frames)
    if sheet is None:
        return None
    root = root or user_data_path(DEBUG_DIR_NAME)
    stamp = datetime.now().strftime("%y%m%d%H%M%S")
    folder = os.path.join(root, datetime.now().strftime("%y%m%d"))
    os.makedirs(folder, exist_ok=True)
    image_path = _unique_path(folder, f"wire_debug_{stamp}.png")
    meta_path = os.path.splitext(image_path)[0] + ".json"
    _write_png(image_path, sheet)
    with open(meta_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "wires": [
                    {
                        "id": wire.id,
                        "tracking_id": wire.tracking_id,
                        "representative_uv": list(wire.representative_uv),
                        "representative_xyz": (
                            list(wire.representative_xyz)
                            if wire.representative_xyz is not None else None
                        ),
                        "length_px": wire.length_px,
                        "thickness_px": wire.thickness_px,
                        "confidence": wire.confidence,
                        "confidence_parts": wire.confidence_parts,
                        "segment_count": wire.segment_count,
                        "depth_mm": wire.depth_mm,
                        "layer_index": wire.layer_index,
                        "is_pick_candidate": wire.is_pick_candidate,
                        "pick_score": wire.pick_score,
                    }
                    for wire in wires
                ]
            },
            fp,
            ensure_ascii=False,
            indent=2,
        )
    return image_path


def _draw_wire(image, wire, color, label_prefix="W", thickness=2):
    points = np.asarray(wire.points, dtype=np.int32)
    if len(points) >= 2:
        cv2.polylines(image, [points.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)
    u, v = (int(round(value)) for value in wire.representative_uv)
    radius = 6 if getattr(wire, "is_pick_candidate", True) else 4
    cv2.circle(image, (u, v), radius, (255, 255, 255), -1)
    cv2.circle(image, (u, v), radius + 2, color, 2)
    label_id = wire.tracking_id if wire.tracking_id is not None else wire.id
    label = f"{label_prefix}{label_id}"
    if getattr(wire, "layer_index", None) is not None:
        label += f" L{wire.layer_index + 1}"
    cv2.putText(
        image,
        label,
        (u + 8, v - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"{wire.confidence:.2f}",
        (u + 8, v + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def _color_for_id(index):
    palette = [
        (40, 80, 255),
        (40, 180, 255),
        (0, 220, 220),
        (70, 220, 70),
        (255, 180, 40),
        (255, 90, 180),
        (180, 100, 255),
        (255, 255, 255),
    ]
    return palette[(max(1, int(index or 1)) - 1) % len(palette)]


def _write_png(path, bgr):
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise OSError("디버그 이미지를 PNG로 변환하지 못했습니다.")
    with open(path, "wb") as fp:
        fp.write(buffer.tobytes())


def _unique_path(folder, filename):
    base, ext = os.path.splitext(filename)
    path = os.path.join(folder, filename)
    suffix = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{base}_{suffix}{ext}")
        suffix += 1
    return path
