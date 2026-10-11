"""Generate synthetic wire color/depth logs for offline testing.

The output matches the virtual camera folder layout:

    <output>/261011/
        color/color_*.png
        depth/depth_*.png
        depth_raw/depth_raw_*.png
        depth_scale.json

Run:
    python wire_sample_data.py
"""

import argparse
import json
import os
from datetime import datetime

import cv2
import numpy as np

from app_paths import user_data_path
from depth_view import colorize_depth
from image_saver import save_frames

WIDTH = 1280
HEIGHT = 720
DEPTH_SCALE_MM = 1.0
SAMPLE_DATE = datetime(2026, 10, 11, 10, 0, 0)


def _background(seed, palette):
    rng = np.random.default_rng(seed)
    y, x = np.indices((HEIGHT, WIDTH), dtype=np.float32)
    base = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    base[:] = np.array(palette, dtype=np.float32)

    vignette = 0.92 + 0.12 * (1.0 - np.hypot((x - WIDTH / 2) / WIDTH, (y - HEIGHT / 2) / HEIGHT))
    base *= vignette[..., None]
    base += 9.0 * np.sin(x / 38.0)[..., None]
    base += 5.0 * np.sin(y / 53.0)[..., None]
    base += rng.normal(0, 4.0, base.shape)

    for sx in range(180, 1130, 18):
        color = tuple(int(v) for v in np.clip(np.array(palette) * 0.82, 0, 255))
        cv2.line(base, (sx, 0), (sx, HEIGHT), color, 1, cv2.LINE_AA)

    for center, axes, alpha in [
        ((410, 190), (150, 85), 0.22),
        ((860, 260), (180, 110), 0.12),
    ]:
        overlay = base.copy()
        cv2.ellipse(overlay, center, axes, 0, 0, 360, (230, 235, 238), -1, cv2.LINE_AA)
        base = cv2.addWeighted(overlay, alpha, base, 1.0 - alpha, 0)

    depth = 310.0 + 0.012 * (x - WIDTH / 2) + 0.018 * (y - HEIGHT / 2)
    depth += rng.normal(0, 0.7, depth.shape)
    return np.clip(base, 0, 255).astype(np.uint8), depth.astype(np.float32)


def _draw_label(img, center, angle):
    box = cv2.boxPoints((center, (120, 34), angle)).astype(np.int32)
    cv2.drawContours(img, [box], 0, (215, 215, 205), -1, cv2.LINE_AA)
    cv2.drawContours(img, [box], 0, (160, 160, 150), 1, cv2.LINE_AA)
    cv2.line(img, tuple(box[0]), tuple(box[2]), (160, 160, 150), 1, cv2.LINE_AA)


def _set_depth_mask(depth, mask, closer_mm):
    target = depth - float(closer_mm)
    depth[mask > 0] = np.minimum(depth[mask > 0], target[mask > 0])


def _draw_connector(img, depth, center, angle, size=(38, 56), depth_closer_mm=0.0):
    mask = np.zeros(depth.shape, dtype=np.uint8)
    center = (float(center[0]), float(center[1]))
    box = cv2.boxPoints((center, size, angle)).astype(np.int32)
    cv2.drawContours(mask, [box], 0, 255, -1, cv2.LINE_AA)
    cv2.drawContours(img, [box], 0, (28, 33, 35), -1, cv2.LINE_AA)
    cv2.drawContours(img, [box], 0, (94, 104, 108), 2, cv2.LINE_AA)
    cv2.circle(img, tuple(np.round(center).astype(int)), 4, (85, 95, 98), -1, cv2.LINE_AA)
    if depth_closer_mm > 0.0:
        _set_depth_mask(depth, mask, depth_closer_mm)


def _draw_wire(img, depth, points, thickness=18, closer_mm=18.0, tape=True):
    pts = np.array(points, dtype=np.int32)
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.polylines(mask, [pts], False, 255, thickness, cv2.LINE_AA)
    _set_depth_mask(depth, mask, closer_mm)

    cv2.polylines(img, [pts], False, (30, 29, 26), thickness, cv2.LINE_AA)
    cv2.polylines(img, [pts], False, (75, 68, 58), max(2, thickness // 5), cv2.LINE_AA)
    cv2.polylines(img, [pts], False, (10, 10, 10), max(1, thickness // 8), cv2.LINE_AA)

    if tape:
        for idx in (0, -1):
            center = pts[idx]
            cv2.circle(img, tuple(center), max(8, thickness), (30, 125, 69), -1, cv2.LINE_AA)
            tm = np.zeros(depth.shape, dtype=np.uint8)
            cv2.circle(tm, tuple(center), max(8, thickness), 255, -1, cv2.LINE_AA)
            _set_depth_mask(depth, tm, closer_mm + 2.0)

    _draw_connector(img, depth, pts[0], -25, size=(34, 48))
    _draw_connector(img, depth, pts[-1], 20, size=(34, 48))


def _draw_fixture(img, depth):
    cv2.line(img, (505, 335), (860, 310), (29, 34, 33), 13, cv2.LINE_AA)
    cv2.rectangle(img, (782, 302), (793, 320), (58, 73, 74), -1, cv2.LINE_AA)
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.line(mask, (505, 335), (860, 310), 255, 15, cv2.LINE_AA)
    _set_depth_mask(depth, mask, 2.0)


def _single_wire():
    img, depth = _background(4, (25, 124, 169))
    _draw_label(img, (130, 160), 48)
    _draw_label(img, (230, 415), 20)
    _draw_label(img, (235, 650), -12)
    _draw_fixture(img, depth)
    _draw_wire(
        img,
        depth,
        [(735, 35), (706, 150), (682, 300), (650, 430), (656, 535)],
        thickness=17,
        closer_mm=18.0,
    )
    cv2.circle(img, (748, 725), 90, (54, 61, 53), -1, cv2.LINE_AA)
    return img, np.clip(depth, 0, 65535).astype(np.uint16)


def _multi_wire():
    img, depth = _background(8, (36, 133, 162))
    _draw_label(img, (130, 160), 48)
    _draw_label(img, (230, 415), 20)
    _draw_fixture(img, depth)

    starts = [(440 + i * 34, 648 - (i % 3) * 18) for i in range(13)]
    ends = [(555 + i * 28, 26 + (i % 4) * 11) for i in range(13)]
    for i, (start, end) in enumerate(zip(starts, ends)):
        mid_x = int((start[0] + end[0]) / 2 + np.sin(i) * 38)
        points = [start, (mid_x, 455 - i * 8), (mid_x + 18, 250 - i * 5), end]
        _draw_wire(img, depth, points, thickness=16 + (i % 2), closer_mm=17.0 + (i % 3))

    _draw_wire(img, depth, [(820, 640), (785, 510), (752, 360), (760, 220), (820, 72)], thickness=15)
    cv2.circle(img, (760, 725), 88, (55, 60, 52), -1, cv2.LINE_AA)
    return img, np.clip(depth, 0, 65535).astype(np.uint16)


def _gray_table_multi_wire():
    img, depth = _background(12, (116, 122, 112))
    _draw_fixture(img, depth)
    for points in [
        [(330, 630), (410, 490), (520, 340), (590, 105)],
        [(430, 650), (490, 500), (565, 335), (650, 85)],
        [(560, 645), (600, 480), (675, 315), (780, 92)],
        [(690, 620), (700, 470), (760, 335), (920, 130)],
    ]:
        _draw_wire(img, depth, points, thickness=20, closer_mm=20.0)
    return img, np.clip(depth, 0, 65535).astype(np.uint16)


def generate_samples(output_root=None):
    output_root = output_root or user_data_path("SampleData", "WireSamples")
    samples = [
        ("single_wire", _single_wire),
        ("multi_wire", _multi_wire),
        ("multi_wire_gray_table", _gray_table_multi_wire),
    ]
    manifest = []
    for index, (name, factory) in enumerate(samples, start=1):
        color, depth = factory()
        ok, saved = save_frames(
            output_root,
            color_image=color,
            depth_image=colorize_depth(depth),
            raw_depth_image=depth,
            when=SAMPLE_DATE.replace(second=index),
            depth_scale_mm=DEPTH_SCALE_MM,
        )
        if not ok:
            raise OSError(saved)
        manifest.append({"name": name, "files": saved})

    folder = os.path.join(output_root, SAMPLE_DATE.strftime("%y%m%d"))
    manifest_path = os.path.join(folder, "sample_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "description": "Synthetic wire color/depth frames for virtual camera tests.",
                "depth_scale_mm": DEPTH_SCALE_MM,
                "samples": manifest,
            },
            fp,
            indent=2,
            ensure_ascii=False,
        )
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default=None,
        help="Output root. The selectable virtual-camera folder is created below it.",
    )
    args = parser.parse_args()
    folder = generate_samples(args.out)
    print(folder)


if __name__ == "__main__":
    main()
