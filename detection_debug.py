"""Shared detection debug-frame helpers.

Detection methods can attach intermediate images as DebugFrame instances.
The UI can then render them as one contact-sheet image without knowing the
details of each algorithm.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class DebugFrame:
    title: str
    image: np.ndarray
    note: str = ""


@dataclass
class ContactSheetCell:
    frame_index: int
    rect: tuple


def to_bgr(image):
    """Return a displayable BGR image."""
    if image is None:
        return None
    array = np.asarray(image)
    if array.ndim == 2:
        if array.dtype != np.uint8:
            array = normalize_gray(array)
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 3:
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array.copy()
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    return None


def normalize_gray(image):
    array = np.asarray(image)
    if array.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape[:2], dtype=np.uint8)
    low = float(np.percentile(finite, 2))
    high = float(np.percentile(finite, 98))
    if high <= low:
        high = low + 1.0
    scaled = (array.astype(np.float32) - low) * (255.0 / (high - low))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def make_contact_sheet(frames, cell_size=(420, 236), columns=2, output_size=None):
    """Build one BGR image containing all debug frames.

    When output_size is given, the sheet keeps that exact width/height so the
    image board does not resize when the user switches between color/debug
    views.
    """
    valid = [frame for frame in frames or [] if to_bgr(frame.image) is not None]
    if not valid:
        return None

    layout = contact_sheet_layout(
        frames,
        cell_size=cell_size,
        columns=columns,
        output_size=output_size,
    )
    width = layout["width"]
    height = layout["height"]
    header_h = layout["header_h"]
    cell_w = layout["cell_w"]
    cell_h = layout["cell_h"]
    sheet = np.full((height, width, 3), 32, dtype=np.uint8)

    for cell in layout["cells"]:
        x, y, tile_w, tile_h = cell.rect
        frame = frames[cell.frame_index]
        image = to_bgr(frame.image)
        if image is None:
            continue
        resized = resize_letterbox(image, (cell_w, cell_h))
        sheet[y + header_h:y + header_h + cell_h, x:x + cell_w] = resized
        cv2.rectangle(sheet, (x, y), (x + tile_w, y + header_h), (18, 18, 18), -1)
        cv2.putText(
            sheet,
            frame.title,
            (x + 10, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if frame.note:
            cv2.putText(
                sheet,
                frame.note,
                (x + 10, y + 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (210, 230, 255),
                1,
                cv2.LINE_AA,
            )
    return sheet


def contact_sheet_layout(frames, cell_size=(420, 236), columns=2, output_size=None):
    valid_indices = [
        index for index, frame in enumerate(frames or [])
        if to_bgr(frame.image) is not None
    ]
    if not valid_indices:
        return {"width": 0, "height": 0, "header_h": 0, "cell_w": 0,
                "cell_h": 0, "cells": []}

    header_h = 42
    pad = 12
    count = len(valid_indices)
    if output_size is None:
        cell_w, cell_h = cell_size
        rows = int(np.ceil(count / float(columns)))
        width = columns * cell_w + (columns + 1) * pad
        height = rows * (cell_h + header_h) + (rows + 1) * pad
    else:
        width, height = (int(output_size[0]), int(output_size[1]))
        width = max(64, width)
        height = max(64, height)
        columns = _choose_columns(count, width, height, pad, header_h)
        rows = int(np.ceil(count / float(columns)))
        cell_w = max(1, int((width - (columns + 1) * pad) / columns))
        row_h = max(1, int((height - (rows + 1) * pad) / rows))
        header_h = min(header_h, max(16, row_h // 4))
        cell_h = max(1, row_h - header_h)

    cells = []
    for display_index, frame_index in enumerate(valid_indices):
        row, col = divmod(display_index, columns)
        x = pad + col * (cell_w + pad)
        y = pad + row * (cell_h + header_h + pad)
        cells.append(ContactSheetCell(frame_index, (x, y, cell_w, cell_h + header_h)))
    return {
        "width": width,
        "height": height,
        "header_h": header_h,
        "cell_w": cell_w,
        "cell_h": cell_h,
        "cells": cells,
    }


def debug_frame_at(frames, point, output_size=None):
    if point is None:
        return None
    x, y = int(point[0]), int(point[1])
    layout = contact_sheet_layout(frames, output_size=output_size)
    for cell in layout["cells"]:
        left, top, width, height = cell.rect
        if left <= x < left + width and top <= y < top + height:
            return frames[cell.frame_index]
    return None


def _choose_columns(count, width, height, pad, header_h):
    target_aspect = 16.0 / 9.0
    best_columns = 1
    best_score = float("inf")
    for columns in range(1, count + 1):
        rows = int(np.ceil(count / float(columns)))
        cell_w = (width - (columns + 1) * pad) / float(columns)
        row_h = (height - (rows + 1) * pad) / float(rows)
        cell_h = row_h - header_h
        if cell_w <= 1 or cell_h <= 1:
            continue
        cell_aspect = cell_w / cell_h
        score = abs(np.log(cell_aspect / target_aspect))
        # Prefer fuller grids when aspect is similar; empty cells look odd.
        score += (columns * rows - count) * 0.03
        if score < best_score:
            best_score = score
            best_columns = columns
    return best_columns


def resize_letterbox(image, size):
    target_w, target_h = size
    h, w = image.shape[:2]
    if min(h, w) <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas
