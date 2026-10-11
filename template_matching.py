"""OpenCV 템플릿 매칭 기반 홀 후보 검출."""

import cv2
import numpy as np

DEFAULT_MATCH_THRESHOLD = 0.75
MAX_MATCHES = 100


class TemplateMatcher:
    def __init__(self, template, threshold=DEFAULT_MATCH_THRESHOLD):
        if template is None or template.size == 0:
            raise ValueError("템플릿 이미지가 없습니다.")
        self.template = _gray(template)
        self.threshold = min(max(float(threshold), 0.0), 1.0)

    def analyze(self, bgr_image, roi=None):
        gray = _gray(bgr_image)
        target, offset_x, offset_y = _crop(gray, roi)
        template_h, template_w = self.template.shape[:2]
        if target.shape[0] < template_h or target.shape[1] < template_w:
            return [], {"binary": np.zeros(gray.shape, dtype=np.uint8)}

        response = cv2.matchTemplate(target, self.template, cv2.TM_CCOEFF_NORMED)
        response = np.nan_to_num(response, nan=-1.0)
        ys, xs = np.where(response >= self.threshold)
        candidates = sorted(
            ((float(response[y, x]), int(x), int(y)) for y, x in zip(ys, xs)),
            reverse=True,
        )

        accepted = []
        for score, x, y in candidates:
            box = (x, y, template_w, template_h)
            if any(_intersection_over_union(box, item[1]) > 0.3 for item in accepted):
                continue
            accepted.append((score, box))
            if len(accepted) >= MAX_MATCHES:
                break

        radius = min(template_w, template_h) / 2.0
        holes = [
            (int(round(offset_x + x + template_w / 2.0)),
             int(round(offset_y + y + template_h / 2.0)),
             radius)
            for _score, (x, y, _width, _height) in accepted
        ]
        binary = np.zeros(gray.shape, dtype=np.uint8)
        for _score, (x, y, width, height) in accepted:
            cv2.rectangle(
                binary,
                (offset_x + x, offset_y + y),
                (offset_x + x + width - 1, offset_y + y + height - 1),
                255,
                -1,
            )
        return holes, {"binary": binary}


def _gray(image):
    if image is None or image.size == 0:
        raise ValueError("분석할 이미지가 없습니다.")
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _crop(image, roi):
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        return image, 0, 0
    x, y, width, height = (int(value) for value in roi)
    image_h, image_w = image.shape[:2]
    x0 = min(max(x, 0), image_w)
    y0 = min(max(y, 0), image_h)
    x1 = min(max(x + width, x0), image_w)
    y1 = min(max(y + height, y0), image_h)
    if x1 <= x0 or y1 <= y0:
        return image, 0, 0
    return image[y0:y1, x0:x1], x0, y0


def _intersection_over_union(first, second):
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0
