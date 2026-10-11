"""뎁스 맵 시각화 (여러 페이지가 공유).

Camera Capture 페이지와 Hole Detection 페이지가 같은 방식으로 뎁스를 그려야
두 화면을 비교할 수 있으므로, 색칠 로직을 이 모듈 하나에 모아 둔다.
페이지를 넣고 빼도 이 모듈만 남아 있으면 나머지는 그대로 동작한다.
"""

import cv2
import numpy as np

# 뎁스 → 색 매핑. JET은 파랑(가까움) → 초록 → 노랑 → 빨강(멈) 순서다.
DEPTH_COLORMAP = cv2.COLORMAP_JET


def colorize_depth(depth):
    """uint16 뎁스 배열을 컬러맵 BGR 이미지로 변환한다.

    유효한 픽셀(0 초과)의 실제 min~max로 정규화하므로, 화면에 들어온 물체의
    거리 범위에 맞춰 대비가 자동으로 최적화된다.

    Args:
        depth: (H, W) uint16 배열 (mm 단위). None이거나 비어 있으면 None 반환.

    Returns:
        (H, W, 3) BGR 이미지, 또는 None
    """
    if depth is None or depth.size == 0:
        return None

    # 유효한 뎁스 값만 고려 (0은 측정 실패 픽셀)
    valid_mask = depth > 0
    if valid_mask.any():
        valid_depth = depth[valid_mask]
        depth_min = valid_depth.min()
        depth_max = valid_depth.max()
    else:
        depth_min = 0
        depth_max = 1

    # 0-255 범위로 정규화 (유효하지 않은 픽셀은 0으로 남겨 검게 나온다)
    normalized = np.zeros_like(depth, dtype=np.uint8)
    if depth_max > depth_min:
        normalized[valid_mask] = (
            ((depth[valid_mask] - depth_min) / (depth_max - depth_min)) * 255
        ).astype(np.uint8)

    return cv2.applyColorMap(normalized, DEPTH_COLORMAP)
