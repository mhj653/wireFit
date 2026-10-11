"""페이지들이 공유하는 ROI(관심 영역) 상태.

Hole Detection 페이지에서 지정한 ROI를 이후 페이지(3D 좌표 계산 등)에서도
그대로 쓰기 위한 단일 보관소다. 페이지끼리 서로를 import 하지 않고
이 모듈만 참조하므로, 어떤 페이지를 빼도 나머지는 그대로 동작한다.

ROI는 항상 '원본 이미지 픽셀 좌표계'의 (x, y, w, h)로 보관한다.
화면 위젯 크기나 표시 배율과 무관한 값이라, 창을 줄이거나 다른 페이지에서
다른 크기로 띄워도 같은 영역을 가리킨다.

사용 예:
    from roi_state import roi_state

    roi = roi_state.clamped_to(width, height)   # 이미지 크기에 맞춰 잘라낸 ROI
    if roi is not None:
        x, y, w, h = roi
"""

MIN_ROI_SIZE = 5   # 이보다 작은 드래그는 ROI로 보지 않는다(오클릭 방지)


class _RoiState:
    """ROI 한 개를 들고 있는 보관소. 모듈 하단의 roi_state 인스턴스를 쓴다."""

    def __init__(self):
        self._rect = None   # (x, y, w, h) 또는 None(전체 영역)

    def is_set(self):
        return self._rect is not None

    def get(self):
        """설정된 ROI를 (x, y, w, h)로 반환. 없으면 None."""
        return self._rect

    def set(self, x, y, w, h):
        """ROI를 설정한다. 너무 작으면 무시하고 False를 반환."""
        if w < MIN_ROI_SIZE or h < MIN_ROI_SIZE:
            return False
        self._rect = (int(x), int(y), int(w), int(h))
        return True

    def clear(self):
        self._rect = None

    def clamped_to(self, width, height):
        """이미지 크기(width x height) 안으로 잘라낸 ROI를 반환.

        ROI를 지정할 때와 다른 해상도의 이미지가 들어와도 안전하게 쓰도록
        경계를 넘는 부분을 잘라낸다. 겹치는 영역이 없으면 None.
        """
        if self._rect is None:
            return None
        x, y, w, h = self._rect
        x0 = max(0, min(x, width))
        y0 = max(0, min(y, height))
        x1 = max(0, min(x + w, width))
        y1 = max(0, min(y + h, height))
        if x1 - x0 < MIN_ROI_SIZE or y1 - y0 < MIN_ROI_SIZE:
            return None
        return (x0, y0, x1 - x0, y1 - y0)

    def crop(self, image):
        """이미지를 ROI로 잘라 (잘린 이미지, x오프셋, y오프셋)을 반환.

        ROI가 없으면 원본과 (0, 0)을 그대로 돌려준다. 오프셋은 잘린 이미지에서
        찾은 좌표를 원본 좌표로 되돌릴 때 더해 주면 된다.
        """
        height, width = image.shape[:2]
        roi = self.clamped_to(width, height)
        if roi is None:
            return image, 0, 0
        x, y, w, h = roi
        return image[y:y + h, x:x + w], x, y


# 앱 전체가 공유하는 단일 인스턴스
roi_state = _RoiState()
