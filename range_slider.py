"""손잡이가 두 개인 레인지 슬라이더.

Qt5에는 구간(하한~상한)을 지정하는 슬라이더가 없어서 직접 그린다.
파라미터 튜닝이 필요한 페이지라면 어디서든 쓸 수 있도록 페이지 밖 모듈로 뒀다.

    from range_slider import RangeSlider

    slider = RangeSlider(0, 255)
    slider.set_values(50, 255)
    slider.range_changed.connect(...)     # 드래그 중 실시간
    slider.range_committed.connect(...)   # 손잡이를 놓았을 때

값은 항상 low <= high - MIN_GAP 를 유지하므로, 두 손잡이가 겹쳐 구간이
비어 버리는 상태가 생기지 않는다.
"""

from PyQt5.QtCore import QPoint, QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy, QWidget

from theme import (
    RANGE_GROOVE_ACTIVE,
    RANGE_GROOVE_BG,
    RANGE_GROOVE_HEIGHT,
    RANGE_HANDLE_BG,
    RANGE_HANDLE_BORDER,
    RANGE_HANDLE_BORDER_WIDTH,
    RANGE_HANDLE_RADIUS,
    RANGE_SLIDER_HEIGHT,
    RANGE_SLIDER_MIN_WIDTH,
)

MIN_GAP = 1   # 하한과 상한 사이 최소 간격 (구간이 비는 것을 막는다)


class RangeSlider(QWidget):
    """[low, high] 구간을 두 손잡이로 지정하는 슬라이더."""

    range_changed = pyqtSignal(int, int)     # 드래그 중 값이 바뀔 때마다
    range_committed = pyqtSignal(int, int)   # 손잡이에서 손을 뗐을 때 한 번

    def __init__(self, minimum=0, maximum=255, parent=None):
        super().__init__(parent)
        self._minimum = int(minimum)
        self._maximum = int(maximum)
        self._low = self._minimum
        self._high = self._maximum
        self._active = None   # 'low' | 'high' | None (드래그 중인 손잡이)

        self.setMinimumWidth(RANGE_SLIDER_MIN_WIDTH)
        self.setFixedHeight(RANGE_SLIDER_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)

    # ===== 값 =====
    def values(self):
        return self._low, self._high

    def set_values(self, low, high):
        """두 값을 한 번에 설정한다. 범위와 최소 간격을 지켜 보정한다."""
        low = self._clamp(low)
        high = self._clamp(high)
        if high - low < MIN_GAP:
            high = min(self._maximum, low + MIN_GAP)
            low = max(self._minimum, high - MIN_GAP)
        changed = (low, high) != (self._low, self._high)
        self._low, self._high = low, high
        if changed:
            self.update()
        return changed

    def _clamp(self, value):
        return max(self._minimum, min(int(value), self._maximum))

    # ===== 좌표 변환 =====
    def _usable_width(self):
        """손잡이 반지름을 뺀, 값이 실제로 매핑되는 가로 길이."""
        return max(1, self.width() - 2 * RANGE_HANDLE_RADIUS)

    def _value_to_x(self, value):
        span = max(1, self._maximum - self._minimum)
        ratio = (value - self._minimum) / span
        return RANGE_HANDLE_RADIUS + int(round(ratio * self._usable_width()))

    def _x_to_value(self, x):
        span = self._maximum - self._minimum
        ratio = (x - RANGE_HANDLE_RADIUS) / self._usable_width()
        return self._clamp(self._minimum + int(round(ratio * span)))

    # ===== 마우스 =====
    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        # 클릭 지점에서 더 가까운 손잡이를 잡는다(홈을 눌러도 그 손잡이가 따라온다).
        x = event.pos().x()
        self._active = (
            "low"
            if abs(x - self._value_to_x(self._low))
            <= abs(x - self._value_to_x(self._high))
            else "high"
        )
        self._drag_to(x)

    def mouseMoveEvent(self, event):
        if self._active is None:
            super().mouseMoveEvent(event)
            return
        self._drag_to(event.pos().x())

    def mouseReleaseEvent(self, event):
        if self._active is None or event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        self._active = None
        self.range_committed.emit(self._low, self._high)

    def _drag_to(self, x):
        """잡고 있는 손잡이를 x 위치로 옮긴다. 상대 손잡이를 넘지 못한다."""
        value = self._x_to_value(x)
        if self._active == "low":
            low = min(value, self._high - MIN_GAP)
            high = self._high
        else:
            low = self._low
            high = max(value, self._low + MIN_GAP)
        if self.set_values(low, high):
            self.range_changed.emit(self._low, self._high)

    # ===== 그리기 =====
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        center_y = self.height() // 2
        groove_top = center_y - RANGE_GROOVE_HEIGHT // 2
        radius = RANGE_GROOVE_HEIGHT / 2
        x_low = self._value_to_x(self._low)
        x_high = self._value_to_x(self._high)

        # 홈 배경
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(RANGE_GROOVE_BG))
        painter.drawRoundedRect(
            QRect(
                RANGE_HANDLE_RADIUS,
                groove_top,
                self._usable_width(),
                RANGE_GROOVE_HEIGHT,
            ),
            radius,
            radius,
        )

        # 선택된 구간
        painter.setBrush(QColor(RANGE_GROOVE_ACTIVE))
        painter.drawRoundedRect(
            QRect(x_low, groove_top, max(1, x_high - x_low), RANGE_GROOVE_HEIGHT),
            radius,
            radius,
        )

        # 두 손잡이
        painter.setPen(QPen(QColor(RANGE_HANDLE_BORDER), RANGE_HANDLE_BORDER_WIDTH))
        painter.setBrush(QColor(RANGE_HANDLE_BG))
        for x in (x_low, x_high):
            painter.drawEllipse(
                QPoint(x, center_y), RANGE_HANDLE_RADIUS - 1, RANGE_HANDLE_RADIUS - 1
            )
