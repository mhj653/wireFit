"""좌우로 미끄러지는 on/off 스위치.

Qt5에는 이런 스위치가 없어서 직접 그린다. 손잡이가 왼쪽에 있으면 꺼짐(빨강),
오른쪽에 있으면 켜짐(초록)이고, 비활성 상태에서는 회색이 되어 조작할 수 없음이
한눈에 드러난다. 파라미터를 켜고 끄는 행이라면 어디서든 쓸 수 있도록
페이지 밖 모듈로 뒀다.

    from toggle_switch import ToggleSwitch

    switch = ToggleSwitch()
    switch.setChecked(True)
    switch.toggled.connect(...)

QAbstractButton을 상속하므로 setCheckable/isChecked/toggled 등
표준 체크 버튼 API를 그대로 쓴다.
"""

from PyQt5.QtCore import QPropertyAnimation, QRectF, QSize, Qt, pyqtProperty
from PyQt5.QtGui import QColor, QPainter
from PyQt5.QtWidgets import QAbstractButton

from theme import (
    SWITCH_ANIM_MS,
    SWITCH_DISABLED_BG,
    SWITCH_HEIGHT,
    SWITCH_KNOB_BG,
    SWITCH_KNOB_MARGIN,
    SWITCH_OFF_BG,
    SWITCH_ON_BG,
    SWITCH_WIDTH,
)


class ToggleSwitch(QAbstractButton):
    """켜짐/꺼짐을 좌우 위치로 보여주는 스위치."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(SWITCH_WIDTH, SWITCH_HEIGHT)
        self.setCursor(Qt.PointingHandCursor)

        self._offset = 0.0   # 0.0 = 왼쪽(꺼짐), 1.0 = 오른쪽(켜짐)
        self._animation = QPropertyAnimation(self, b"offset", self)
        self._animation.setDuration(SWITCH_ANIM_MS)
        self.toggled.connect(self._animate_to)

    def sizeHint(self):
        return QSize(SWITCH_WIDTH, SWITCH_HEIGHT)

    # ===== 손잡이 위치 (애니메이션 대상) =====
    def _get_offset(self):
        return self._offset

    def _set_offset(self, value):
        self._offset = value
        self.update()

    offset = pyqtProperty(float, fget=_get_offset, fset=_set_offset)

    def _animate_to(self, checked):
        self._animation.stop()
        self._animation.setStartValue(self._offset)
        self._animation.setEndValue(1.0 if checked else 0.0)
        self._animation.start()

    def set_checked_silently(self, checked):
        """toggled 신호 없이 상태를 맞춘다(다른 대상의 상태를 표시할 때).

        애니메이션도 건너뛰고 손잡이를 바로 그 위치에 둔다.
        """
        blocked = self.blockSignals(True)
        self.setChecked(checked)
        self.blockSignals(blocked)
        self._animation.stop()
        self._set_offset(1.0 if checked else 0.0)

    # ===== 그리기 =====
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 트랙: 꺼짐이면 빨강, 켜짐이면 초록, 비활성이면 회색.
        # 손잡이가 덮지 않은 여백이 그대로 상태 색으로 보인다.
        if not self.isEnabled():
            track_color = SWITCH_DISABLED_BG
        else:
            track_color = SWITCH_ON_BG if self.isChecked() else SWITCH_OFF_BG

        radius = self.height() / 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(track_color))
        painter.drawRoundedRect(
            QRectF(0, 0, self.width(), self.height()), radius, radius
        )

        # 손잡이
        diameter = self.height() - 2 * SWITCH_KNOB_MARGIN
        travel = self.width() - diameter - 2 * SWITCH_KNOB_MARGIN
        x = SWITCH_KNOB_MARGIN + self._offset * travel
        painter.setBrush(QColor(SWITCH_KNOB_BG))
        painter.drawEllipse(QRectF(x, SWITCH_KNOB_MARGIN, diameter, diameter))
