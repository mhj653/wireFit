"""오퍼레이터 화면의 상단 상태 카드와 모델 탭.

    ┌───────────────╲   ┌───────────────╲   ┌───────────────╲
    │ [i] 카메라 모델 │  │ [i] TCP Server │  │ [i] 데이터 저장 │
    │     연결됨      │  │     연결됨      │  │     저장됨      │
    └────────────────┘  └────────────────┘  └────────────────┘

    ╱현재 모델╲ ╱다른 모델╲        ← 모델 탭 (고른 것만 밝다)
    ┌──────────────────────────────┐
    │        이미지 보드            │

모서리 하나를 잘라낸(책갈피) 모양은 QSS 로 그릴 수 없다 — border-radius 는 네
모서리를 둥글게만 만든다. 그래서 이 두 위젯은 paintEvent 에서 QPainterPath 로
직접 그린다. 색과 크기는 전부 theme.py 에서 온다.

왜 카드인가
───────────
현장에서 촬영이 안 될 때 원인은 거의 셋 중 하나다 — 카메라가 안 붙었거나, 로봇이
접속하지 않았거나, 저장 경로가 없거나. 그 셋을 로그에서 찾게 하지 않고 늘 보이는
자리에 못 박아 둔다. 상태 글자는 좋은 상태일 때 초록, 손봐야 할 때 빨강이라
글을 읽지 않고 색만 봐도 갈린다.
"""

from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QAbstractButton, QHBoxLayout, QWidget

import nav_icons
from theme import (
    CARD_BG,
    CARD_BORDER,
    CARD_CHAMFER,
    CARD_HEIGHT,
    CARD_ICON_PLATE,
    CARD_ICON_SIZE,
    CARD_MIN_WIDTH,
    CARD_NG_TEXT,
    CARD_OK_TEXT,
    CARD_TEXT,
    MODEL_TAB_BG,
    MODEL_TAB_BG_HOVER,
    MODEL_TAB_BG_SELECTED,
    MODEL_TAB_BORDER,
    MODEL_TAB_CHAMFER,
    MODEL_TAB_HEIGHT,
    MODEL_TAB_MIN_WIDTH,
    MODEL_TAB_SPACING,
    MODEL_TAB_TEXT,
    MODEL_TAB_TEXT_SELECTED,
    NAV_PLATE_BORDER,
)

CARD_PADDING = 8       # 카드 테두리와 내용 사이
CARD_TEXT_GAP = 10     # 아이콘 판과 글자 사이
CARD_TITLE_SIZE = 12   # 첫 줄(무엇인지) 글자 크기
CARD_STATE_SIZE = 12   # 둘째 줄(어떤 상태인지)
CARD_LINE_GAP = 3


def _chamfered_path(rect, cut, corner):
    """모서리 하나를 잘라낸 사각형 경로.

    Args:
        rect: QRectF
        cut: 잘라내는 크기(px)
        corner: "topright" 또는 "bottomright"

    폭이 좁으면 잘라내는 크기를 줄인다 — 안 줄이면 경로가 자기를 파고들어
    모양이 뒤집힌다.
    """
    cut = max(0.0, min(float(cut), rect.width() / 2.0, rect.height() / 2.0))
    path = QPainterPath()
    if corner == "topright":
        path.moveTo(rect.left(), rect.top())
        path.lineTo(rect.right() - cut, rect.top())
        path.lineTo(rect.right(), rect.top() + cut)
        path.lineTo(rect.right(), rect.bottom())
        path.lineTo(rect.left(), rect.bottom())
    else:   # bottomright
        path.moveTo(rect.left(), rect.top())
        path.lineTo(rect.right(), rect.top())
        path.lineTo(rect.right(), rect.bottom() - cut)
        path.lineTo(rect.right() - cut, rect.bottom())
        path.lineTo(rect.left(), rect.bottom())
    path.closeSubpath()
    return path


class StatusCard(QWidget):
    """아이콘 + 두 줄(무엇인지 / 어떤 상태인지) 카드 하나.

    둘째 줄은 `set_state(text, ok)` 로 갱신한다. ok 가 None 이면 중립(흰 글자)
    이라, '켜짐/꺼짐'처럼 좋고 나쁨을 가릴 수 없는 값에 쓴다.
    """

    def __init__(self, title, icon_name, parent=None):
        super().__init__(parent)
        self.setFixedHeight(CARD_HEIGHT)
        self.setMinimumWidth(CARD_MIN_WIDTH)
        self._title = title
        self._state_text = ""
        self._state_ok = None
        self._icon = nav_icons.render(icon_name, CARD_ICON_SIZE, NAV_PLATE_BORDER)

    def set_title(self, text):
        if text != self._title:
            self._title = text
            self.update()

    def set_state(self, text, ok=None):
        """둘째 줄을 갈아 끼운다. ok: True(초록) / False(빨강) / None(흰색)."""
        if (text, ok) == (self._state_text, self._state_ok):
            return
        self._state_text = text
        self._state_ok = ok
        self.setToolTip(f"{self._title}\n{text}")
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # 테두리를 반 픽셀 안으로 넣어야 1px 선이 흐려지지 않는다.
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = _chamfered_path(rect, CARD_CHAMFER, "bottomright")
        painter.fillPath(path, QColor(CARD_BG))
        painter.setPen(QPen(QColor(CARD_BORDER), 1))
        painter.drawPath(path)

        # ── 아이콘 판 ──
        plate_top = (self.height() - CARD_ICON_PLATE) / 2.0
        plate = QRectF(
            CARD_PADDING + 0.5, plate_top, CARD_ICON_PLATE, CARD_ICON_PLATE
        )
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(CARD_BORDER), 1))
        painter.drawRect(plate)
        painter.drawPixmap(
            int(round(plate.center().x() - self._icon.width() / 2.0)),
            int(round(plate.center().y() - self._icon.height() / 2.0)),
            self._icon,
        )

        # ── 두 줄 글자 ──
        text_left = plate.right() + CARD_TEXT_GAP
        text_width = max(10.0, rect.right() - text_left - CARD_PADDING)
        line_height = CARD_TITLE_SIZE + 6
        block_top = (
            self.height() - (line_height * 2 + CARD_LINE_GAP)
        ) / 2.0

        font = QFont(painter.font())
        font.setPixelSize(CARD_TITLE_SIZE)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(CARD_TEXT)))
        # 카드 폭은 고정이고 글자는 카메라 모델명처럼 길 수 있다. 넘치면 잘라야
        # 한다 — 안 자르면 글자가 카드 테두리를 넘어가 옆 카드에 겹친다.
        # 뒤를 줄인다: 모델명은 앞쪽('RealSense D405')이 앞자리 정보다.
        painter.drawText(
            QRectF(text_left, block_top, text_width, line_height),
            Qt.AlignLeft | Qt.AlignVCenter,
            painter.fontMetrics().elidedText(
                self._title, Qt.ElideRight, int(text_width)
            ),
        )

        font.setPixelSize(CARD_STATE_SIZE)
        font.setBold(False)
        painter.setFont(font)
        if self._state_ok is None:
            color = CARD_TEXT
        else:
            color = CARD_OK_TEXT if self._state_ok else CARD_NG_TEXT
        painter.setPen(QPen(QColor(color)))
        # 상태는 가운데를 줄인다. 저장 경로가 들어오는데 그때는 앞(드라이브)과
        # 뒤(날짜 폴더)가 둘 다 필요하다.
        painter.drawText(
            QRectF(
                text_left,
                block_top + line_height + CARD_LINE_GAP,
                text_width,
                line_height,
            ),
            Qt.AlignLeft | Qt.AlignVCenter,
            painter.fontMetrics().elidedText(
                self._state_text, Qt.ElideMiddle, int(text_width)
            ),
        )
        painter.end()


class _ModelTab(QAbstractButton):
    """모델 탭 하나. 오른쪽 위를 잘라낸 책갈피 모양."""

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(MODEL_TAB_HEIGHT)
        self.setMinimumWidth(MODEL_TAB_MIN_WIDTH)
        self.setText(name)
        self.setToolTip(name)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        selected = self.isChecked()
        if selected:
            background = MODEL_TAB_BG_SELECTED
        elif self.underMouse():
            background = MODEL_TAB_BG_HOVER
        else:
            background = MODEL_TAB_BG

        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        path = _chamfered_path(rect, MODEL_TAB_CHAMFER, "topright")
        painter.fillPath(path, QColor(background))
        painter.setPen(QPen(QColor(MODEL_TAB_BORDER), 1))
        painter.drawPath(path)

        font = QFont(painter.font())
        font.setPixelSize(13)
        font.setBold(selected)
        painter.setFont(font)
        painter.setPen(
            QPen(QColor(MODEL_TAB_TEXT_SELECTED if selected else MODEL_TAB_TEXT))
        )
        metrics = painter.fontMetrics()
        inner = rect.adjusted(10.0, 0.0, -(MODEL_TAB_CHAMFER + 6.0), 0.0)
        painter.drawText(
            inner,
            Qt.AlignCenter,
            metrics.elidedText(self.text(), Qt.ElideRight, int(inner.width())),
        )
        painter.end()

    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)


class ModelTabBar(QWidget):
    """모델 탭 줄. 표 위(세팅)와 이미지 보드 위(오퍼레이터)가 같은 것을 쓴다.

    두 화면의 차이는 **누를 수 있는지**뿐이다.

      세팅 페이지     누르면 그 모델로 넘어간다(표 내용이 갈린다).
      오퍼레이터 화면  잠근다(set_locked). 라벨을 눌러 제품이 바뀌면 안 된다 —
                      260910 사용자 요구, 안정성. 바꾸려면 '모델 변경' 창을 쓴다.

    탭은 만든 순서대로 왼쪽부터 쌓이고, 고른 것만 밝은 회색이라 어느 것이 켜져
    있는지 색으로 바로 갈린다.
    """

    model_selected = pyqtSignal(int)

    LOCKED_HINT = (
        "여기서는 모델을 바꿀 수 없습니다.\n"
        "'모델 변경' 버튼으로 바꾸세요 (실수로 제품이 바뀌는 것을 막습니다)."
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(MODEL_TAB_HEIGHT)
        self._tabs = []
        self._current = -1
        self._locked = False

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(MODEL_TAB_SPACING)
        self._layout.addStretch(1)

    def set_locked(self, locked):
        """누를 수 없게 잠근다(오퍼레이터 화면). 보이는 모양은 그대로다.

        모양을 그대로 두는 이유: 회색으로 흐려 놓으면 '지금 고른 모델'이 어느
        것인지도 함께 흐려진다. 현장에서 알아야 하는 것은 그 색이다.
        """
        self._locked = bool(locked)
        for tab in self._tabs:
            self._apply_lock(tab)

    def _apply_lock(self, tab):
        tab.setEnabled(not self._locked)
        tab.setCursor(Qt.ArrowCursor if self._locked else Qt.PointingHandCursor)
        tab.setToolTip(
            f"{tab.text()}\n{self.LOCKED_HINT}" if self._locked else tab.text()
        )

    def set_models(self, names, selected=0):
        """탭을 다시 짠다. names 가 비면 줄 자체를 숨긴다."""
        for tab in self._tabs:
            self._layout.removeWidget(tab)
            tab.setParent(None)
            tab.deleteLater()
        self._tabs = []
        self._current = -1

        names = [str(name) for name in (names or [])]
        self.setVisible(bool(names))
        if not names:
            return

        for index, name in enumerate(names):
            tab = _ModelTab(name, self)
            tab.clicked.connect(
                lambda _checked, widget=tab: self._on_clicked(widget)
            )
            self._apply_lock(tab)
            self._layout.insertWidget(self._layout.count() - 1, tab)
            self._tabs.append(tab)
        self.set_current(min(max(selected, 0), len(names) - 1))

    def names(self):
        return [tab.text() for tab in self._tabs]

    def set_current(self, index):
        if not 0 <= index < len(self._tabs) or index == self._current:
            return
        self._current = index
        for position, tab in enumerate(self._tabs):
            tab.setChecked(position == index)
            tab.update()
        self.model_selected.emit(index)

    def current(self):
        return self._current

    def count(self):
        return len(self._tabs)

    def _on_clicked(self, widget):
        index = self._tabs.index(widget)
        if index == self._current:
            # 이미 고른 탭이면 체크가 풀리지 않게 되돌린다.
            widget.setChecked(True)
            return
        self.set_current(index)
