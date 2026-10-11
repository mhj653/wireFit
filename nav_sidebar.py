"""좌측 사이드바 — 아이콘(위) + 라벨(아래) 한 칸씩.

    ┌──────────┐
    │ ┌──────┐ │   아이콘 판 (nav_icons 가 그린다)
    │ └──────┘ │
    │  라벨    │
    ├──────────┤
    │   ...    │

QListWidget 을 쓰지 않고 직접 만든 이유: 목록 항목 하나 안에 '아이콘 위 · 글자
아래'를 넣으려면 IconMode 로 바꾸거나 delegate 를 붙여야 하는데, 그러면 선택
표시와 여백을 QSS 로 다루기가 오히려 어려워진다. 칸 하나를 버튼으로 그리면
상태별 색을 한 군데(_NavItem.paintEvent)에서 다 정할 수 있다.

MainWindow 는 예전 QListWidget 과 같은 이름으로 쓴다(addItem / setCurrentRow /
currentRow / count / currentRowChanged). 그래서 PythonUI 쪽 호출부는 사이드바를
갈아 끼운 뒤에도 그대로다.
"""

from PyQt5.QtCore import QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QAbstractButton, QFrame, QVBoxLayout

import nav_icons
from theme import (
    NAV_ICON_SIZE,
    NAV_ITEM_HEIGHT,
    NAV_LABEL_GAP,
    NAV_LABEL_SIZE,
    NAV_PLATE_BG,
    NAV_PLATE_BG_SELECTED,
    NAV_PLATE_BORDER,
    NAV_PLATE_HEIGHT,
    NAV_PLATE_TOP,
    NAV_PLATE_WIDTH,
    SIDEBAR_BG,
    SIDEBAR_HOVER,
    SIDEBAR_LINE,
    SIDEBAR_SELECTED,
    SIDEBAR_TEXT,
)


class _NavItem(QAbstractButton):
    """사이드바 탭 한 칸.

    QPushButton 이 아니라 QAbstractButton 을 직접 그리는 이유: 아이콘 판과 라벨의
    자리를 픽셀로 잡아야 칸마다 같은 곳에 놓인다. QPushButton 의 icon + text 배치는
    글자 길이에 따라 아이콘이 좌우로 밀린다('Camera Selection' 과 'Pose Setting'
    처럼 길이가 다르면 아이콘 줄이 어긋난다).
    """

    def __init__(self, title, icon_name, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(NAV_ITEM_HEIGHT)
        self.setToolTip(title)
        self._title = title
        self._icon_name = icon_name
        # 아이콘은 크기가 정해져 있으므로 한 번만 그려 두고 다시 쓴다.
        self._icon = nav_icons.render(icon_name, NAV_ICON_SIZE, NAV_PLATE_BORDER)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        selected = self.isChecked()
        hovered = self.underMouse() and not selected

        # ── 칸 배경 ──
        if selected:
            background = SIDEBAR_SELECTED
        elif hovered:
            background = SIDEBAR_HOVER
        else:
            background = SIDEBAR_BG
        painter.fillRect(self.rect(), QColor(background))

        # ── 아래쪽 구분선 ──
        painter.setPen(QPen(QColor(SIDEBAR_LINE), 1))
        painter.drawLine(
            0, self.height() - 1, self.width(), self.height() - 1
        )

        # ── 아이콘 판 ──
        plate_x = (self.width() - NAV_PLATE_WIDTH) / 2.0
        plate = QRectF(
            plate_x, float(NAV_PLATE_TOP), NAV_PLATE_WIDTH, NAV_PLATE_HEIGHT
        )
        painter.setPen(QPen(QColor(NAV_PLATE_BORDER), 1))
        painter.setBrush(
            QColor(NAV_PLATE_BG_SELECTED if selected else NAV_PLATE_BG)
        )
        painter.drawRect(plate)

        icon_x = plate.x() + (plate.width() - self._icon.width()) / 2.0
        icon_y = plate.y() + (plate.height() - self._icon.height()) / 2.0
        painter.drawPixmap(int(round(icon_x)), int(round(icon_y)), self._icon)

        # ── 라벨 ──
        font = QFont(painter.font())
        font.setPixelSize(NAV_LABEL_SIZE)
        font.setBold(selected)
        painter.setFont(font)
        painter.setPen(QPen(QColor(SIDEBAR_TEXT)))
        label_top = plate.bottom() + NAV_LABEL_GAP
        painter.drawText(
            QRectF(2.0, label_top, self.width() - 4.0, self.height() - label_top - 6.0),
            Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
            self._title,
        )
        painter.end()

    # 마우스가 들어오고 나갈 때 배경색이 바뀌므로 다시 그리게 한다.
    def enterEvent(self, event):
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.update()
        super().leaveEvent(event)


class NavSidebar(QFrame):
    """탭 칸들을 세로로 쌓은 사이드바.

    currentRowChanged 는 QListWidget 과 같은 뜻이다(고른 줄이 바뀔 때 그 번호).
    같은 칸을 다시 눌러도 신호는 나가지 않는다 — 페이지를 다시 만들 이유가 없다.
    """

    currentRowChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self._items = []
        self._current = -1

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        # 칸이 적으면 아래가 비는데, 그 자리는 사이드바 색으로 남는다.
        self._layout.addStretch(1)

    def addItem(self, title, icon_name="default"):
        item = _NavItem(title, icon_name, self)
        item.clicked.connect(lambda _checked, widget=item: self._on_clicked(widget))
        # 마지막 stretch 앞에 넣어 칸들이 위에서부터 쌓이게 한다.
        self._layout.insertWidget(self._layout.count() - 1, item)
        self._items.append(item)
        return item

    def count(self):
        return len(self._items)

    def currentRow(self):
        return self._current

    def setCurrentRow(self, row):
        if not 0 <= row < len(self._items) or row == self._current:
            return
        self._current = row
        for index, item in enumerate(self._items):
            item.setChecked(index == row)
            item.update()
        self.currentRowChanged.emit(row)

    def _on_clicked(self, widget):
        # 눌린 칸이 이미 고른 칸이면 체크가 풀리지 않게 되돌린다.
        row = self._items.index(widget)
        if row == self._current:
            widget.setChecked(True)
            return
        self.setCurrentRow(row)
