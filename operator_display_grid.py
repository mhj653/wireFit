"""체결 세팅 개수에 맞춰 나뉘는 운영 화면.

체결 위치가 몇 개인지에 따라 화면을 나눈다. 남는 칸은 검정으로 채워서
칸 크기가 들쭉날쭉해지지 않게 한다(1개가 넓고 1개가 좁으면 같은 배율로
보고 비교할 수 없다).

    1개        2개              3개              4개
    ┌─────┐   ┌────┬────┐   ┌────┬────┐   ┌────┬────┐
    │  1  │   │ 1  │ 2  │   │ 1  │ 2  │   │ 1  │ 2  │
    │     │   │    │    │   ├────┼────┤   ├────┼────┤
    └─────┘   └────┴────┘   │ 3  │검정│   │ 3  │ 4  │
                            └────┴────┘   └────┴────┘

칸 수는 grid_shape() 가 정한다. 가로로 먼저 채우고(세로로 자르고) 모자라면
줄을 늘린다. 2개일 때 위아래로 나누고 싶으면 grid_shape() 만 고치면 된다.

칸에는 테두리로 두 가지를 표시한다.

    고른 칸(파랑)   눌러서 고른 칸. 촬영 버튼을 누르면 이 칸에 담긴다.
    최신 칸(초록)   가장 마지막에 촬영된 칸. 트리거로 찍을 때 어디가 찍혔는지
                    한눈에 보이게 한다.

둘 다인 칸은 파랑으로 보인다(theme.py 에서 고른 칸 규칙을 뒤에 두었다).
수동으로 고르고 찍는 것과 트리거로 찍는 것은 보통 같이 쓰지 않으므로
겹치는 일이 드물다.

OK / NG 배지는 **칸마다 하나씩** 그 칸의 우측 상단에 얹힌다(260910 요구).
화면 한 군데에 두면 네 칸을 찍었을 때 마지막 판정만 남아, 어느 칸이 NG 였는지
알 수 없다. 배지는 레이아웃에 넣지 않고 칸 위에 겹쳐 놓으므로(자식 위젯 +
resizeEvent 에서 자리 잡기) 사진이 차지하는 자리를 줄이지 않는다.
"""

import math

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QGridLayout, QLabel, QSizePolicy, QWidget

from theme import JUDGE_BADGE_MARGIN

CELL_SPACING = 8
CELL_MIN_WIDTH = 160
CELL_MIN_HEIGHT = 120


def grid_shape(count):
    """칸 수에 맞는 (줄 수, 칸 수) 를 돌려준다.

    가로 칸 수를 먼저 √(개수) 로 잡고 줄 수를 맞춘다.
      1 -> (1, 1)      2 -> (1, 2)      3 -> (2, 2)      4 -> (2, 2)
      5 -> (2, 3)      6 -> (2, 3)      9 -> (3, 3)

    2개를 위아래로 나누고 싶으면 여기서 (2, 1) 을 돌려주면 된다.
    """
    if count <= 1:
        return 1, 1
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    return rows, columns


class _DisplayCell(QLabel):
    """눌러서 고를 수 있는 화면 칸.

    그림은 칸 크기에 맞춰 비율을 지켜 줄여 보여 준다. 원본을 따로 들고 있다가
    칸 크기가 바뀔 때마다 원본에서 다시 줄이므로, 줄인 그림을 또 줄여서
    화질이 쌓여 나빠지는 일이 없다(다른 페이지의 이미지 보드와 같은 방식).
    """

    clicked = pyqtSignal(int)   # 체결 순서 (1부터)

    def __init__(self, sequence, parent=None):
        super().__init__(parent)
        self.sequence = sequence
        self._source_pixmap = None
        self.setObjectName("operatorDisplay")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(CELL_MIN_WIDTH, CELL_MIN_HEIGHT)
        # 그림 크기가 칸 크기를 밀어내지 않게 sizeHint 를 무시한다.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setCursor(Qt.PointingHandCursor)

        # 이 칸의 판정 배지. 사진 위에 겹쳐 놓는다(레이아웃에 넣지 않는다).
        # 판정이 없으면 숨기므로 '아직 안 찍음' 색은 따로 두지 않는다.
        self.badge = QLabel("", self)
        self.badge.setObjectName("judgeBadge")
        self.badge.setAlignment(Qt.AlignCenter)
        # 배지를 눌러도 칸이 선택되게 마우스를 통과시킨다.
        self.badge.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.badge.hide()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.sequence)
        super().mousePressEvent(event)

    def setPixmap(self, pixmap):
        self._source_pixmap = pixmap
        self._rescale()

    def setText(self, text):
        self._source_pixmap = None
        super().setPixmap(QPixmap())
        super().setText(text)

    def set_verdict(self, verdict):
        """판정 배지를 갈아 끼운다. verdict: True(OK) / False(NG) / None(숨김).

        Qt 는 속성으로 스타일이 갈리는 위젯을 다시 그리려면 unpolish/polish 로
        스타일을 새로 물려야 한다(setProperty 만으로는 색이 바뀌지 않는다).
        """
        if verdict is None:
            self.badge.hide()
            return
        self.badge.setText("OK" if verdict else "NG")
        self.badge.setProperty("judge", "ok" if verdict else "ng")
        self.badge.style().unpolish(self.badge)
        self.badge.style().polish(self.badge)
        self.badge.adjustSize()
        self._place_badge()
        self.badge.show()
        self.badge.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()
        self._place_badge()

    def _place_badge(self):
        """배지를 칸의 우측 상단에 붙인다."""
        self.badge.move(
            max(0, self.width() - self.badge.width() - JUDGE_BADGE_MARGIN),
            JUDGE_BADGE_MARGIN,
        )

    def _rescale(self):
        if self._source_pixmap is None:
            return
        super().setPixmap(
            self._source_pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )


class OperatorDisplayGrid(QWidget):
    """운영 화면 칸들.

    set_count(n) 으로 칸을 다시 짠다. 앞의 n 칸이 체결 순서 1..n 이고,
    격자를 채우고 남은 칸은 검정이다.
    """

    cell_selected = pyqtSignal(int)   # 칸을 눌렀을 때 (체결 순서)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cells = []      # 체결 순서에 해당하는 칸 (_DisplayCell)
        self._fillers = []    # 남는 자리를 메우는 검정 칸
        self._selected = 0    # 고른 칸 (0 = 없음)
        self._latest = 0      # 가장 마지막에 촬영된 칸 (0 = 없음)

        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(0, 0, 0, 0)
        self._grid.setSpacing(CELL_SPACING)

        self.set_count(0)

    # ===== 칸 짜기 =====
    def set_count(self, count):
        """체결 위치 개수에 맞춰 화면을 다시 나눈다."""
        self._clear()
        self._selected = 0
        self._latest = 0

        if count <= 0:
            # 체결 세팅이 없으면 한 칸만 두고 그렇다고 알려 준다.
            # 고를 수는 없으므로 누르는 칸으로 만들지 않는다.
            empty = QLabel("체결 세팅이 없습니다")
            empty.setObjectName("operatorDisplay")
            empty.setAlignment(Qt.AlignCenter)
            empty.setMinimumSize(CELL_MIN_WIDTH, CELL_MIN_HEIGHT)
            empty.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._fillers.append(empty)
            self._grid.addWidget(empty, 0, 0)
            self._grid.setRowStretch(0, 1)
            self._grid.setColumnStretch(0, 1)
            return

        rows, columns = grid_shape(count)
        for position in range(rows * columns):
            row, column = divmod(position, columns)
            if position < count:
                cell = _DisplayCell(position + 1)
                cell.setText(str(position + 1))
                cell.clicked.connect(self.select)
                self._cells.append(cell)
            else:
                cell = self._make_filler()
                self._fillers.append(cell)
            self._grid.addWidget(cell, row, column)

        # 칸이 같은 크기로 나뉘게 한다.
        for row in range(rows):
            self._grid.setRowStretch(row, 1)
        for column in range(columns):
            self._grid.setColumnStretch(column, 1)

        # 속성을 처음부터 채워 둔다(안 채우면 '없음' 상태로 남아 나중에
        # 켜고 끌 때만 스타일이 물린다).
        self._restyle()

        # 한 칸뿐이면 고를 것이 없으니 미리 골라 둔다.
        if count == 1:
            self.select(1)

    def set_points(self, points):
        """체결 위치 목록으로 칸을 짠다(개수만 본다)."""
        self.set_count(len(points or []))

    def _clear(self):
        for cell in self._cells + self._fillers:
            self._grid.removeWidget(cell)
            cell.setParent(None)
            cell.deleteLater()
        self._cells = []
        self._fillers = []
        # 지난번에 준 stretch 가 남아 칸 비율이 어긋나지 않게 되돌린다.
        for row in range(self._grid.rowCount()):
            self._grid.setRowStretch(row, 0)
        for column in range(self._grid.columnCount()):
            self._grid.setColumnStretch(column, 0)

    def _make_filler(self):
        label = QLabel()
        label.setObjectName("operatorFiller")
        label.setMinimumSize(CELL_MIN_WIDTH, CELL_MIN_HEIGHT)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return label

    # ===== 고르기 / 최신 표시 =====
    def select(self, sequence):
        """칸을 고른다. 없는 순서면 아무 것도 하지 않는다."""
        if self.cell(sequence) is None:
            return False
        self._selected = sequence
        self._restyle()
        self.cell_selected.emit(sequence)
        return True

    def selected(self):
        """고른 칸의 체결 순서. 없으면 0."""
        return self._selected

    def mark_latest(self, sequence):
        """가장 마지막에 촬영된 칸으로 표시한다."""
        if self.cell(sequence) is None:
            return False
        self._latest = sequence
        self._restyle()
        return True

    def latest(self):
        return self._latest

    def _restyle(self):
        """테두리를 다시 물린다.

        Qt 는 속성으로 스타일이 갈리는 위젯을 다시 그리려면 unpolish/polish 로
        스타일을 새로 물려야 한다(setProperty 만으로는 테두리가 바뀌지 않는다).
        """
        for cell in self._cells:
            cell.setProperty(
                "selected", "true" if cell.sequence == self._selected else "false"
            )
            cell.setProperty(
                "latest", "true" if cell.sequence == self._latest else "false"
            )
            cell.style().unpolish(cell)
            cell.style().polish(cell)

    # ===== 그림 넣기 =====
    def count(self):
        return len(self._cells)

    def cell(self, sequence):
        """체결 순서(1부터)에 해당하는 칸. 없으면 None."""
        index = sequence - 1
        if 0 <= index < len(self._cells):
            return self._cells[index]
        return None

    def set_pixmap(self, sequence, pixmap):
        """칸에 그림을 넣는다. 없는 순서면 아무 것도 하지 않는다."""
        cell = self.cell(sequence)
        if cell is None:
            return False
        cell.setPixmap(pixmap)
        return True

    def set_text(self, sequence, text):
        cell = self.cell(sequence)
        if cell is None:
            return False
        cell.setText(text)
        return True

    # ===== 판정 배지 / 비우기 =====
    def set_verdict(self, sequence, verdict):
        """그 칸의 OK/NG 배지를 갈아 끼운다. None 이면 숨긴다."""
        cell = self.cell(sequence)
        if cell is None:
            return False
        cell.set_verdict(verdict)
        return True

    def clear_cells(self, keep=None):
        """칸을 비운다(사진과 배지를 지우고 순서 번호만 남긴다).

        Args:
            keep: 이 순서의 칸은 건드리지 않는다. 새 주기의 첫 촬영이 들어올 때
                지난 주기의 사진을 지우면서 방금 찍은 칸은 남겨야 한다.

        번호를 남기는 이유: 통째로 빈 칸으로 두면 어느 칸이 몇 번인지 알 수
        없어 촬영할 칸을 고를 수가 없다.
        """
        for cell in self._cells:
            if keep is not None and cell.sequence == keep:
                continue
            cell.setText(str(cell.sequence))
            cell.set_verdict(None)
