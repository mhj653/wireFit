"""체결 위치 표.

    체결 세팅                                        [+] [−]
    ┌──────┬──────────────────┬───────────────────────┐
    │ 순서 │ Blob 세팅        │ 캘리브레이션          │
    ├──────┼──────────────────┼───────────────────────┤
    │  1   │ [BlobSetting  ▾] │ [EIH_20260810_...  ▾] │
    │  2   │ [First        ▾] │ [EIH_20260818_...  ▾] │
    └──────┴──────────────────┴───────────────────────┘

- [+] 로 행을 늘리고, 행을 고른 뒤 [−] 로 지운다.
- 순서 칸을 잡아 끌면 행이 옮겨진다. 칸의 콤보가 마우스를 먼저 받기 때문에
  끌기는 순서 칸에서만 시작된다(순서 칸이 손잡이 역할).
- 번호는 늘 위에서부터 1, 2, 3... 이다. 행을 옮겨도 번호는 그대로고
  옮겨 다니는 건 행의 내용뿐이다(fastening_data_model 참고).
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from fastening_data_model import FasteningDataModel, FasteningPoint
from theme import ACCENT, CONTENT_SPACING

ORDER_COLUMN_WIDTH = 56
TABLE_MIN_HEIGHT = 190
ROW_HEIGHT = 34
DRAG_HANDLE_COLUMN = 0   # 순서 칸 = 끌기 손잡이
DROP_LINE_WIDTH = 2


class _ReorderTable(QTableWidget):
    """순서 칸을 잡아 끌어서 행을 옮길 수 있는 표.

    Qt 의 드래그앤드롭(InternalMove)은 쓰지 않는다. 두 가지가 걸린다.

    1. QTableWidget 의 모델은 '행 사이에 끼워 넣는' 드롭을 받지 못한다.
       그래서 dragMoveEvent 단계에서 거부되고 dropEvent 가 아예 불리지 않는다.
    2. 칸에 콤보 위젯을 얹어 두었으므로 Qt 가 행을 옮기면 글자만 옮겨지고
       콤보는 제자리에 남아 표가 어긋난다.

    그래서 마우스를 직접 받아서 어디서 어디로 갔는지만 알리고(row_moved),
    목록을 고친 뒤 표를 다시 그린다. 끼울 자리는 파란 선으로 보여 준다.

    끌기는 순서 칸에서만 시작된다. 나머지 칸은 콤보가 마우스를 먼저 받으므로
    애초에 여기까지 오지 않는다. 순서 칸이 손잡이 역할이다.
    """

    row_moved = pyqtSignal(int, int)   # from_row, to_row

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.verticalHeader().setVisible(False)   # 왼쪽 기본 행번호를 숨긴다
        self.verticalHeader().setDefaultSectionSize(ROW_HEIGHT)
        # 손잡이 위에서 커서를 바꿔 주려면 버튼을 누르지 않은 움직임도 받아야 한다.
        self.setMouseTracking(True)

        self._press_row = -1     # 순서 칸을 누른 행 (-1 = 안 누름)
        self._press_y = 0
        self._drag_row = -1      # 실제로 끌기 시작한 행 (-1 = 끌지 않음)
        self._drop_row = -1      # 끼울 자리 (0..rowCount)

    # ===== 마우스 =====
    def mousePressEvent(self, event):
        super().mousePressEvent(event)   # 행 선택은 Qt 가 하던 대로
        if event.button() != Qt.LeftButton:
            return
        index = self.indexAt(event.pos())
        if index.isValid() and index.column() == DRAG_HANDLE_COLUMN:
            self._press_row = index.row()
            self._press_y = event.pos().y()

    def mouseMoveEvent(self, event):
        if self._press_row >= 0 and event.buttons() & Qt.LeftButton:
            if self._drag_row < 0:
                # 눌린 채 조금 움직인 것과 끌기를 구분한다.
                if abs(event.pos().y() - self._press_y) < QApplication.startDragDistance():
                    return
                self._drag_row = self._press_row
                self.setCursor(Qt.ClosedHandCursor)
            self._drop_row = self._insert_row_at(event.pos().y())
            self.viewport().update()
            return

        super().mouseMoveEvent(event)
        self._sync_hover_cursor(event.pos())

    def mouseReleaseEvent(self, event):
        if self._drag_row >= 0:
            from_row, drop_row = self._drag_row, self._drop_row
            self._reset_drag()
            self._request_move(from_row, drop_row)
            event.accept()
            return
        self._reset_drag()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        if self._drag_row < 0:
            self.unsetCursor()
        super().leaveEvent(event)

    def _sync_hover_cursor(self, pos):
        """손잡이(순서 칸) 위에서는 잡을 수 있다는 걸 커서로 알린다."""
        index = self.indexAt(pos)
        if index.isValid() and index.column() == DRAG_HANDLE_COLUMN:
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.unsetCursor()

    def _reset_drag(self):
        was_dragging = self._drag_row >= 0
        self._press_row = -1
        self._drag_row = -1
        self._drop_row = -1
        if was_dragging:
            self.unsetCursor()
            self.viewport().update()

    # ===== 자리 계산 =====
    def _insert_row_at(self, y):
        """y 좌표가 '몇 번째 행 앞에 끼우는 것'인지 (0..rowCount)."""
        if self.rowCount() == 0:
            return 0
        row = self.rowAt(y)
        if row < 0:
            # 첫 행 위쪽이면 맨 앞, 마지막 행 아래쪽이면 맨 끝
            return 0 if y < self.rowViewportPosition(0) else self.rowCount()
        middle = self.rowViewportPosition(row) + self.rowHeight(row) / 2
        return row + 1 if y >= middle else row

    def _request_move(self, from_row, drop_row):
        """끌어온 행(from_row)을 drop_row 앞에 끼운다는 뜻으로 읽는다.

        drop_row 는 '몇 번째 앞에 끼우는지'라서 옮긴 뒤의 자리와 다르다.
        원래 자리보다 뒤에 끼우면 자기 자신이 빠지면서 한 칸 당겨진다.
        """
        if from_row < 0 or drop_row < 0:
            return
        to_row = drop_row - 1 if drop_row > from_row else drop_row
        to_row = max(0, min(to_row, self.rowCount() - 1))
        if to_row != from_row:
            self.row_moved.emit(from_row, to_row)

    # ===== 끼울 자리 표시 =====
    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drag_row < 0 or self._drop_row < 0:
            return

        if self._drop_row >= self.rowCount():
            last = self.rowCount() - 1
            y = self.rowViewportPosition(last) + self.rowHeight(last)
        else:
            y = self.rowViewportPosition(self._drop_row)
        y = max(1, min(y, self.viewport().height() - 1))

        painter = QPainter(self.viewport())
        pen = QPen(QColor(ACCENT))
        pen.setWidth(DROP_LINE_WIDTH)
        painter.setPen(pen)
        painter.drawLine(0, y, self.viewport().width(), y)


class FasteningTable(QWidget):
    """[모델 탭] ... 제목 + [+] [−] 버튼 + 체결 위치 표.

    표 위 줄의 **왼쪽은 비어 있고**(set_leading_widget 으로 채운다) 제목과
    버튼은 오른쪽에 붙는다. 260910 에 모델 개념이 들어오면서, 그 왼쪽 자리에
    모델 탭이 놓이게 되었기 때문이다 — 모델 탭이 표 바로 위에 붙어 있어야
    '이 표가 그 모델의 것'으로 읽힌다.
    """

    # 값이 바뀐 모든 변경(콤보 선택까지). 부르는 쪽이 이걸 받아 파일에 쓴다.
    data_changed = pyqtSignal()

    def __init__(self, title="체결 포인트 추가", parent=None):
        super().__init__(parent)
        self._model = FasteningDataModel()
        self._blob_options = []
        self._calibration_options = []

        # 페이지에서 stretch 를 주면 세로로 늘어나게 한다.
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)   # 모델 탭이 표에 붙어야 책갈피로 보인다

        # ── 표 위: (왼쪽은 비워 둔다) 오른쪽에 제목 + 행 추가 / 삭제 ──
        self._top_row = QHBoxLayout()
        self._top_row.setContentsMargins(0, 0, 0, CONTENT_SPACING // 2)
        self._top_row.setSpacing(CONTENT_SPACING // 2)
        self._top_row.addStretch()

        caption = QLabel(title)
        caption.setObjectName("sectionLabel")
        self._top_row.addWidget(caption)

        self.btn_add = QPushButton("+")
        self.btn_add.setObjectName("iconBtn")
        self.btn_add.setToolTip("체결 포인트 추가")
        self.btn_add.clicked.connect(self.add_point)
        self._top_row.addWidget(self.btn_add)

        self.btn_remove = QPushButton("−")
        self.btn_remove.setObjectName("iconBtn")
        self.btn_remove.setToolTip("고른 체결 포인트 삭제")
        self.btn_remove.clicked.connect(self._on_remove)
        self._top_row.addWidget(self.btn_remove)

        layout.addLayout(self._top_row)

        # ── 표 ──
        self.table = _ReorderTable()
        self.table.setObjectName("sampleTable")   # theme.py 의 표 스타일 재사용
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["순서", "비전 세팅", "캘리브레이션"])
        self.table.setColumnWidth(0, ORDER_COLUMN_WIDTH)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Fixed)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)

        self.table.setMinimumHeight(TABLE_MIN_HEIGHT)
        self.table.row_moved.connect(self._on_row_moved)
        self.table.itemSelectionChanged.connect(self._sync_remove_button)
        # 표가 남는 세로 공간을 차지한다(제목 줄은 제 높이만 쓴다).
        layout.addWidget(self.table, 1)

        self._model.on_changed(self._refresh_table)
        self._model.on_data_changed(self.data_changed.emit)
        self._sync_remove_button()

    def set_leading_widget(self, widget):
        """표 위 줄의 **왼쪽 끝**에 위젯을 넣는다(모델 탭이 들어온다)."""
        self._top_row.insertWidget(0, widget)
        return widget

    # ===== 선택 목록 =====
    def set_blob_options(self, options):
        self._blob_options = list(options)
        self._refresh_table()

    def set_calibration_options(self, options):
        self._calibration_options = list(options)
        self._refresh_table()

    # ===== 행 추가 / 삭제 / 이동 =====
    def add_point(self):
        self._model.add_point()
        last = self._model.count() - 1
        if last >= 0:
            self.table.selectRow(last)

    def _on_remove(self):
        row = self.table.currentRow()
        if row < 0:
            return
        self._model.remove_point(row)
        # 지운 자리에 오는 행을 골라 둔다(마지막 행이었다면 그 앞).
        remaining = self._model.count()
        if remaining:
            self.table.selectRow(min(row, remaining - 1))

    def _on_row_moved(self, from_row, to_row):
        if self._model.move_point(from_row, to_row):
            self.table.selectRow(to_row)   # 옮긴 행을 계속 고른 상태로 둔다

    def _sync_remove_button(self):
        self.btn_remove.setEnabled(self.table.currentRow() >= 0)

    # ===== 그리기 =====
    def _refresh_table(self):
        # 탭을 나갔다 돌아오면 목록을 다시 읽으면서 표를 새로 그린다.
        # 그때 고르고 있던 행이 풀리지 않게 기억해 두고 되돌린다.
        selected = self.table.currentRow()
        self.table.setRowCount(0)

        for index, point in enumerate(self._model.get_points()):
            self.table.insertRow(index)

            # 순서: 자리대로 매기므로 목록에 따로 담아 두지 않는다.
            item = QTableWidgetItem(str(index + 1))
            item.setTextAlignment(Qt.AlignCenter)
            item.setToolTip("끌어서 순서를 바꿉니다.")
            self.table.setItem(index, 0, item)

            self.table.setCellWidget(
                index, 1,
                self._make_combo(self._blob_options, point.blob_setting,
                                 index, "blob"),
            )
            self.table.setCellWidget(
                index, 2,
                self._make_combo(self._calibration_options, point.calibration,
                                 index, "calibration"),
            )

        if 0 <= selected < self.table.rowCount():
            self.table.selectRow(selected)
        self._sync_remove_button()

    def _make_combo(self, options, value, row, field):
        combo = QComboBox()
        combo.addItem("")
        combo.addItems(options)

        # 골라 둔 값이 목록에서 사라졌어도(세팅을 지운 경우) 화면에는 남긴다.
        # 조용히 빈 값으로 바뀌면 표에 보이는 것과 담긴 값이 어긋난다.
        if value and value not in options:
            combo.addItem(value)
        combo.setCurrentText(value)

        # 값을 맞춘 뒤에 연결한다. 먼저 연결하면 addItems/setCurrentText 가
        # 쏘는 신호까지 '사용자가 고른 것'으로 잡힌다.
        combo.currentTextChanged.connect(
            lambda text, r=row, f=field: self._on_combo_changed(r, f, text)
        )
        return combo

    def _on_combo_changed(self, row, field, text):
        if field == "blob":
            self._model.update_point(row, blob_setting=text)
        else:
            self._model.update_point(row, calibration=text)

    # ===== 값 =====
    def get_points(self):
        return self._model.get_points()

    def get_data(self):
        return self._model.to_list()

    def set_data(self, points_data, notify=True):
        """저장된 목록을 표에 싣는다.

        notify=False 는 모델을 갈아 끼울 때 쓴다. 그때 data_changed 가 나가면
        '방금 불러온 것'을 사용자가 고친 것으로 보고 파일에 되쓰게 되는데,
        모델 전환 중이라 아직 어느 모델에 써야 할지 정해지지 않은 시점이다.
        """
        points = [FasteningPoint.from_dict(item) for item in points_data]
        if notify:
            self._model.set_points(points)
            return
        blocked = self.blockSignals(True)
        try:
            self._model.set_points(points)
        finally:
            self.blockSignals(blocked)

    def clear(self):
        self._model.clear()
