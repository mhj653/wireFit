"""프레임리스 창의 커스텀 상단 바.

두 모드가 같은 바를 나눠 쓴다(set_mode 로 갈아 끼운다).

  세팅     [아이콘] IT                     [–][□][✕]
                                        Last update: 26.09.10

  오퍼레이터        [아이콘] 공정명       [◀ 홈 인터페이스][종료하기]
                                        Last update: 26.09.10

오퍼레이터 모드에서 달라지는 것 (260910)
────────────────────────────────────────
  - 아이콘과 공정명이 **창 가운데**로 간다. 왼쪽 아이콘은 숨는다.
  - 최소화·최대화 버튼이 없다. 그 모드는 전체화면으로 고정해 운영하므로 창
    크기를 바꿀 일이 없고, 실수로 창이 작아지면 현장에서 되돌리기 번거롭다.
  - 대신 '홈 인터페이스'(세팅으로 복귀)와 '종료하기'만 둔다.
  - 창 드래그 이동과 더블클릭 최대화도 막는다 — 전체화면에서는 뜻이 없고,
    끌면 전체화면에서 빠져나와 작아진다.

가운데 정렬을 레이아웃이 아니라 **오버레이**로 하는 이유: 좌우에 놓인 것들의
폭이 다르면(왼쪽은 아이콘, 오른쪽은 버튼 둘) 레이아웃 가운데는 창 가운데가
아니다. 바 전체에 겹쳐 놓고 AlignCenter 를 주면 늘 창 기준 가운데다.
"""

import os

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app_paths import resource_path
from theme import HEADER_HEIGHT, ICON_SIZE, WINDOW_BTN_HEIGHT

ICON_PATH = resource_path("bolt.png")
CENTER_ICON_GAP = 14   # 가운데 아이콘과 공정명 사이


def _icon_pixmap(icon_path, size):
    if not icon_path or not os.path.exists(icon_path):
        return None
    return QPixmap(icon_path).scaled(
        size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


class TitleBar(QFrame):
    """프레임리스 창의 커스텀 상단 바 (모듈 설명 참고)."""

    def __init__(self, window, title="", subtitle="", icon_path=ICON_PATH):
        super().__init__(window)
        self._window = window
        self._drag_pos = None
        self._mode = "setup"
        self._on_home = None
        self._on_quit = None

        self.setObjectName("header")
        self.setFixedHeight(HEADER_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        pixmap = _icon_pixmap(icon_path, ICON_SIZE)

        self.icon_label = QLabel()
        self.icon_label.setObjectName("headerIcon")
        if pixmap is not None:
            self.icon_label.setPixmap(pixmap)
        layout.addWidget(self.icon_label)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("headerTitle")

        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setObjectName("headerSubtitle")

        # ── 오퍼레이터 모드용 가운데 묶음 (아이콘 + 공정명) ──
        # 레이아웃에 넣지 않고 바 전체에 겹쳐 창 기준 가운데에 놓는다.
        self.center_box = QWidget(self)
        self.center_box.setAttribute(Qt.WA_TransparentForMouseEvents)
        center_layout = QHBoxLayout(self.center_box)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(CENTER_ICON_GAP)
        center_layout.addStretch()

        self.center_icon = QLabel()
        if pixmap is not None:
            self.center_icon.setPixmap(pixmap)
        center_layout.addWidget(self.center_icon)

        self.center_title = QLabel("")
        self.center_title.setObjectName("headerCenterTitle")
        center_layout.addWidget(self.center_title)
        center_layout.addStretch()
        self.center_box.hide()

        # ── 세팅 모드 창 버튼 ──
        self.btn_min = QPushButton("–")
        self.btn_min.setObjectName("winBtn")
        self.btn_min.setFixedHeight(WINDOW_BTN_HEIGHT)
        self.btn_min.clicked.connect(self._window.showMinimized)

        self.btn_max = QPushButton("□")
        self.btn_max.setObjectName("winBtn")
        self.btn_max.setFixedHeight(WINDOW_BTN_HEIGHT)
        self.btn_max.clicked.connect(self._toggle_max_restore)

        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("winCloseBtn")
        self.btn_close.setFixedHeight(WINDOW_BTN_HEIGHT)
        self.btn_close.clicked.connect(self._window.close)

        # ── 오퍼레이터 모드 버튼 ──
        self.btn_home = QPushButton("◀  홈 인터페이스")
        self.btn_home.setObjectName("headerAction")
        self.btn_home.setToolTip("세팅 화면으로 돌아갑니다 (서버를 내려놓습니다).")
        self.btn_home.clicked.connect(self._handle_home)
        self.btn_home.hide()

        self.btn_quit = QPushButton("종료하기")
        self.btn_quit.setObjectName("headerAction")
        self.btn_quit.setToolTip("프로그램을 종료합니다.")
        self.btn_quit.clicked.connect(self._handle_quit)
        self.btn_quit.hide()

        layout.addWidget(self.title_label)
        layout.addStretch()
        # 오른쪽은 두 줄로 쌓는다: 위에 버튼, 아래에 부제.
        # 한 줄에 나란히 두면 창 버튼이 헤더 높이만큼 커져서, 창을 닫으려다
        # 부제를 누르는 일이 생긴다(버튼이 너무 크면 경계가 보이지 않는다).
        right = QVBoxLayout()
        right.setContentsMargins(0, 6, 8, 6)
        right.setSpacing(4)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(4)
        button_row.addStretch()
        button_row.addWidget(self.btn_home)
        button_row.addWidget(self.btn_quit)
        button_row.addWidget(self.btn_min)
        button_row.addWidget(self.btn_max)
        button_row.addWidget(self.btn_close)
        right.addLayout(button_row)

        subtitle_row = QHBoxLayout()
        subtitle_row.setContentsMargins(0, 0, 0, 0)
        subtitle_row.addStretch()
        subtitle_row.addWidget(self.subtitle_label)
        right.addLayout(subtitle_row)

        layout.addLayout(right)

    # ===== 오퍼레이터 버튼이 할 일 =====
    def bind_operator_actions(self, on_home=None, on_quit=None):
        """'홈 인터페이스' / '종료하기' 가 부를 함수를 받아 둔다.

        타이틀바는 서버를 내려놓는 절차를 모른다(그건 OperatorView 의 일이다).
        그래서 MainWindow 가 두 함수를 여기에 꽂아 준다.
        """
        self._on_home = on_home
        self._on_quit = on_quit

    def _handle_home(self):
        if self._on_home is not None:
            self._on_home()

    def _handle_quit(self):
        if self._on_quit is not None:
            self._on_quit()
            return
        self._window.close()

    # ===== 모드 전환 =====
    def set_mode(self, mode, process_name=""):
        """타이틀바 표시 모드 전환.

        - "setup"    : 좌측 정렬 앱 타이틀 + 최소화·최대화·닫기
        - "operator" : 가운데 아이콘 + 공정명, 홈 인터페이스 / 종료하기
        """
        self._mode = mode
        operator = mode == "operator"

        # 아이콘은 한 번에 한 자리에만 있어야 한다. 오퍼레이터에서는 가운데다.
        self.icon_label.setVisible(not operator)
        self.title_label.setVisible(not operator)
        for button in (self.btn_min, self.btn_max, self.btn_close):
            button.setVisible(not operator)
        for button in (self.btn_home, self.btn_quit):
            button.setVisible(operator)

        if operator:
            self.center_title.setText(process_name)
            self.center_box.show()
            self.center_box.raise_()
            self._reposition_center_box()
        else:
            self.center_box.hide()

    def _reposition_center_box(self):
        # 타이틀바 전체 폭에 맞춰 가운데 정렬 (창 기준 중앙)
        self.center_box.setGeometry(0, 0, self.width(), self.height())

    def resizeEvent(self, event):
        self._reposition_center_box()
        super().resizeEvent(event)

    def _toggle_max_restore(self):
        if self._window.isMaximized():
            self._window.showNormal()
            self.btn_max.setText("□")
        else:
            self._window.showMaximized()
            self.btn_max.setText("❐")   # restore

    # ===== 창 드래그 이동 =====
    def _can_move_window(self):
        """지금 창을 끌어 옮길 수 있는가.

        오퍼레이터 모드는 전체화면 고정이라 옮기지 않는다. 끌면 전체화면에서
        빠져나와 창이 작아지는데, 현장에서는 그것을 되돌릴 방법을 찾아야 한다.
        """
        return self._mode != "operator" and not self._window.isFullScreen()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        if not self._can_move_window():
            self._drag_pos = None
            event.accept()
            return
        handle = self._window.windowHandle()
        if (not self._window.isMaximized()
                and handle is not None
                and handle.startSystemMove()):
            self._drag_pos = None
            event.accept()
            return
        self._drag_pos = (
            event.globalPos() - self._window.frameGeometry().topLeft()
        )
        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            if not self._can_move_window():
                self._drag_pos = None
                return
            if self._window.isMaximized():
                # 최대화 상태에서 드래그 시작 시 복원 후 이동
                self._toggle_max_restore()
                self._drag_pos = (
                    event.globalPos() - self._window.frameGeometry().topLeft()
                )
            self._window.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self._can_move_window():
            self._toggle_max_restore()
