"""오퍼레이터 진입 설정.

두 줄로 나눈다. 공정명은 늘 적는 값이고 저장은 켰다 껐다 하는 값이라
성격이 달라서, 한 줄에 몰아 두면 눈이 어디까지가 한 묶음인지 헷갈린다.

    공정명 [____]
    데이터 저장 [ ●]   [경로 선택] [D:\\...\\Logs                        ]

경로는 길어지기 쉬우므로 남는 가로 공간을 전부 쓰는 입력창에 담는다
(줄바꿈으로 아래로 흐르지 않게 한다). 직접 고치지는 못하고 '경로 선택'
으로만 바꾼다 — 손으로 적은 경로가 틀렸을 때 저장 시점에야 알게 되는 걸
막으려는 것이다.
"""

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from folder_dialog import select_folder
from image_saver import default_log_root
from theme import CONTENT_SPACING
from toggle_switch import ToggleSwitch


class OperatorControlPanel(QWidget):
    """공정명 줄 + 저장 설정 줄."""

    process_name_changed = pyqtSignal(str)
    save_enabled_changed = pyqtSignal(bool)
    save_path_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._save_path = default_log_root()

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(CONTENT_SPACING)

        # ── 1줄: 공정명 ──
        name_row = QHBoxLayout()
        name_row.setSpacing(CONTENT_SPACING)
        name_row.addWidget(QLabel("공정명"))

        self.input_process = QLineEdit()
        self.input_process.setObjectName("processInput")
        self.input_process.setPlaceholderText("공정명을 입력하세요.")
        # 너비는 theme.py 의 QLineEdit#processInput(min-width)이 정한다.
        self.input_process.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.input_process.textChanged.connect(self.process_name_changed.emit)
        name_row.addWidget(self.input_process)
        name_row.addStretch()
        outer.addLayout(name_row)

        # ── 2줄: 저장 on/off + 저장 경로 ──
        save_row = QHBoxLayout()
        save_row.setSpacing(CONTENT_SPACING)

        save_row.addWidget(QLabel("데이터 저장"))
        self.toggle_save = ToggleSwitch()
        self.toggle_save.toggled.connect(self.save_enabled_changed.emit)
        save_row.addWidget(self.toggle_save)

        save_row.addSpacing(CONTENT_SPACING)
        self.btn_path = QPushButton("경로 선택")
        self.btn_path.setObjectName("secondaryBtn")
        self.btn_path.clicked.connect(self._on_select_path)
        save_row.addWidget(self.btn_path)

        self.path_display = QLineEdit(self._save_path)
        self.path_display.setObjectName("pathDisplay")
        self.path_display.setReadOnly(True)
        self.path_display.setCursorPosition(0)
        self.path_display.setToolTip(self._save_path)
        # 남는 가로 공간을 전부 차지하게 해서 경로가 아래로 밀리지 않게 한다.
        self.path_display.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        save_row.addWidget(self.path_display, 1)
        outer.addLayout(save_row)

    # ===== 사용자 조작 =====
    def _on_select_path(self):
        path = select_folder(self, "이미지 저장 경로 선택", self._save_path)
        if not path:
            return
        self.set_save_path(path)
        self.save_path_changed.emit(path)

    # ===== 읽기 =====
    def get_process_name(self):
        return self.input_process.text().strip() or "공정 미지정"

    def get_process_text(self):
        return self.input_process.text()

    def is_save_enabled(self):
        return self.toggle_save.isChecked()

    def get_save_path(self):
        return self._save_path

    # ===== 쓰기 =====
    def set_process_name(self, name):
        self.input_process.setText(name)

    def set_save_enabled(self, enabled):
        # 프로그램이 상태를 맞추는 경우이므로 toggled 를 다시 쏘지 않는다.
        self.toggle_save.set_checked_silently(bool(enabled))

    def set_save_path(self, path):
        self._save_path = path
        self.path_display.setText(path)
        self.path_display.setCursorPosition(0)   # 긴 경로면 앞부터 보이게
        self.path_display.setToolTip(path)
