"""뎁스 후처리 필터 패널 — Camera Selection 페이지의 뎁스맵 오른쪽에 붙는다.

RealSense Viewer 의 Post-Processing 패널과 같은 내용을 같은 차례로 보여 준다.
구성만 닮게 하고 색·간격·위젯 모양은 theme.py 를 그대로 따른다.

    ▼ 데시메이션                    [ ]
        배율            [ 2 ]
    ▼ 거리 threshold                [x]
        최소 거리       [  50 mm ]
        최대 거리       [ 600 mm ]
    ...

위젯은 realsense_postprocess.FILTERS 스펙에서 자동으로 만든다. 필터를 더하거나
옵션을 바꿀 때 이 파일은 손대지 않아도 되게 하려는 것이다.

값이 바뀌면 곧바로 realsense_postprocess.set_settings() 로 저장한다. 저장이 곧
공유이므로(모듈 전역), 다른 페이지가 다음 프레임부터 같은 뎁스를 본다.
"""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

import realsense_postprocess as postprocess
from theme import CONTENT_SPACING

PANEL_WIDTH = 250          # 뎁스맵 옆에 붙는 폭
OPTION_INDENT = 12         # 필터 이름 아래 옵션을 들여쓰는 정도
ROW_SPACING = 4


class _FilterBlock(QFrame):
    """필터 하나 — 제목 줄(이름 + 켜기 토글)과 그 아래 옵션들."""

    changed = pyqtSignal()

    def __init__(self, spec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.setObjectName("paramCard")
        self._editors = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(ROW_SPACING)

        # 제목 줄: 이름과 켜기 버튼. 버튼을 눌러 켜고 끈다.
        self.toggle = QPushButton(spec.label)
        self.toggle.setObjectName("filterToggle")
        self.toggle.setCheckable(True)
        self.toggle.setCursor(Qt.PointingHandCursor)
        if spec.note:
            self.toggle.setToolTip(spec.note)
        self.toggle.toggled.connect(self._on_toggled)
        layout.addWidget(self.toggle)

        # 옵션들. 들여써서 어느 필터에 속하는지 보이게 한다.
        self._options = QWidget()
        option_layout = QVBoxLayout(self._options)
        option_layout.setContentsMargins(OPTION_INDENT, 0, 0, 0)
        option_layout.setSpacing(ROW_SPACING)
        for option in spec.options:
            option_layout.addWidget(self._build_option(option))
        layout.addWidget(self._options)

    def _build_option(self, option):
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)

        label = QLabel(option.label)
        label.setObjectName("paramLabel")
        box.addWidget(label)

        if option.choices:
            editor = QComboBox()
            editor.setObjectName("viewCombo")
            for text, value in option.choices:
                editor.addItem(text, value)
            editor.currentIndexChanged.connect(lambda _i: self.changed.emit())
        else:
            editor = QDoubleSpinBox()
            editor.setObjectName("paramSpin")
            editor.setRange(option.minimum, option.maximum)
            editor.setSingleStep(option.step)
            editor.setDecimals(option.decimals)
            editor.setSuffix(option.suffix)
            editor.setKeyboardTracking(False)   # 타이핑 도중 매 글자마다 저장하지 않게
            editor.valueChanged.connect(lambda _v: self.changed.emit())
        editor.setToolTip(option.label)
        self._editors[option.key] = editor
        box.addWidget(editor)
        return holder

    def _on_toggled(self, checked):
        # 꺼 둔 필터의 옵션까지 만질 수 있으면 무엇이 실제로 걸려 있는지 헷갈린다.
        self._options.setEnabled(checked)
        self.changed.emit()

    def load(self, state):
        """설정 dict 한 조각을 위젯에 싣는다."""
        enabled = bool(state.get("enabled", self.spec.default_enabled))
        self.toggle.setChecked(enabled)
        self._options.setEnabled(enabled)
        values = state.get("values") or {}
        for option in self.spec.options:
            editor = self._editors[option.key]
            value = values.get(option.key, option.default)
            if isinstance(editor, QComboBox):
                index = editor.findData(int(value))
                editor.setCurrentIndex(max(0, index))
            else:
                editor.setValue(float(value))

    def collect(self):
        """위젯에서 설정 dict 한 조각을 만든다."""
        values = {}
        for option in self.spec.options:
            editor = self._editors[option.key]
            values[option.key] = (
                editor.currentData()
                if isinstance(editor, QComboBox)
                else editor.value()
            )
        return {"enabled": self.toggle.isChecked(), "values": values}


class PostProcessPanel(QScrollArea):
    """후처리 필터 전체 패널. 값이 바뀌면 곧바로 저장하고 changed 를 낸다."""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("postProcessPanel")
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setFixedWidth(PANEL_WIDTH)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, CONTENT_SPACING, 0)
        layout.setSpacing(CONTENT_SPACING)

        title = QLabel("뎁스 후처리")
        title.setObjectName("paramTitle")
        layout.addWidget(title)

        self._blocks = []
        self._loading = False
        for spec in postprocess.FILTERS:
            block = _FilterBlock(spec)
            block.changed.connect(self._on_changed)
            layout.addWidget(block)
            self._blocks.append(block)

        if not self._blocks:
            # pyrealsense2 가 없는 PC. 빈 패널만 두면 왜 비었는지 알 수 없다.
            note = QLabel("pyrealsense2 가 없어 후처리를 쓸 수 없습니다.")
            note.setObjectName("paramLabel")
            note.setWordWrap(True)
            layout.addWidget(note)

        self.btn_reset = QPushButton("기본값으로")
        self.btn_reset.setObjectName("secondaryBtn")
        self.btn_reset.clicked.connect(self._on_reset)
        layout.addWidget(self.btn_reset)

        layout.addStretch()
        body.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.setWidget(body)

        self.reload()

    def reload(self):
        """저장된 설정을 위젯에 싣는다."""
        self._loading = True
        try:
            state = postprocess.settings()
            for block in self._blocks:
                block.load(state.get(block.spec.key, {}))
        finally:
            self._loading = False

    def _on_changed(self):
        if self._loading:
            return
        postprocess.set_settings(
            {block.spec.key: block.collect() for block in self._blocks}
        )
        self.changed.emit()

    def _on_reset(self):
        postprocess.reset_to_defaults()
        self.reload()
        self.changed.emit()
