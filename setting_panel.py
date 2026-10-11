"""이미지 보드 오른쪽에 세우는 설정 패널.

검출 방식(Hole Detection)이나 자세 기준(Pose Setting)을 바꾸면 필요한 파라미터가
통째로 달라진다. 그 줄들을 보드 **위**에 두면 방식을 바꿀 때마다 줄 수가 변해
이미지 보드의 높이가 함께 움직인다 — 값을 바꿔 가며 결과를 비교하는 화면에서
사진 크기가 매번 달라지면 눈으로 견줄 수가 없다.

그래서 방식별 파라미터는 보드 **오른쪽**의 고정 폭 패널에 세로로 쌓는다. 패널
폭이 고정이므로 어느 방식을 골라도 보드가 차지하는 자리는 그대로다. 내용이
길어지면 패널만 스크롤된다(보드를 밀어내지 않는다).

    ┌───────────────────────┐  ┌──────────────┐
    │                       │  │ Parameter    │
    │      이미지 보드       │  │  Setting     │
    │                       │  ├──────────────┤
    │                       │  │ Flange Pose  │
    └───────────────────────┘  └──────────────┘
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from theme import (
    SETTING_GROUP_SPACING,
    SETTING_PANEL_WIDTH,
    SETTING_ROW_SPACING,
)

GROUP_MARGINS = (10, 8, 10, 10)


class SettingGroup(QFrame):
    """제목 하나 + 그 아래 줄들. 흰 판으로 띄워 회색 바탕과 갈라 둔다.

    body 에 위젯이나 레이아웃을 넣는다. 줄 하나를 만들 때는 row() 를 쓰면
    '라벨 [입력]' 형태로 가로로 붙는다.
    """

    def __init__(self, caption, parent=None):
        super().__init__(parent)
        self.setObjectName("settingGroup")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(*GROUP_MARGINS)
        outer.setSpacing(SETTING_ROW_SPACING)

        self.caption = QLabel(caption)
        self.caption.setObjectName("groupCaption")
        outer.addWidget(self.caption)

        self.body = QVBoxLayout()
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(SETTING_ROW_SPACING)
        outer.addLayout(self.body)

    def add_widget(self, widget):
        self.body.addWidget(widget)
        return widget

    def add_layout(self, layout):
        self.body.addLayout(layout)
        return layout

    def row(self, label_text, *widgets, stretch_label=False):
        """'라벨 [위젯] [위젯]' 한 줄.

        라벨이 없으면(빈 문자열) 위젯만 나란히 놓는다. stretch_label 을 켜면
        라벨이 남는 폭을 먹어 입력이 오른쪽에 붙는다 — 입력칸 여러 줄의 왼쪽
        끝을 맞추고 싶을 때 쓴다.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SETTING_ROW_SPACING)
        if label_text:
            label = QLabel(label_text)
            label.setObjectName("paramLabel")
            row.addWidget(label, 1 if stretch_label else 0)
        elif stretch_label:
            row.addStretch(1)
        for widget in widgets:
            row.addWidget(widget)
        if not stretch_label:
            row.addStretch(1)
        self.body.addLayout(row)
        return row

    def note(self, text):
        """설명 한 덩어리. 좁은 패널이라 줄바꿈을 허용한다."""
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        label.setWordWrap(True)
        label.setMaximumWidth(SETTING_PANEL_WIDTH - GROUP_MARGINS[0] - GROUP_MARGINS[2] - 12)
        self.body.addWidget(label)
        return label


class SettingPanel(QScrollArea):
    """고정 폭 세로 패널. 안에 SettingGroup 들을 쌓는다.

    스크롤 영역인 이유: 방식마다 줄 수가 달라 어떤 방식은 창 높이를 넘길 수
    있다. 그때 패널만 스크롤되면 이미지 보드는 영향을 받지 않는다 — 반대로
    패널이 늘어나게 두면 보드가 눌려 작아진다.
    """

    def __init__(self, width=SETTING_PANEL_WIDTH, parent=None):
        super().__init__(parent)
        self.setObjectName("settingPanel")
        self.setWidgetResizable(True)
        self.setFixedWidth(width)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)

        self._content = QWidget()
        self._content.setFixedWidth(max(1, width - 14))
        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(SETTING_GROUP_SPACING)
        self._layout.addStretch(1)
        self.setWidget(self._content)

    def add_group(self, group):
        """그룹을 맨 아래(스트레치 앞)에 붙인다."""
        self._layout.insertWidget(self._layout.count() - 1, group)
        return group

    def add_widget(self, widget):
        self._layout.insertWidget(self._layout.count() - 1, widget)
        return widget
