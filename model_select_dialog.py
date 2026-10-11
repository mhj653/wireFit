"""오퍼레이터 화면의 '모델 변경' 창.

    ┌ 모델 변경 ──────────────────────────────┐
    │ 지금 모델: Model1                        │
    │ ┌──────────────────────────────────────┐ │
    │ │ 1.  Model1  — 체결 포인트 2개  (지금) │ │
    │ │ 2.  Model2  — 체결 포인트 4개         │ │
    │ └──────────────────────────────────────┘ │
    │ 모델을 바꾸면 찍어 둔 이미지와 판정을     │
    │ 모두 버립니다.                           │
    │                        [취소] [변경]     │
    └──────────────────────────────────────────┘

왜 창을 띄우는가
────────────────
오퍼레이터 화면의 모델 탭은 **누를 수 없다**(260910 사용자 요구). 라벨을 잘못
눌러 제품이 바뀌면, 다른 제품의 체결 좌표가 로봇으로 나간다. 모델을 바꾸는 것은
'다른 제품이 들어왔다'는 뜻이라 자주 있는 일이 아니므로, 창을 띄워 한 번 더
확인받는 편이 맞다.

체결 포인트 개수를 함께 보여 준다. 이름만으로는 어느 것이 지금 라인에 흐르는
제품인지 헷갈릴 수 있고, 개수는 눈으로 바로 맞춰 볼 수 있는 값이다.

앞에 붙는 번호는 **모델 순서**다. 로봇이 모델 변경(103,1,n)을 보낼 때 쓰는 n 이
바로 이 번호라서, 사람이 손으로 바꿀 때와 트리거로 바꿀 때가 같은 번호를 보게
해 둔다. 목록에서 모델을 지우거나 더하면 번호가 밀리므로, 로봇 쪽 번호도 함께
맞춰야 한다.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from theme import CONTENT_SPACING, PAGE_SPACING

DIALOG_MIN_WIDTH = 380
LIST_MIN_HEIGHT = 160


class ModelSelectDialog(QDialog):
    """모델 하나를 고르는 모달 창. → selected_name() 또는 None."""

    def __init__(self, models, active_name=None, parent=None):
        """
        Args:
            models: [{"name": str, "points": [...]}, ...] 또는
                    [{"name": str, "fastening_points": [...]}, ...]
            active_name: 지금 쓰고 있는 모델 이름
        """
        super().__init__(parent)
        self.setWindowTitle("모델 변경")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setModal(True)
        self.setMinimumWidth(DIALOG_MIN_WIDTH)

        self._models = list(models or [])
        self._active_name = active_name

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(CONTENT_SPACING)

        current = QLabel(f"지금 모델: {active_name or '없음'}")
        current.setObjectName("dialogLabel")
        layout.addWidget(current)

        self.list_view = QListWidget()
        self.list_view.setObjectName("sampleTable")   # 표와 같은 흰 판 스타일
        self.list_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_view.setMinimumHeight(LIST_MIN_HEIGHT)
        layout.addWidget(self.list_view, 1)

        note = QLabel("모델을 바꾸면 찍어 둔 이미지와 OK/NG 판정을 모두 버립니다.")
        note.setObjectName("cameraStatus")
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QHBoxLayout()
        buttons.setSpacing(CONTENT_SPACING)
        buttons.addStretch()

        self.btn_cancel = QPushButton("취소")
        self.btn_cancel.setObjectName("secondaryBtn")
        self.btn_cancel.clicked.connect(self.reject)
        buttons.addWidget(self.btn_cancel)

        self.btn_ok = QPushButton("변경")
        self.btn_ok.setObjectName("primaryBtn")
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self._accept)
        buttons.addWidget(self.btn_ok)

        layout.addSpacing(PAGE_SPACING - CONTENT_SPACING)
        layout.addLayout(buttons)

        # ── 목록 채우기 ──
        # **버튼을 다 만든 뒤에** 채운다. addItem 은 currentRowChanged 를 쏘고
        # 그 처리기(_sync_button)가 btn_ok 를 만지는데, 버튼이 아직 없으면
        # AttributeError 가 난다. 슬롯에서 난 예외는 PyQt 가 프로그램을 통째로
        # 죽이므로(추적 없이 종료된다) 순서를 지켜야 한다.
        for order, entry in enumerate(self._models, start=1):
            name = entry.get("name", "")
            points = entry.get("points")
            if points is None:
                points = entry.get("fastening_points") or []
            # 번호는 모델 변경 트리거(103,1,n)의 n 과 같은 순서다.
            text = f"{order}.  {name}  —  체결 포인트 {len(points)}개"
            if name == active_name:
                text += "   (지금)"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, name)
            self.list_view.addItem(item)

        if active_name is not None:
            for row in range(self.list_view.count()):
                if self.list_view.item(row).data(Qt.UserRole) == active_name:
                    self.list_view.setCurrentRow(row)
                    break
        elif self.list_view.count():
            self.list_view.setCurrentRow(0)

        # 더블클릭으로도 바로 바꿀 수 있게 한다(창을 여는 것 자체가 이미
        # 의도적인 조작이므로, 그 안에서 한 번 더 막을 이유는 없다).
        self.list_view.itemDoubleClicked.connect(lambda _item: self._accept())
        self.list_view.currentRowChanged.connect(self._sync_button)
        self._sync_button()

    def _sync_button(self):
        """지금 쓰는 모델을 고른 상태면 '변경'을 잠근다(바꿀 것이 없다)."""
        name = self.selected_name()
        self.btn_ok.setEnabled(name is not None and name != self._active_name)

    def selected_name(self):
        item = self.list_view.currentItem()
        return None if item is None else item.data(Qt.UserRole)

    def _accept(self):
        if self.btn_ok.isEnabled():
            self.accept()

    @staticmethod
    def get_model(models, active_name=None, parent=None):
        """창을 띄워 고른 모델 이름을 돌려준다. 취소하면 None."""
        dialog = ModelSelectDialog(models, active_name, parent)
        if dialog.exec_() != QDialog.Accepted:
            return None
        return dialog.selected_name()
