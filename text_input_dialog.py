"""이름 하나를 받는 간단한 입력 창.

'세팅 추가'처럼 짧은 텍스트를 받아야 할 때 쓴다. 확인 버튼을 누르면 그 자리에서
검사해서, 잘못된 값이면 창을 닫지 않고 이유를 아래에 띄운다. 창이 닫힌 뒤에
따로 오류를 알리면 방금 입력한 내용이 사라져 다시 쳐야 하기 때문이다.

    name = TextInputDialog.get_text(
        parent, "세팅 추가", "세팅 이름", "생성", "취소", validator=my_check
    )
    if name is not None:
        ...

validator는 (ok, 정리된값 또는 오류메시지) 를 돌려주는 함수다.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from theme import CONTENT_SPACING, PAGE_SPACING


class TextInputDialog(QDialog):
    """텍스트 한 줄 + [확인] [취소] 로 이루어진 창."""

    def __init__(
        self,
        title,
        label,
        ok_text="확인",
        cancel_text="취소",
        initial="",
        validator=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        # 물음표 버튼은 하는 일이 없으므로 없앤다.
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setModal(True)

        self._validator = validator
        self._value = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(CONTENT_SPACING)

        caption = QLabel(label)
        caption.setObjectName("dialogLabel")
        layout.addWidget(caption)

        self.input = QLineEdit(initial)
        self.input.setObjectName("nameInput")
        self.input.returnPressed.connect(self._on_accept)
        layout.addWidget(self.input)

        # 평소에는 자리만 잡고 있다가 잘못된 입력일 때 사유를 띄운다.
        self.error_label = QLabel("")
        self.error_label.setObjectName("dialogError")
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(CONTENT_SPACING)

        self.btn_ok = QPushButton(ok_text)
        self.btn_ok.setObjectName("primaryBtn")
        self.btn_ok.setDefault(True)
        self.btn_ok.clicked.connect(self._on_accept)

        self.btn_cancel = QPushButton(cancel_text)
        self.btn_cancel.setObjectName("secondaryBtn")
        self.btn_cancel.clicked.connect(self.reject)

        buttons.addStretch()
        buttons.addWidget(self.btn_cancel)
        buttons.addWidget(self.btn_ok)
        layout.addSpacing(PAGE_SPACING - CONTENT_SPACING)
        layout.addLayout(buttons)

        self.input.setFocus()

    def value(self):
        """검사를 통과한 값. 취소했으면 None."""
        return self._value

    def _on_accept(self):
        text = self.input.text()
        if self._validator is None:
            self._value = text.strip()
            self.accept()
            return

        ok, result = self._validator(text)
        if not ok:
            # 창을 닫지 않고 사유만 보여 준다(입력한 내용을 지키기 위해).
            self.error_label.setText(result)
            self.input.setFocus()
            self.input.selectAll()
            return
        self._value = result
        self.accept()

    @staticmethod
    def get_text(
        parent,
        title,
        label,
        ok_text="확인",
        cancel_text="취소",
        initial="",
        validator=None,
    ):
        """창을 띄우고 입력값을 돌려준다. 취소했으면 None."""
        dialog = TextInputDialog(
            title, label, ok_text, cancel_text, initial, validator, parent
        )
        if dialog.exec_() == QDialog.Accepted:
            return dialog.value()
        return None
