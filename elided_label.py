"""글자가 길어도 페이지 폭을 밀어내지 않는 라벨.

QLabel 은 자기 글자 길이를 최소 폭으로 요구한다. 상태문구·처리 시간·연결 메시지처럼
내용이 길어지는 라벨을 그대로 두면 **페이지 최소 폭이 글자 길이가 되어** 창이
모니터보다 넓어진다. 260907 에 두 페이지에서 실제로 그랬다.

  - Hole Detection : 상태문구에 벌어짐(mm)을, 처리 시간에 단계별 소요를 덧붙이자
    오른쪽이 잘렸다.
  - Camera Selection : 연결 메시지(장치·프리셋·해상도 경고, 약 1800px)를 넣으면
    페이지 최소 폭이 1576 → 3212px 로 튀었다.

여기서는 최소 폭을 0 으로 내려놓고, 주어진 폭에 안 들어가면 '…' 로 줄인다. 원문은
툴팁에 남겨 마우스를 올리면 다 읽을 수 있다. text() 는 원문을 돌려준다 — 시험과
로그가 화면에 보이는 글자가 아니라 뜻을 읽어야 한다.

여러 페이지가 쓰므로 페이지 모듈이 아니라 여기 둔다(range_slider, text_input_dialog
와 같은 자리). hole_detection_page 가 camera_capture_page 를 import 하므로 한쪽
페이지에 두면 순환 import 가 된다.
"""

from PyQt5.QtCore import QSize, Qt
from PyQt5.QtWidgets import QLabel


class ElidedLabel(QLabel):
    def __init__(self, text="", parent=None, elide=Qt.ElideMiddle):
        """
        Args:
            elide: 어디를 줄일지. 상태문구는 앞(홀 수)과 뒤(경고)가 모두 중요해
                가운데를 줄이고(ElideMiddle), 처리 시간은 앞(캡쳐/분석)이 먼저
                읽혀야 해서 뒤를 줄인다(ElideRight).
        """
        super().__init__("", parent)
        self._full = ""
        self._elide = elide
        self.setText(text)

    def setText(self, text):
        self._full = text or ""
        self.setToolTip(self._full)
        self._apply()

    def text(self):
        return self._full

    def minimumSizeHint(self):
        # 폭은 요구하지 않는다. 이것이 이 클래스의 존재 이유다.
        return QSize(0, super().minimumSizeHint().height())

    def sizeHint(self):
        # 선호 폭은 **원문** 기준으로 준다. 줄인 글자 기준으로 주면 레이아웃이
        # 그만큼만 내주고, 그러면 더 줄이고, 다시 덜 내주는 되먹임이 생긴다.
        width = self.fontMetrics().horizontalAdvance(self._full) + 4
        return QSize(width, super().sizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply()

    def _apply(self):
        width = self.contentsRect().width()
        if width <= 0:
            shown = self._full          # 아직 레이아웃 전이면 그대로 둔다
        else:
            shown = self.fontMetrics().elidedText(self._full, self._elide, width)
        super().setText(shown)
