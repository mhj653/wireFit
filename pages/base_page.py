from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from theme import CONTENT_SPACING, PAGE_MARGINS, PAGE_SPACING


class BasePage(QWidget):
    """모든 기능 탭 페이지가 상속하는 기본 클래스.

    새 기능을 추가할 때는 이 클래스를 상속하고 title을 지정한 뒤,
    PythonUI.py의 PAGES 목록에 등록하면 사이드바 탭과 메인 화면에 자동으로 연결된다.

    content_layout에 위젯을 추가하면 페이지 제목 아래에 배치된다.

    expand_content = True 로 설정하면 콘텐츠가 남는 공간을 모두 채우도록
    세로로 확장된다(영상/캔버스 등 창 크기에 맞춰 늘어나야 하는 페이지용).
    기본값(False)은 콘텐츠를 상단에 정렬한다.
    """

    title = "Base"
    expand_content = False
    # 사이드바 탭에 그릴 아이콘 이름 (nav_icons._PAINTERS 의 키).
    # 표에 없는 이름이면 기본 아이콘(동그라미)이 그려진다.
    nav_icon = "default"

    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(*PAGE_MARGINS)
        outer.setSpacing(PAGE_SPACING)

        page_title = QLabel(self.title)
        page_title.setObjectName("pageTitle")
        outer.addWidget(page_title)

        # 하위 클래스는 self.content_layout 에 위젯을 추가한다.
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(CONTENT_SPACING)

        if self.expand_content:
            # 콘텐츠가 남는 공간을 모두 차지하도록 stretch=1 로 추가
            outer.addLayout(self.content_layout, 1)
        else:
            self.content_layout.setAlignment(Qt.AlignTop)
            outer.addLayout(self.content_layout)
            outer.addStretch()
