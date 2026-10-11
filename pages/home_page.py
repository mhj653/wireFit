"""홈(시작) 페이지 — 사이드바의 첫 번째 탭.

BasePage를 상속한 가장 단순한 예시. 안내 문구만 표시한다.
"""

from PyQt5.QtWidgets import QLabel

from pages.base_page import BasePage


class HomePage(BasePage):
    title = "Home"  # 사이드바 탭에 표시될 이름
    nav_icon = "home"  # 사이드바 탭 아이콘 (nav_icons)

    # BasePage가 제목 라벨과 self.content_layout(빈 세로 레이아웃)을 만들어 주므로,
    # 여기서는 content_layout에 위젯을 추가하기만 하면 된다.

    def __init__(self, parent=None):
        super().__init__(parent)

        self.content_layout.addWidget(
            QLabel("StreoVision Fasten Automation S/W\n 1. 사용 카메라 : D405\n 2. 필요 기능 : EIH Calib, Hole Detection\n" \
            " 3. 홀 인식 알고리즘 : 블롭 분석\n 4. EIH 캘리브레이션 기능 추가 (체크보드 간격 30mm)")
        )
