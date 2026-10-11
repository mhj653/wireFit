"""애플리케이션 전역 테마.

디자인 관련 설정은 모두 이 파일에 모여 있다.
- 색상 팔레트  : 색감 변경
- 레이아웃 수치 : 크기/여백 변경 (사이드바 너비, 헤더 높이, 아이콘 크기 등)
- STYLESHEET   : 폰트/모양/효과 (QSS)

각 모듈(PythonUI.py, titlebar.py 등)은 이 상수들을 import 해서 사용하므로,
디자인을 바꿀 때는 원칙적으로 이 파일만 수정하면 된다.

260910 에 색감을 통째로 바꿨다(밝은 흰 바탕 -> 회색 바탕 + 검정 헤더). 이유는
현장 조명 아래에서 흰 바탕이 이미지 보드와 구분되지 않아, 어디까지가 사진이고
어디부터가 UI 인지 눈으로 가릴 수 없었기 때문이다. 이제 UI 는 회색, 사진이
들어가는 자리만 흰색 + 파란 테두리다.
"""

# ======================================================
# 색상 팔레트
# ======================================================
HEADER_BG = "#262626"          # 상단 타이틀바 (검정)
HEADER_TEXT = "#ffffff"
HEADER_SUBTEXT = "#d9d9d9"     # 'Last update' 처럼 덜 중요한 글자

SIDEBAR_BG = "#8c8c8c"         # 좌측 사이드바 (중간 회색)
SIDEBAR_TEXT = "#1c1c1c"
SIDEBAR_HOVER = "#9b9b9b"
SIDEBAR_SELECTED = "#c8c8c8"   # 고른 탭 (밝게)
SIDEBAR_LINE = "#6e6e6e"       # 탭 사이 구분선

ACCENT = "#4472c4"             # 강조(사진 테두리, 포커스)
CONTENT_BG = "#a6a6a6"         # 본문 바탕 (회색)
CONTENT_TEXT = "#111111"
SURFACE = "#ffffff"            # 사진/입력칸처럼 '내용이 들어가는' 흰 바탕
BORDER = "#666666"

CONTROL_BG = "#d9d9d9"         # 버튼/콤보 바탕
CONTROL_HOVER = "#e8e8e8"
CONTROL_PRESSED = "#c2c2c2"
CONTROL_BORDER = "#4d4d4d"
CONTROL_DISABLED_BG = "#b0b0b0"
CONTROL_DISABLED_TEXT = "#7a7a7a"

MUTED_TEXT = "#3a3a3a"         # 부제/설명 글자 (회색 바탕 위라 너무 옅으면 안 읽힌다)

# ======================================================
# 레이아웃 수치 (px)
# ======================================================
SIDEBAR_WIDTH = 124       # 좌측 사이드바 너비 (아이콘 + 라벨이 들어간다)
HEADER_HEIGHT = 64        # 상단 타이틀바 높이
NAV_HEIGHT = 56           # 하단 페이지 네비게이션 바 높이
NAV_INDICATOR_WIDTH = 120  # 하단 네비 "현재/전체" 표시기 너비
ICON_SIZE = 30            # 헤더 좌측 아이콘 크기
WINDOW_BTN_WIDTH = 40     # 최소화/최대화/닫기 버튼 너비
WINDOW_BTN_HEIGHT = 26    # 창 버튼은 헤더보다 낮게 (헤더 오른쪽 위에 붙는다)
RESIZE_MARGIN = 6         # 프레임리스 창 가장자리 리사이즈 감지 두께

# ===== 사이드바 탭 한 칸 =====
# 아이콘을 위에, 라벨을 아래에 둔다. 칸이 세로로 길어진 만큼 아이콘이 들어갈
# 자리가 생겼고, 그 자리를 nav_icons 가 그려 채운다.
NAV_ITEM_HEIGHT = 100     # 탭 한 칸 높이
NAV_PLATE_WIDTH = 78      # 아이콘을 담는 네모(판) 너비
NAV_PLATE_HEIGHT = 46     # 아이콘 판 높이
NAV_ICON_SIZE = 26        # 판 안에 그려지는 아이콘 한 변
NAV_PLATE_TOP = 12        # 칸 위쪽에서 판까지 여백
NAV_LABEL_GAP = 10        # 판과 라벨 사이
NAV_PLATE_BG = "#d9d9d9"
NAV_PLATE_BG_SELECTED = "#f2f2f2"
NAV_PLATE_BORDER = "#3f3f3f"
NAV_LABEL_SIZE = 11       # 라벨 글자 크기(pt 아님, px)

PAGE_MARGINS = (24, 18, 24, 18)  # 페이지 바깥 여백 (좌, 상, 우, 하)
PAGE_SPACING = 14                # 페이지 제목과 콘텐츠 사이 간격
CONTENT_SPACING = 10             # 콘텐츠 내부 위젯 간 간격

VIDEO_MIN_WIDTH = 320      # 카메라 영상 최소 너비
VIDEO_MIN_HEIGHT = 240     # 카메라 영상 최소 높이

# ===== 이미지 보드 휠 확대 / 우클릭 드래그 이동 =====
# 오퍼레이터 화면을 뺀 모든 이미지 보드가 같은 값을 쓴다(zoom_pan.py).
ZOOM_MIN = 1.0             # 1.0 = 화면에 꽉 맞춘 상태. 그보다 작게는 줄이지 않는다.
ZOOM_MAX = 20.0            # 1픽셀을 20배까지
ZOOM_STEP = 1.15           # 휠 한 칸당 배율
ZOOM_RESET_HINT = "휠: 확대/축소 · 우클릭 드래그: 이동 · 더블클릭: 원래대로"

LED_SIZE = 14              # 연결 상태 표시등 지름
LED_OFF = "#6f6f6f"        # 미연결 (회색 바탕에서 보이도록 진하게)
LED_ON = "#22a45d"         # 연결 (녹색)

COMBO_MIN_WIDTH = 260      # 카메라 선택 콤보 최소 너비 (장치명 + 시리얼이 들어감)
VIEW_COMBO_MIN_WIDTH = 110  # 이미지 보드에 띄울 화면 종류 선택 콤보 최소 너비

# 좌우로 미끄러지는 on/off 스위치: 직접 그리는 위젯이라 QSS가 아닌 상수로 둔다.
SWITCH_WIDTH = 46
SWITCH_HEIGHT = 24
SWITCH_KNOB_MARGIN = 3         # 손잡이와 트랙 가장자리 사이 여백
SWITCH_ANIM_MS = 140           # 좌우로 미끄러지는 시간
SWITCH_ON_BG = "#22a45d"       # 켜짐: 초록
SWITCH_OFF_BG = "#d0342c"      # 꺼짐: 빨강
SWITCH_DISABLED_BG = "#8f8f8f" # 비활성: 회색
SWITCH_KNOB_BG = "#ffffff"     # 손잡이

# 레인지 슬라이더(두 손잡이): 직접 그리는 위젯이라 QSS가 아닌 상수로 둔다.
RANGE_SLIDER_MIN_WIDTH = 150   # 슬라이더 최소 너비
RANGE_SLIDER_MAX_WIDTH = 220   # 한 행에 다른 입력이 함께 들어가므로 너무 넓어지지 않게
RANGE_SLIDER_HEIGHT = 26       # 슬라이더 전체 높이 (손잡이가 들어갈 세로 공간)
RANGE_GROOVE_HEIGHT = 4        # 홈 두께
RANGE_HANDLE_RADIUS = 8        # 손잡이 반지름
RANGE_GROOVE_BG = "#7d7d7d"    # 홈 배경 (선택 구간 밖)
RANGE_GROOVE_ACTIVE = ACCENT   # 선택된 구간
RANGE_HANDLE_BG = "#ffffff"    # 손잡이 안쪽
RANGE_HANDLE_BORDER = ACCENT   # 손잡이 테두리
RANGE_HANDLE_BORDER_WIDTH = 2

PARAM_BOX_WIDTH = 76           # 파라미터 입력 박스 너비
FILTER_COMBO_MIN_WIDTH = 190   # 필터 종류 선택 콤보 (항목명이 길다)
PROFILE_COMBO_MIN_WIDTH = 160  # 세팅(체결 위치) 선택 콤보
BOARD_COMBO_MIN_WIDTH = 190    # 체커보드 크기 선택 콤보
POSE_BOX_WIDTH = 92            # 플랜지 포즈 입력칸 너비
POSE_PANEL_WIDTH = 190         # 이미지 보드 우측 포즈 입력 패널 너비
SAMPLE_TABLE_WIDTH = 150       # 이미지 보드 좌측 수집 목록 표 (이름 한 열만)
ICON_BTN_SIZE = 28             # +/- 같은 정사각 아이콘 버튼 한 변
NAME_INPUT_MIN_WIDTH = 260     # 이름 입력 창의 텍스트 박스 최소 너비

# ===== 이미지 보드 오른쪽 설정 패널 =====
# 검출 방식(또는 자세 기준)을 바꿔도 **이미지 보드 크기가 변하지 않게** 하려고
# 방식별 파라미터를 보드 위가 아니라 오른쪽에 세로로 세운다. 폭이 고정이므로
# 방식이 바뀌어도 보드가 차지하는 자리는 그대로다.
SETTING_PANEL_WIDTH = 320      # 파라미터 패널 너비
SETTING_GROUP_SPACING = 8      # 패널 안 그룹 사이 간격
SETTING_ROW_SPACING = 6        # 그룹 안 줄 사이 간격

# ROI(관심 영역) 오버레이: 이미지 위에 직접 그리는 색이라 QSS가 아닌 상수로 둔다.
ROI_BORDER = "#ffb020"        # 확정된 ROI 테두리 (주황)
ROI_DRAG_BORDER = "#2b6cff"   # 드래그 중인 임시 박스 (파랑)
ROI_BORDER_WIDTH = 2          # ROI 테두리 두께
ROI_DIM_ALPHA = 90            # ROI 바깥을 어둡게 덮는 정도 (0~255)

# 뎁스 직선 프로파일 선택/그래프
DEPTH_PROFILE_LINE = "#ff3b30"
DEPTH_PROFILE_GRID = "#c9c9c9"

# 토글 버튼 글자 영역 너비를 고정한다. 라벨이 '카메라 연결'(74px) <-> '연결 해제'로
# 바뀌어도 버튼 크기가 흔들리지 않도록 가장 긴 라벨 + 여유를 준 값.
TOGGLE_BTN_TEXT_WIDTH = 80

# 연결 토글 버튼이 '연결됨' 상태일 때의 강조색
CONNECTED_BG = "#cdeeda"
CONNECTED_TEXT = "#136c3c"
CONNECTED_BORDER = "#22a45d"

# ===== 오퍼레이터 OK / NG 판정 =====
# 촬영 결과가 로봇으로 나갈 값인지(OK) 실패 신호인지(NG)를 한눈에 보여 준다.
# 배지는 **이미지 보드마다 하나씩** 그 보드의 우측 상단에 얹힌다(260910 요구) —
# 화면 한 군데에 두면 네 칸을 찍었을 때 마지막 판정만 남아, 어느 칸이 NG 였는지
# 알 수 없다. 로그를 읽을 틈이 없는 현장에서 이 색만 보고 판단한다.
JUDGE_BADGE_WIDTH = 92
JUDGE_BADGE_HEIGHT = 40
JUDGE_BADGE_MARGIN = 6     # 보드 테두리와 배지 사이
JUDGE_OK_BG = "#00a651"
JUDGE_NG_BG = "#ec1c24"
JUDGE_TEXT = "#ffffff"

# ===== 오퍼레이터 상단 상태 카드 =====
# 타이틀바 바로 아래 왼쪽에 나란히 놓이는 카드. 모서리 하나를 잘라낸(책갈피)
# 모양이라 QSS 로는 그릴 수 없어 operator_panels 가 QPainter 로 그린다.
CARD_HEIGHT = 62
CARD_MIN_WIDTH = 230
CARD_CHAMFER = 14          # 잘라내는 모서리 크기
CARD_BG = "#262626"
CARD_BORDER = "#d9d9d9"
CARD_TEXT = "#ffffff"
CARD_OK_TEXT = "#3ddc84"   # '연결됨' 처럼 좋은 상태
CARD_NG_TEXT = "#ff6b6b"   # '연결 안 됨' 처럼 손을 봐야 하는 상태
CARD_ICON_PLATE = 40       # 카드 안 아이콘 판 한 변
CARD_ICON_SIZE = 24

# ===== 오퍼레이터 모델 탭 (책갈피) =====
# 이미지 보드 위에 모델 이름을 늘어놓는다. 고른 모델만 밝은 회색이라 어느 것이
# 켜져 있는지 색으로 바로 갈린다.
MODEL_TAB_HEIGHT = 34
MODEL_TAB_MIN_WIDTH = 150
MODEL_TAB_CHAMFER = 16     # 오른쪽 위를 잘라 책갈피 모양으로
MODEL_TAB_SPACING = 4
MODEL_TAB_BG = "#262626"
MODEL_TAB_BG_HOVER = "#3d3d3d"
MODEL_TAB_BG_SELECTED = "#8c8c8c"
MODEL_TAB_TEXT = "#e6e6e6"
MODEL_TAB_TEXT_SELECTED = "#111111"
MODEL_TAB_BORDER = "#d9d9d9"

# ===== 오퍼레이터 로그 창 =====
# 흰 판이 아니라 회색이다(260910 요구: 테마와 통일). 사진이 들어가는 자리만
# 흰색이라는 이 앱의 규칙과도 맞는다 — 로그는 사진이 아니다.
OPERATOR_LOG_BG = "#9e9e9e"
OPERATOR_LOG_TEXT = "#141414"

STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "Malgun Gothic", sans-serif;
    font-size: 12px;
}}

QMainWindow, QWidget#central {{
    background: {CONTENT_BG};
}}

/* ===== 상단 헤더바 ===== */
QFrame#header {{
    background: {HEADER_BG};
    border: none;
}}

QLabel#headerIcon {{
    padding-left: 16px;
    padding-right: 0px;
}}

QLabel#headerTitle {{
    color: {HEADER_TEXT};
    font-size: 22px;
    font-weight: 600;
    padding-left: 12px;
}}

QLabel#headerSubtitle {{
    color: {HEADER_SUBTEXT};
    font-size: 13px;
    padding-right: 14px;
}}

/* 오퍼레이터 모드: 중앙 공정명 */
QLabel#headerCenterTitle {{
    color: {HEADER_TEXT};
    font-size: 22px;
    font-weight: 600;
}}

/* ===== 윈도우 컨트롤 버튼 (최소화/최대화/닫기) ===== */
QPushButton#winBtn {{
    background: transparent;
    border: 1px solid #6b6b6b;
    color: #e6e6e6;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
    min-width: {WINDOW_BTN_WIDTH}px;
    max-width: {WINDOW_BTN_WIDTH}px;
}}

QPushButton#winBtn:hover {{
    background: #3d3d3d;
    color: #ffffff;
}}

QPushButton#winBtn:pressed {{
    background: #2a2f36;
}}

QPushButton#winCloseBtn {{
    background: transparent;
    border: 1px solid #6b6b6b;
    color: #e6e6e6;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
    min-width: {WINDOW_BTN_WIDTH}px;
    max-width: {WINDOW_BTN_WIDTH}px;
}}

QPushButton#winCloseBtn:hover {{
    background: #e04150;
    color: #ffffff;
}}

QPushButton#winCloseBtn:pressed {{
    background: #c2333f;
}}

/* 오퍼레이터 모드 타이틀바 버튼 (홈 인터페이스 / 종료하기).
   최소화·최대화는 그 모드에서 아예 없앴다 — 전체화면으로 고정해 운영하므로
   창 크기를 바꿀 일이 없고, 실수로 창이 작아지면 현장에서 되돌리기 번거롭다. */
QPushButton#headerAction {{
    background: {HEADER_BG};
    color: {HEADER_TEXT};
    border: 1px solid {CARD_BORDER};
    border-radius: 0px;
    padding: 7px 16px;
    font-size: 14px;
    font-weight: 600;
}}

QPushButton#headerAction:hover {{
    background: #3d3d3d;
}}

QPushButton#headerAction:pressed {{
    background: #1a1a1a;
}}

/* ===== 좌측 사이드바 ===== */
/* 탭 한 칸은 nav_sidebar._NavItem 이 직접 그린다(아이콘 판 + 라벨). 여기서는
   사이드바 전체 바탕과 오른쪽 경계만 잡는다. */
QFrame#sidebar {{
    background: {SIDEBAR_BG};
    border: none;
    border-right: 1px solid {SIDEBAR_LINE};
}}

/* ===== 메인 콘텐츠 ===== */
QStackedWidget#stack {{
    background: {CONTENT_BG};
}}

QStackedWidget#stack QLabel {{
    color: {CONTENT_TEXT};
}}

QLabel#pageTitle {{
    font-size: 22px;
    font-weight: 700;
    color: {CONTENT_TEXT};
}}

/* ===== 공용 컨트롤 (버튼 / 콤보 / 영상 영역) ===== */
QPushButton#primaryBtn {{
    background: {CONTROL_BG};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 7px 16px;
    font-weight: 600;
}}

QPushButton#primaryBtn:hover {{
    background: {CONTROL_HOVER};
}}

QPushButton#primaryBtn:pressed {{
    background: {CONTROL_PRESSED};
}}

QPushButton#primaryBtn:disabled {{
    background: {CONTROL_DISABLED_BG};
    color: {CONTROL_DISABLED_TEXT};
    border: 1px solid #8a8a8a;
}}

QPushButton#secondaryBtn {{
    background: #cccccc;
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 7px 16px;
}}

QPushButton#secondaryBtn:hover {{
    background: {CONTROL_HOVER};
}}

QPushButton#secondaryBtn:pressed {{
    background: {CONTROL_PRESSED};
}}

QPushButton#secondaryBtn:disabled {{
    background: {CONTROL_DISABLED_BG};
    color: {CONTROL_DISABLED_TEXT};
    border: 1px solid #8a8a8a;
}}

QComboBox#cameraCombo {{
    background: {CONTROL_BG};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 6px 10px;
    min-width: {COMBO_MIN_WIDTH}px;
}}

QComboBox#cameraCombo:focus {{
    border: 1px solid {ACCENT};
}}

/* drop-down 서브컨트롤은 스타일을 주지 않는다.
   QSS로 지정하는 순간 Qt 기본 화살표 렌더링이 꺼져서 화살표가 사라진다. */

QComboBox#cameraCombo:disabled {{
    background: {CONTROL_DISABLED_BG};
    color: {CONTROL_DISABLED_TEXT};
}}

/* 콤보를 펼쳤을 때 나오는 목록. 회색 바탕에 묻히지 않게 흰 판으로 둔다. */
QComboBox QAbstractItemView {{
    background: {SURFACE};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    selection-background-color: {ACCENT};
    selection-color: #ffffff;
    outline: 0;
}}

/* 연결/연결해제 토글 버튼: 켜짐(연결됨) 시 녹색 강조.
   라벨 길이가 바뀌어도 크기가 변하지 않도록 너비를 고정한다. */
QPushButton#connectToggle {{
    background: {CONTROL_BG};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 7px 16px;
    min-width: {TOGGLE_BTN_TEXT_WIDTH}px;
    max-width: {TOGGLE_BTN_TEXT_WIDTH}px;
}}

QPushButton#connectToggle:hover {{
    background: {CONTROL_HOVER};
}}

QPushButton#connectToggle:checked {{
    background: {CONNECTED_BG};
    color: {CONNECTED_TEXT};
    border: 1px solid {CONNECTED_BORDER};
    font-weight: 600;
}}

QPushButton#connectToggle:disabled {{
    background: {CONTROL_DISABLED_BG};
    color: {CONTROL_DISABLED_TEXT};
    border: 1px solid #8a8a8a;
}}

/* 사진이 들어가는 자리. 회색 UI 와 분명히 갈라 두려고 흰 바탕 + 파란 테두리다. */
QLabel#videoView {{
    background: {SURFACE};
    border: 1px solid {ACCENT};
    border-radius: 0px;
    color: #6f6f6f;
}}

/* 블롭 파라미터 입력 박스와 라벨 (정수/실수 입력 모두 같은 모양) */
QSpinBox#paramSpin, QDoubleSpinBox#paramSpin {{
    background: {SURFACE};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 5px 6px;
    min-width: {PARAM_BOX_WIDTH}px;
    max-width: {PARAM_BOX_WIDTH}px;
}}

QSpinBox#paramSpin:focus, QDoubleSpinBox#paramSpin:focus {{
    border: 1px solid {ACCENT};
}}

/* 스위치가 꺼져 있으면 회색으로 비활성임을 드러낸다 */
QSpinBox#paramSpin:disabled, QDoubleSpinBox#paramSpin:disabled {{
    background: {CONTROL_DISABLED_BG};
    color: {CONTROL_DISABLED_TEXT};
    border: 1px solid #8a8a8a;
}}

QLabel#paramLabel {{
    color: {MUTED_TEXT};
    font-size: 12px;
}}

/* 이미지 보드 우측 상단에 붙는 처리 시간 표시 */
QLabel#elapsedLabel {{
    color: {MUTED_TEXT};
    font-size: 12px;
    padding-right: 2px;
}}

QLabel#paramLabel:disabled {{
    color: {CONTROL_DISABLED_TEXT};
}}

/* ===== 이미지 보드 오른쪽 설정 패널 ===== */
/* 검출 방식/자세 기준이 바뀌면 이 패널 안의 내용만 갈린다. 폭이 고정이라
   이미지 보드 크기는 그대로다. */
QScrollArea#settingPanel {{
    background: transparent;
    border: none;
}}

QScrollArea#settingPanel > QWidget > QWidget {{
    background: transparent;
}}

/* 패널 안의 한 덩어리. 흰 판으로 띄워 회색 바탕과 갈라 둔다. */
QFrame#settingGroup {{
    background: {SURFACE};
    border: 1px solid {ACCENT};
}}

QFrame#settingGroup QLabel {{
    color: {CONTENT_TEXT};
}}

QLabel#groupCaption {{
    color: {CONTENT_TEXT};
    font-size: 13px;
    font-weight: 600;
}}

/* ===== 뎁스 후처리 패널 (Camera Selection) ===== */
/* 뎁스맵 오른쪽에 붙는 세로 패널. 항목이 많아 스크롤되므로 배경을 본문과
   같게 두어 스크롤해도 경계가 눈에 걸리지 않게 한다. */
QScrollArea#postProcessPanel {{
    background: transparent;
    border: none;
}}

QScrollArea#postProcessPanel > QWidget > QWidget {{
    background: transparent;
}}

QLabel#paramTitle {{
    color: {CONTENT_TEXT};
    font-size: 14px;
    font-weight: 600;
    padding-bottom: 2px;
}}

/* 필터 하나를 감싸는 칸 */
QFrame#paramCard {{
    background: transparent;
}}

/* 필터 제목 겸 켜기 버튼. 눌린 상태가 곧 '켜짐'이라, 켜고 끈 것이 한눈에
   보이도록 배경까지 바꾼다(체크박스보다 목록에서 훑기 쉽다). */
QPushButton#filterToggle {{
    background: {CONTROL_BG};
    color: {MUTED_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 5px 8px;
    text-align: left;
    font-size: 12px;
}}

QPushButton#filterToggle:hover {{
    border: 1px solid {ACCENT};
}}

QPushButton#filterToggle:checked {{
    background: {ACCENT};
    color: #ffffff;
    border: 1px solid {ACCENT};
    font-weight: 600;
}}

/* ===== EIH 캘리브레이션 페이지 ===== */
QComboBox#boardCombo {{
    background: {CONTROL_BG};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 6px 10px;
    min-width: {BOARD_COMBO_MIN_WIDTH}px;
}}

QComboBox#boardCombo:focus {{
    border: 1px solid {ACCENT};
}}

QComboBox#boardCombo:disabled {{
    background: {CONTROL_DISABLED_BG};
    color: {CONTROL_DISABLED_TEXT};
}}

/* 플랜지 포즈 입력칸: 체커보드를 못 찾으면 통째로 회색이 된다 */
QDoubleSpinBox#poseInput {{
    background: {SURFACE};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 5px 8px;
    min-width: {POSE_BOX_WIDTH}px;
    max-width: {POSE_BOX_WIDTH}px;
}}

QDoubleSpinBox#poseInput:focus {{
    border: 1px solid {ACCENT};
}}

QDoubleSpinBox#poseInput:disabled {{
    background: {CONTROL_DISABLED_BG};
    color: {CONTROL_DISABLED_TEXT};
    border: 1px solid #8a8a8a;
}}

QLabel#poseLabel {{
    color: {CONTENT_TEXT};
    font-size: 12px;
}}

QLabel#poseLabel:disabled {{
    color: {CONTROL_DISABLED_TEXT};
}}

QLabel#sectionLabel {{
    color: {MUTED_TEXT};
    font-size: 12px;
    font-weight: 600;
}}

QTableWidget#sampleTable {{
    background: {SURFACE};
    border: 1px solid {ACCENT};
    border-radius: 0px;
    gridline-color: #c9c9c9;
    font-size: 12px;
}}

QTableWidget#sampleTable::item:selected {{
    background: {ACCENT};
    color: #ffffff;
}}

/* 표 우클릭 메뉴 */
QMenu {{
    background: {SURFACE};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 4px;
}}

QMenu::item {{
    color: {CONTENT_TEXT};
    padding: 7px 18px;
    border-radius: 2px;
}}

QMenu::item:selected {{
    background: #fdecea;
    color: #d0342c;
}}

QTableWidget#sampleTable QHeaderView::section {{
    background: #6f6f6f;
    color: #ffffff;
    border: none;
    border-right: 1px solid #8a8a8a;
    padding: 6px 4px;
    font-size: 12px;
}}

/* 세팅(체결 위치) 선택 콤보 */
QComboBox#profileCombo {{
    background: {CONTROL_BG};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 6px 10px;
    min-width: {PROFILE_COMBO_MIN_WIDTH}px;
}}

QComboBox#profileCombo:focus {{
    border: 1px solid {ACCENT};
}}

QComboBox#profileCombo:disabled {{
    background: {CONTROL_DISABLED_BG};
    color: {CONTROL_DISABLED_TEXT};
}}

/* +/- 정사각 버튼 */
QComboBox#compactCombo {{
    background: {CONTROL_BG};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 6px 8px;
    min-width: 100px;
}}

QComboBox#compactCombo:focus {{
    border: 1px solid {ACCENT};
}}

QComboBox#compactCombo:disabled {{
    background: {CONTROL_DISABLED_BG};
    color: {CONTROL_DISABLED_TEXT};
}}

QPushButton#iconBtn {{
    background: {CONTROL_BG};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 0px;
    font-size: 16px;
    font-weight: 600;
    min-width: {ICON_BTN_SIZE}px;
    max-width: {ICON_BTN_SIZE}px;
    min-height: {ICON_BTN_SIZE}px;
    max-height: {ICON_BTN_SIZE}px;
}}

QPushButton#iconBtn:hover {{
    background: {CONTROL_HOVER};
    border: 1px solid {ACCENT};
}}

QPushButton#iconBtn:pressed {{
    background: {CONTROL_PRESSED};
}}

QPushButton#iconBtn:disabled {{
    background: {CONTROL_DISABLED_BG};
    color: {CONTROL_DISABLED_TEXT};
    border: 1px solid #8a8a8a;
}}

/* 이름 입력 창 */
QDialog {{
    background: {CONTENT_BG};
}}

QLineEdit#nameInput {{
    background: {SURFACE};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 7px 10px;
    min-width: {NAME_INPUT_MIN_WIDTH}px;
}}

QLineEdit#nameInput:focus {{
    border: 1px solid {ACCENT};
}}

QLabel#dialogLabel {{
    color: {CONTENT_TEXT};
}}

QLabel#dialogError {{
    color: #b3261e;
    font-size: 12px;
}}

/* 타겟 필터링 종류 선택 */
QComboBox#filterCombo {{
    background: {CONTROL_BG};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 6px 10px;
    min-width: {FILTER_COMBO_MIN_WIDTH}px;
}}

QComboBox#filterCombo:focus {{
    border: 1px solid {ACCENT};
}}

QComboBox#filterCombo:disabled {{
    background: {CONTROL_DISABLED_BG};
    color: {CONTROL_DISABLED_TEXT};
}}

/* 이미지 보드에 띄울 화면 종류 선택 (컬러맵 / 뎁스맵 / 이진화 ...) */
QComboBox#viewCombo {{
    background: {CONTROL_BG};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 6px 10px;
    min-width: {VIEW_COMBO_MIN_WIDTH}px;
}}

QComboBox#viewCombo:focus {{
    border: 1px solid {ACCENT};
}}

QComboBox#viewCombo:disabled {{
    background: {CONTROL_DISABLED_BG};
    color: {CONTROL_DISABLED_TEXT};
}}

/* ROI 설정 토글: 켜짐(설정 모드) 시 주황 강조 */
QPushButton#roiToggle {{
    background: {CONTROL_BG};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 7px 16px;
}}

QPushButton#roiToggle:hover {{
    background: {CONTROL_HOVER};
}}

QPushButton#roiToggle:checked {{
    background: #ffe2b8;
    color: #8a4b00;
    border: 1px solid {ROI_BORDER};
    font-weight: 600;
}}

QPushButton#roiToggle:disabled {{
    background: {CONTROL_DISABLED_BG};
    color: {CONTROL_DISABLED_TEXT};
    border: 1px solid #8a8a8a;
}}

QLabel#cameraStatus {{
    color: {MUTED_TEXT};
    font-size: 12px;
}}

/* 연결 상태 표시등: 회색(미연결) -> 녹색(연결) */
QLabel#statusLed {{
    border-radius: {LED_SIZE // 2}px;
    min-width: {LED_SIZE}px;
    max-width: {LED_SIZE}px;
    min-height: {LED_SIZE}px;
    max-height: {LED_SIZE}px;
    background: {LED_OFF};
}}

QLabel#statusLed[connected="true"] {{
    background: {LED_ON};
}}

/* 켜짐/꺼짐 표시등: 빨강(꺼짐) -> 초록(켜짐).
   statusLed 와 따로 둔 이유는 '아직 연결 안 됨'과 '꺼져 있음'이 다르기
   때문이다. 전자는 회색이 맞고(할 일이 남은 것), 후자는 빨강이 맞다.
   색은 토글 스위치와 같은 값을 써서 한 화면에서 어긋나지 않게 한다. */
QLabel#stateLed {{
    border-radius: {LED_SIZE // 2}px;
    min-width: {LED_SIZE}px;
    max-width: {LED_SIZE}px;
    min-height: {LED_SIZE}px;
    max-height: {LED_SIZE}px;
    background: {SWITCH_OFF_BG};
}}

QLabel#stateLed[on="true"] {{
    background: {SWITCH_ON_BG};
}}

/* ===== 하단 페이지 네비게이션 ===== */
QFrame#navBar {{
    background: {CONTENT_BG};
    border-top: 1px solid {SIDEBAR_LINE};
}}

QPushButton#navBtn {{
    background: #6f6f6f;
    color: #ffffff;
    border: 1px solid #4d4d4d;
    border-radius: 2px;
    padding: 7px 22px;
    font-weight: 600;
}}

QPushButton#navBtn:hover {{
    background: #7d7d7d;
}}

QPushButton#navBtn:pressed {{
    background: #5c5c5c;
}}

QPushButton#navBtn:disabled {{
    background: {CONTROL_DISABLED_BG};
    color: {CONTROL_DISABLED_TEXT};
    border: 1px solid #8a8a8a;
}}

QLabel#navIndicator {{
    color: {CONTENT_TEXT};
    font-size: 13px;
    font-weight: 600;
}}

/* ===== 오퍼레이터 모드 ===== */
QLineEdit#processInput {{
    background: {SURFACE};
    color: {CONTENT_TEXT};
    border: 1px solid {ACCENT};
    border-radius: 2px;
    padding: 6px 10px;
    min-width: 300px;
}}

QLineEdit#processInput:focus {{
    border: 1px solid {ACCENT};
}}

/* 저장 경로 표시 (읽기 전용, '경로 선택'으로만 바뀐다) */
QLineEdit#pathDisplay {{
    background: {SURFACE};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 6px 10px;
}}

/* TCP 서버 IP / 포트 입력 */
QLineEdit#ipInput, QLineEdit#portInput {{
    background: {SURFACE};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 6px 10px;
}}

QLineEdit#ipInput:focus, QLineEdit#portInput:focus {{
    border: 1px solid {ACCENT};
}}

QLabel#operatorDisplay {{
    background: {SURFACE};
    border: 1px solid {ACCENT};
    border-radius: 0px;
    color: #6f6f6f;
    font-size: 16px;
}}

/* 가장 마지막에 촬영된 칸: 초록 테두리 */
QLabel#operatorDisplay[latest="true"] {{
    border: 2px solid {LED_ON};
}}

/* 눌러서 고른 칸: 파랑 테두리.
   '최신' 규칙보다 뒤에 두어, 둘 다인 칸은 파랑으로 보이게 한다. */
QLabel#operatorDisplay[selected="true"] {{
    border: 2px solid {ACCENT};
}}

/* 화면 분할에서 남는 자리를 메우는 검정 칸 (체결 순서가 없는 칸) */
QLabel#operatorFiller {{
    background: #4d4d4d;
    border: 1px solid {SIDEBAR_LINE};
}}

/* 촬영 결과 판정 배지 (OK / NG). 이미지 보드마다 하나씩 우측 상단에 얹힌다.
   판정이 없을 때는 아예 숨기므로 '아직 안 찍음' 색은 두지 않는다. */
QLabel#judgeBadge {{
    background: {JUDGE_NG_BG};
    color: {JUDGE_TEXT};
    border: 1px solid #1a1a1a;
    font-size: 22px;
    font-weight: 700;
    min-width: {JUDGE_BADGE_WIDTH}px;
    max-width: {JUDGE_BADGE_WIDTH}px;
    min-height: {JUDGE_BADGE_HEIGHT}px;
    max-height: {JUDGE_BADGE_HEIGHT}px;
}}

QLabel#judgeBadge[judge="ok"] {{
    background: {JUDGE_OK_BG};
}}

QLabel#judgeBadge[judge="ng"] {{
    background: {JUDGE_NG_BG};
}}

/* 오퍼레이터 상단: 현재 시각 + 실행/모델 변경 버튼 */
QLabel#operatorClock {{
    background: {HEADER_BG};
    color: {HEADER_TEXT};
    border: 1px solid {ACCENT};
    font-size: 17px;
    font-weight: 600;
    padding: 6px 12px;
}}

QPushButton#operatorAction {{
    background: {HEADER_BG};
    color: {HEADER_TEXT};
    border: 1px solid {ACCENT};
    border-radius: 0px;
    padding: 10px 18px;
    font-size: 14px;
    font-weight: 600;
}}

QPushButton#operatorAction:hover {{
    background: #3d3d3d;
}}

QPushButton#operatorAction:pressed {{
    background: #1a1a1a;
}}

QPushButton#operatorAction:disabled {{
    background: #5c5c5c;
    color: #a6a6a6;
    border: 1px solid #7a7a7a;
}}

/* 오퍼레이터 화면 오른쪽 로그 창 (테마와 같은 회색) */
QPlainTextEdit#operatorLog {{
    background: {OPERATOR_LOG_BG};
    color: {OPERATOR_LOG_TEXT};
    border: 1px solid {SIDEBAR_LINE};
    border-radius: 0px;
    padding: 6px;
    font-size: 11px;
}}

QLabel#operatorLogCaption {{
    color: {CONTENT_TEXT};
    font-size: 13px;
    font-weight: 600;
}}

QFrame#operatorPoseBox {{
    background: {BORDER};
    border: 1px solid {ACCENT};
    border-radius: 0px;
}}

QLabel#operatorPoseCell {{
    background: {SURFACE};
    color: {CONTENT_TEXT};
    border: none;
    padding: 7px 2px;
    font-size: 11px;
    font-weight: 600;
}}

/* ===== SAM2 시험 페이지 ===== */
QLabel#sam2Status {{
    color: {MUTED_TEXT};
    font-size: 12px;
}}

QComboBox#sam2ModelCombo {{
    background: {CONTROL_BG};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 5px 8px;
    min-width: 110px;
}}

QComboBox#sam2ModelCombo:focus {{
    border: 1px solid {ACCENT};
}}

QComboBox#sam2ClassCombo {{
    background: {CONTROL_BG};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 5px 8px;
    min-width: 160px;
}}

QComboBox#sam2ClassCombo:focus {{
    border: 1px solid {ACCENT};
}}

/* 박스 삭제 토글: 켜짐(활성) 시 빨간 강조 */
QPushButton#dangerToggle {{
    background: {CONTROL_BG};
    color: {CONTENT_TEXT};
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    padding: 7px 16px;
}}

QPushButton#dangerToggle:hover {{
    background: {CONTROL_HOVER};
}}

QPushButton#dangerToggle:checked {{
    background: #fdecea;
    color: #d0342c;
    border: 1px solid #d0342c;
    font-weight: 600;
}}

QPushButton#dangerToggle:disabled {{
    background: {CONTROL_DISABLED_BG};
    color: {CONTROL_DISABLED_TEXT};
    border: 1px solid #8a8a8a;
}}

QTableWidget#sam2Table {{
    background: {SURFACE};
    border: 1px solid {ACCENT};
    border-radius: 0px;
    gridline-color: #c9c9c9;
}}

QTableWidget#sam2Table::item:selected {{
    background: {ACCENT};
    color: #ffffff;
}}

QTableWidget#sam2Table QHeaderView::section {{
    background: #6f6f6f;
    color: #ffffff;
    border: none;
    border-right: 1px solid #8a8a8a;
    padding: 6px 8px;
}}

QProgressBar {{
    border: 1px solid {CONTROL_BORDER};
    border-radius: 2px;
    background: {CONTROL_BG};
    text-align: center;
    color: {CONTENT_TEXT};
    height: 18px;
}}

QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 0px;
}}

/* 스크롤바: 회색 바탕에서 기본 흰 스크롤바가 튀어 보이므로 맞춰 둔다. */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background: #7a7a7a;
    min-height: 24px;
    border-radius: 5px;
}}

QScrollBar::handle:vertical:hover {{
    background: #666666;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0px;
}}

QScrollBar::handle:horizontal {{
    background: #7a7a7a;
    min-width: 24px;
    border-radius: 5px;
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}
"""
