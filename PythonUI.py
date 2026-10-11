"""애플리케이션 진입점 & 메인 창(MainWindow).

전체 화면 구조를 조립하는 최상위 파일이다.

  ┌─ TitleBar (커스텀 상단 바: 아이콘/제목/최소·최대·닫기) ────────────────────┐
  │                                                                      │
  │  mode_stack (최상위 화면 전환)                                          │
  │   ├ index 0: setup_view  = [사이드바 탭] + [페이지 스택 + 하단 네비]      │
  │   └ index 1: operator_view = 현장 운영자용 단순 화면                    │
  └──────────────────────────────────────────────────────────────────────┘

- '세팅 모드'와 '오퍼레이터 모드'를 mode_stack으로 전환한다.
- 세팅 모드의 각 기능은 pages/ 폴더의 BasePage 하위 클래스 하나 = 사이드바 탭 하나.
- 새 기능을 추가하려면 pages/에 페이지 파일을 만들고 아래 PAGES에 등록만 하면 된다.

실행: python PythonUI.py  (pages, theme 등을 상대 import 하므로 이 폴더 안에서 실행)
"""

import os
import sys

# ===== 창 모드 exe 의 stdout/stderr 를 되살린다 =====
# PyInstaller 로 console=False 로 묶으면 sys.stdout 과 sys.stderr 가 **None** 이다.
# 그러면 거기에 쓰려는 라이브러리가 AttributeError 로 죽는다:
#   "분석 실패: 'NoneType' object has no attribute 'write'"
#
# 실제로 걸린 것은 tqdm 이다. sam2 의 SAM2VideoPredictor.propagate_in_video 가
# 진행률을 tqdm 으로 찍는데(sam2_video_predictor.py 의 tqdm 호출), tqdm 은 기본
# 출력이 sys.stderr 다. 그래서 'SAM2 비디오 추적'과 '법선 매칭점'이 exe 에서만
# 통째로 실패했다 — 소스로 실행하면 콘솔이 있어 안 보인다.
#
# 라이브러리를 하나씩 고치는 대신 여기서 자리를 만들어 준다. 남의 코드가 어디서
# 무엇을 찍을지 우리가 다 알 수는 없기 때문이다. 값은 버린다 — 창 모드에는 보여
# 줄 콘솔이 없고, 우리 쪽 진단은 화면 상태문구와 오퍼레이터 로그로 남는다.
#
# import 보다 먼저 해야 한다. torch/tqdm 이 올라오는 순간부터 필요하다.
if sys.stdout is None or sys.stderr is None:
    _devnull = open(os.devnull, "w", encoding="utf-8")
    if sys.stdout is None:
        sys.stdout = _devnull
    if sys.stderr is None:
        sys.stderr = _devnull

# Windows must know that this process handles DPI changes per monitor before
# Qt creates any native windows. This also covers development runs through
# python/debugpy, where there is no application manifest of our own.
if sys.platform == "win32":
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ.setdefault("QT_SCALE_FACTOR_ROUNDING_POLICY", "PassThrough")
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        pass

# torch 는 PyQt5 보다 반드시 먼저 올려야 한다. 반대 순서로 부르면 Windows 에서
# c10.dll 초기화가 실패한다(OSError WinError 1114). Hole Detection 의 SAM2
# 검출이 torch 를 쓰는데, 그 모듈은 PyQt5 가 올라온 뒤에야 import 되므로
# 여기서 미리 한 번 올려 순서를 못 박는다.
#
# torch 가 없는 설치(SAM2 를 안 쓰는 경우)도 정상이므로 실패는 그냥 지나간다.
# 그때는 sam2_detector 가 '쓸 수 없음'으로 처리하고 앱은 그대로 뜬다.
try:
    import torch  # noqa: F401
except Exception:
    pass

from app_paths import is_frozen

# 소스로 실행할 때만 venv 안의 Qt 플러그인 위치를 잡아 준다.
# exe로 묶으면 PyInstaller가 플러그인을 _internal 에 넣고 경로도 직접 잡으므로,
# 여기서 QT_PLUGIN_PATH 를 덮어쓰면 오히려 플러그인을 못 찾는다.
if not is_frozen():
    _qt_plugins = os.path.join(
        os.path.dirname(sys.executable),
        "..", "Lib", "site-packages", "PyQt5", "Qt5", "plugins",
    )
    _qt_platform_plugins = os.path.join(_qt_plugins, "platforms")
    os.environ.setdefault("QT_PLUGIN_PATH", _qt_plugins)
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", _qt_platform_plugins)

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from native_resize import NativeResizeMixin
from nav_sidebar import NavSidebar
from operator_view import OperatorRunConfig, OperatorView
from pages.camera_capture_page import CameraCapturePage
from pages.home_page import HomePage
from pages.operator_launch_page import OperatorLaunchPage
from pages.pose_setting_page import PoseSettingPage
from pages.hole_detection_page import HoleDetectionPage
from pages.eih_calib_page import EIHCalibPage
from session_store import load_session, save_session
from theme import (
    NAV_HEIGHT,
    NAV_INDICATOR_WIDTH,
    SIDEBAR_WIDTH,
    STYLESHEET,
)
from titlebar import TitleBar

# 세팅 UI 페이지 목록. 새 기능 모듈을 추가하려면:
# 1) pages/ 폴더에 BasePage를 상속한 새 클래스 파일을 만든다.
# 2) 여기서 import 하고 아래 PAGES 목록에 추가한다.
# OperatorLaunchPage는 항상 마지막에 두어 오퍼레이터 모드로의 전환을 제공한다.
PAGES = [
    #HomePage,
    CameraCapturePage,
    HoleDetectionPage,
    PoseSettingPage,
    EIHCalibPage,
    OperatorLaunchPage,
]

APP_TITLE = "IT"
APP_SUBTITLE = "Last update: 26.09.14"

class MainWindow(NativeResizeMixin, QMainWindow):
    """앱의 메인 창.

    NativeResizeMixin을 함께 상속해 프레임리스(테두리 없는) 창이면서도
    가장자리 드래그 리사이즈가 되도록 한다(native_resize.py 참고).
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1280, 800)

        # OS 기본 타이틀바 제거 (커스텀 헤더바 사용)
        self.setWindowFlags(Qt.FramelessWindowHint)

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)

        # 전체: 헤더(상단) + 본문(사이드바 + 메인)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.titlebar = TitleBar(self, APP_TITLE, APP_SUBTITLE)
        root_layout.addWidget(self.titlebar)

        # 최상위 모드 스택: 세팅 뷰(사이드 탭) <-> 오퍼레이터 뷰
        self.mode_stack = QStackedWidget()
        self.setup_view = self._build_body()
        self.operator_view = OperatorView(on_back=self.enter_setup_mode)
        # 촬영할 때 Camera Capture 페이지의 카메라를 빌려 쓰므로 창을 넘겨 준다.
        self.operator_view.bind_window(self)
        self.mode_stack.addWidget(self.setup_view)      # index 0
        self.mode_stack.addWidget(self.operator_view)   # index 1
        root_layout.addWidget(self.mode_stack)

        # 오퍼레이터 모드 타이틀바 버튼. '홈 인터페이스'는 서버를 내려놓고
        # 나가야 하므로 OperatorView.leave() 를 부른다 — 타이틀바는 그 절차를
        # 모른다. 세팅으로 돌아가는 것은 leave() 가 on_back 으로 이어 준다.
        self.titlebar.bind_operator_actions(
            on_home=self.operator_view.leave, on_quit=self.close
        )
        # 오퍼레이터 모드에 들어가기 전 창 상태. 나올 때 그대로 되돌린다.
        self._pre_operator_maximized = False

        self.camera_page = None
        self.operator_launch_page = None
        self.hole_page = None
        self.pose_page = None
        self._load_pages(PAGES)
        self._link_pages()
        self.sidebar.setCurrentRow(0)
        self._restore_session()

    # ===== 모드 전환 (세팅 <-> 오퍼레이터) =====
    def enter_operator_mode(self, process_name="", save_enabled=False,
                            save_path=None, fastening_points=None,
                            server_enabled=False, server_config=None,
                            models=None, active_model=None):
        """오퍼레이터 모드 진입.

        운영에 쓸 값은 세팅의 Operator Mode 페이지에서 다 정해서 넘어온다.
        오퍼레이터 화면은 그 값을 들고만 있고 설정 UI를 다시 두지 않는다.

        **전체화면으로 들어간다**(260910 요구). 현장 화면은 한 번 띄우면 계속
        그대로 두고 쓰므로 창 크기를 조절할 이유가 없고, 화면을 꽉 채워야 이미지
        보드가 가장 크게 나온다. 돌아갈 때 원래 크기로 되돌리려고 지금 상태를
        적어 둔다.
        """
        self.operator_view.set_run_config(OperatorRunConfig(
            process_name=process_name,
            save_enabled=save_enabled,
            save_path=save_path,
            fastening_points=fastening_points,
            server_enabled=server_enabled,
            server_config=server_config,
            models=models,
            active_model=active_model,
        ))
        self.titlebar.set_mode("operator", process_name)
        self.mode_stack.setCurrentWidget(self.operator_view)
        if not self.isFullScreen():
            self._pre_operator_maximized = self.isMaximized()
            self.showFullScreen()

    def enter_setup_mode(self):
        self.titlebar.set_mode("setup")
        self.mode_stack.setCurrentWidget(self.setup_view)
        if self.isFullScreen():
            # 들어올 때의 크기로 되돌린다. 최대화였으면 최대화로.
            if self._pre_operator_maximized:
                self.showMaximized()
            else:
                self.showNormal()

    def closeEvent(self, event):
        # 듣고 있던 포트를 놓고 나간다. 안 내려놓으면 바로 다시 실행할 때
        # "포트가 이미 사용 중"으로 막힐 수 있다.
        self._save_session()
        self.operator_view.stop_server()
        super().closeEvent(event)

    def _save_session(self):
        state = {
            "view_mode": (
                "operator"
                if self.mode_stack.currentWidget() is self.operator_view
                else "setup"
            ),
        }
        if self.camera_page is not None:
            state["camera"] = self.camera_page.session_state()
        if self.operator_launch_page is not None:
            state["operator"] = self.operator_launch_page.session_state()
        save_session(state)

    def _restore_session(self):
        state = load_session()
        if self.operator_launch_page is not None:
            self.operator_launch_page.restore_session_state(state.get("operator"))
        if self.camera_page is not None:
            self.camera_page.restore_session_state(state.get("camera"))

        if (state.get("view_mode") == "operator"
                and self.operator_launch_page is not None):
            page = self.operator_launch_page
            self.enter_operator_mode(
                process_name=page.control_panel.get_process_name(),
                save_enabled=page.control_panel.is_save_enabled(),
                save_path=page.control_panel.get_save_path(),
                fastening_points=page.table.get_points(),
                server_enabled=page.tcp_panel.is_server_enabled(),
                server_config=page.tcp_panel.get_server_config(),
                models=page.model_entries(),
                active_model=page.current_model_name(),
            )

    def _build_body(self):
        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 좌측 사이드 탭. 칸마다 아이콘(위) + 라벨(아래)이 들어간다(nav_sidebar).
        self.sidebar = NavSidebar()
        self.sidebar.setFixedWidth(SIDEBAR_WIDTH)
        self.sidebar.currentRowChanged.connect(self.change_page)

        layout.addWidget(self.sidebar)
        layout.addWidget(self._build_content())
        return body

    def _build_content(self):
        """메인 영역: 페이지 스택(위) + 하단 네비게이션(아래)."""
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 탭에 대응하는 메인 화면 (탭 전환 시 이 위젯이 바뀐다)
        self.stack = QStackedWidget()
        self.stack.setObjectName("stack")

        layout.addWidget(self.stack)
        layout.addWidget(self._build_nav())
        return content

    def _build_nav(self):
        """하단 중앙 이전/다음 페이지 네비게이션."""
        nav = QFrame()
        nav.setObjectName("navBar")
        nav.setFixedHeight(NAV_HEIGHT)

        layout = QHBoxLayout(nav)
        layout.setContentsMargins(0, 0, 0, 0)

        self.btn_prev = QPushButton("◀  이전")
        self.btn_prev.setObjectName("navBtn")
        self.btn_prev.clicked.connect(self.go_prev)

        self.nav_indicator = QLabel("")
        self.nav_indicator.setObjectName("navIndicator")
        self.nav_indicator.setAlignment(Qt.AlignCenter)
        self.nav_indicator.setFixedWidth(NAV_INDICATOR_WIDTH)

        self.btn_next = QPushButton("다음  ▶")
        self.btn_next.setObjectName("navBtn")
        self.btn_next.clicked.connect(self.go_next)

        layout.addStretch()
        layout.addWidget(self.btn_prev)
        layout.addWidget(self.nav_indicator)
        layout.addWidget(self.btn_next)
        layout.addStretch()
        return nav

    def _load_pages(self, page_classes):
        """PAGES의 각 페이지 클래스를 인스턴스화해 사이드바 탭 + 페이지 스택에 등록.

        사이드바 항목 순서와 stack 위젯 순서가 1:1로 대응한다(같은 index).
        """
        for page_cls in page_classes:
            page = page_cls(self)
            # MainWindow 참조가 필요한 페이지(예: 오퍼레이터 전환)에 자신을 주입
            if hasattr(page, "bind_window"):
                page.bind_window(self)
            # 사이드바 탭 = 페이지 title + 페이지가 지정한 아이콘 이름(nav_icons)
            self.sidebar.addItem(page.title, getattr(page, "nav_icon", "default"))
            self.stack.addWidget(page)          # 탭 선택 시 보여줄 실제 화면
            if isinstance(page, CameraCapturePage):
                self.camera_page = page
            elif isinstance(page, OperatorLaunchPage):
                self.operator_launch_page = page
            elif isinstance(page, HoleDetectionPage):
                self.hole_page = page
            elif isinstance(page, PoseSettingPage):
                self.pose_page = page

    def _link_pages(self):
        """Hole Detection 과 Pose Setting 을 서로 이어 준다.

        둘은 세팅 파일 하나를 나눠 쓰고(검출 키 / 포즈 키), Pose Setting 은
        저쪽의 검출 결과를 받아 포즈를 세운다. 그런데 서로를 import 하면 순환이
        되므로, 둘 다 이미 알고 있는 MainWindow 가 여기서 참조를 건네준다.

        페이지를 다 만든 **뒤에** 해야 한다. PAGES 순서가 바뀌어도 상관없도록.

        프로파일 불러오기(init_profiles)를 여기서 하는 것도 순서 때문이다. 두
        페이지가 이어진 뒤에 읽어야 첫 실행에 만드는 기본 프로파일에 포즈 키가
        함께 들어간다. Hole Detection 의 __init__ 에서 읽으면 그때는 Pose
        Setting 이 없어서 그 키들이 빠진 파일이 만들어진다.
        """
        if self.hole_page is None or self.pose_page is None:
            # Pose Setting 을 PAGES 에서 뺀 구성이어도 프로파일은 읽어야 한다.
            if self.hole_page is not None:
                self.hole_page.init_profiles()
            return
        self.pose_page.bind_hole_page(self.hole_page)
        self.hole_page.bind_pose_page(self.pose_page)
        self.hole_page.init_profiles()

    def change_page(self, index):
        # 사이드바에서 탭이 바뀌면(currentRowChanged) 해당 페이지로 전환
        if index >= 0:
            self.stack.setCurrentIndex(index)
            self._update_nav(index)

    def _update_nav(self, index):
        count = self.sidebar.count()
        self.btn_prev.setEnabled(index > 0)
        self.btn_next.setEnabled(index < count - 1)
        self.nav_indicator.setText(f"{index + 1} / {count}")

    def go_prev(self):
        index = self.sidebar.currentRow()
        if index > 0:
            self.sidebar.setCurrentRow(index - 1)

    def go_next(self):
        index = self.sidebar.currentRow()
        if index < self.sidebar.count() - 1:
            self.sidebar.setCurrentRow(index + 1)


def main():
    # 고해상도(HiDPI) 대응: 반드시 QApplication 생성 전에 설정해야 적용된다.
    # 이걸 켜지 않으면 Windows 배율(예: 200%)을 무시하고 물리 픽셀로 그려서
    # 앱 UI만 작게 나오고, 배율을 지키는 OS 파일 선택창은 상대적으로 거대하게
    # 보인다(= 다른 프로그램과 이질적으로 보이는 원인).
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
