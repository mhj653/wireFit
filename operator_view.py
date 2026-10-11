"""오퍼레이터용 UI 뷰.

세팅 UI(좌측 사이드 탭 + 페이지)와 달리, 현장 운영자가 사용하는 단순한 화면.
사이드 탭이 없고, 아이콘과 공정명은 상단 타이틀바 **가운데**에 놓인다(타이틀바가
담당). 이 위젯은 본문만 맡는다.

구성 (260910 개편):

    ┌ 카메라 ╲ ┌ TCP ╲ ┌ 저장 ╲            Date : 26-09-10 10:50:31
    └─────────┘└──────┘└───────┘            [ 실행 ] [ 모델 변경 ]
    ╱현재 모델╲
    ┌───────────────────────────────┐   로그
    │  이미지 보드 (칸마다 OK/NG)     │   ┌────────────────┐
    │                        [ OK ] │   │ 10:50:31 촬영… │
    └───────────────────────────────┘   └────────────────┘

  - 상단 카드 셋(카메라 / TCP 서버 / 데이터 저장)은 촬영이 안 될 때 원인이 되는
    셋을 늘 보이게 둔 것이다(operator_panels.StatusCard). 로그에서 찾게 하지
    않는다.
  - 오른쪽 위에 현재 시각과 버튼 둘(실행 / 모델 변경). '실행'이 예전의 '촬영'
    이다 — 고른 칸에 한 장 찍는다.
  - 모델 탭은 이미지 보드 위에 놓인다. 모델을 추가하는 기능은 아직 없어서 지금은
    공정명 하나가 유일한 모델이다(operator_panels.ModelTabBar).

공정명·저장 여부·저장 경로·체결 위치 목록은 세팅의 Operator Mode 페이지에서
정하고, 이 화면은 그 값을 받아 들고만 있는다(run_config). 화면에 설정 UI를
다시 두지 않는 이유는, 현장에서 운영 중에 설정이 바뀌면 앞서 찍은 이미지와
뒤에 찍은 이미지가 다른 기준으로 남기 때문이다.

TCP 서버는 이 화면에 들어올 때 뜨고 나갈 때 내려간다(운영 중에만 트리거를
받으면 되므로). 서버가 받는 일은 다른 스레드에서 돌기 때문에, 화면을 고칠 때는
반드시 Qt 신호로 넘긴다 — 다른 스레드에서 위젯을 직접 만지면 안 된다.

촬영 이미지는 체결 순번에 지정된 블랍 세팅으로 분석해 검출 결과를 각 칸에 표시한다.

OK / NG 판정 (260910)
─────────────────────
촬영이 끝나면 **그 칸의 우측 상단**에 OK 또는 NG 가 뜨고 그대로 남는다.

    OK   홀이 **정확히 하나** 찾아지고 그 홀의 좌표까지 나왔을 때
    NG   그 밖의 모두 (0개, 2개 이상, 좌표를 못 구한 경우)

배지를 화면 한 군데가 아니라 칸마다 두는 이유: 네 칸을 찍으면 판정도 네 개다.
한 군데에 두면 마지막 것만 남아 어느 칸이 NG 였는지 알 수 없다.

법선 매칭점은 홀 수에 세지 않는다 — 그것은 자세를 어디서 받을지 정하는 자리라
검출된 홀이 아니다(애초에 홀 목록에 들어오지 않는다).

판정이 곧 로봇에 나가는 값이다. OK 일 때만 정상 응답을 보내고, NG 면 촬영
응답(101)과 결과 응답(102) 모두 실패 코드 1003 이 나간다. 홀이 둘 이상일 때
그중 하나를 골라 보내면 어느 것을 골랐는지 알 수 없고, 로봇은 엉뚱한 자리로
간다 — 나가지 않는 편이 안전하다.

모델도 트리거로 바꾼다 (103)
────────────────────────────
`103,1,n` 이 오면 n 번째 모델로 갈아 끼운다. n 은 '모델 변경' 창에 늘어선 순서
(=run_config.models 순서)이고, 손으로 바꿀 때와 같은 `apply_model()` 을 지나
찍어 둔 이미지·판정·좌표를 모두 버린다. 성공이면 `103,1107`, 없는 번호이거나
바꾸지 못했으면 `103,1003` 이다. 이미 그 모델이면 바꿀 것 없이 성공으로 본다.

1번 칸을 찍으면 새 주기다
─────────────────────────
1번 칸이 찍히는 순간 나머지 칸의 사진·판정·좌표를 모두 버린다(260910 요구).
지난 주기의 사진이 남아 있으면 이번 주기에 아직 안 찍은 칸을 찍은 것으로
착각한다. 좌표까지 함께 버리는 것이 중요하다 — 화면만 지우고 좌표를 남기면,
그 칸에 결과(102)를 물어볼 때 **지난 주기 값이 정상 응답으로 나간다.**
"""

from datetime import datetime
from threading import Event
import time

import cv2
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QFrame,
    QGridLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import protocol
from calibration_store import calibration_warnings, load_cam2gripper
from depth_view import colorize_depth
from fastening_data_model import FasteningPoint
from image_saver import date_dir, default_log_root, display_path, save_frames
from model_select_dialog import ModelSelectDialog
from operator_display_grid import OperatorDisplayGrid
from operator_panels import ModelTabBar, StatusCard
from hole_pose import PLANE_FROM_SURFACE, base_target_pose, flange_target_pose
from pages.camera_capture_page import CameraCapturePage
from pages.hole_detection_page import HoleDetectionPage, load_flange_pose
from pages.pose_setting_page import PoseSettingPage
from pose_pipeline import load_approach_offset
from tcp_server import FasteningTCPServer
from theme import CONTENT_SPACING, PAGE_MARGINS, PAGE_SPACING

SIDE_PANEL_WIDTH = 280    # 오른쪽 로그 칸 너비
MAX_LOG_LINES = 500       # 로그 창에 남겨 둘 줄 수
CLOCK_INTERVAL_MS = 1000  # 시각과 상태 카드를 새로 고치는 주기

# OK 판정 기준: 홀이 정확히 이만큼 찾아져야 한다. 둘 이상이면 어느 것을 보낼지
# 정할 수 없고, 없으면 보낼 것이 없다(이 파일 맨 위 설명 참고).
OK_HOLE_COUNT = 1

# 이 순번을 찍으면 새 주기로 본다(나머지 칸을 비운다).
FIRST_SEQUENCE = 1


def to_pixmap(bgr):
    """OpenCV BGR 이미지를 화면에 띄울 QPixmap 으로 바꾼다."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    image = QImage(
        rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888
    ).copy()   # numpy 버퍼와 분리해 두어야 배열이 해제돼도 안전하다.
    return QPixmap.fromImage(image)

CAPTURE_TRIGGER_TIMEOUT = 30.0

# 모델 변경(103)은 카메라를 쓰지 않고 화면만 다시 짜므로 금방 끝난다. 그래도
# 화면 스레드가 촬영 중이면 순서가 밀리므로 촬영 한 번만큼은 기다려 준다.
MODEL_SWITCH_TIMEOUT = 30.0


class OperatorRunConfig:
    """오퍼레이터 모드로 들어올 때 넘겨받은 설정."""

    def __init__(self, process_name="", save_enabled=False,
                 save_path=None, fastening_points=None,
                 server_enabled=False, server_config=None,
                 models=None, active_model=None):
        self.process_name = process_name
        self.save_enabled = save_enabled
        self.save_path = save_path or default_log_root()
        # **고른 모델의** 체결 포인트. 모델을 바꾸면 이 목록이 갈린다.
        self.fastening_points = list(fastening_points or [])
        # TCP 서버 설정 (tcp_server.FasteningServerConfig)
        self.server_enabled = server_enabled
        self.server_config = server_config
        # 모델 목록 [{"name": str, "fastening_points": [dict, ...]}, ...].
        # '모델 변경' 창이 이 목록을 보여 준다.
        self.models = list(models or [])
        self.active_model = active_model


class OperatorView(QWidget):
    # 서버 스레드에서 화면 스레드로 넘기는 신호.
    _event_logged = pyqtSignal(str)
    _capture_requested = pyqtSignal(object)
    _model_switch_requested = pyqtSignal(object)

    def __init__(self, on_back=None, parent=None):
        super().__init__(parent)
        self._on_back = on_back
        self._window = None          # bind_window()로 주입되는 MainWindow
        self.run_config = OperatorRunConfig()
        self._poses_by_sequence = {}
        self._pose_lists_by_sequence = {}
        # 촬영 칸마다 마지막 판정 (True=OK). 결과 응답(102)도 이 값을 본다 —
        # 트리거가 촬영과 결과를 따로 보내므로, 판정을 들고 있지 않으면 NG 였던
        # 칸의 좌표가 뒤늦게 정상 응답으로 나갈 수 있다.
        self._judgements_by_sequence = {}
        # 캘리브레이션 경고를 이미 적어 둔 세션 이름 (촬영마다 반복하지 않게)
        self._warned_calibrations = set()
        # 서버를 띄우지 못했는지. 1초 타이머가 실패 사유를 덮어쓰지 않게 한다.
        self._server_failed = False

        self._server = FasteningTCPServer()
        self._server.on_event = self._event_logged.emit   # 스레드 경계
        self._server.set_handler(protocol.CMD_CAPTURE, self._on_capture)
        self._server.set_handler(protocol.CMD_RESULT, self._on_result)
        self._server.set_handler(protocol.CMD_SWITCH_MODEL, self._on_switch_model)

        self._event_logged.connect(self._show_server_event)
        self._capture_requested.connect(self._mark_capture)
        self._model_switch_requested.connect(self._switch_model_by_order)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(*PAGE_MARGINS)
        outer.setSpacing(PAGE_SPACING)
        outer.addLayout(self._build_top_row())

        # 본문: 왼쪽에 모델 탭 + 나뉜 화면, 오른쪽에 로그
        main_row = QHBoxLayout()
        main_row.setSpacing(PAGE_SPACING)

        board_column = QVBoxLayout()
        board_column.setSpacing(0)   # 탭이 보드에 붙어야 책갈피로 보인다
        # 탭은 **잠근다**(260910 사용자 요구). 라벨을 잘못 눌러 제품이 바뀌면
        # 다른 제품의 체결 좌표가 로봇으로 나간다. 바꾸려면 '모델 변경' 창을 쓴다.
        self.model_tabs = ModelTabBar()
        self.model_tabs.set_locked(True)
        board_column.addWidget(self.model_tabs)

        self.display_grid = OperatorDisplayGrid()
        self.display_grid.cell_selected.connect(self._on_cell_selected)
        board_column.addWidget(self.display_grid, 1)
        main_row.addLayout(board_column, 1)

        main_row.addLayout(self._build_side_panel())
        outer.addLayout(main_row, 1)

        # 시각과 상태 카드를 1초마다 새로 고친다. 카메라가 빠지거나 로봇이
        # 접속을 끊는 일은 우리에게 알림이 오지 않으므로 주기적으로 물어봐야 한다.
        self._clock_timer = QTimer(self)
        self._clock_timer.setInterval(CLOCK_INTERVAL_MS)
        self._clock_timer.timeout.connect(self._tick)

        self._tick()

    def _build_top_row(self):
        """타이틀바 바로 아래 줄: 상태 카드 셋(좌) + 시각·버튼(우)."""
        row = QHBoxLayout()
        row.setSpacing(CONTENT_SPACING)

        self.card_camera = StatusCard("카메라", "camera")
        self.card_server = StatusCard("TCP Server", "network")
        self.card_save = StatusCard("데이터 저장", "save")
        for card in (self.card_camera, self.card_server, self.card_save):
            row.addWidget(card)
        row.addStretch()

        right = QVBoxLayout()
        right.setSpacing(CONTENT_SPACING // 2)

        self.clock_label = QLabel("")
        self.clock_label.setObjectName("operatorClock")
        self.clock_label.setAlignment(Qt.AlignCenter)
        right.addWidget(self.clock_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(CONTENT_SPACING)

        # '실행'이 예전의 '촬영'이다. 고른 칸에 한 장 찍어 판정까지 낸다.
        self.btn_run = QPushButton("실행")
        self.btn_run.setObjectName("operatorAction")
        self.btn_run.setToolTip("고른 이미지 보드에 한 장 찍고 OK/NG 를 낸다.")
        self.btn_run.clicked.connect(self._on_capture_clicked)
        buttons.addWidget(self.btn_run)

        # 모델(제품) 전환. 탭을 눌러서는 못 바꾸고 이 버튼으로만 바꾼다.
        self.btn_change_model = QPushButton("모델 변경")
        self.btn_change_model.setObjectName("operatorAction")
        self.btn_change_model.setToolTip(
            "다른 제품 모델로 바꿉니다.\n"
            "찍어 둔 이미지와 OK/NG 판정은 모두 버립니다."
        )
        self.btn_change_model.clicked.connect(self._on_change_model_clicked)
        buttons.addWidget(self.btn_change_model)

        right.addLayout(buttons)
        row.addLayout(right)
        return row

    def _build_side_panel(self):
        """오른쪽 세로 칸: 로그 창 + 좌표 칸.

        촬영 버튼은 여기 없다 — 260910 에 오른쪽 위 '실행'으로 옮겼다. 시각과
        버튼을 한 덩어리로 모아 두면 운영 중에 눈이 한 자리만 보면 된다.
        """
        side = QVBoxLayout()
        side.setSpacing(CONTENT_SPACING // 2)

        caption = QLabel("로그")
        caption.setObjectName("operatorLogCaption")
        side.addWidget(caption)

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("operatorLog")
        self.log_view.setReadOnly(True)
        self.log_view.setFixedWidth(SIDE_PANEL_WIDTH)
        # 오래 돌리면 줄이 끝없이 쌓이므로 최근 것만 남긴다.
        self.log_view.setMaximumBlockCount(MAX_LOG_LINES)
        side.addWidget(self.log_view, 1)

        pose_box = QFrame()
        pose_box.setObjectName("operatorPoseBox")
        pose_box.setFixedWidth(SIDE_PANEL_WIDTH)
        pose_layout = QGridLayout(pose_box)
        pose_layout.setContentsMargins(1, 1, 1, 1)
        pose_layout.setHorizontalSpacing(1)
        pose_layout.setVerticalSpacing(1)

        self.pose_labels = {}
        for row, names in enumerate((("x", "y", "z"), ("rx", "ry", "rz"))):
            for column, name in enumerate(names):
                label = QLabel()
                label.setObjectName("operatorPoseCell")
                label.setAlignment(Qt.AlignCenter)
                pose_layout.addWidget(label, row, column)
                pose_layout.setColumnStretch(column, 1)
                self.pose_labels[name] = label
        side.addWidget(pose_box)
        self._show_pose(None)

        return side

    # ===== 상단 카드 / 시각 =====
    def showEvent(self, event):
        """오퍼레이터 화면이 보일 때만 시계를 돌린다."""
        super().showEvent(event)
        self._tick()
        self._clock_timer.start()

    def hideEvent(self, event):
        self._clock_timer.stop()
        super().hideEvent(event)

    def _tick(self):
        """시각을 갱신하고 상태 카드를 다시 읽는다(1초마다)."""
        self.clock_label.setText(
            f"Date : {datetime.now():%y-%m-%d %H:%M:%S}"
        )
        self._refresh_status_cards()

    def _refresh_status_cards(self):
        """카메라 / TCP 서버 / 데이터 저장 상태를 카드에 싣는다.

        촬영이 안 될 때 원인은 거의 이 셋 중 하나다. 세 가지를 늘 보이게 둬야
        현장에서 로그를 뒤지지 않는다.
        """
        # ── 카메라 ──
        camera_page = self._find_camera_page()
        if camera_page is None:
            self.card_camera.set_title("카메라")
            self.card_camera.set_state("Camera Selection 페이지 없음", False)
        else:
            name, connected = camera_page.camera_status()
            self.card_camera.set_title(name or "카메라 미선택")
            self.card_camera.set_state(
                "연결됨" if connected else "연결 안 됨", connected
            )

        # ── TCP 서버 ──
        config = self.run_config.server_config
        where = (
            f"{config.host}:{config.port}"
            if config is not None
            else "IP 미지정"
        )
        if not self.run_config.server_enabled:
            self.card_server.set_state("사용 안 함", None)
        elif not self._server.is_running():
            # 시작 실패 사유는 _restart_server 가 이미 카드에 실어 두었다.
            # 여기서 덮어쓰면 사유가 1초 만에 사라진다.
            if not self._server_failed:
                self.card_server.set_state(f"{where} 꺼짐", False)
        else:
            clients = self._server.client_count()
            if clients:
                self.card_server.set_state(f"{where} 연결됨 ({clients})", True)
            else:
                self.card_server.set_state(f"{where} 대기 중", None)

        # ── 데이터 저장 ──
        if self.run_config.save_enabled:
            self.card_save.set_state(date_dir(self.run_config.save_path), True)
        else:
            self.card_save.set_state("저장 안 함", None)

    @staticmethod
    def _judge(hole_count, pose):
        """이번 촬영이 로봇에 나갈 값인지 가린다. → (OK 여부, 사유)

        사유는 NG 일 때만 채운다. 무엇 때문에 막혔는지 로그에 남아야 현장에서
        조명을 고칠지 세팅을 고칠지 판단할 수 있다.
        """
        if hole_count is None:
            return False, "분석하지 못했습니다"
        if hole_count != OK_HOLE_COUNT:
            return False, f"홀 {hole_count}개 (1개여야 합니다)"
        if pose is None:
            return False, "좌표를 구하지 못했습니다"
        return True, ""

    # ===== 설정 받기 =====
    def set_run_config(self, config):
        """세팅 페이지에서 정한 값을 받아 반영한다."""
        self.run_config = config
        self._reset_results()
        self.display_grid.set_points(config.fastening_points)
        self._reload_model_tabs()
        self._restart_server()
        self._tick()

    def _reset_results(self):
        """찍어 둔 것을 통째로 버린다(모델이 바뀌거나 새로 들어올 때).

        좌표까지 버리는 것이 요점이다 — 화면만 지우고 남겨 두면 그 칸에
        결과(102)를 물어볼 때 지난 값이 정상 응답으로 나간다.
        """
        self._poses_by_sequence.clear()
        self._pose_lists_by_sequence.clear()
        self._judgements_by_sequence.clear()
        self._show_pose(None)

    def _reload_model_tabs(self):
        """모델 탭을 다시 짠다. 전체 모델을 늘어놓고 지금 것만 밝게 둔다.

        모델 목록이 없는 구성(시험 등)에서는 공정명 하나를 탭으로 둔다 — 탭 줄이
        통째로 비면 보드 위가 어색하게 잘려 보인다.
        """
        names = [entry.get("name", "") for entry in self.run_config.models]
        if not names:
            names = [self.run_config.process_name or "모델 1"]
            selected = 0
        else:
            active = self.run_config.active_model
            selected = names.index(active) if active in names else 0
        self.model_tabs.set_models(names, selected=selected)

    # ===== 모델 변경 =====
    def _on_change_model_clicked(self):
        """'모델 변경' 창을 띄우고, 고른 모델로 갈아 끼운다."""
        models = self.run_config.models
        if not models:
            self._log("바꿀 모델이 없습니다 — 세팅에서 모델을 먼저 만드세요")
            return
        name = ModelSelectDialog.get_model(
            models, self.run_config.active_model, self
        )
        if name is None:
            return   # 취소했거나 같은 모델을 고름
        self.apply_model(name, source="모델 변경")

    def apply_model(self, name, source="모델 변경"):
        """이름으로 모델을 갈아 끼운다. → 바뀌었는지

        찍어 둔 이미지·판정·좌표를 **모두 버린다**(260910 사용자 결정). 모델이
        바뀌면 체결 포인트 수가 달라져 보드 칸도 다시 짜이는데, 지난 제품의
        사진과 좌표가 남아 있으면 그 칸에 결과를 물어볼 때 엉뚱한 값이 나간다.

        세팅 페이지의 탭도 같이 옮긴다 — 프로그램 전체에서 '현재 모델'은 하나다.
        """
        entry = next(
            (item for item in self.run_config.models
             if item.get("name") == name),
            None,
        )
        if entry is None or name == self.run_config.active_model:
            return False

        points = [
            FasteningPoint.from_dict(item)
            for item in (entry.get("fastening_points") or [])
        ]
        self.run_config.active_model = name
        self.run_config.fastening_points = points

        self._reset_results()
        self.display_grid.set_points(points)
        self._reload_model_tabs()
        self._log(f"{source} — '{name}' (체결 포인트 {len(points)}개, 결과 초기화)")

        # 세팅 페이지도 같은 모델을 열게 한다.
        page = getattr(self._window, "operator_launch_page", None)
        if page is not None:
            page.select_model(name)
        return True

    # ===== 서버 =====
    def _restart_server(self):
        self._server.stop()
        self._server_failed = False

        if not self.run_config.server_enabled:
            return

        if self.run_config.server_config is not None:
            self._server.config = self.run_config.server_config

        ok, reason = self._server.start()
        if not ok:
            # 포트가 이미 쓰이는 등의 이유를 카드에 그대로 남긴다. 1초 타이머가
            # 이 문구를 덮어쓰지 않도록 실패했다는 것을 기억해 둔다.
            self._server_failed = True
            self.card_server.set_state(f"서버 실패 — {reason}", False)
            self._log(f"서버를 띄우지 못했습니다 — {reason}")

    def stop_server(self):
        """창을 닫거나 세팅으로 돌아갈 때 부른다."""
        self._server.stop()
        self._server_failed = False

    # ===== 트리거 처리 (서버 스레드에서 불린다) =====
    def _valid_sequence(self, sequence):
        """체결 세팅에 있는 순서인지 확인한다.

        세팅한 개수보다 큰 번호가 오면 실패로 돌려준다. 없는 자리를 찍은 척
        하고 좌표를 주면 로봇이 엉뚱한 곳으로 간다.
        """
        count = len(self.run_config.fastening_points)
        return isinstance(sequence, int) and 1 <= sequence <= count

    def _on_capture(self, request):
        if not self._valid_sequence(request.sequence):
            return None
        task = {
            "sequence": request.sequence,
            "done": Event(),
            "success": False,
        }
        self._capture_requested.emit(task)   # 카메라와 화면은 UI 스레드에서 처리

        deadline = time.monotonic() + CAPTURE_TRIGGER_TIMEOUT
        while not task["done"].wait(0.1):
            if not self._server.is_running() or time.monotonic() >= deadline:
                return None
        return {} if task["success"] else None

    def _on_result(self, request):
        if not self._valid_sequence(request.sequence):
            return None
        # NG 였던 칸의 좌표는 내보내지 않는다. 촬영과 결과가 따로 오므로,
        # 여기서 다시 판정을 보지 않으면 NG 뒤에 정상 좌표가 나갈 수 있다.
        if not self._judgements_by_sequence.get(request.sequence, False):
            return None
        pose = self._poses_by_sequence.get(request.sequence)
        if pose is None:
            return None
        # 회전 세 값을 어느 순서로 실을지는 설정이 정한다(로봇마다 다르다).
        # 각도 값은 그대로이고 나가는 자리만 바뀐다.
        a, b, c = protocol.rotation_values(pose, self._rotation_order())
        return {
            "reserved1": 1,
            "reserved2": 1,
            "reserved3": 0,
            "x": pose.x,
            "y": pose.y,
            "z": pose.z,
            "a": a,
            "b": b,
            "c": c,
            "reserved4": 0,
            "reserved5": 0,
        }

    def _on_switch_model(self, request):
        """103 — n 번째 모델로 갈아 끼운다.

        n 은 '모델 변경' 창에 늘어선 순서(=run_config.models 순서)다. 목록에
        없는 번호면 실패로 돌려준다 — 없는 제품으로 바꾼 척하면 그다음 촬영이
        엉뚱한 제품의 체결 좌표를 내보낸다.

        이미 그 모델이면 바꿀 것이 없지만 **성공(1107)** 이다. 로봇이 주기마다
        같은 번호를 보내도 되게 하려는 것이라, '요청한 모델 상태가 됐는가'로
        본다(사용자 결정). 이때는 찍어 둔 것을 버리지 않는다.
        """
        order = request.values.get("model")
        models = self.run_config.models
        if not isinstance(order, int) or not 1 <= order <= len(models):
            return None

        name = models[order - 1].get("name")
        if not name:
            return None
        if name == self.run_config.active_model:
            return {}

        task = {"name": name, "done": Event(), "success": False}
        self._model_switch_requested.emit(task)   # 화면은 UI 스레드에서 짠다

        deadline = time.monotonic() + MODEL_SWITCH_TIMEOUT
        while not task["done"].wait(0.1):
            if not self._server.is_running() or time.monotonic() >= deadline:
                return None
        return {} if task["success"] else None

    def _rotation_order(self):
        """지금 쓸 회전 출력 순서. 설정이 없으면 쿠카 순서(ZYX)."""
        config = self.run_config.server_config
        return getattr(config, "rotation_order", protocol.ROTATION_ZYX)

    # ===== 카메라 =====
    def bind_window(self, window):
        """MainWindow 를 받아 둔다. 카메라 페이지를 찾을 때 쓴다."""
        self._window = window

    def _find_camera_page(self):
        """세팅 페이지 스택에서 Camera Capture 페이지를 찾는다.

        카메라는 그 페이지 하나만 열고 나머지는 빌려 쓴다(D405 는 한 번에
        한 곳만 열 수 있다). Hole Detection 도 같은 방식으로 쓴다.
        PAGES 순서가 바뀌어도 되도록 index 를 고정하지 않고 훑는다.
        """
        stack = getattr(self._window, "stack", None)
        if stack is None:
            return None
        for index in range(stack.count()):
            page = stack.widget(index)
            if isinstance(page, CameraCapturePage):
                return page
        return None

    def _find_hole_detection_page(self):
        """현재 블랍 세팅을 가진 Hole Detection 페이지를 찾는다."""
        return self._find_page(HoleDetectionPage)

    def _find_pose_setting_page(self):
        """진입 각도 오프셋을 가진 Pose Setting 페이지를 찾는다."""
        return self._find_page(PoseSettingPage)

    def _find_page(self, page_type):
        stack = getattr(self._window, "stack", None)
        if stack is None:
            return None
        for index in range(stack.count()):
            page = stack.widget(index)
            if isinstance(page, page_type):
                return page
        return None

    def _visualize_blobs(
        self, sequence, color, depth, intrinsics, depth_scale_mm
    ):
        """체결 순번에 지정된 블랍 세팅으로 오버레이를 만든다.

        Returns:
            (오버레이 이미지, 홀 개수, 포즈, 포즈, 법선 문구, 실패 사유).
            셋째와 넷째는 같은 목록이다('포즈 조정'이 사라진 뒤로 '조정 전/후'
            구분이 없어졌지만 자리는 남겨 두었다). 법선 문구는 매칭점 세 곳을 다
            찾았는지 로그로 남기는 데 쓴다 — 오퍼레이터에는 뎁스 보드가 없어서
            못 찾은 것을 눈으로 볼 방법이 없다.
        """
        point = self.run_config.fastening_points[sequence - 1]
        profile_name = getattr(point, "blob_setting", "")
        if not profile_name:
            return color, None, [], [], None, "비전 세팅이 지정되지 않았습니다"

        page = self._find_hole_detection_page()
        if page is None:
            return color, None, [], [], None, "Hole Detection 페이지를 찾을 수 없습니다"
        try:
            visualized, holes, poses, raw_poses, note = page.visualize_for_operator(
                color,
                profile_name,
                depth=depth,
                intrinsics=intrinsics,
                depth_scale_mm=depth_scale_mm,
                previous_poses=self._pose_lists_by_sequence.get(sequence),
            )
        except Exception as exc:
            return color, None, [], [], None, str(exc)
        return visualized, len(holes), poses, raw_poses, note, ""

    # ===== 촬영 =====
    def _grab_frame(self):
        """카메라에서 한 장 얻는다.

        Returns:
            (컬러 BGR, 뎁스, 내부 파라미터, 뎁스 단위, 실패 사유).
            성공이면 사유가 빈 문자열이고,
            실패면 컬러가 None 이다.
        """
        camera_page = self._find_camera_page()
        if camera_page is None:
            return None, None, None, 1.0, "Camera Capture 페이지를 찾을 수 없습니다"

        try:
            rgb, depth = camera_page.grab_frames()
        except Exception as exc:
            # 미연결이면 grab_frames 가 RuntimeError 를 올린다.
            return None, None, None, 1.0, str(exc)

        # 이후 처리(저장/OpenCV)는 BGR 기준이라 여기서 바꿔 둔다.
        return (
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            depth,
            camera_page.color_intrinsics(),
            camera_page.depth_scale_mm(),
            "",
        )

    def _capture_into(self, sequence, source):
        """지정한 칸에 촬영 결과를 담는다.

        Args:
            sequence: 체결 순서 (1부터)
            source: 로그에 남길 촬영 계기 ("수동" / "트리거")

        Returns:
            OK 판정 여부. 이 값이 곧 촬영 응답(101)의 성공/실패다 —
            NG 면 부르는 쪽이 실패 코드 1003 을 내보낸다.
        """
        if self.display_grid.cell(sequence) is None:
            self._log(f"{source} 촬영 실패 — {sequence}번 칸이 없습니다")
            return False

        capture_started = time.perf_counter()
        color, depth, intrinsics, depth_scale_mm, reason = self._grab_frame()
        capture_ms = (time.perf_counter() - capture_started) * 1000.0

        if color is None:
            # 찍히지 않았으므로 최신 표시를 옮기지 않는다. 옮기면 이전에
            # 제대로 찍힌 칸을 못 찾게 되고, 안 찍힌 칸이 찍힌 것처럼 보인다.
            #
            # 지난 주기를 비우지도 않는다(아래 _start_new_cycle 을 부르지 않는다).
            # 한 장 못 받았다고 앞서 제대로 찍어 둔 것을 버리면, 카메라가 잠깐
            # 끊긴 것만으로 주기 전체를 다시 찍어야 한다.
            self._log(
                f"{source} 촬영 실패 {sequence}번 — 획득 {capture_ms:.1f} ms · {reason}"
            )
            self._judgements_by_sequence[sequence] = False
            self.display_grid.set_verdict(sequence, False)
            return False

        # 1번 칸이 찍혔으면 새 주기다 — 나머지 칸을 비운다(260910 요구).
        # 사진을 실제로 받은 뒤에 비운다. 위에서 실패했을 때 비우면 카메라가
        # 잠깐 끊긴 것만으로 지난 주기가 사라진다.
        if sequence == FIRST_SEQUENCE:
            self._start_new_cycle(sequence)

        analyze_started = time.perf_counter()
        (
            visualized,
            blob_count,
            poses,
            raw_poses,
            normal_note,
            analyze_error,
        ) = self._visualize_blobs(
            sequence, color, depth, intrinsics, depth_scale_mm
        )
        analyze_ms = (time.perf_counter() - analyze_started) * 1000.0
        self.display_grid.set_pixmap(sequence, to_pixmap(visualized))
        self.display_grid.mark_latest(sequence)
        camera_pose = next((item for item in poses if item is not None), None)
        self._log_pose_source(normal_note, poses, camera_pose)
        pose = None
        if camera_pose is not None:
            point = self.run_config.fastening_points[sequence - 1]
            try:
                cam2gripper = load_cam2gripper(point.calibration)
                self._warn_about_calibration(point.calibration)
                # 플랜지 포즈와 진입 오프셋은 이 체결 지점의 세팅에서 가져온다.
                # 화면 값을 그대로 읽으면 지점마다 다른 값이 하나로 뭉개진다.
                hole_page = self._find_hole_detection_page()
                flange_pose = self._flange_pose(point, hole_page)
                offset = self._approach_offset(point, hole_page)
                if flange_pose is not None:
                    # T_base_target = T_base_flange × T_flange_target
                    pose = base_target_pose(
                        flange_pose, cam2gripper, camera_pose, offset
                    )
                else:
                    # 플랜지 포즈가 없으면 플랜지 기준 좌표만 넘긴다.
                    pose = flange_target_pose(camera_pose, cam2gripper, offset)
                if any(offset):
                    self._log(
                        f"  진입 오프셋 적용 RX {offset[0]:+.2f}° "
                        f"RY {offset[1]:+.2f}° RZ {offset[2]:+.2f}°"
                    )
            except ValueError as exc:
                analyze_error = str(exc)
        # 다음 촬영의 previous_poses 로 넘어가는 자리(오퍼레이터 계약).
        self._pose_lists_by_sequence[sequence] = raw_poses
        self._poses_by_sequence[sequence] = pose
        self._show_pose(pose)
        if analyze_error:
            self._log(
                f"{source} 촬영 {sequence}번 — 획득 {capture_ms:.1f} ms · "
                f"분석 실패: {analyze_error}"
            )
        else:
            self._log(
                f"{source} 촬영 {sequence}번 — 획득 {capture_ms:.1f} ms · "
                f"분석 {analyze_ms:.1f} ms · 블랍 {blob_count}개"
            )

        # 판정. 저장보다 먼저 세운다 — 저장이 실패해도 판정은 남아야 한다
        # (사진을 못 남긴 것과 체결 좌표가 틀린 것은 다른 일이다).
        verdict, why = self._judge(blob_count, pose)
        if analyze_error:
            verdict, why = False, analyze_error
        self._judgements_by_sequence[sequence] = verdict
        # 배지는 그 칸에 남는다. 네 칸을 찍으면 판정도 네 개다.
        self.display_grid.set_verdict(sequence, verdict)
        self._log(f"  판정 {'OK' if verdict else 'NG'}" + (f" — {why}" if why else ""))

        if self.run_config.save_enabled:
            # 화면 확인용 컬러맵과 가상 카메라 재사용용 원본 뎁스를 함께 남긴다.
            ok, result = save_frames(
                self.run_config.save_path,
                color_image=color,
                depth_image=colorize_depth(depth),
                raw_depth_image=depth,
                depth_scale_mm=depth_scale_mm,
            )
            if ok:
                self._log(f"  저장 {display_path(result['color'])}")
            else:
                self._log(f"  저장 실패 — {result}")
        # OK 일 때만 촬영 응답이 성공으로 나간다(NG 면 부르는 쪽이 1003 을 낸다).
        return verdict

    def _start_new_cycle(self, sequence):
        """1번 칸을 새로 찍었을 때 나머지 칸을 통째로 비운다.

        사진만 지우면 안 된다. 좌표와 판정을 남기면 그 칸에 결과(102)를 물어볼 때
        **지난 주기 값이 정상 응답으로 나간다** — 화면은 비어 있는데 로봇은
        좌표를 받는 상태가 된다.
        """
        if self.display_grid.count() <= 1:
            return
        self.display_grid.clear_cells(keep=sequence)
        for other in list(self._judgements_by_sequence):
            if other != sequence:
                self._judgements_by_sequence.pop(other, None)
        for other in list(self._poses_by_sequence):
            if other != sequence:
                self._poses_by_sequence.pop(other, None)
                self._pose_lists_by_sequence.pop(other, None)
        self._log("새 주기 — 나머지 보드를 비웠습니다")

    def _on_capture_clicked(self):
        sequence = self.display_grid.selected()
        if sequence <= 0:
            # 칸이 여러 개면 어디에 담을지 알 수 없다.
            self._log("촬영할 이미지 보드를 먼저 누르세요")
            return
        self._capture_into(sequence, "수동")

    def _on_cell_selected(self, sequence):
        self._log(f"{sequence}번 보드 선택")
        self._show_pose(self._poses_by_sequence.get(sequence))

    def _show_pose(self, pose):
        """좌표 전용 칸에 첫 번째 유효 홀 포즈를 표시한다."""
        if pose is None:
            for name, label in self.pose_labels.items():
                label.setText(f"{name}: -")
            return
        for name, label in self.pose_labels.items():
            label.setText(f"{name}: {getattr(pose, name):.1f}")

    # ===== 로그 =====
    def _log(self, text):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"{stamp}  {text}")

    @staticmethod
    def _flange_pose(point, hole_page):
        """이 체결 지점에 쓸 로봇 플랜지 포즈 (x, y, z, rx, ry, rz). 없으면 None.

        로봇이 서는 자리는 체결 지점마다 다르므로 프로파일에 저장된 값이
        기준이다. 화면 값을 그냥 읽으면 Hole Detection 이 열어 둔 프로파일의
        포즈가 모든 지점에 적용돼, 1번 지점의 좌표가 2번 지점의 플랜지 포즈로
        계산된다.

        단, 화면이 이 지점과 같은 프로파일을 열어 두고 있으면 화면 값을 먼저
        쓴다. 티칭할 때 저장을 누르지 않아도 바로 확인할 수 있어야 한다
        (_approach_offset 과 같은 규칙이다).
        """
        if (
            hole_page is not None
            and point.blob_setting
            and hole_page.current_profile() == point.blob_setting
        ):
            return hole_page.current_flange_pose()
        return load_flange_pose(point.blob_setting)

    def _approach_offset(self, point, hole_page):
        """이 체결 지점에 쓸 진입 각도 오프셋 (rx, ry, rz) [도].

        오프셋은 체결 지점마다 다르므로 프로파일에 저장된 값이 기준이다. 단,
        세팅 화면이 같은 프로파일을 열어 두고 있으면 화면의 값을 먼저 쓴다.
        티칭할 때 저장을 누르지 않아도 바로 확인할 수 있어야 한다(플랜지 포즈를
        화면에서 바로 읽는 것과 같은 이유다).

        '같은 프로파일인가'는 여전히 Hole Detection 에게 묻는다 — 프로파일의
        주인이 저쪽이고, Pose Setting 은 같은 것을 비춰 볼 뿐이다. 값 자체는
        Pose Setting 이 들고 있으므로 두 페이지를 모두 봐야 한다.
        """
        pose_page = self._find_pose_setting_page()
        if (
            hole_page is not None
            and pose_page is not None
            and point.blob_setting
            and hole_page.current_profile() == point.blob_setting
        ):
            return pose_page.current_approach_offset()
        return load_approach_offset(point.blob_setting)

    def _log_pose_source(self, normal_note, poses, camera_pose):
        """자세를 어디서 받았는지 촬영마다 남긴다.

        자세가 튈 때 원인이 '매칭점을 못 찾아 홀 주변 고리로 떨어진 것'인지
        '비전 자체가 흔들린 것'인지를 로그만 보고 가릴 수 있어야 한다. 세팅
        화면은 상태문구로 같은 내용을 늘 보여 주는데 오퍼레이터에는 그 자리가
        없어서, 같은 세팅에서 결과가 달라도 짚을 단서가 없었다.

        매칭점 기준 홀 수가 0 으로 떨어지면 고리 기준으로 밀린 것이다
        (고리는 ±8도까지 흔들린다).
        """
        posed = [item for item in poses if item is not None]
        if not posed:
            return

        if normal_note:
            self._log(f"  {normal_note}")
        from_surface = sum(
            1 for item in posed if item.plane_source == PLANE_FROM_SURFACE
        )
        if from_surface:
            self._log(f"  z축 매칭점 기준 {from_surface}/{len(posed)}개")
        else:
            self._log(f"  z축은 홀 주변 고리 기준 {len(posed)}개 (매칭점 없음)")

        if camera_pose is not None:
            # 로봇 좌표가 튀었을 때 비전이 튄 것인지 변환이 튄 것인지 가르는 값
            self._log(
                f"  카메라 기준 자세 RX {camera_pose.rx:+.2f}° "
                f"RY {camera_pose.ry:+.2f}° RZ {camera_pose.rz:+.2f}°"
            )

    def _warn_about_calibration(self, session_name):
        """실측이 아닌 캘리브레이션으로 좌표가 나가고 있으면 로그에 짚어 준다.

        캘리브레이션마다 한 번만 적는다. 촬영마다 되풀이하면 로그가 경고로
        덮여 정작 봐야 할 촬영 기록이 묻힌다.
        """
        if session_name in self._warned_calibrations:
            return
        self._warned_calibrations.add(session_name)
        for warning in calibration_warnings(session_name):
            self._log(f"[주의] 캘리브레이션 '{session_name}' — {warning}")

    # ===== 화면 갱신 (화면 스레드) =====
    def _show_server_event(self, text):
        """서버가 알려 온 일을 로그에 남기고 카드를 다시 읽는다.

        패킷 내용은 **로그에만** 둔다. 카드에는 '지금 듣고 있나 / 몇 대가
        붙어 있나'가 있어야 한다 — 방금 지나간 패킷 한 줄을 카드에 띄우면
        접속이 끊겨도 마지막 패킷이 그대로 남아 붙어 있는 것처럼 보인다.
        """
        self._log(text)   # 주고받은 패킷을 되짚을 수 있게 로그에 남긴다
        self._refresh_status_cards()

    def _mark_capture(self, task):
        """트리거로 들어온 촬영. 화면 스레드에서 불린다(신호로 넘어온다)."""
        try:
            sequence = task["sequence"]
            self._poses_by_sequence.pop(sequence, None)
            # 지난 판정도 함께 버린다. 남겨 두면 이번 촬영이 도중에 끊겼을 때
            # 결과 응답(102)이 옛 OK 를 보고 옛 좌표를 내보낸다.
            self._judgements_by_sequence.pop(sequence, None)
            task["success"] = self._capture_into(sequence, "트리거")
        finally:
            task["done"].set()

    def _switch_model_by_order(self, task):
        """트리거(103)로 들어온 모델 변경. 화면 스레드에서 불린다."""
        try:
            task["success"] = self.apply_model(task["name"], source="트리거")
        finally:
            task["done"].set()

    def leave(self):
        """세팅(홈 인터페이스)으로 돌아간다. 타이틀바의 버튼이 부른다.

        버튼이 타이틀바로 옮겨 갔으므로(260910) 이 자리가 공개 창구다. 서버를
        먼저 내려놓는 것이 중요하다 — 안 내려놓으면 세팅 화면에서 트리거를
        받아 카메라를 건드린다.
        """
        self.stop_server()
        if self._on_back is not None:
            self._on_back()
