"""포즈 세팅 페이지 — 찾아 둔 홀에 자세를 세우고, 그 자세를 손본다.

Hole Detection 은 '홀이 어디 있는가'까지만 답한다. 거기서 나온 픽셀 좌표에
방향과 높이를 붙여 (x, y, z, rx, ry, rz) 로 만드는 일은 전부 이 페이지가 한다.
법선을 만드는 것도 결국 포즈를 만들기 위한 것이므로 여기 있다.

구성:

    [자세 기준 ▼] [이미지 캡쳐] [컬러맵 ▼]        세팅 [프로파일 ▼] [저장]
                              ● 전체 1280×720 — 홀 26개 · 포즈 26/26 · 법선 …
                                              분석 48.7 ms  [법선 2010 · 포즈 12]
    ┌───────────────────────────────┐  ┌ Parameter Setting ─┐
    │  매칭점 + 홀 + 좌표축          │  │ 고른 기준의 값만    │
    │                               │  ├ Rotational Offset ─┤
    │                               │  │  RX / RY / RZ      │
    └───────────────────────────────┘  └────────────────────┘

자세 기준별 파라미터는 보드 **오른쪽** 패널에 있다(260910). 방식마다 필요한
줄 수가 크게 달라서 — 플랜지 고정은 0줄, 매칭점 3곳은 5줄 — 보드 위에 두면
기준을 바꿀 때마다 사진 크기가 달라졌다.

이미지는 두 갈래로 들어온다(둘 다 쓴다):
  - '이미지 캡쳐': 이 페이지에서 직접 찍는다. 홀 검출만 Hole Detection 에
    위임한다. 진입 각도 티칭은 촬영→로봇 이동→조그→오프셋 입력→재확인을
    되풀이하는 작업이라, 그 고리가 한 페이지 안에서 닫혀야 쓸 만하다.
  - 탭으로 들어올 때 Hole Detection 이 새로 찍었으면 그것을 자동으로 이어받는다.
    260910 에 '검출 결과 가져오기' 버튼을 없앤 것은 이 자동 이어받기가 있어
    손으로 누를 일이 없었기 때문이다.

세팅(프로파일)은 Hole Detection 과 **하나를 함께 쓴다.** 파일 하나에 검출 키와
포즈 키가 같이 들어 있고, 각 페이지는 자기 키만 읽고 쓴다. 오퍼레이터의 체결
위치 테이블이 이름 하나로 프로파일을 지목하고 있어서 쪼갤 수 없다.

회전 오프셋은 '진입 각도 오프셋' 하나다. 로봇으로 넘길 때 공구를 더 기울이는
값이고 화면 자세는 바뀌지 않는다. 260910 에 '포즈 조정'(화면 자세를 직접 돌리는
값)을 없앴다 — 둘이 겹치면 어느 쪽이 얼마를 돌렸는지 가릴 수 없었다.
"""

import time

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from depth_view import colorize_depth
from elided_label import ElidedLabel
from hole_pose import (
    NORMAL_MODE_FLANGE,
    NORMAL_MODE_POINTS,
    NORMAL_MODE_RING,
    RING_RADIUS_MM_DEFAULT,
    RING_RADIUS_MM_MAX,
    RING_RADIUS_MM_MIN,
    estimate_hole_poses,
)
from image_board import ResultBoard
from normal_plane import mask_to_depth_size, plane_from_normal_points
from pages.base_page import BasePage
# 홀을 그리는 일은 Hole Detection 이 정본이다. 두 화면의 홀이 같아 보여야 하므로
# 따로 그리지 않고 같은 것을 쓴다. 저쪽은 이 모듈을 import 하지 않으므로 순환이 아니다.
from pages.hole_detection_page import BlobResultVisualizer
from pose_pipeline import (
    APPROACH_OFFSET_FIELDS,
    APPROACH_OFFSET_HINT,
    NORMAL_MODE_CHOICES,
    NORMAL_MODE_HINTS,
    NORMAL_POINTS_HINT,
    as_float,
    draw_normal_points,
    draw_ring_regions,
    normal_plane_from_masks,
    normal_points_from_settings,
    normal_points_masks,
    normal_source_from_settings,
    spread_note,
)
from roi_state import roi_state
import sam2_detector
from sam2_video_tracker import NORMAL_POINTS, measure_normal_point_areas
from template_store import (
    list_normal_points,
    save_normal_points,
    validate_template_name,
)
from text_input_dialog import TextInputDialog
from setting_panel import SettingGroup, SettingPanel
from theme import CONTENT_SPACING, SETTING_ROW_SPACING
from toggle_switch import ToggleSwitch

# ===== 이미지 보드에 띄울 화면 =====
# 검출 마스크는 여기 없다 — 그건 '홀을 어떻게 찾았는가'라서 Hole Detection 몫이다.
VIEW_COLOR = "color"
VIEW_DEPTH = "depth"
VIEW_CHOICES = [
    ("컬러맵", VIEW_COLOR),
    ("뎁스맵", VIEW_DEPTH),
]


class PoseSettingPage(BasePage):
    """홀 자세(법선)를 세우고 오프셋으로 손보는 페이지."""

    title = "Pose Setting"
    nav_icon = "pose"
    expand_content = True   # 결과 보드가 남는 공간을 모두 채우도록

    def __init__(self, parent=None):
        super().__init__(parent)

        self._window = None          # bind_window()로 주입되는 MainWindow
        self._hole_page = None       # bind_hole_page()로 주입되는 Hole Detection

        self._last_captured_image = None   # 마지막 이미지 (BGR 원본)
        self._last_depth = None            # 같은 시점의 뎁스 (없으면 None)
        self._last_intrinsics = None       # (camera_matrix, dist_coeffs)
        self._depth_scale_mm = 1.0         # 뎁스 값 1당 mm
        self._last_holes = []              # [(x, y, radius), ...]
        self._last_hole_masks = []         # 홀마다 모델이 문 경계 (없으면 빈 목록)
        self._last_poses = []              # 홀별 3D 포즈 (못 구한 자리는 None)
        # 오퍼레이터가 previous_poses 로 받아 가는 자리. '포즈 조정'이 사라진
        # 뒤로는 _last_poses 와 같은 값이지만, 부르는 쪽 계약을 그대로 두려고
        # 이름은 남겨 두었다.
        self._last_raw_poses = []
        # 이번 촬영의 법선 매칭점 (마스크, 문구, 네모). 촬영마다 비운다 —
        # 비디오 추적을 촬영 한 장에 한 번만 돌리려는 것.
        self._normal_cache = None
        # 세 점이 벌어진 정도(mm). 막지 않는 대신 문구로 알리는 값이다.
        self._normal_spread_mm = None
        self._stage_seconds = {}
        self._syncing = False        # 위젯끼리 값을 맞출 때 되먹임 방지
        # 지금 들고 있는 이미지가 Hole Detection 의 몇 번째 촬영에서 온 것인지.
        # 탭으로 들어올 때 저쪽 번호와 견주어 새로 찍혔으면 이어받는다.
        self._ingested_seq = None

        self.content_layout.addLayout(self._build_controls())
        self.content_layout.addLayout(self._build_status_row())
        self.content_layout.addLayout(self._build_elapsed_row())

        # 결과 보드(좌) + 설정 패널(우).
        # 보드는 이미지 비율까지만 자라므로 남는 폭이 생긴다. 그 폭을 둘 사이에
        # 두면 가운데가 빈 구멍처럼 보이므로, 스트레치를 양 끝에만 둔다.
        board_row = QHBoxLayout()
        board_row.setSpacing(CONTENT_SPACING)

        self.result_board = ResultBoard(
            "이미지 캡쳐 버튼을 누르면 여기에 표시됩니다."
        )
        self.result_board.region_selected.connect(self._on_region_dragged)

        board_row.addStretch(1)
        board_row.addWidget(self.result_board, 100)
        board_row.addWidget(self._build_setting_panel())
        board_row.addStretch(1)
        self.content_layout.addLayout(board_row, 1)

        self._reload_normal_list()
        self._update_normal_controls()
        self._update_mode_rows()

    # ===== 다른 페이지와의 연결 =====
    def bind_window(self, window):
        """MainWindow가 페이지 등록 시 자신을 주입한다(PythonUI._load_pages)."""
        self._window = window

    def bind_hole_page(self, page):
        """Hole Detection 페이지를 주입받는다(PythonUI._load_pages).

        isinstance 로 스택을 훑지 않고 명시적으로 받는 이유: Hole Detection 이
        이 페이지를 import 하면 순환이 된다(이 페이지가 저쪽을 import 한다).
        MainWindow 는 이미 둘 다 알고 있으므로 거기서 이어 주는 것이 싸다.
        """
        self._hole_page = page
        self._reload_profile_list()

    def showEvent(self, event):
        """탭으로 들어올 때, Hole Detection 이 새로 찍었으면 그것을 이어받는다.

        '아직 아무것도 없을 때만' 이어받게 두면 안 된다. 그러면 부품을 옮겨
        다시 찍고 이 탭으로 와도 **첫 촬영이 계속 남아** 있어서, 위치 변화에
        반응하지 않는 것처럼 보인다(260908 에 실제로 그렇게 만들어 놓았다).

        그래서 촬영 번호를 견준다. 이 페이지에서 직접 찍은 뒤에는 저쪽 번호를
        따라 적어 두므로, 저쪽이 **또** 찍기 전까지는 내 촬영이 유지된다 —
        오프셋을 맞추는 중에 화면이 바뀌면 비교 대상이 사라지기 때문이다.
        """
        super().showEvent(event)
        self._sync_profile_selection()
        if self._hole_page is None:
            return
        if (self._last_captured_image is None
                or self._hole_page.capture_sequence() != self._ingested_seq):
            self._pull_from_hole_page(quiet=True)

    # ===== UI 구성 =====
    def _build_controls(self):
        """맨 위 한 줄: [자세 기준 ▼] [이미지 캡쳐] [화면 ▼]   세팅 [▼] [저장].

        '검출 결과 가져오기' 버튼은 260910 에 없앴다. 탭으로 들어올 때 저쪽이
        새로 찍었으면 자동으로 이어받으므로(showEvent), 손으로 누를 일이 없었다.
        """
        row = QHBoxLayout()
        row.setSpacing(CONTENT_SPACING)

        # 자세를 어디서 받을지. Hole Detection 의 '검출 방식' 콤보와 같은 자리다 —
        # 이 화면에서 가장 크게 갈리는 선택이라 맨 앞에 둔다.
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("viewCombo")
        for caption, key in NORMAL_MODE_CHOICES:
            self.mode_combo.addItem(caption, key)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)

        self.btn_capture = QPushButton("이미지 캡쳐")
        self.btn_capture.setObjectName("primaryBtn")
        self.btn_capture.setToolTip(
            "여기서 직접 찍는다. 홀 검출은 Hole Detection 세팅을 그대로 쓴다."
        )
        self.btn_capture.clicked.connect(self._on_capture_clicked)

        self.view_combo = QComboBox()
        self.view_combo.setObjectName("viewCombo")
        for label, key in VIEW_CHOICES:
            self.view_combo.addItem(label, key)
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)

        profile_label = QLabel("세팅")
        profile_label.setObjectName("paramLabel")

        # Hole Detection 과 같은 프로파일을 본다. 여기서 바꾸면 저쪽도 따라온다.
        self.profile_combo = QComboBox()
        self.profile_combo.setObjectName("profileCombo")
        self.profile_combo.setToolTip(
            "Hole Detection 과 같은 세팅 파일을 쓴다.\n"
            "검출 값과 포즈 값이 한 파일에 함께 들어 있다."
        )
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)

        self.btn_save = QPushButton("저장")
        self.btn_save.setObjectName("secondaryBtn")
        self.btn_save.setToolTip("지금 세팅에 포즈 값까지 함께 저장한다.")
        self.btn_save.clicked.connect(self._on_save_clicked)

        row.addWidget(self.mode_combo)
        row.addWidget(self.btn_capture)
        row.addWidget(self.view_combo)
        row.addStretch()
        row.addWidget(profile_label)
        row.addWidget(self.profile_combo)
        row.addWidget(self.btn_save)
        return row

    def _build_setting_panel(self):
        """이미지 보드 오른쪽 패널: 자세 기준별 파라미터 + 진입 각도 오프셋.

        자세 기준을 바꿔도 **보드 크기가 변하지 않게** 하려고 파라미터를 보드
        위가 아니라 여기에 세운다(Hole Detection 과 같은 구조다). 방식마다
        필요한 줄 수가 크게 다르므로 — 플랜지 고정은 0줄, 매칭점은 5줄 — 보드
        위에 두면 방식을 바꿀 때마다 사진 크기가 달라졌다.
        """
        panel = SettingPanel()

        group = SettingGroup("Parameter Setting")
        # 고른 방식의 설명을 먼저 적는다. 콤보 툴팁만으로는 세 방식의 차이를
        # 비교할 수 없고, 이 선택이 로봇으로 나가는 자세를 통째로 가른다.
        self.mode_note = group.note("")
        self.mode_note.setObjectName("sectionLabel")
        # 방식별 줄은 상자에 담아 통째로 보이고 감춘다.
        self._ring_box = group.add_widget(self._build_ring_box())
        self._normal_box = group.add_widget(self._build_normal_box())
        panel.add_group(group)

        panel.add_group(self._build_offset_group())
        return panel

    def _build_ring_box(self):
        """홀 주변 영역 상자: 영역 반지름 [15.0 mm].

        mm 로 받는 이유는 그것이 부품 상의 크기라서다 — 카메라가 멀어져도 같은
        자리를 보게 된다. 화면의 노란 고리가 실제 계산 범위이므로, 값을 바꾸면
        고리가 곧바로 따라 움직인다.
        """
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SETTING_ROW_SPACING)

        label = QLabel("영역 반지름")
        label.setObjectName("paramLabel")
        label.setToolTip(NORMAL_MODE_HINTS[NORMAL_MODE_RING])

        self.ring_radius = QDoubleSpinBox()
        self.ring_radius.setObjectName("paramSpin")
        self.ring_radius.setRange(RING_RADIUS_MM_MIN, RING_RADIUS_MM_MAX)
        self.ring_radius.setDecimals(1)
        self.ring_radius.setSingleStep(1.0)
        self.ring_radius.setSuffix(" mm")
        self.ring_radius.setValue(RING_RADIUS_MM_DEFAULT)
        self.ring_radius.setAlignment(Qt.AlignCenter)
        self.ring_radius.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.ring_radius.setToolTip(NORMAL_MODE_HINTS[NORMAL_MODE_RING])
        # 홀 검출은 그대로 두고 포즈만 다시 그린다(수십 ms).
        self.ring_radius.valueChanged.connect(self._on_ring_radius_changed)

        hint = QLabel("화면의 노란 고리가 실제 계산 범위")
        hint.setObjectName("sectionLabel")
        hint.setWordWrap(True)

        radius_row = QHBoxLayout()
        radius_row.setContentsMargins(0, 0, 0, 0)
        radius_row.setSpacing(SETTING_ROW_SPACING)
        radius_row.addWidget(label, 1)
        radius_row.addWidget(self.ring_radius)
        column.addLayout(radius_row)
        column.addWidget(hint)
        return box

    def current_normal_source(self):
        """지금 고른 자세 출처. → {"mode", "ring_radius_mm"}"""
        return {
            "mode": self.mode_combo.currentData(),
            "ring_radius_mm": self.ring_radius.value(),
        }

    def _on_mode_changed(self):
        self._update_mode_rows()
        if self._syncing:
            return
        # 매칭점 방식을 껐다 켜면 추적 결과를 다시 받아야 한다.
        self._normal_cache = None
        self._refresh_poses()

    def _on_ring_radius_changed(self):
        if self._syncing:
            return
        self._refresh_poses()

    def _update_mode_rows(self):
        """고른 방식에 해당하는 상자만 보이게 한다.

        둘 다 오른쪽 패널 안에 있고 패널 폭이 고정이라, 방식을 바꿔도 이미지
        보드 크기는 변하지 않는다.
        """
        mode = self.mode_combo.currentData()
        self._ring_box.setVisible(mode == NORMAL_MODE_RING)
        self._normal_box.setVisible(mode == NORMAL_MODE_POINTS)
        # 좁은 패널이라 첫 두 줄까지 적는다(나머지는 툴팁에).
        hint = NORMAL_MODE_HINTS.get(mode, "")
        self.mode_note.setText(" ".join(hint.split("\n")[:2]).strip())
        self.mode_note.setToolTip(hint)

    def _build_normal_box(self):
        """법선 매칭점 상자: [스위치] [1] [2] [3] [저장] 기준 [▼] SAM2 모델 [▼].

        자세(법선)를 세울 자리 세 곳을 SAM2 로 따 둔다. **방향만** 여기서 받고,
        홀 위치와 높이(z)는 건드리지 않는다 — 세 점 밖의 홀도 좌표가 그대로 나온다.

        칸이 셋으로 고정인 이유는 NORMAL_POINTS_HINT 에 적어 두었다. 세 칸은
        따로따로 다시 지정할 수 있다 — 한 곳만 잘못 잡았을 때 처음부터 세 번
        드래그하게 만들지 않으려는 것이다.

        SAM2 모델 콤보를 여기에 따로 두는 이유: 매칭점은 홀 검출 방식과 무관하게
        늘 SAM2 를 쓴다(블랍으로 홀을 찾더라도 법선의 출처는 똑같이 정해야 한다).
        Hole Detection 의 모델 콤보를 빌려 오면, 검출을 블랍으로 바꾼 순간 그 줄이
        숨어서 여기서 어느 모델을 쓰는지 볼 수 없게 된다.
        """
        box = QWidget()
        column = QVBoxLayout(box)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(SETTING_ROW_SPACING)

        label = QLabel("법선 매칭점")
        label.setObjectName("paramLabel")
        label.setToolTip(NORMAL_POINTS_HINT)

        self.normal_switch = ToggleSwitch()
        self.normal_switch.setChecked(False)
        self.normal_switch.toggled.connect(self._on_normal_switch_toggled)

        # 지정 대기 중인 네모 세 개. 한 화면에서 셋을 다 잡아야 하므로
        # 새로 캡쳐하면 비운다(_reset_normal_boxes).
        self._normal_boxes = [None] * NORMAL_POINTS
        # 티칭할 때 SAM2 가 딴 마스크. 저장 전에도 화면에 보여 주려고 들고 있는다.
        # 다시 따려면 촬영당 1.2초가 또 드는데, 티칭 때 이미 계산한 값이다.
        self._normal_taught_masks = [None] * NORMAL_POINTS
        self._normal_slot = None       # 지금 드래그를 기다리는 칸 (0-based)
        self.btn_normal_slots = []

        for index in range(NORMAL_POINTS):
            button = QPushButton(str(index + 1))
            button.setObjectName("roiToggle")
            button.setCheckable(True)
            button.setEnabled(False)   # 이미지가 있어야 드래그한다
            button.setToolTip(
                f"{index + 1}번 매칭점을 네모로 드래그한다. "
                "이미 잡혀 있으면 그 자리를 덮어쓴다."
            )
            button.toggled.connect(
                lambda checked, slot=index: self._on_normal_slot_toggled(slot, checked)
            )
            self.btn_normal_slots.append(button)

        self.btn_normal_save = QPushButton("저장")
        self.btn_normal_save.setObjectName("roiToggle")
        self.btn_normal_save.setEnabled(False)
        self.btn_normal_save.setToolTip(
            "세 칸이 다 차면 이름을 붙여 저장한다. 오퍼레이터는 저장된 것을 쓴다."
        )
        self.btn_normal_save.clicked.connect(self._on_normal_save_clicked)

        reference_label = QLabel("기준")
        reference_label.setObjectName("paramLabel")

        self.normal_reference_combo = QComboBox()
        self.normal_reference_combo.setObjectName("profileCombo")
        self.normal_reference_combo.setToolTip(
            "저장해 둔 매칭점 세 곳. 오퍼레이터도 이것으로 같은 자리를 찾는다."
        )
        self.normal_reference_combo.currentIndexChanged.connect(
            self._on_normal_reference_changed
        )

        model_label = QLabel("SAM2 모델")
        model_label.setObjectName("paramLabel")

        self.sam2_model_combo = QComboBox()
        self.sam2_model_combo.setObjectName("sam2ModelCombo")
        for name in sam2_detector.list_models():
            self.sam2_model_combo.addItem(sam2_detector.model_label(name), name)
        # 어느 장치에서 도는지 여기에 띄운다. 휠에 이 GPU 커널이 없으면 CPU 로
        # 물러서는데, 그때 무엇을 깔아야 하는지가 이 문구에 그대로 나온다.
        self.sam2_model_combo.setToolTip(
            "매칭점을 따고 추적할 때 쓰는 모델. 클수록 정확하지만 느리다.\n"
            f"지금 연산 장치: {sam2_detector.device_note()}"
        )
        self.sam2_model_combo.currentIndexChanged.connect(
            self._on_normal_reference_changed
        )

        # 스위치 줄 → 칸 세 개 + 저장 → 기준 → 모델 순으로 위에서 아래로 쌓는다.
        switch_row = QHBoxLayout()
        switch_row.setContentsMargins(0, 0, 0, 0)
        switch_row.setSpacing(SETTING_ROW_SPACING)
        switch_row.addWidget(label, 1)
        switch_row.addWidget(self.normal_switch)
        column.addLayout(switch_row)

        slot_row = QHBoxLayout()
        slot_row.setContentsMargins(0, 0, 0, 0)
        slot_row.setSpacing(SETTING_ROW_SPACING)
        for button in self.btn_normal_slots:
            slot_row.addWidget(button)
        slot_row.addWidget(self.btn_normal_save)
        column.addLayout(slot_row)

        column.addWidget(reference_label)
        column.addWidget(self.normal_reference_combo)
        column.addWidget(model_label)
        column.addWidget(self.sam2_model_combo)
        return box

    def _build_status_row(self):
        """이미지 보드 위 우측: 상태등 + 결과 문구."""
        row = QHBoxLayout()
        row.setSpacing(CONTENT_SPACING)

        self.status_led = QLabel()
        self.status_led.setObjectName("statusLed")

        # 상태문구는 길어진다(홀 수 · 포즈 · 법선 매칭점 · 벌어짐 …). 보통 QLabel
        # 이면 그 길이가 페이지 최소 폭이 되어 창이 모니터를 넘는다.
        self.status_text = ElidedLabel("이미지를 캡쳐하세요.")
        self.status_text.setObjectName("cameraStatus")

        row.addStretch()
        row.addWidget(self.status_led)
        row.addWidget(self.status_text)
        return row

    def _build_elapsed_row(self):
        """이미지 보드 바로 위 우측에 처리 시간을 붙이는 행."""
        row = QHBoxLayout()
        row.setSpacing(0)

        # 단계별 소요([법선 · 포즈])가 붙어 길어진다. 상태문구와 같은 이유로
        # 폭을 요구하지 않는 라벨을 쓴다.
        self.elapsed_label = ElidedLabel("", elide=Qt.ElideRight)
        self.elapsed_label.setObjectName("elapsedLabel")
        self.elapsed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        row.addStretch()
        row.addWidget(self.elapsed_label)
        return row

    def _build_offset_group(self):
        """우측 패널 아래 그룹: 진입 각도 오프셋 (Rotational Offset).

        260910 에 '포즈 조정'이 사라져 이제 회전 오프셋은 이것 하나다. 둘을
        나란히 두었던 동안에는 어느 쪽이 자세를 얼마나 돌렸는지 가릴 수 없었고,
        화면이 도는 쪽(포즈 조정)과 안 도는 쪽(이것)이 섞여 오해를 불렀다.
        """
        group = SettingGroup("Rotational Offset")
        group.caption.setToolTip(APPROACH_OFFSET_HINT)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.offset_inputs = {}
        for line, (key, label, low, high, decimals, suffix) in enumerate(
            APPROACH_OFFSET_FIELDS
        ):
            name = QLabel(label)
            name.setObjectName("poseLabel")

            box = QDoubleSpinBox()
            box.setObjectName("poseInput")
            box.setRange(low, high)
            box.setDecimals(decimals)
            box.setSuffix(f" {suffix}")
            box.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            box.setButtonSymbols(QDoubleSpinBox.NoButtons)
            box.setToolTip(APPROACH_OFFSET_HINT)

            grid.addWidget(name, line, 0)
            grid.addWidget(box, line, 1)
            self.offset_inputs[key] = box

        group.add_layout(grid)

        # 이 값이 화면에 안 보인다는 것을 적어 둔다. 좌표축이 즉시 돌 것으로
        # 여겨 '왜 안 움직이지' 하고 헤매기 쉽다.
        note = group.note("화면 자세는 바뀌지 않는다 (로봇으로 넘길 때만 적용)")
        note.setToolTip(APPROACH_OFFSET_HINT)
        return group

    # ===== 현재 값 (오퍼레이터/저장이 읽는다) =====
    def current_normal_points(self):
        """현재 입력된 법선 매칭점 세팅."""
        return {
            "enabled": self.normal_switch.isChecked(),
            "reference": self.normal_reference_combo.currentData(),
            "model": self.sam2_model_combo.currentData(),
        }

    def current_approach_offset(self):
        """현재 입력된 진입 각도 오프셋 (rx, ry, rz) [도]."""
        return tuple(
            self.offset_inputs[key].value() for key, *_ in APPROACH_OFFSET_FIELDS
        )

    def hole_poses(self):
        """마지막 분석의 홀 포즈 목록 (카메라 좌표계, 위치 mm / 각도 도).

        다음 공정(로봇 이송 등)이 쓰는 공개 API다. 홀 개수만큼 돌려주되 포즈를
        구하지 못한 자리에는 None이 들어가므로, hole_positions()와 인덱스로
        짝지어 쓸 수 있다.

        규약은 hand_eye 모듈과 같다. 손-눈 결과가 있으면
        hole_pose.pose_matrix(pose)로 4x4 행렬을 만들어 그대로 곱하면 된다.
        """
        return list(self._last_poses)

    def hole_positions(self):
        """마지막 분석의 홀 픽셀 좌표 목록 [(x, y, radius), ...]."""
        return list(self._last_holes)

    # ===== 세팅 (Hole Detection 과 한 프로파일을 나눠 쓴다) =====
    def collect_pose_settings(self):
        """저장할 포즈 키만 모은다. Hole Detection 이 자기 키와 합쳐 저장한다."""
        return {
            # 자세를 어디서 받을지. 오퍼레이터는 촬영 보드마다 다른 프로파일을
            # 읽으므로, 보드마다 법선을 뽑는 방식이 달라도 된다.
            "normal_source": self.current_normal_source(),
            "normal_points": self.current_normal_points(),
            # 티칭한 진입 각도 오프셋 (공구 기준, 도)
            "approach_offset": {
                key: self.offset_inputs[key].value()
                for key, *_ in APPROACH_OFFSET_FIELDS
            },
        }

    def apply_pose_settings(self, data):
        """읽어 들인 세팅에서 포즈 키만 위젯에 얹는다.

        저장 파일을 손으로 고쳤을 수도 있으므로 값 하나하나를 방어적으로 읽는다.
        """
        self._syncing = True
        try:
            # 자세 출처 복원. 이 키가 없는 옛 프로파일은 매칭점 방식으로 읽힌다
            # (그때 쓰던 방식이 그것이다).
            source = normal_source_from_settings(data)
            index = self.mode_combo.findData(source["mode"])
            if index >= 0:
                self.mode_combo.setCurrentIndex(index)
            self.ring_radius.setValue(source["ring_radius_mm"])

            # 법선 매칭점 복원. 저장된 이름이 이 PC에 없으면 콤보가 -1 로
            # 남고, normal_points_masks() 가 그때 사유를 문구로 알려 준다.
            normal = normal_points_from_settings(data)
            self.normal_switch.setChecked(normal["enabled"])
            self.normal_reference_combo.setCurrentIndex(
                self.normal_reference_combo.findData(normal["reference"])
            )
            # 저장된 모델이 이 PC에 없으면(체크포인트를 안 받았거나 다른 PC에서
            # 만든 세팅이면) 건드리지 않고 기본 모델을 그대로 둔다.
            model_index = self.sam2_model_combo.findData(normal["model"])
            if model_index >= 0:
                self.sam2_model_combo.setCurrentIndex(model_index)

            # 진입 각도 오프셋 복원 (없으면 0 = 보정 없음)
            offset = data.get("approach_offset") if isinstance(data, dict) else None
            for key, _label, low, high, *_rest in APPROACH_OFFSET_FIELDS:
                value = 0.0
                if isinstance(offset, dict):
                    value = as_float(offset.get(key), 0.0, low, high)
                self.offset_inputs[key].setValue(value)

            self._reset_normal_boxes()
        finally:
            self._syncing = False

        self._normal_cache = None
        self._update_normal_controls()
        self._update_mode_rows()
        self._sync_profile_selection()
        if self._last_holes:
            self._refresh_poses()

    def reset_pose_settings(self):
        """포즈 값을 초기값으로 되돌린다(깨진 파일을 읽었을 때)."""
        self._syncing = True
        try:
            for box in self.offset_inputs.values():
                box.setValue(0.0)
            self.normal_switch.setChecked(False)
            self.normal_reference_combo.setCurrentIndex(-1)
            if self.sam2_model_combo.count():
                self.sam2_model_combo.setCurrentIndex(0)
            index = self.mode_combo.findData(NORMAL_MODE_POINTS)
            if index >= 0:
                self.mode_combo.setCurrentIndex(index)
            self.ring_radius.setValue(RING_RADIUS_MM_DEFAULT)
            self._reset_normal_boxes()
        finally:
            self._syncing = False
        self._normal_cache = None
        self._update_normal_controls()
        self._update_mode_rows()

    def _reload_profile_list(self, select=None):
        """Hole Detection 이 가진 프로파일 목록을 그대로 비춘다."""
        if self._hole_page is None:
            return
        self._syncing = True
        try:
            keep = select or self._hole_page.current_profile()
            self.profile_combo.clear()
            for name in self._hole_page.profile_names():
                self.profile_combo.addItem(name, name)
            index = self.profile_combo.findData(keep)
            if index >= 0:
                self.profile_combo.setCurrentIndex(index)
        finally:
            self._syncing = False
        self.btn_save.setEnabled(self.profile_combo.count() > 0)

    def _sync_profile_selection(self):
        """Hole Detection 에서 프로파일이 바뀐 것을 콤보에 반영한다."""
        if self._hole_page is None:
            return
        current = self._hole_page.current_profile()
        if self.profile_combo.count() != len(self._hole_page.profile_names()):
            self._reload_profile_list(select=current)
            return
        index = self.profile_combo.findData(current)
        if index >= 0 and index != self.profile_combo.currentIndex():
            self._syncing = True
            try:
                self.profile_combo.setCurrentIndex(index)
            finally:
                self._syncing = False

    def _on_profile_changed(self):
        if self._syncing or self._hole_page is None:
            return
        name = self.profile_combo.currentData()
        if name is not None:
            # 프로파일 주인은 Hole Detection 이다. 저쪽이 파일을 읽고 나서
            # apply_pose_settings() 로 이쪽 값을 다시 채워 준다.
            self._hole_page.select_profile(name)

    def _on_save_clicked(self):
        if self._hole_page is None:
            self._set_status("Hole Detection 페이지를 찾을 수 없습니다.", "error")
            return
        ok, message = self._hole_page.save_current_profile()
        self._set_status(message, "success" if ok else "error")

    # ===== 이미지 받기 =====
    def _on_capture_clicked(self):
        """여기서 직접 한 장 찍고, 홀 검출만 Hole Detection 에 맡긴다."""
        if self._hole_page is None:
            self._set_status("Hole Detection 페이지를 찾을 수 없습니다.", "error")
            return
        camera_page = self._hole_page.camera_page()
        if camera_page is None:
            self._set_status("Camera Capture 페이지를 찾을 수 없습니다.", "error")
            return

        capture_started = time.perf_counter()
        try:
            rgb, depth = camera_page.grab_frames()
        except Exception as exc:
            self._set_status(f"캡쳐 실패: {exc}", "error")
            self.elapsed_label.setText("")
            return
        capture_seconds = time.perf_counter() - capture_started

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        try:
            holes, debug_info = self._hole_page.detect_holes(bgr)
        except Exception as exc:
            self._set_status(f"홀 검출 실패: {exc}", "error")
            self.elapsed_label.setText("")
            return

        self._ingest(
            bgr,
            depth,
            camera_page.color_intrinsics(),
            camera_page.depth_scale_mm(),
            holes,
            debug_info.get("hole_masks") or [],
        )
        self.analyze(capture_seconds)

    def _pull_from_hole_page(self, quiet):
        """Hole Detection 의 마지막 캡쳐와 검출 결과를 그대로 이어받는다."""
        context = None if self._hole_page is None else self._hole_page.capture_context()
        if context is None:
            if not quiet:
                self._set_status(
                    "Hole Detection 에서 먼저 이미지를 캡쳐하세요.", "error"
                )
            return
        self._ingest(
            context["image"],
            context["depth"],
            context["intrinsics"],
            context["depth_scale_mm"],
            context["holes"],
            context["hole_masks"],
            sequence=context["sequence"],
        )
        self.analyze()

    def _ingest(self, bgr, depth, intrinsics, depth_scale_mm, holes, hole_masks,
                sequence=None):
        """새 장면을 받아 상태를 갈아 끼운다.

        지정 대기 중이던 네모는 버린다. 세 점은 **한 화면에서** 잡아야 하고,
        캡쳐에 걸쳐 잡으면 서로 다른 프레임의 좌표가 섞여 비디오 추적이 엉뚱한
        자리를 기준으로 삼는다.

        sequence 는 이 이미지가 Hole Detection 의 몇 번째 촬영인지다. 이 페이지가
        직접 찍었을 때는 None 이 오는데, 그때는 **저쪽의 지금 번호**를 적어 둔다 —
        내 촬영이 저쪽 최신보다 새것이므로, 저쪽이 또 찍기 전까지 유지되어야 한다.
        """
        if sequence is None and self._hole_page is not None:
            sequence = self._hole_page.capture_sequence()
        self._ingested_seq = sequence
        self._last_captured_image = bgr
        self._last_depth = depth
        self._last_intrinsics = intrinsics
        self._depth_scale_mm = depth_scale_mm
        self._last_holes = list(holes)
        self._last_hole_masks = list(hole_masks)
        self._normal_cache = None   # 새 장면이므로 매칭점을 다시 찾아야 한다
        self._reset_normal_boxes()
        self._update_normal_controls()
        # 확대/이동을 되돌린다. 장면이 바뀌면 확대해 둔 자리가 무엇을 가리키는지
        # 알 수 없다(zoom_pan.reset_view 설명 참고).
        self.result_board.reset_view()

    # ===== 분석 =====
    def _matching_points_active(self):
        """매칭점을 지금 찾아야 하는가.

        방식이 '매칭점 3곳'일 때만이다. 다른 방식에서 찾으면 촬영당 2초를 쓰고
        결과는 아무도 안 본다.
        """
        return (
            self.mode_combo.currentData() == NORMAL_MODE_POINTS
            and self.normal_switch.isChecked()
        )

    def analyze(self, capture_seconds=None):
        """받아 둔 홀에 포즈를 세우고 보드에 그린다."""
        if self._last_captured_image is None:
            self._set_status("먼저 이미지를 캡쳐하세요.", "error")
            return

        analysis_started = time.perf_counter()
        # 지난 촬영의 벌어짐이 남아 문구가 거짓말하지 않게 매번 비운다.
        # 뎁스가 없으면 _normal_plane 이 호출되지 않아 다시 채워지지 않는다.
        self._normal_spread_mm = None
        self._stage_seconds = {}
        try:
            # 매칭점을 **포즈보다 먼저** 찾는다. 이유가 둘이다.
            #   1. _estimate_poses 안에서만 찾으면 뎁스나 내부 파라미터가 없을 때
            #      아예 호출되지 않아, 추적이 됐는지 화면으로 확인할 수가 없다.
            #   2. 여기서 먼저 해 두면 포즈 단계는 캐시를 쓰므로, 아래 시간이
            #      '추적'과 '포즈 계산'으로 깨끗하게 갈린다.
            # (티칭 중이면 화면이 티칭 결과를 쓰므로 이것도 건너뛴다.)
            stage_started = time.perf_counter()
            if self._matching_points_active() and not any(self._normal_boxes):
                self._normal_masks_for(self._last_captured_image)
            self._stage_seconds["법선"] = time.perf_counter() - stage_started

            stage_started = time.perf_counter()
            poses = self._estimate_poses(self._last_holes)
            self._stage_seconds["포즈"] = time.perf_counter() - stage_started
        except Exception as exc:
            self._set_status(f"분석 실패: {exc}", "error")
            self.elapsed_label.setText("")
            return

        self._last_poses = poses
        self._set_elapsed(capture_seconds, time.perf_counter() - analysis_started)

        if not self._render_current_view():
            return   # 표시 실패 사유가 이미 상태문구에 있으므로 덮어쓰지 않는다
        self._set_result_status()

    def _pose_inputs_ready(self):
        """포즈를 계산할 재료(뎁스 + 내부 파라미터)가 갖춰졌는지.

        가상 카메라는 depth_raw가 있어도 카메라 내부 파라미터가 없으므로
        3D 포즈는 계산하지 않는다. 둘 중 하나만 없어도 포즈는 나오지 않는다.
        """
        return self._last_depth is not None and self._last_intrinsics is not None

    def _estimate_poses(self, holes):
        """홀마다 (x, y, z, rx, ry, rz)를 구한다. 재료가 없으면 전부 None.

        법선 매칭점 세 곳을 지정해 두면 **법선만** 그 세 점의 평면에서 받는다.
        홀은 원래 자리에서 그대로 찾고 높이(z)도 홀 주변 고리에서 잡는다 —
        매칭점은 표본을 거르는 것이 아니라 방향의 출처를 바꾸는 것이다.
        그래서 세 점에서 멀리 떨어진 홀도 좌표가 그대로 나온다.

        매칭점이 없거나 못 찾았으면 법선도 홀 주변 고리에서 잡는다. 정밀도는
        떨어지지만(260901 실측 흔들림 ±8도) 좌표가 아예 안 나오는 것보다 낫고,
        상태문구에 그 사실이 남는다.
        """
        if not self._pose_inputs_ready():
            return [None] * len(holes)

        camera_matrix, dist_coeffs = self._last_intrinsics
        height, width = self._last_captured_image.shape[:2]
        source = self.current_normal_source()
        mode = source["mode"]
        # 매칭점 평면은 그 방식일 때만 세운다. 다른 방식에서 세우면 촬영당 2초를
        # 쓰고 결과는 버리게 된다.
        reference = (
            self._normal_plane(camera_matrix, dist_coeffs, (width, height))
            if mode == NORMAL_MODE_POINTS
            else None
        )
        poses = estimate_hole_poses(
            holes,
            self._last_depth,
            camera_matrix,
            dist_coeffs,
            color_size=(width, height),
            depth_scale_mm=self._depth_scale_mm,
            surface_plane=reference,
            mode=mode,
            ring_radius_mm=source["ring_radius_mm"],
        )
        # 이전 촬영과 섞지 않는다(stabilize_hole_poses 를 부르지 않는다).
        # 260908 사용자 결정: 세 방식 모두 '이번 촬영만' 쓴다. EMA 는 흔들림을
        # 줄이지만 한 촬영에 3도만 움직여서, 부품이 실제로 움직였을 때 여러 장을
        # 찍어야 따라오고 '이전 캡쳐 기준으로 도는' 것처럼 보였다.
        self._last_raw_poses = poses
        return poses

    def _refresh_poses(self):
        """이미 받아 둔 홀들의 포즈를 다시 계산하고 시각화한다."""
        if not self._last_holes:
            return
        try:
            self._last_poses = self._estimate_poses(self._last_holes)
        except Exception as exc:
            self._set_status(f"포즈 계산 실패: {exc}", "error")
            return
        if not self._render_current_view():
            return   # 표시 실패 사유가 이미 상태문구에 있으므로 덮어쓰지 않는다
        self._set_result_status()

    # ===== 법선 매칭점 (세 곳) =====
    def _normal_masks_for(self, bgr_image):
        """이 촬영의 매칭점 마스크 세 개. 못 찾은 자리는 None.

        촬영 한 장 안에서 여러 번(법선 평면 / 화면 그리기) 필요하므로 결과를
        붙잡아 둔다. 비디오 추적을 두 번 돌리면 2초 더 쓸 뿐 답은 같다.
        """
        if self._normal_cache is None:
            self._normal_cache = normal_points_masks(
                self.current_normal_points(), bgr_image
            )
        return self._normal_cache   # (masks, note, boxes)

    def _normal_plane(self, camera_matrix, dist_coeffs, color_size):
        """저장해 둔 매칭점 세 곳에서 세운 평면. 못 세우면 None.

        홀 주변 고리는 구멍이 주변보다 깊어 모따기와 단차가 표본에 섞이고,
        폭이 좁아 법선의 지렛대도 짧다(260901 실측 ±8도). 그래서 사람이 짚어 둔
        세 자리에서 **방향만** 받아 온다. 위치와 높이는 홀 자리 그대로다.
        """
        masks, _note, _boxes = self._normal_masks_for(self._last_captured_image)
        plane, spread = normal_plane_from_masks(
            masks,
            self._last_depth,
            camera_matrix,
            dist_coeffs,
            color_size=color_size,
            depth_scale_mm=self._depth_scale_mm,
        )
        # 문구에 실어 보내려고 붙잡아 둔다. 막지 않는 대신 알려 주는 값이다.
        self._normal_spread_mm = spread
        return plane

    def _normal_plane_from(self, boxes, camera_matrix, dist_coeffs, color_size):
        """아직 저장하지 않은 네모 세 개로 평면을 세워 본다. 안 되면 None.

        저장 전 확인과 화면 미리보기가 같은 계산을 쓰게 하려고 따로 두었다.
        추적을 거치지 않고 티칭 프레임에서 곧장 마스크를 딴다 — 지금 보고 있는
        화면이 곧 티칭 프레임이라 추적할 것이 없다.
        """
        if any(box is None for box in boxes) or self._last_depth is None:
            return None
        model = self.sam2_model_combo.currentData()
        masks = []
        for box in boxes:
            try:
                mask = sam2_detector.segmenter(model).segment_box(
                    self._last_captured_image, box
                )
            except Exception:
                return None
            resized = mask_to_depth_size(mask, self._last_depth.shape)
            if resized is None:
                return None
            masks.append(resized)
        return plane_from_normal_points(
            masks,
            self._last_depth,
            camera_matrix,
            dist_coeffs,
            color_size=color_size,
            depth_scale_mm=self._depth_scale_mm,
        )

    def _on_normal_switch_toggled(self, _checked):
        self._update_normal_controls()
        if self._syncing:
            return
        self._normal_cache = None
        self._refresh_poses()

    def _on_normal_reference_changed(self):
        if self._syncing:
            return
        self._normal_cache = None
        self._refresh_poses()

    def _update_normal_controls(self):
        """세 칸과 저장 버튼의 상태를 지금 형편에 맞춘다."""
        ready = sam2_detector.available() and self._last_captured_image is not None
        for index, button in enumerate(self.btn_normal_slots):
            button.setEnabled(ready)
            # 잡힌 칸은 눈에 보이게 표시한다. 세 칸 중 어디가 비었는지
            # 한눈에 알아야 저장 버튼이 왜 안 켜지는지 납득이 된다.
            taught = self._normal_boxes[index] is not None
            button.setText(f"{index + 1} ✓" if taught else str(index + 1))
        filled = sum(box is not None for box in self._normal_boxes)
        self.btn_normal_save.setEnabled(ready and filled == NORMAL_POINTS)
        self.normal_reference_combo.setEnabled(self.normal_switch.isChecked())

    def _reset_normal_boxes(self):
        """지정 대기 중인 네모를 비운다. 새로 캡쳐할 때 부른다."""
        self._normal_boxes = [None] * NORMAL_POINTS
        self._normal_taught_masks = [None] * NORMAL_POINTS
        self._normal_slot = None
        for button in self.btn_normal_slots:
            if button.isChecked():
                button.setChecked(False)

    def _on_normal_slot_toggled(self, slot, checked):
        if checked:
            # 세 칸이 같은 보드의 같은 드래그를 쓴다. 둘이 함께 켜져 있으면
            # 드래그가 어디로 갈지 알 수 없으므로 나머지를 내린다.
            for index, button in enumerate(self.btn_normal_slots):
                if index != slot and button.isChecked():
                    button.setChecked(False)
            # 네모 드래그를 받는다. 점 프롬프트는 SAM2가 면인지 부품 전체인지
            # 스스로 골라야 해서 뜻과 다른 것이 잡히곤 한다.
            self._normal_slot = slot
            self.result_board.set_selection_mode("region")
            self._set_status(
                f"{slot + 1}번 매칭점이 될 자리를 네모로 드래그하세요 "
                "(평평하고 특징이 뚜렷한 곳)."
            )
        elif self._normal_slot == slot:
            self._normal_slot = None
            self.result_board.set_selection_mode(None)

    def _on_region_dragged(self, rect):
        """보드에서 네모를 그렸다. 지금 기다리는 칸에 넣는다."""
        slot = self._normal_slot
        if slot is None or self._last_captured_image is None:
            return
        self.btn_normal_slots[slot].setChecked(False)
        left, top, width, height = rect
        self._teach_normal_point(slot, (left, top, left + width, top + height))

    def _teach_normal_point(self, slot, box):
        """드래그한 네모를 slot 번 매칭점으로 잡는다.

        SAM2 로 한 번 따 보는 이유는 두 가지다. 사람이 넉넉히 그린 네모를 실제
        경계에 맞게 좁혀 주고(그 여백이 넓으면 다음 촬영에서 옆 물체까지 프롬프트에
        들어온다), 그 자리를 애초에 SAM2 가 잡을 수 있는지 지금 알려 준다.
        """
        self._set_status(f"SAM2로 {slot + 1}번 매칭점을 따는 중입니다...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            mask, taught = sam2_detector.teach_region(
                self._last_captured_image, box, self.sam2_model_combo.currentData()
            )
        except Exception as exc:
            self._set_status(f"{slot + 1}번 매칭점 지정 실패: {exc}", "error")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._normal_boxes[slot] = tuple(taught)
        self._normal_taught_masks[slot] = mask
        self._update_normal_controls()
        self._normal_cache = None
        filled = sum(item is not None for item in self._normal_boxes)
        remaining = NORMAL_POINTS - filled
        # 분석을 먼저 돌리고 나서 문구를 세운다. 순서를 바꾸면 분석 결과가
        # 티칭 확인 문구를 덮어써, 기다린 작업이 성공했는지 알 수 없다.
        self.analyze()
        tail = (
            f"{remaining}곳 더 지정하면 저장할 수 있습니다."
            if remaining
            else "'저장'을 눌러 이름을 붙이세요."
        )
        self._set_status(
            f"{slot + 1}번 매칭점을 잡았습니다 ({int(mask.sum()):,} px) — {tail}",
            "success",
        )

    def _on_normal_save_clicked(self):
        """잡아 둔 세 곳을 이름 붙여 저장한다."""
        if self._last_captured_image is None:
            return
        if any(box is None for box in self._normal_boxes):
            self._set_status("세 곳을 모두 지정해야 저장할 수 있습니다.", "error")
            return

        # 저장 전에 실제로 평면이 세워지는지 확인한다. 일직선이나 너무 붙은
        # 배치는 그럴듯한 숫자를 내므로, 저장한 뒤 오퍼레이터에서 처음 알게
        # 되면 원인을 짚기 어렵다.
        if self._pose_inputs_ready():
            camera_matrix, dist_coeffs = self._last_intrinsics
            height, width = self._last_captured_image.shape[:2]
            plane = self._normal_plane_from(
                self._normal_boxes, camera_matrix, dist_coeffs, (width, height)
            )
            if plane is None:
                self._set_status(
                    "이 세 자리로는 평면을 세울 수 없습니다 — 완전히 일직선이거나 "
                    "그 자리의 뎁스가 비어 있습니다.",
                    "error",
                )
                return

        # 재검출이 쓸 기준 넓이를 **재검출과 같은 방법으로** 재 둔다.
        #
        # 티칭은 이미지 예측기로 하고 재검출은 비디오 추적으로 하는데 둘이
        # 내놓는 마스크 넓이가 다르다. 티칭 쪽 값을 기준으로 삼으면 문턱이
        # 엉뚱하게 잡힌다(실측: 같은 자리가 3613px 대 174px).
        #
        # 여기서 한 번 돌려 두면 '티칭은 됐는데 추적은 못 하는' 자리를 지금
        # 잡아낸다. 그대로 저장하면 촬영마다 조용히 실패하고, 화면에는 자홍
        # 네모만 떠서 원인을 짚기 어렵다.
        self._set_status("매칭점 기준 넓이를 재는 중입니다...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            areas = measure_normal_point_areas(
                self._last_captured_image,
                self._normal_boxes,
                self.sam2_model_combo.currentData(),
            )
        except Exception as exc:
            self._set_status(f"기준 넓이를 재지 못했습니다: {exc}", "error")
            return
        finally:
            QApplication.restoreOverrideCursor()

        untrackable = [str(i + 1) for i, area in enumerate(areas) if area <= 0]
        if untrackable:
            self._set_status(
                f"{', '.join(untrackable)}번 매칭점은 재검출이 다시 찾지 못합니다 "
                "— 더 넓고 특징이 뚜렷한 자리로 다시 지정하세요.",
                "error",
            )
            return

        name = TextInputDialog.get_text(
            self,
            "법선 매칭점 저장",
            "지정한 세 자리의 이름을 입력하세요.",
            ok_text="저장",
            cancel_text="취소",
            validator=validate_template_name,
        )
        if name is None:
            return
        ok, result = save_normal_points(
            name, self._last_captured_image, self._normal_boxes, areas
        )
        if not ok:
            self._set_status(f"법선 매칭점 저장 실패: {result}", "error")
            return

        self._reload_normal_list(select=name)
        self._syncing = True
        try:
            self.normal_switch.setChecked(True)
        finally:
            self._syncing = False
        self._update_normal_controls()
        self._normal_cache = None
        # 분석을 먼저 돌리고 문구를 세운다(위와 같은 이유).
        self.analyze()
        self._set_status(f"'{name}' 법선 매칭점 세 곳을 저장했습니다.", "success")

    def _reload_normal_list(self, select=None):
        """저장된 매칭점 목록을 콤보에 다시 채운다."""
        self._syncing = True
        try:
            keep = (
                select
                if select is not None
                else self.normal_reference_combo.currentData()
            )
            self.normal_reference_combo.clear()
            for name in list_normal_points():
                self.normal_reference_combo.addItem(name, name)
            index = self.normal_reference_combo.findData(keep)
            if index >= 0:
                self.normal_reference_combo.setCurrentIndex(index)
        finally:
            self._syncing = False

    def _normal_overlay(self):
        """화면에 그릴 매칭점. → (마스크 목록, 티칭 네모 목록)

        지금 티칭 중이면 **그 결과를 보여 준다.** 저장한 기준을 추적한 결과를
        보여 주면, 방금 드래그한 자리가 화면에 안 나타나 제대로 잡혔는지 알 수
        없다. 티칭은 지금 보고 있는 이 화면에서 하는 것이므로 추적할 것도 없다.

        티칭 중이 아니면 저장한 기준을 추적한 결과다. 못 찾은 자리는 마스크가
        None 으로 오고, draw_normal_points 가 티칭 네모로 표시한다.
        """
        if any(box is not None for box in self._normal_boxes):
            return list(self._normal_taught_masks), list(self._normal_boxes)
        if self._normal_cache is None:
            return [], []
        masks, _note, boxes = self._normal_cache
        return masks, boxes

    # ===== 화면 =====
    def _on_view_changed(self):
        if self._last_captured_image is None:
            return
        if self._render_current_view():
            # 직전 화면에서 띄운 경고가 남지 않도록 결과 문구로 되돌린다.
            self._set_result_status()

    def _build_view_image(self, view):
        """선택된 화면의 바탕 이미지(BGR)를 만든다. 만들 수 없으면 None."""
        if view == VIEW_COLOR:
            return self._last_captured_image.copy()
        if view == VIEW_DEPTH:
            bgr = colorize_depth(self._last_depth)
            if bgr is None:
                return None
            # 뎁스와 컬러 해상도가 다르면 홀 좌표가 어긋나므로 컬러 기준으로 맞춘다.
            height, width = self._last_captured_image.shape[:2]
            if bgr.shape[:2] != (height, width):
                bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_NEAREST)
            return bgr
        return None

    def _render_current_view(self):
        """고른 화면에 매칭점 + 홀 + 좌표축을 덧그려 보드에 표시한다.

        표시할 수 없으면 사유를 상태문구로 알리고 False를 반환한다(호출한 쪽이
        그 위에 다른 문구를 덮어쓰지 않도록).
        """
        view = self.view_combo.currentData()
        base = self._build_view_image(view)
        if base is None:
            self.result_board.clear_image()
            self._set_status(
                "뎁스 데이터가 없습니다 (가상 카메라 로그에는 depth_raw가 필요합니다)."
                if view == VIEW_DEPTH
                else "표시할 이미지가 없습니다.",
                "error",
            )
            return False

        camera_matrix, dist_coeffs = self._last_intrinsics or (None, None)
        mode = self.mode_combo.currentData()

        # 고른 방식이 실제로 보는 자리만 그린다. 안 쓰는 표시를 남겨 두면 무엇이
        # 자세를 정하고 있는지 화면에서 가릴 수 없다.
        overlaid = base
        if mode == NORMAL_MODE_POINTS:
            # 매칭점을 먼저 깔고 홀을 그 위에 그린다(옅은 덮개가 홀과 축을
            # 가리지 않게). 컬러 화면이 아닐 때도 그린다 — 뎁스맵에서 자리가
            # 옳게 잡혔는지 보는 것이 오히려 이 기능의 요점이다.
            normal_masks, normal_boxes = self._normal_overlay()
            overlaid = draw_normal_points(base, normal_masks, normal_boxes)
        elif mode == NORMAL_MODE_RING:
            overlaid = draw_ring_regions(
                base,
                self._last_holes,
                self._last_poses,
                camera_matrix,
                self.ring_radius.value(),
            )

        self.result_board.set_image_bgr(
            BlobResultVisualizer.draw_holes(
                overlaid,
                self._last_holes,
                self._last_poses,
                camera_matrix,
                dist_coeffs,
                hole_masks=self._last_hole_masks,
            )
        )
        return True

    # ===== 상태 표시 =====
    def _set_elapsed(self, capture_seconds, analysis_seconds):
        """걸린 시간을 띄운다. 단계별로 쪼개야 어디가 느린지 짚을 수 있다."""
        text = f"분석 {analysis_seconds * 1000:.1f} ms"
        # 0.1ms 미만 단계는 뺀다(꺼진 기능까지 0.0 으로 늘어놓으면 읽기 나쁘다).
        parts = [
            f"{name} {seconds * 1000:.0f}"
            for name, seconds in self._stage_seconds.items()
            if seconds >= 0.0001
        ]
        if parts:
            text += "  [" + " · ".join(parts) + "]"
        if capture_seconds is not None:
            total = capture_seconds + analysis_seconds
            text = (
                f"캡쳐 {capture_seconds * 1000:.1f} ms + {text}"
                f"  =  {total * 1000:.1f} ms"
            )
        self.elapsed_label.setText(text)

    def _set_result_status(self):
        """홀 수 · 포즈 수 · 법선 출처를 한 줄로 띄운다."""
        if self._last_captured_image is None:
            return
        height, width = self._last_captured_image.shape[:2]
        roi = roi_state.clamped_to(width, height)
        scope = f"ROI {roi[2]}×{roi[3]}" if roi else f"전체 {width}×{height}"
        count = len(self._last_holes)

        # 포즈는 뎁스가 비면 일부만 나올 수 있다. 몇 개가 나왔는지 함께 보여 줘야
        # '홀은 받았는데 좌표가 없다'를 바로 알아챌 수 있다.
        if not self._pose_inputs_ready():
            pose_note = " · 포즈 없음 (뎁스/내부 파라미터 없음)"
        else:
            posed = sum(1 for pose in self._last_poses if pose is not None)
            pose_note = f" · 포즈 {posed}/{count}"

        self._set_status(
            f"{scope} — 홀 {count}개{pose_note}{self._normal_note()}",
            "success" if count else "error",
        )

    def _normal_note(self):
        """법선을 어디서 받았는지, 얼마나 미더운지 알려 주는 꼬리말.

        매칭점을 켜 뒀는데 못 찾았다면 반드시 보여야 한다. 그때 법선은 홀 고리로
        조용히 물러서므로, 문구가 없으면 정밀도가 떨어진 것을 알아챌 수 없다.
        """
        mode = self.mode_combo.currentData()
        if mode != NORMAL_MODE_POINTS:
            # 다른 방식에서는 매칭점을 아예 안 본다. 대신 무엇으로 자세를
            # 세웠는지 한 마디 남긴다 — 화면만 보고는 가릴 수 없다.
            return (
                " · 자세: 플랜지 고정 (x·y·z만)"
                if mode == NORMAL_MODE_FLANGE
                else f" · 자세: 홀 주변 {self.ring_radius.value():.0f}mm 영역"
            )
        if not self.normal_switch.isChecked() or self._normal_cache is None:
            return ""
        _masks, note, _boxes = self._normal_cache
        parts = [part for part in (note, spread_note(self._normal_spread_mm)) if part]
        return "".join(f" · {part}" for part in parts)

    def _set_status(self, text, status_type="info"):
        self.status_text.setText(text)
        self.status_led.setProperty(
            "connected", "true" if status_type == "success" else "false"
        )
        self.status_led.style().unpolish(self.status_led)
        self.status_led.style().polish(self.status_led)
