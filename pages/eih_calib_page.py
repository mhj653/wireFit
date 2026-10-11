"""EIH(Eye-in-Hand) 캘리브레이션 페이지.

체커보드를 고정해 둔 채 로봇을 여러 자세로 옮겨 가며 찍고, 매 장면의 플랜지
포즈를 손으로 입력해 '카메라가 플랜지에 대해 어디에 붙어 있는지'를 구한다.

구성:

    [6축 로봇 ▼] [체커보드 9×6 ▼] [칸 25.0mm] [캡쳐]          [계산] [저장]
    ┌──────────────┬────────────────────────┬──────────────┐
    │ 수집 목록 표  │      이미지 보드        │ 플랜지 포즈   │
    │ # / 포즈      │  (코너 선·점 시각화)    │ x y z rx ry rz│
    │ 행 클릭 →     │                        │   [확인]      │
    │ 그때 이미지    │                        │ (4축: Z 기준) │
    └──────────────┴────────────────────────┴──────────────┘

동작 규칙
  - 4축 로봇을 고르면 포즈 입력은 X/Y/Z/RZ만 남고(RX = RY = 0), 카메라 Z
    오프셋을 정할 '보드 높이' 입력이 붙는다. 4축 움직임만으로는 그 값이
    원리상 정해지지 않기 때문이다(hand_eye._solve_ax_xb_4axis 참고).
  - 체커보드를 못 찾으면 포즈 입력이 잠긴다(쓸 수 없는 장면을 못 넣게).
  - '확인'을 눌러야 표에 한 줄 쌓인다.
  - '저장'은 계산까지 하고 Calibration/EIH_<시각>/ 에 사진·포즈·결과를 남긴다.

단위/규약은 hand_eye 모듈과 동일하다: 위치 mm, 각도 도(deg),
회전은 ZYX 오일러각 (R = Rz·Ry·Rx).
"""

import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import hand_eye
from calibration_store import display_path, save_session
# 이미지 보드는 다른 페이지와 같은 것을 쓴다. 휠 확대와 우클릭 드래그 이동이
# 여기에도 필요한데(코너가 제대로 잡혔는지 보려면 확대해야 한다), 보드를 따로
# 두면 그 동작을 두 벌 유지해야 한다.
from image_board import ImageBoard
from pages.base_page import BasePage
from pages.camera_capture_page import CameraCapturePage
from theme import (
    CONTENT_SPACING,
    POSE_PANEL_WIDTH,
    SAMPLE_TABLE_WIDTH,
)

# ===== 로봇 종류 =====
# (표시 이름, hand_eye 로봇 종류)
ROBOT_CHOICES = [
    ("6축 로봇", hand_eye.ROBOT_6AXIS),
    ("4축 로봇", hand_eye.ROBOT_4AXIS),
]
# 4축에서 숨기는 입력 (4축 로봇에는 이 값이 없다)
FOUR_AXIS_HIDDEN_FIELDS = ("rx", "ry")

# 4축 카메라 Z 오프셋을 정하는 방법 (표시 이름, hand_eye Z 기준)
Z_SOURCE_CHOICES = [
    ("미보정 (Z = 0)", hand_eye.Z_SOURCE_UNSET),
    ("보드 높이 입력", hand_eye.Z_SOURCE_BOARD_HEIGHT),
]
FOUR_AXIS_GUIDE = (
    "4축: RZ를 서로 다른 각도(권장 ±30°)로 돌린 자세와, RZ는 그대로 두고 "
    "X·Y로 옮긴 자세를 섞어 찍으세요.\n"
    "보드 높이: 툴 끝을 보드 면에 닿게 했을 때 로봇이 보여 주는 Z."
)

# ===== 체커보드 =====
# (표시 이름, 안쪽 코너 가로, 안쪽 코너 세로)
# OpenCV 가 세는 건 사각형 개수가 아니라 '안쪽 코너' 개수다.
BOARD_CHOICES = [
    ("8 × 5 (안쪽 코너)", 8, 5),
    ("9 × 6 (안쪽 코너)", 9, 6),
    ("8 × 6 (안쪽 코너)", 8, 6),
    ("7 × 6 (안쪽 코너)", 7, 6),
    ("7 × 5 (안쪽 코너)", 7, 5),
    ("6 × 5 (안쪽 코너)", 6, 5),
    ("6 × 4 (안쪽 코너)", 6, 4),
    ("10 × 7 (안쪽 코너)", 10, 7),
    ("11 × 8 (안쪽 코너)", 11, 8),
]
DEFAULT_SQUARE_SIZE_MM = 25.0
SQUARE_SIZE_MIN_MM = 0.1
SQUARE_SIZE_MAX_MM = 500.0

# ===== 플랜지 포즈 입력 =====
# (키, 라벨, 최소, 최대, 소수점자리, 단위)
POSE_FIELDS = [
    ("x", "X", -100000.0, 100000.0, 3, "mm"),
    ("y", "Y", -100000.0, 100000.0, 3, "mm"),
    ("z", "Z", -100000.0, 100000.0, 3, "mm"),
    ("rx", "RX", -360.0, 360.0, 3, "°"),
    ("ry", "RY", -360.0, 360.0, 3, "°"),
    ("rz", "RZ", -360.0, 360.0, 3, "°"),
]

# 표는 이름 한 열만 둔다. 포즈 값은 행을 고르면 상태문구에, 마우스를 올리면
# 툴팁에 나오므로 표에 늘어놓아 자리를 차지할 이유가 없다.
TABLE_HEADERS = ["자세"]


# ======================================================
# 계산 결과 창
# ======================================================
class _ResultDialog(QDialog):
    """저장 전에 계산 값과 오차를 확인하는 창."""

    def __init__(self, result, board, quality, parent=None):
        super().__init__(parent)
        self.setWindowTitle("캘리브레이션 결과")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(CONTENT_SPACING)

        source = {
            "provided": "카메라 내장값",
            "estimated_from_images": "촬영 이미지로 추정",
        }.get(result["intrinsics_source"], result["intrinsics_source"])

        robot_name = dict(
            (value, label) for label, value in ROBOT_CHOICES
        ).get(result.get("robot_type"), result.get("robot_type"))

        layout.addWidget(self._section("입력"))
        layout.addLayout(self._rows([
            ("로봇", robot_name),
            ("사용한 자세", f"{result['sample_count']} 개"),
            ("체커보드", f"{board['cols']} × {board['rows']} "
                        f"(칸 {board['square_size_mm']:g} mm)"),
            ("카메라 내부 파라미터", source),
        ]))

        x, y, z, rx, ry, rz = result["pose"]
        layout.addWidget(self._section("카메라 위치 · 자세 (플랜지 기준)"))
        layout.addLayout(self._rows([
            ("X", f"{x:10.3f} mm"), ("Y", f"{y:10.3f} mm"), ("Z", f"{z:10.3f} mm"),
            ("RX", f"{rx:10.3f} °"), ("RY", f"{ry:10.3f} °"), ("RZ", f"{rz:10.3f} °"),
        ]))

        z_reference = result.get("z_reference")
        if z_reference:
            if z_reference["source"] == hand_eye.Z_SOURCE_BOARD_HEIGHT:
                z_text = f"보드 높이 {z_reference['board_height_mm']:.3f} mm 로 맞춤"
            else:
                z_text = "미보정 (Z = 0)"
            layout.addLayout(self._rows([("Z 기준 (4축)", z_text)]))

        consistency = result["consistency"]
        layout.addWidget(self._section("오차"))
        layout.addLayout(self._rows([
            ("재투영 오차 (평균)", f"{result['reprojection_error_px']:.3f} px"),
            ("재투영 오차 (최대)", f"{result['reprojection_error_max_px']:.3f} px"),
            ("체커보드 위치 편차 (최대)", f"{consistency['max_mm']:.3f} mm"),
            ("체커보드 위치 편차 (표준편차)", f"{consistency['std_mm']:.3f} mm"),
        ]))

        layout.addWidget(self._section(f"판정 — {quality['summary']}"))
        for note in quality["notes"]:
            item = QLabel("· " + note)
            item.setObjectName("cameraStatus")
            item.setWordWrap(True)
            layout.addWidget(item)

        buttons = QHBoxLayout()
        close_button = QPushButton("닫기")
        close_button.setObjectName("primaryBtn")
        close_button.setDefault(True)
        close_button.clicked.connect(self.accept)
        buttons.addStretch()
        buttons.addWidget(close_button)
        layout.addSpacing(6)
        layout.addLayout(buttons)

    @staticmethod
    def _section(text):
        label = QLabel(text)
        label.setObjectName("sectionLabel")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _rows(pairs):
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        for line, (name, value) in enumerate(pairs):
            caption = QLabel(name)
            caption.setObjectName("cameraStatus")
            amount = QLabel(value)
            amount.setObjectName("dialogLabel")
            amount.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            # 값을 복사해 다른 곳에 붙여 넣을 수 있게 한다.
            amount.setTextInteractionFlags(Qt.TextSelectableByMouse)
            grid.addWidget(caption, line, 0)
            grid.addWidget(amount, line, 1)
        return grid


# ======================================================
# 페이지
# ======================================================
class EIHCalibPage(BasePage):
    """로봇 플랜지와 카메라 사이 변환을 구하는 캘리브레이션 페이지."""

    title = "EIH Calibration"
    nav_icon = "calibration"
    expand_content = True

    def __init__(self, parent=None):
        super().__init__(parent)

        self._window = None        # bind_window()로 주입되는 MainWindow
        self._samples = []         # [{"image", "corners", "pose"}, ...]
        self._pending_image = None   # 방금 캡쳐한 원본 (아직 확인 전)
        self._pending_corners = None # 방금 캡쳐에서 찾은 코너
        self._intrinsics = None      # (camera_matrix, dist) 또는 None
        self._result = None          # 계산 결과 (확인 전에는 저장 불가)

        self.content_layout.addLayout(self._build_top_row())
        self.content_layout.addLayout(self._build_body(), 1)
        self.content_layout.addLayout(self._build_status_row())

        self._apply_robot_type()
        self._update_pose_enabled()
        self._update_button_states()

    def bind_window(self, window):
        """MainWindow가 페이지 등록 시 자신을 주입한다(PythonUI._load_pages)."""
        self._window = window

    # ===== UI 구성 =====
    def _build_top_row(self):
        row = QHBoxLayout()
        row.setSpacing(CONTENT_SPACING)

        # 로봇 종류에 따라 입력 칸과 풀이 방식이 달라진다.
        self.robot_combo = QComboBox()
        self.robot_combo.setObjectName("boardCombo")
        for label, robot_type in ROBOT_CHOICES:
            self.robot_combo.addItem(label, robot_type)
        self.robot_combo.currentIndexChanged.connect(self._on_robot_changed)

        self.board_combo = QComboBox()
        self.board_combo.setObjectName("boardCombo")
        for label, cols, rows in BOARD_CHOICES:
            self.board_combo.addItem(label, (cols, rows))
        self.board_combo.currentIndexChanged.connect(self._on_board_changed)

        # 칸 크기가 있어야 결과가 mm 단위로 나온다(없으면 '칸 수' 단위가 된다).
        square_label = QLabel("칸 크기")
        square_label.setObjectName("poseLabel")
        self.square_size = QDoubleSpinBox()
        self.square_size.setObjectName("poseInput")
        self.square_size.setRange(SQUARE_SIZE_MIN_MM, SQUARE_SIZE_MAX_MM)
        self.square_size.setDecimals(2)
        self.square_size.setValue(DEFAULT_SQUARE_SIZE_MM)
        self.square_size.setSuffix(" mm")
        self.square_size.setAlignment(Qt.AlignCenter)

        self.btn_capture = QPushButton("캡쳐")
        self.btn_capture.setObjectName("primaryBtn")
        self.btn_capture.clicked.connect(self._on_capture_clicked)

        self.btn_calculate = QPushButton("계산")
        self.btn_calculate.setObjectName("primaryBtn")
        self.btn_calculate.clicked.connect(self._on_calculate_clicked)

        self.btn_save = QPushButton("저장")
        self.btn_save.setObjectName("secondaryBtn")
        self.btn_save.clicked.connect(self._on_save_clicked)

        # 칸 크기가 바뀌면 이미 계산해 둔 값이 더는 맞지 않는다.
        self.square_size.valueChanged.connect(self._invalidate_result)

        row.addWidget(self.robot_combo)
        row.addWidget(self.board_combo)
        row.addWidget(square_label)
        row.addWidget(self.square_size)
        row.addWidget(self.btn_capture)
        row.addStretch()
        row.addWidget(self.btn_calculate)
        row.addWidget(self.btn_save)
        return row

    def _build_body(self):
        """좌: 수집 목록 표 / 중: 이미지 보드 / 우: 포즈 입력."""
        row = QHBoxLayout()
        row.setSpacing(CONTENT_SPACING)
        row.addWidget(self._build_sample_table())
        row.addWidget(self._build_image_board(), 1)
        row.addWidget(self._build_pose_panel())
        return row

    def _build_sample_table(self):
        panel = QWidget()
        panel.setFixedWidth(SAMPLE_TABLE_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        caption = QLabel("수집한 자세")
        caption.setObjectName("sectionLabel")

        self.sample_table = QTableWidget(0, len(TABLE_HEADERS))
        self.sample_table.setObjectName("sampleTable")
        self.sample_table.setHorizontalHeaderLabels(TABLE_HEADERS)
        self.sample_table.verticalHeader().setVisible(False)
        self.sample_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.sample_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sample_table.setSelectionMode(QAbstractItemView.SingleSelection)
        # 행을 고르면 그때 찍은 사진을 보드에 띄운다.
        self.sample_table.itemSelectionChanged.connect(self._on_sample_selected)
        # 잘못 넣은 자세는 우클릭 메뉴로 지운다.
        self.sample_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sample_table.customContextMenuRequested.connect(self._on_table_menu)

        self.sample_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )

        layout.addWidget(caption)
        layout.addWidget(self.sample_table, 1)
        return panel

    def _build_image_board(self):
        self.board = ImageBoard("캡쳐 버튼을 누르면 여기에 표시됩니다.")
        return self.board

    def _build_pose_panel(self):
        panel = QWidget()
        panel.setFixedWidth(POSE_PANEL_WIDTH)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        caption = QLabel("플랜지 포즈")
        caption.setObjectName("sectionLabel")
        layout.addWidget(caption)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.pose_inputs = {}
        self.pose_labels = {}
        for line, (key, label, low, high, decimals, suffix) in enumerate(POSE_FIELDS):
            name = QLabel(label)
            name.setObjectName("poseLabel")

            box = QDoubleSpinBox()
            box.setObjectName("poseInput")
            box.setRange(low, high)
            box.setDecimals(decimals)
            box.setSuffix(f" {suffix}")
            box.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            box.setButtonSymbols(QDoubleSpinBox.NoButtons)

            grid.addWidget(name, line, 0)
            grid.addWidget(box, line, 1)
            self.pose_inputs[key] = box
            self.pose_labels[key] = name

        layout.addLayout(grid)

        self.btn_confirm = QPushButton("확인")
        self.btn_confirm.setObjectName("primaryBtn")
        self.btn_confirm.clicked.connect(self._on_confirm_clicked)
        layout.addWidget(self.btn_confirm)

        self.pose_hint = QLabel("")
        self.pose_hint.setObjectName("cameraStatus")
        self.pose_hint.setWordWrap(True)
        layout.addWidget(self.pose_hint)

        layout.addWidget(self._build_z_reference_panel())

        layout.addStretch()
        return panel

    def _build_z_reference_panel(self):
        """4축 전용: 카메라 Z 오프셋을 정할 보드 높이 입력.

        자세마다 받는 값이 아니라 계산 한 번에 하나라, 캡쳐 여부와 상관없이
        늘 입력할 수 있다.
        """
        self.z_reference_panel = QWidget()
        layout = QVBoxLayout(self.z_reference_panel)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(6)

        caption = QLabel("Z 기준 (4축)")
        caption.setObjectName("sectionLabel")

        self.z_source_combo = QComboBox()
        self.z_source_combo.setObjectName("boardCombo")
        for label, source in Z_SOURCE_CHOICES:
            self.z_source_combo.addItem(label, source)
        self.z_source_combo.currentIndexChanged.connect(self._on_z_source_changed)

        height_row = QHBoxLayout()
        height_row.setSpacing(8)
        height_label = QLabel("보드 높이")
        height_label.setObjectName("poseLabel")
        self.board_height = QDoubleSpinBox()
        self.board_height.setObjectName("poseInput")
        self.board_height.setRange(-100000.0, 100000.0)
        self.board_height.setDecimals(3)
        self.board_height.setSuffix(" mm")
        self.board_height.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.board_height.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.board_height.valueChanged.connect(self._invalidate_result)
        height_row.addWidget(height_label)
        height_row.addWidget(self.board_height, 1)

        guide = QLabel(FOUR_AXIS_GUIDE)
        guide.setObjectName("cameraStatus")
        guide.setWordWrap(True)

        layout.addWidget(caption)
        layout.addWidget(self.z_source_combo)
        layout.addLayout(height_row)
        layout.addWidget(guide)

        self._on_z_source_changed()
        return self.z_reference_panel

    def _build_status_row(self):
        row = QHBoxLayout()
        row.setSpacing(CONTENT_SPACING)

        self.status_led = QLabel()
        self.status_led.setObjectName("statusLed")

        self.status_text = QLabel("체커보드 크기를 고르고 캡쳐하세요.")
        self.status_text.setObjectName("cameraStatus")

        row.addWidget(self.status_led)
        row.addWidget(self.status_text)
        row.addStretch()
        return row

    # ===== 로봇 종류 =====
    def _robot_type(self):
        return self.robot_combo.currentData()

    def _is_4axis(self):
        return self._robot_type() == hand_eye.ROBOT_4AXIS

    def _apply_robot_type(self):
        """4축이면 RX/RY 입력을 숨기고 Z 기준 입력을 띄운다."""
        four_axis = self._is_4axis()
        for key in FOUR_AXIS_HIDDEN_FIELDS:
            self.pose_inputs[key].setVisible(not four_axis)
            self.pose_labels[key].setVisible(not four_axis)
        self.z_reference_panel.setVisible(four_axis)

    def _on_robot_changed(self):
        self._apply_robot_type()
        # 풀이 방식이 바뀌므로 이미 계산한 값은 더는 맞지 않는다.
        self._invalidate_result()
        if self._samples:
            # 6축으로 모은 자세는 RX/RY가 있어 4축으로 풀 수 없고, 4축으로 모은
            # 자세는 회전축이 하나뿐이라 6축으로 풀 수 없다.
            self._set_status(
                "로봇 종류를 바꿨습니다. 이미 모은 자세는 이전 로봇 기준이라 함께 "
                "쓸 수 없습니다. 목록을 비우고 다시 모으세요.",
                "error",
            )

    def _on_z_source_changed(self):
        uses_height = (
            self.z_source_combo.currentData() == hand_eye.Z_SOURCE_BOARD_HEIGHT
        )
        self.board_height.setEnabled(uses_height)
        self._invalidate_result()

    def _board_height_mm(self):
        """4축 계산에 넘길 보드 높이. 미보정을 골랐으면 None."""
        if self.z_source_combo.currentData() != hand_eye.Z_SOURCE_BOARD_HEIGHT:
            return None
        return self.board_height.value()

    # ===== 체커보드 설정 =====
    def _board_size(self):
        return self.board_combo.currentData()

    def _on_board_changed(self):
        """크기를 바꾸면 이미 모은 코너와 규격이 안 맞으므로 알린다."""
        if self._samples:
            self._set_status(
                "체커보드 크기를 바꿨습니다. 이미 모은 자세는 이전 크기로 찾은 "
                "것이라 함께 쓸 수 없습니다. 목록을 비우고 다시 모으세요.",
                "error",
            )
        # 확인 전이던 캡쳐는 새 규격으로 다시 찾아야 하므로 버린다.
        self._pending_image = None
        self._pending_corners = None
        self._update_pose_enabled()

    # ===== 카메라 =====
    def _find_camera_page(self):
        """페이지 스택에서 CameraCapturePage 인스턴스를 찾는다."""
        stack = getattr(self._window, "stack", None)
        if stack is None:
            return None
        for i in range(stack.count()):
            page = stack.widget(i)
            if isinstance(page, CameraCapturePage):
                return page
        return None

    def _on_capture_clicked(self):
        camera_page = self._find_camera_page()
        if camera_page is None:
            self._set_status("Camera Capture 페이지를 찾을 수 없습니다.", "error")
            return

        try:
            rgb, _depth = camera_page.grab_frames()
        except Exception as exc:
            self._set_status(f"캡쳐 실패: {exc}", "error")
            return

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # 내부 파라미터는 연결된 카메라에서 한 번만 받아 둔다(가상 모드면 None).
        if self._intrinsics is None:
            self._intrinsics = camera_page.color_intrinsics()

        cols, rows = self._board_size()
        corners = hand_eye.find_board_corners(bgr, cols, rows)

        # 표에서 과거 항목을 보고 있었다면 선택을 풀어 방금 캡쳐를 띄운다.
        self.sample_table.clearSelection()
        # 새 장면이므로 확대/이동을 되돌린다.
        self.board.reset_view()

        if corners is None:
            # 못 찾은 장면은 포즈를 받아 봐야 쓸 수 없으므로 입력을 잠근다.
            self._pending_image = None
            self._pending_corners = None
            self.board.set_image_bgr(bgr)
            self._set_status(
                f"체커보드({cols}×{rows})를 찾지 못했습니다. "
                "크기 설정과 조명·초점을 확인하세요.",
                "error",
            )
        else:
            self._pending_image = bgr
            self._pending_corners = corners
            self.board.set_image_bgr(
                hand_eye.draw_board_corners(bgr, corners, cols, rows)
            )
            self._set_status(
                f"체커보드 {cols}×{rows} 인식됨 — 플랜지 포즈를 입력하고 확인을 누르세요.",
                "success",
            )

        self._update_pose_enabled()

    # ===== 자세 수집 =====
    def _update_pose_enabled(self):
        """체커보드를 못 찾았으면 포즈 입력을 통째로 잠근다."""
        usable = self._pending_corners is not None
        for box in self.pose_inputs.values():
            box.setEnabled(usable)
        for label in self.pose_labels.values():
            label.setEnabled(usable)
        self.btn_confirm.setEnabled(usable)
        self.pose_hint.setText(
            "" if usable else "체커보드를 인식한 뒤에 입력할 수 있습니다."
        )

    def _invalidate_result(self):
        """표본이나 설정이 바뀌면 계산해 둔 값은 더 이상 그 상태의 결과가 아니다.

        낡은 결과를 저장하면 화면에서 확인한 값과 파일 내용이 달라지므로,
        다시 계산할 때까지 저장을 잠근다.
        """
        self._result = None
        self._update_button_states()

    def _update_button_states(self):
        enough = len(self._samples) >= hand_eye.MIN_SAMPLES
        self.btn_calculate.setEnabled(enough)
        self.btn_calculate.setToolTip(
            "" if enough
            else f"자세가 최소 {hand_eye.MIN_SAMPLES}개는 있어야 계산할 수 있습니다."
        )

        # 저장은 계산 결과를 눈으로 확인한 뒤에만 열린다.
        self.btn_save.setEnabled(self._result is not None)
        self.btn_save.setToolTip(
            "" if self._result is not None
            else "먼저 계산을 눌러 결과를 확인하세요."
        )

    def _current_pose(self):
        # 4축은 RX/RY가 없다. 숨긴 칸에 예전 값이 남아 있어도 0으로 넣는다.
        hidden = FOUR_AXIS_HIDDEN_FIELDS if self._is_4axis() else ()
        return tuple(
            0.0 if key in hidden else self.pose_inputs[key].value()
            for key, *_rest in POSE_FIELDS
        )

    def _on_confirm_clicked(self):
        if self._pending_corners is None:
            self._set_status("체커보드를 인식한 캡쳐가 없습니다.", "error")
            return

        self._samples.append({
            "image": self._pending_image,
            "corners": self._pending_corners,
            "pose": self._current_pose(),
        })
        # 같은 장면을 두 번 넣지 못하도록 확인한 캡쳐는 비운다.
        self._pending_image = None
        self._pending_corners = None

        self._append_table_row(self._samples[-1])
        self._update_pose_enabled()
        self._invalidate_result()   # 표본이 늘었으니 다시 계산해야 한다

        count = len(self._samples)
        remaining = hand_eye.RECOMMENDED_SAMPLES - count
        hint = f" (권장 {hand_eye.RECOMMENDED_SAMPLES}개까지 {remaining}개 남음)" if remaining > 0 else ""
        self._set_status(f"{count}번째 자세를 저장했습니다.{hint}", "success")

    @staticmethod
    def _pose_tooltip(pose):
        x, y, z, rx, ry, rz = pose
        return (f"X {x:.3f} mm\nY {y:.3f} mm\nZ {z:.3f} mm\n"
                f"RX {rx:.3f}°\nRY {ry:.3f}°\nRZ {rz:.3f}°")

    def _append_table_row(self, sample):
        line = self.sample_table.rowCount()
        self.sample_table.insertRow(line)

        item = QTableWidgetItem(f"pose_{line + 1}")
        item.setTextAlignment(Qt.AlignCenter)
        # 표는 이름만 보여 주되, 값이 궁금하면 마우스를 올려 볼 수 있게 한다.
        item.setToolTip(self._pose_tooltip(sample["pose"]))
        self.sample_table.setItem(line, 0, item)

    def _on_table_menu(self, position):
        """표에서 우클릭하면 그 행을 지울 수 있는 메뉴를 띄운다."""
        line = self.sample_table.rowAt(position.y())
        if not 0 <= line < len(self._samples):
            return   # 빈 곳을 눌렀다

        menu = QMenu(self)
        # 어떤 행이 지워질지 이름에 박아 둔다(우클릭한 행과 선택된 행이 다를 수 있다).
        delete_action = menu.addAction(f"{line + 1}번 자세 삭제")
        chosen = menu.exec_(self.sample_table.viewport().mapToGlobal(position))
        if chosen is delete_action:
            self._delete_sample(line)

    def _delete_sample(self, line):
        """수집한 자세 하나를 목록과 표에서 뺀다."""
        # 표를 먼저 건드리면 선택 변경 신호가 아직 안 지워진 목록을 가리키므로 막아 둔다.
        blocked = self.sample_table.blockSignals(True)
        self.sample_table.removeRow(line)
        del self._samples[line]
        # 뒤 행들이 한 칸씩 당겨졌으니 이름과 툴팁을 다시 매긴다.
        for row in range(self.sample_table.rowCount()):
            item = self.sample_table.item(row, 0)
            item.setText(f"pose_{row + 1}")
            item.setToolTip(self._pose_tooltip(self._samples[row]["pose"]))
        self.sample_table.clearSelection()
        self.sample_table.blockSignals(blocked)

        # 지운 자세를 보고 있었을 수 있으므로 보드를 비운다(엉뚱한 사진 방지).
        self.board.clear_image()
        self._invalidate_result()   # 표본이 줄었으니 다시 계산해야 한다

        remaining = len(self._samples)
        short = hand_eye.MIN_SAMPLES - remaining
        note = f" 계산하려면 {short}개 더 필요합니다." if short > 0 else ""
        self._set_status(f"{line + 1}번 자세를 삭제했습니다. 남은 자세 {remaining}개.{note}")

    def _on_sample_selected(self):
        """표에서 행을 고르면 그때 찍은 사진을 코너와 함께 보여 준다."""
        rows = self.sample_table.selectionModel().selectedRows()
        if not rows:
            return
        line = rows[0].row()
        if not 0 <= line < len(self._samples):
            return

        sample = self._samples[line]
        cols, board_rows = self._board_size()
        try:
            preview = hand_eye.draw_board_corners(
                sample["image"], sample["corners"], cols, board_rows
            )
        except cv2.error:
            # 체커보드 크기를 바꾼 뒤라 코너 개수가 안 맞는 경우
            preview = sample["image"]
        # 다른 자세의 사진으로 갈아 끼우는 것이므로 확대를 되돌린다.
        self.board.reset_view()
        self.board.set_image_bgr(preview)

        x, y, z, rx, ry, rz = sample["pose"]
        self._set_status(
            f"{line + 1}번 자세 — X {x:.1f} Y {y:.1f} Z {z:.1f} / "
            f"RX {rx:.1f} RY {ry:.1f} RZ {rz:.1f}"
        )

    # ===== 계산 =====
    def _board_spec(self):
        cols, rows = self._board_size()
        return {"cols": cols, "rows": rows, "square_size_mm": self.square_size.value()}

    def _on_calculate_clicked(self):
        """캘리브레이션을 계산하고 결과를 새 창에 보여 준다(저장은 하지 않는다)."""
        board = self._board_spec()
        camera_matrix, dist = (self._intrinsics or (None, None))

        image_size = None
        if self._samples:
            height, width = self._samples[0]["image"].shape[:2]
            image_size = (width, height)

        try:
            result = hand_eye.calibrate_eye_in_hand(
                self._samples,
                board["cols"], board["rows"], board["square_size_mm"],
                camera_matrix=camera_matrix,
                dist_coeffs=dist,
                image_size=image_size,
                robot_type=self._robot_type(),
                board_height_mm=self._board_height_mm() if self._is_4axis() else None,
            )
        except (ValueError, RuntimeError) as exc:
            self._result = None
            self._update_button_states()
            self._set_status(f"캘리브레이션 실패: {exc}", "error")
            return

        # 저장은 지금 확인한 이 결과를 그대로 쓴다(다시 계산하지 않는다).
        self._result = result
        self._result_board = board
        self._update_button_states()

        quality = hand_eye.assess_quality(result)
        _ResultDialog(result, board, quality, self).exec_()

        x, y, z, rx, ry, rz = result["pose"]
        self._set_status(
            f"계산 완료 ({quality['summary'].split(' —')[0]}) — "
            f"X {x:.1f} Y {y:.1f} Z {z:.1f} mm / "
            f"RX {rx:.1f} RY {ry:.1f} RZ {rz:.1f}° · 확인 후 저장하세요.",
            "success" if quality["level"] != "poor" else "error",
        )

    # ===== 저장 =====
    def _on_save_clicked(self):
        if self._result is None:
            self._set_status("먼저 계산을 눌러 결과를 확인하세요.", "error")
            return

        ok, target = save_session(self._samples, self._result, self._result_board)
        if not ok:
            self._set_status(f"저장 실패: {target}", "error")
            return

        x, y, z, rx, ry, rz = self._result["pose"]
        self._set_status(
            f"저장됨 {display_path(target)} — "
            f"카메라 위치 X {x:.1f} Y {y:.1f} Z {z:.1f} mm / "
            f"RX {rx:.1f} RY {ry:.1f} RZ {rz:.1f}°",
            "success",
        )

    # ===== 상태 표시 =====
    def _set_status(self, text, status_type="info"):
        self.status_text.setText(text)
        self.status_led.setProperty(
            "connected", "true" if status_type == "success" else "false"
        )
        self.status_led.style().unpolish(self.status_led)
        self.status_led.style().polish(self.status_led)
