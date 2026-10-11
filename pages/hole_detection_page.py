"""홀 탐지 페이지 — 한 장 찍어 '홀이 어디 있는가'까지 답한다.

Camera Capture 페이지의 카메라로 한 장 찍어 홀을 찾고, 탐지된 홀을 이미지
위에 표시한다. 홀을 찾는 방법은 네 가지 중에 고른다(맨 왼쪽 콤보).

    블랍 검출    이진화 + 블롭 검출기. 원형 홀 여러 개를 한 번에, 수십 ms.
    템플릿 매칭  잘라 둔 모양을 이미지에서 찾는다.
    SAM2 홀 지정 홀 하나를 네모로 드래그해 티칭하고, 이후 그 홀만 다시 찾는다.
    SAM2 비디오 추적  어디를 볼지도 SAM2가 정한다. 거리·회전이 바뀌어도 따라간다.

어느 방법을 쓰든 결과는 [(x, y, 반지름), ...] 하나로 모인다.

**자세(법선)와 포즈는 여기서 만들지 않는다.** 법선 매칭점과 진입 각도
오프셋은 전부 Pose Setting 페이지에 있다 — 법선을 만드는 것도 결국 포즈를 만들기
위한 것이라 한 화면에 모아 두는 편이 낫다. 이 페이지는 홀 좌표까지만 책임지고,
Pose Setting 이 capture_context() / detect_holes() 로 그 결과를 받아 간다.

세팅(프로파일) 파일은 두 페이지가 **하나를 함께 쓴다.** 이 페이지가 주인이라
읽고 쓰는 일을 맡고, 포즈 키는 Pose Setting 에 물어 합친다(_collect_settings).

구성:

    [블랍 검출 ▼] [이미지 캡쳐] [ROI 설정] [컬러맵 ▼]  [세팅 ▼] [+] [−] [저장]
                              ● 전체 1280×720 — 홀 26개 탐지
                          캡쳐 6.8 ms + 분석 48.7 ms  =  55.5 ms
    ┌───────────────────────────────┐  ┌ Parameter Setting ┐
    │  검출 결과 (원 + 중심점)       │  │ 고른 방식의 값만   │
    │                               │  ├ Flange Pose ──────┤
    │                               │  │ X Y Z RX RY RZ    │
    └───────────────────────────────┘  └───────────────────┘

방식별 파라미터는 보드 **오른쪽** 패널에 있다(260910). 보드 위에 두면 방식을
바꿀 때마다 줄 수가 달라져 이미지 보드 높이가 함께 움직였다 — 값을 바꿔 가며
결과를 견주는 화면에서 사진 크기가 매번 달라지면 비교할 수가 없다. 패널 폭이
고정이므로 이제 어느 방식을 골라도 보드가 차지하는 자리는 그대로다.

ROI 설정:
  버튼을 켜고 결과 보드를 드래그하면 그 영역에 바운딩 박스가 그려지고,
  이후 분석은 박스 안에서만 수행된다. 아주 작게 드래그(클릭)하면 해제된다.
  ROI는 roi_state 모듈에 원본 이미지 좌표로 보관되므로 다른 페이지에서도
  같은 영역을 그대로 쓴다. '저장'을 누르면 블롭 파라미터와 함께
  Settings/BlobSetting.json 에 남아 다음 실행에서도 복원된다.

블롭 분석 파라미터:
- 이진화: 적응형 임계값 (Adaptive Threshold)
- 필터링: 면적, 원형도(Circularity), 볼록도(Convexity) 기준

SAM2 홀 지정:
  '홀 지정'을 누르고 이미지에서 홀을 네모로 드래그하면, SAM2가 그 네모 안에서
  홀의 경계를 따서 중심과 반지름을 잡고 그 경계의 외접 사각형을 '기준 모양'
  으로 저장한다. 260914 까지는 홀 안쪽을 클릭했는데, 점 하나로는 SAM2 가
  홀인지 판재인지 스스로 골라야 해서 뜻한 것보다 큰 물체가 잡히곤 했다.
  네모는 범위를 함께 알려 주므로 그 일이 없다(법선 매칭점과 같은 방식이다).

  ROI·템플릿 지정·홀 지정이 같은 드래그를 쓰므로, 하나를 켜면 나머지는
  내려간다(_release_drag_buttons). 이후 촬영에서는 저장한 네모를 프롬프트로
  써서 같은 홀을 다시 찾는다 — 부품이 촬영마다 수~수십 px 흔들려도 짚는다.

  기준 모양은 템플릿과 같은 창고(Templates/)에 둔다. 둘 다 '이 모양을 다시
  찾아라'는 뜻이고, 실제로 대략 위치를 잡을 때 템플릿 매칭을 쓴다.

  torch/sam2 가 없으면 이 항목은 비활성으로 남고 사유가 툴팁에 뜬다. 자세한
  내용과 성능(GPU 유무에 따라 촬영당 0.1~1초)은 sam2_detector 모듈에 있다.

오퍼레이터 경로:
  visualize_for_operator() 는 검출부터 포즈까지 한 번에 돈다. 화면 위젯을 하나도
  보지 않고 프로파일 파일만 읽으므로, 체결 지점마다 다른 세팅을 그대로 쓴다.
  포즈 부분은 Pose Setting 화면과 같은 함수(pose_pipeline)를 써서 두 경로가
  같은 답을 내도록 맞춰 둔다.
"""

import time
from collections import namedtuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from depth_view import colorize_depth
from detection_debug import DebugFrame, debug_frame_at, make_contact_sheet, to_bgr
from hole_pose import (
    NORMAL_MODE_FLANGE,
    NORMAL_MODE_POINTS,
    NORMAL_MODE_RING,
    estimate_hole_poses,
    rpy_zyx_deg_to_matrix,
)
from image_board import ImageBoard, ResultBoard
from pages.base_page import BasePage
from pages.camera_capture_page import CameraCapturePage
from elided_label import ElidedLabel
# 포즈를 만드는 값과 계산은 Pose Setting 페이지와 함께 쓴다. 아래 오퍼레이터
# 경로가 세팅 화면과 같은 답을 내야 하므로, 두 곳이 같은 함수를 부른다.
from pose_pipeline import (
    as_float,
    as_int,
    draw_normal_points,
    draw_ring_regions,
    normal_plane_from_masks,
    normal_points_from_settings,
    normal_points_masks,
    normal_source_from_settings,
    spread_note,
)
from range_slider import RangeSlider
from roi_state import roi_state
from setting_panel import SettingGroup, SettingPanel
from settings_store import (
    DEFAULT_PROFILE_NAME,
    active_profile,
    delete_profile,
    display_path,
    list_profiles,
    load_profile,
    save_profile,
    set_active_profile,
    validate_profile_name,
)
import sam2_detector
import sam2_video_tracker
from template_matching import DEFAULT_MATCH_THRESHOLD, TemplateMatcher
from template_store import (
    delete_template,
    list_holes,
    list_templates,
    load_hole,
    load_template,
    save_hole,
    save_template,
    validate_template_name,
)
from text_input_dialog import TextInputDialog
from theme import (
    CONTENT_SPACING,
    PARAM_BOX_WIDTH,
    RANGE_SLIDER_MAX_WIDTH,
)
from toggle_switch import ToggleSwitch
from wire_detector import (
    REP_BOTTOM,
    REP_CENTER,
    REP_CHOICES,
    REP_FIRST_SORT,
    REP_LEFT,
    REP_RIGHT,
    REP_TOP,
    SORT_CHOICES,
    SORT_CUSTOM,
    SORT_X,
    WIRE_MODE_BIN_TOP,
    WIRE_MODE_CHOICES,
    WIRE_MODE_EDGE,
    WIRE_MODE_PLANE,
    WireDetectionConfig,
    WireDetector,
    draw_wires,
)
from wire_tracker import WireTracker

# ===== 플랜지 포즈 입력 필드 =====
# (키, 라벨, 최소, 최대, 소수점자리, 단위)
POSE_FIELDS = [
    ("x", "X", -100000.0, 100000.0, 3, "mm"),
    ("y", "Y", -100000.0, 100000.0, 3, "mm"),
    ("z", "Z", -100000.0, 100000.0, 3, "mm"),
    ("rx", "RX", -360.0, 360.0, 3, "°"),
    ("ry", "RY", -360.0, 360.0, 3, "°"),
    ("rz", "RZ", -360.0, 360.0, 3, "°"),
]

# ===== 블롭 분석 파라미터 (동작 값이므로 이 파일에 둔다) =====
ADAPTIVE_THRESHOLD_BLOCK_SIZE = 21
ADAPTIVE_THRESHOLD_C = 5
MIN_BLOB_AREA = 50          # 최소 영역 (픽셀)
MAX_BLOB_AREA = 50000       # 최대 영역 (픽셀)
MIN_CIRCULARITY = 0.6       # 최소 원형도 (0~1)
MIN_CONVEXITY = 0.7         # 최소 볼록도 (0~1)

# OpenCV 5의 SimpleBlobDetector는 검출기를 만들 때 0<min<=max 를 요구하고,
# filterBy* 를 꺼 놔도 검사는 그대로 한다. 그래서 비율 계열 기준값의 하한은
# 0 이 아니라 이 값으로 둔다(0 을 넣으면 생성 단계에서 예외가 난다).
RATIO_FLOOR = 0.01

# 검출기가 훑을 임계값 구간과 간격 (UI에서 조정 가능한 기본값)
THRESHOLD_ABS_MIN = 0       # 슬라이더가 허용하는 하한
THRESHOLD_ABS_MAX = 255     # 슬라이더가 허용하는 상한
MIN_THRESHOLD = 50
MAX_THRESHOLD = 255
THRESHOLD_STEP = 10
THRESHOLD_STEP_MIN = 1
THRESHOLD_STEP_MAX = 255

# 이보다 가까운 블롭은 하나로 합쳐진다
MIN_DIST_BETWEEN_BLOBS = 10
MIN_DIST_MIN = 1
MIN_DIST_MAX = 2000

PARAM_APPLY_DELAY_MS = 250  # 값을 계속 바꿀 때 매번 재분석하지 않도록 두는 지연

DETECTION_BLOB = "blob"
DETECTION_TEMPLATE = "template"
DETECTION_SAM2_VIDEO = "sam2_video"
DETECTION_WIRE = "wire"

# 260908 에 'SAM2 홀 지정'(2단계)을 걷어냈다. 그것은 cv2.matchTemplate 으로
# 대략 위치를 잡고 SAM2 로 경계만 따는 방식이었는데, 매칭이 배율도 회전도
# 보지 않아 카메라 거리가 20%만 바뀌어도 판재를 물고 **그럴듯한 오답**을 냈다
# (실측: 반지름 20px 이 나와야 할 자리에 257px. 상한 검사도 통과했다).
# 못 찾는 것보다 나쁘므로 SAM2 경로는 비디오 추적 하나로 통일했다.
SAM2_METHODS = (DETECTION_SAM2_VIDEO,)

# SAM2 재검출은 CPU 에서 촬영 한 장당 1초 남짓 걸린다. 블랍/템플릿의
# 수십 ms 와 견주면 느리므로, 그 값을 치를 만한 자리에만 쓴다.
SAM2_HINT = (
    "홀을 네모로 드래그해 티칭해 두면, 이후 촬영에서 같은 홀을 다시 찾는다.\n"
    "\n"
    "찾는 방법:\n"
    "  1. 티칭한 모양으로 대략 위치를 먼저 잡는다 (값싸다).\n"
    "  2. 그 자리에 SAM2를 찍어 홀 경계를 정밀하게 딴다.\n"
    "\n"
    "부품이 촬영마다 수~수십 px 흔들려도 같은 홀을 짚는다.\n"
    "CPU 에서는 촬영 한 장당 1초 남짓, CUDA GPU 에서는 훨씬 빠르다.\n"
    "\n"
    "1단계가 배율도 회전도 보지 않으므로, 카메라 거리가 바뀌면 놓친다.\n"
    "그럴 때는 'SAM2 비디오 추적'을 쓴다."
)

SAM2_VIDEO_HINT = (
    "티칭 프레임과 지금 프레임을 '2장짜리 비디오'로 묶어 SAM2에 넘긴다.\n"
    "어디를 볼지도 SAM2가 정하므로, 카메라 거리와 회전이 바뀌어도 따라간다.\n"
    "\n"
    "참조하는 것은 늘 최초 티칭 프레임이다 (직전 촬영이 아니다).\n"
    "그래서 오래 돌려도 기준이 밀리지 않고 촬영끼리 영향을 주지 않는다.\n"
    "\n"
    "CPU 에서 촬영 한 장당 2초 남짓으로 'SAM2 홀 지정'보다 0.8초쯤 더 든다.\n"
    "CUDA GPU 가 있으면 자동으로 그쪽에서 돈다 (모델 콤보 도움말에 표시).\n"
    "티칭할 때와 촬영 해상도가 같아야 한다."
)

# SAM2 가 실제로 문 홀 경계. 초록 원은 넓이에서 되돌린 '같은 넓이의 원'이라
# 실제 모양이 아니다. 둘을 겹쳐 그려야 추적이 홀을 제대로 물었는지, 아니면
# 옆의 무엇을 물고도 그럴듯한 원을 내놓은 것인지 눈으로 가릴 수 있다.
HOLE_CONTOUR_COLOR = (255, 255, 0)      # BGR = 청록. 초록 원/빨강 중심과 겹치지 않는다.
HOLE_CONTOUR_WIDTH = 1                  # 원보다 얇게. 굵으면 원을 덮어 버린다.

# ===== 포즈 표시 =====
# 좌표축은 홀 크기에 비례해 그린다(멀리 있는 작은 홀에 큰 축이 붙지 않도록).
AXIS_LENGTH_SCALE = 2.5     # 화면에서 홀 반지름의 몇 배로 보이게 할지
AXIS_MIN_MM = 2.0
AXIS_MAX_MM = 30.0

# ===== 타겟 필터링 =====
# 필터마다 켜짐 여부와 기준값을 따로 기억한다. 새 필터를 추가하려면 여기에 한 줄
# 넣고 BlobAnalyzer._create_detector()에 반영 한 줄만 더하면 된다.
FilterSpec = namedtuple(
    "FilterSpec",
    "key label minimum maximum decimals step default_value default_enabled",
)

FILTER_AREA = "area"
FILTER_CIRCULARITY = "circularity"
FILTER_CONVEXITY = "convexity"
FILTER_INERTIA = "inertia"
FILTER_COLOR = "color"

FILTER_SPECS = [
    FilterSpec(FILTER_AREA, "크기 기준 필터링",
               1, 1_000_000, 0, 10, MIN_BLOB_AREA, True),
    FilterSpec(FILTER_CIRCULARITY, "진원도(원형률) 필터링",
               RATIO_FLOOR, 1.0, 2, 0.05, MIN_CIRCULARITY, True),
    FilterSpec(FILTER_CONVEXITY, "볼록성 필터링",
               RATIO_FLOOR, 1.0, 2, 0.05, MIN_CONVEXITY, True),
    FilterSpec(FILTER_INERTIA, "관성비 필터링",
               RATIO_FLOOR, 1.0, 2, 0.05, 0.1, False),
    # 색상은 blobColor: 0 = 어두운 블롭, 255 = 밝은 블롭
    FilterSpec(FILTER_COLOR, "색상 필터링",
               0, 255, 0, 255, 0, False),
]
FILTER_SPEC_BY_KEY = {spec.key: spec for spec in FILTER_SPECS}


def default_filter_states():
    """필터별 {켜짐, 기준값} 초기 상태를 만든다."""
    return {
        spec.key: {"enabled": spec.default_enabled, "value": spec.default_value}
        for spec in FILTER_SPECS
    }


def load_flange_pose(profile_name):
    """프로파일에 저장된 로봇 플랜지 포즈 (x, y, z, rx, ry, rz). 없으면 None.

    오퍼레이터 모드가 쓴다. 체결 지점마다 로봇이 서는 자리가 다르므로 포즈도
    지점마다 다르다. 값이 없으면 0으로 채우지 않고 None 을 돌려준다. 0 은
    '베이스 원점에 플랜지가 있다'는 뜻이라 조용히 틀린 좌표가 로봇으로 나간다.
    None 이면 부르는 쪽이 플랜지 기준 좌표로 물러선다.
    """
    data = load_profile(profile_name)
    saved = data.get("flange_pose") if isinstance(data, dict) else None
    if not isinstance(saved, dict):
        return None
    if any(saved.get(key) is None for key, *_ in POSE_FIELDS):
        return None
    return tuple(
        as_float(saved.get(key), 0.0, low, high)
        for key, _label, low, high, *_rest in POSE_FIELDS
    )


def sam2_video_detector(sam2_settings):
    """세팅의 sam2 블록으로 비디오 추적 검출기를 만든다.

    세팅 화면과 오퍼레이터가 같은 값을 읽도록 두 경로가 이 함수를 함께 쓴다.

    비디오 추적은 크롭이 아니라 티칭 프레임 원본과 그 안의 네모를 쓴다
    (법선 매칭점과 같은 구조다). 260908 이전에 티칭한 홀은 네모 대신 점만
    갖고 있어서 load_hole 이 집지 않으므로, 무엇을 해야 하는지 분명히 알려
    주고 멈춘다 — 조용히 넘기면 프롬프트 없이 추적을 돌리게 된다.
    """
    reference_name = (sam2_settings or {}).get("reference")
    if not reference_name:
        raise RuntimeError("먼저 '홀 지정'으로 찾을 홀을 드래그해 두세요.")
    taught = load_hole(reference_name)
    if taught is None:
        raise RuntimeError(
            f"'{reference_name}' 을 비디오 추적으로 쓸 수 없습니다. "
            "260908 이전에 티칭한 홀이면 '홀 지정'으로 다시 잡아 주세요."
        )
    scene_bgr, box, area = taught
    return sam2_video_tracker.Sam2VideoDetector(
        scene_bgr, box, (sam2_settings or {}).get("model"), taught_area=area
    )


def _wire_debug_info(result):
    pick_candidates = [
        wire for wire in result.wires if getattr(wire, "is_pick_candidate", True)
    ]
    return {
        "binary": result.binary,
        "wire_candidates": result.wires,
        "wire_pick_candidates": pick_candidates,
        "debug_frames": result.debug_frames,
        "wire_note": result.note,
        "debug_path": result.debug_path,
        "blob_count": len(result.holes),
        "wire_count": len(result.wires),
        "pick_count": len(pick_candidates),
    }


def _wire_config_from_settings(data):
    settings = data.get("wire_detection") if isinstance(data, dict) else None
    settings = settings if isinstance(settings, dict) else {}
    sort_axis = settings.get("sort_axis")
    if sort_axis not in SORT_CHOICES:
        sort_axis = SORT_X
    representative = settings.get("representative")
    if representative not in REP_CHOICES:
        representative = REP_CENTER
    foreground_mode = settings.get("foreground_mode")
    if foreground_mode not in WIRE_MODE_CHOICES:
        foreground_mode = WIRE_MODE_BIN_TOP
    depth_band = as_float(
        settings.get("top_band_mm", settings.get("plane_threshold_mm")),
        18.0,
        0.5,
        200.0,
    )
    return WireDetectionConfig(
        min_length_px=as_int(settings.get("min_length_px"), 80),
        min_area_px=as_int(settings.get("min_area_px"), 120),
        min_thickness_px=as_float(settings.get("min_thickness_px"), 2.0, 0.5, 200.0),
        max_thickness_px=as_float(settings.get("max_thickness_px"), 80.0, 1.0, 300.0),
        foreground_mode=foreground_mode,
        top_band_mm=depth_band,
        top_percentile=as_float(settings.get("top_percentile"), 1.0, 0.1, 50.0),
        depth_min_mm=as_float(settings.get("depth_min_mm"), 70.0, 0.0, 3000.0),
        depth_max_mm=as_float(settings.get("depth_max_mm"), 0.0, 0.0, 5000.0),
        plane_threshold_mm=as_float(settings.get("plane_threshold_mm"), depth_band, 0.5, 200.0),
        merge_distance_px=as_float(settings.get("merge_distance_px"), 18.0, 1.0, 200.0),
        merge_angle_deg=as_float(settings.get("merge_angle_deg"), 35.0, 1.0, 90.0),
        sort_axis=sort_axis,
        sort_desc=bool(settings.get("sort_desc", False)),
        sort_vector_xyz=_wire_sort_vector(settings),
        sort_vector_uv=_wire_sort_vector(settings)[:2],
        representative_mode=representative,
        tracking_enabled=bool(settings.get("tracking_enabled", True)),
        confidence_minimum=as_float(settings.get("confidence_minimum"), 0.45, 0.0, 1.0),
        use_depth_foreground=bool(settings.get("use_depth_foreground", True)),
        use_edge_fallback=bool(settings.get("use_edge_fallback", True)),
        apply_morph_open=bool(settings.get("apply_morph_open", True)),
        apply_morph_close=bool(settings.get("apply_morph_close", True)),
        filter_components=bool(settings.get("filter_components", True)),
        skeleton_enabled=bool(settings.get("skeleton_enabled", True)),
        save_debug_images=bool(settings.get("save_debug_images", False)),
        debug_enabled=True,
    )


def _wire_sort_vector(settings):
    vector = settings.get("sort_vector_xyz")
    if not isinstance(vector, (list, tuple)) or len(vector) != 3:
        return (1.0, 0.0, 0.0)
    return tuple(as_float(value, fallback, -1.0, 1.0)
                 for value, fallback in zip(vector, (1.0, 0.0, 0.0)))


# ===== 이미지 보드에 띄울 화면 종류 =====
# (표시 이름, 키) 순서대로 콤보에 들어간다. 새 화면을 추가하려면 여기에 한 줄 넣고
# _build_view_image()에 분기 하나만 더하면 된다.
VIEW_COLOR = "color"
VIEW_DEPTH = "depth"
VIEW_BINARY = "binary"
VIEW_DEBUG = "debug"
VIEW_CHOICES = [
    ("컬러맵", VIEW_COLOR),
    ("뎁스맵", VIEW_DEPTH),
    ("검출 마스크", VIEW_BINARY),
    ("디버그", VIEW_DEBUG),
]

_USE_CURRENT_ROI = object()


# ======================================================
# 블롭 분석
# ======================================================
class BlobAnalyzer:
    """OpenCV의 SimpleBlobDetector를 활용한 블롭 분석."""

    def __init__(self):
        # UI에서 바꿀 수 있는 값은 인스턴스에 둔다(모듈 상수는 초기값 역할).
        self.min_threshold = MIN_THRESHOLD
        self.max_threshold = MAX_THRESHOLD
        self.threshold_step = THRESHOLD_STEP
        self.min_dist_between_blobs = MIN_DIST_BETWEEN_BLOBS
        self.filters = default_filter_states()
        self.detector = self._create_detector()

    def set_thresholds(self, min_threshold, max_threshold, step):
        """임계값 구간과 간격을 바꾼다. 실제로 달라졌으면 검출기를 다시 만든다.

        Returns:
            bool: 값이 바뀌어 검출기를 새로 만들었는지
        """
        new = (int(min_threshold), int(max_threshold), int(step))
        if new == (self.min_threshold, self.max_threshold, self.threshold_step):
            return False
        self.min_threshold, self.max_threshold, self.threshold_step = new
        self.detector = self._create_detector()
        return True

    def set_min_dist(self, distance):
        """블롭 병합 거리를 바꾼다. 달라졌으면 검출기를 다시 만든다."""
        if float(distance) == self.min_dist_between_blobs:
            return False
        self.min_dist_between_blobs = float(distance)
        self.detector = self._create_detector()
        return True

    def set_filter(self, key, enabled, value):
        """필터 하나의 켜짐 여부와 기준값을 바꾼다. 달라졌으면 검출기를 다시 만든다."""
        state = self.filters[key]
        if state["enabled"] == bool(enabled) and state["value"] == float(value):
            return False
        state["enabled"] = bool(enabled)
        state["value"] = float(value)
        self.detector = self._create_detector()
        return True

    def _create_detector(self):
        """SimpleBlobDetector 생성 및 파라미터 설정."""
        params = cv2.SimpleBlobDetector_Params()

        # 이진화 임계값 구간을 thresholdStep 간격으로 훑는다.
        params.minThreshold = float(self.min_threshold)
        params.maxThreshold = float(self.max_threshold)
        params.thresholdStep = float(self.threshold_step)

        # 구간/간격 조합에 따라 훑을 레벨이 하나뿐일 수 있다(예: min=50, max=100,
        # step=60). 그때 minRepeatability 기본값 2는 만족이 불가능해서 OpenCV가
        # detect() 단계에서 예외를 던지므로, 레벨 수에 맞춰 낮춰 준다.
        levels = len(
            range(self.min_threshold, self.max_threshold, max(1, self.threshold_step))
        )
        params.minRepeatability = 1 if levels <= 1 else 2

        # 이보다 가까운 블롭은 하나로 합친다
        params.minDistBetweenBlobs = float(self.min_dist_between_blobs)

        # 타겟 필터링: 각 필터는 UI에서 개별로 켜고 끌 수 있다.
        # 기준값은 0<min<=max 를 만족해야 하고 이 검사는 필터를 꺼도 수행되므로,
        # 저장 파일이 손으로 고쳐진 경우까지 대비해 여기서 한 번 더 여민다.
        area = self.filters[FILTER_AREA]
        min_area = max(1.0, area["value"])
        params.filterByArea = area["enabled"]
        params.minArea = min_area
        params.maxArea = max(float(MAX_BLOB_AREA), min_area)

        circularity = self.filters[FILTER_CIRCULARITY]
        params.filterByCircularity = circularity["enabled"]
        params.minCircularity = max(RATIO_FLOOR, circularity["value"])

        convexity = self.filters[FILTER_CONVEXITY]
        params.filterByConvexity = convexity["enabled"]
        params.minConvexity = max(RATIO_FLOOR, convexity["value"])

        inertia = self.filters[FILTER_INERTIA]
        params.filterByInertia = inertia["enabled"]
        params.minInertiaRatio = max(RATIO_FLOOR, inertia["value"])

        color = self.filters[FILTER_COLOR]
        params.filterByColor = color["enabled"]
        params.blobColor = int(color["value"])

        return cv2.SimpleBlobDetector_create(params)

    def analyze(self, bgr_image, roi=_USE_CURRENT_ROI):
        """이미지에서 홀(블롭)을 탐지한다.

        ROI가 설정돼 있으면 그 영역만 잘라 분석하되, 반환 좌표는 항상
        원본 이미지 기준이라 호출한 쪽에서 보정할 필요가 없다.

        Args:
            bgr_image: BGR 컬러 이미지

        Returns:
            holes: [(x, y, radius), ...]  원본 이미지 좌표
            debug_info: 분석 정보 딕셔너리
        """
        if roi is _USE_CURRENT_ROI:
            target, offset_x, offset_y = roi_state.crop(bgr_image)
            roi_applied = roi_state.is_set()
        else:
            target, offset_x, offset_y = bgr_image, 0, 0
            roi_applied = False
            if isinstance(roi, (list, tuple)) and len(roi) == 4:
                try:
                    x, y, width, height = (int(value) for value in roi)
                except (TypeError, ValueError):
                    pass
                else:
                    image_height, image_width = bgr_image.shape[:2]
                    x = max(0, min(x, image_width))
                    y = max(0, min(y, image_height))
                    width = min(width, image_width - x)
                    height = min(height, image_height - y)
                    if width > 0 and height > 0:
                        target = bgr_image[y:y + height, x:x + width]
                        offset_x, offset_y = x, y
                        roi_applied = True

        # 그레이스케일 변환
        gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

        # 적응형 임계값으로 이진화
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            ADAPTIVE_THRESHOLD_BLOCK_SIZE,
            ADAPTIVE_THRESHOLD_C,
        )

        # 노이즈 제거 (작은 점 제거)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # 블롭 탐지 후 잘라낸 만큼 좌표를 되돌린다.
        keypoints = self.detector.detect(binary)
        holes = [
            (
                int(round(kp.pt[0])) + offset_x,
                int(round(kp.pt[1])) + offset_y,
                kp.size / 2.0,
            )
            for kp in keypoints
        ]

        # 이진화 이미지도 원본 크기 캔버스에 되붙인다. 화면에 띄울 때 다른 화면
        # (컬러맵/뎁스맵)과 좌표계가 같아야 ROI 오버레이가 어긋나지 않는다.
        binary_full = np.zeros(bgr_image.shape[:2], dtype=np.uint8)
        crop_h, crop_w = binary.shape[:2]
        binary_full[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w] = binary

        debug_info = {
            "binary": binary_full,
            "blob_count": len(holes),
            "roi_applied": roi_applied,
        }
        return holes, debug_info


# ======================================================
# 결과 시각화
# ======================================================
class BlobResultVisualizer:
    """블롭 분석 결과를 이미지에 시각화."""

    @staticmethod
    def draw_holes(
        image,
        holes,
        poses=None,
        camera_matrix=None,
        dist_coeffs=None,
        hole_masks=None,
    ):
        """이미지에 탐지된 홀을 그린다.

        Args:
            image: BGR 이미지
            holes: [(x, y, radius), ...] 원본 이미지 좌표
            poses: holes와 같은 길이의 HolePose 리스트(못 구한 자리는 None).
                주면 홀마다 좌표축과 z/자세 값을 함께 그린다.
            camera_matrix, dist_coeffs: 좌표축을 영상에 투영할 카메라 파라미터
            hole_masks: holes와 같은 길이의 bool 마스크 목록(없는 자리는 None).
                주면 모델이 실제로 문 경계를 원 안쪽에 함께 그린다. 블랍/템플릿
                검출은 경계를 내지 않으므로 None 이고, 그때는 원만 그린다.

        Returns:
            result: 홀이 그려진 BGR 이미지
        """
        result = image.copy()
        poses = poses or []
        hole_masks = hole_masks or []

        for index, (x, y, radius) in enumerate(holes):
            # 실제 경계를 먼저 깔고 원을 그 위에 얹는다. 순서를 바꾸면 얇은
            # 경계선이 굵은 원 밑에 깔려 안 보인다.
            mask = hole_masks[index] if index < len(hole_masks) else None
            BlobResultVisualizer._draw_hole_contour(result, mask)

            # 원 그리기 (초록색)
            cv2.circle(result, (x, y), int(round(radius)), (0, 255, 0), 2)

            # 중심점 그리기 (빨강색)
            cv2.circle(result, (x, y), 3, (0, 0, 255), -1)

            pose = poses[index] if index < len(poses) else None
            lines = [f"({x}, {y})"]
            if pose is not None:
                # 픽셀 좌표 옆에 실제 거리와 자세를 붙인다. 축 그림만으로는
                # 값을 읽을 수 없고, 값만으로는 어느 쪽으로 기울었는지 안 보인다.
                lines[0] += f"  z {pose.z:.1f}"
                lines.append(f"r ({pose.rx:.1f}, {pose.ry:.1f}, {pose.rz:.1f})")
                BlobResultVisualizer._draw_axes(
                    result, pose, radius, camera_matrix, dist_coeffs
                )

            for offset, text in enumerate(lines):
                cv2.putText(
                    result,
                    text,
                    (x + 10, y - 10 + offset * 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )

        return result

    @staticmethod
    def _draw_hole_contour(image, mask):
        """모델이 문 홀 경계를 그린다. 마스크가 없거나 안 맞으면 아무것도 안 한다.

        draw_normal_points 와 달리 안쪽을 덮지 않는다. 홀은 작아서 덮으면
        빨강 중심점과 좌표축이 가려진다.
        """
        if mask is None or getattr(mask, "ndim", 0) != 2:
            return
        if mask.shape != image.shape[:2] or not mask.any():
            return
        contours, _hierarchy = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            cv2.drawContours(
                image, contours, -1, HOLE_CONTOUR_COLOR, HOLE_CONTOUR_WIDTH
            )

    @staticmethod
    def _draw_axes(image, pose, radius, camera_matrix, dist_coeffs):
        """홀 자세를 좌표축으로 그린다 (빨강 X, 초록 Y, 파랑 Z=법선).

        z축은 카메라 쪽으로 뻗으므로 정면 홀에서는 점처럼 짧게 보이고, 면이
        기울수록 길게 누워 보인다. 그 자체가 기울기를 읽는 단서가 된다.
        """
        if camera_matrix is None:
            return
        matrix = np.asarray(camera_matrix, dtype=np.float64)
        focal = float(matrix[0, 0])
        if focal <= 0.0:
            return

        # 픽셀 반지름을 그 거리에서의 실제 크기(mm)로 되돌려 축 길이를 정한다.
        length_mm = float(
            np.clip(
                radius * pose.z / focal * AXIS_LENGTH_SCALE, AXIS_MIN_MM, AXIS_MAX_MM
            )
        )
        coeffs = (
            np.zeros(5)
            if dist_coeffs is None
            else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
        )
        # 자세는 rx/ry/rz 에서 세운다. pose_matrix() 는 normal 로 자세를 다시
        # 만드는데, 그 경로는 z축 둘레 회전(rz)을 표현할 수 없어서 옆에 찍히는
        # 숫자와 화면의 축이 어긋난다(회전 오프셋이 축에 안 보이던 이유).
        rotation_vector, _ = cv2.Rodrigues(
            rpy_zyx_deg_to_matrix(pose.rx, pose.ry, pose.rz)
        )
        translation = np.array([pose.x, pose.y, pose.z], dtype=np.float64)
        try:
            cv2.drawFrameAxes(
                image, matrix, coeffs, rotation_vector, translation, length_mm, 2
            )
        except cv2.error:   # 축이 화면 밖으로 크게 벗어난 경우 등
            pass


# ======================================================
# 페이지
# ======================================================
class HoleDetectionPage(BasePage):
    """블롭 분석을 활용한 홀 탐지 페이지."""

    title = "Hole Detection"
    nav_icon = "holes"
    expand_content = True  # 결과 보드가 남는 공간을 모두 채우도록

    def __init__(self, parent=None):
        super().__init__(parent)

        self._window = None               # bind_window()로 주입되는 MainWindow
        self._pose_page = None            # bind_pose_page()로 주입되는 Pose Setting
        self._last_captured_image = None  # 마지막 캡쳐 이미지 (BGR 원본)
        # 네모 드래그를 누가 기다리는지 ("template" / "hole" / None).
        # 템플릿 저장과 홀 티칭이 같은 보드의 같은 드래그를 쓰므로 갈라야 한다.
        self._drag_target = None
        self._last_depth = None           # 같은 시점의 뎁스 (없으면 None)
        self._last_binary = None          # 마지막 분석의 이진화 이미지 (원본 크기)
        self._last_holes = []             # 마지막 분석 결과 [(x, y, radius), ...]
        self._last_debug_info = {}        # 마지막 분석의 공통 디버그 정보
        self._last_debug_frames = []      # 디버그 뷰 contact sheet 원본 단계들
        self._last_capture_save_note = "" # Camera Capture 자동 저장 결과
        self._debug_dialogs = []          # 디버그 타일 확대 창 참조
        # 홀마다 모델이 문 경계. 검출 방식이 경계를 내지 않으면 빈 목록이다.
        self._last_hole_masks = []
        self._last_intrinsics = None      # 캡쳐 시점의 (camera_matrix, dist_coeffs)
        # 촬영마다 하나 오른다. Pose Setting 이 '내가 들고 있는 것이 최신인가'를
        # 이 번호로 가린다 — 이미지 배열을 비교하면 촬영마다 300만 화소를 훑어야
        # 하고, 같은 장면을 두 번 찍으면 새 촬영을 놓친다.
        self._capture_seq = 0
        self._depth_scale_mm = 1.0        # 캡쳐 시점의 뎁스 값 1당 mm
        self._analyzer = BlobAnalyzer()
        self._wire_tracker = WireTracker()
        self._syncing = False             # 위젯끼리 값을 맞출 때 되먹임 방지
        # 필터별 {켜짐, 기준값}. 화면에는 고른 필터 하나만 보이지만 상태는 전부 남는다.
        self._filter_states = default_filter_states()

        # 값을 연달아 바꿀 때 매번 재분석하면 버벅이므로, 마지막 변경 후
        # PARAM_APPLY_DELAY_MS 만큼 잠잠해지면 한 번만 돌린다.
        self._apply_timer = QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(PARAM_APPLY_DELAY_MS)
        self._apply_timer.timeout.connect(self._apply_active_params)

        self.content_layout.addLayout(self._build_controls())
        self.content_layout.addLayout(self._build_status_row())
        self.content_layout.addLayout(self._build_elapsed_row())

        # 컬러 결과 보드(좌) + 설정 패널(우)
        board_row = QHBoxLayout()
        board_row.setSpacing(CONTENT_SPACING)

        self.result_board = ResultBoard("이미지 캡쳐 버튼을 누르면 여기에 표시됩니다.")
        self.result_board.roi_changed.connect(self._on_roi_changed)
        self.result_board.region_selected.connect(self._on_region_dragged)
        self.result_board.image_clicked.connect(self._on_result_board_clicked)

        # 보드와 패널을 **붙여 한 덩어리로** 두고, 그 그룹을 가운데 놓는다.
        #
        # 보드는 이미지 비율까지만 자라므로(_limit_to_aspect) 남는 폭이 생긴다.
        # 그 폭을 보드와 패널 **사이**에 두면 둘이 멀리 떨어져 가운데가 빈 구멍처럼
        # 보인다(면 보드를 걷어내고 실제로 570px 가 벌어졌다). 스트레치를 양 끝에만
        # 두면 여백이 화면 가장자리로 밀려나고 둘은 붙어 있게 된다.
        #
        # 보드 쪽 스트레치를 크게 둬야 상한 아래에서는 보드가 여백 없이 꽉 찬다.
        board_row.addStretch(1)
        board_row.addWidget(self.result_board, 100)
        board_row.addWidget(self._build_setting_panel())
        board_row.addStretch(1)

        self.content_layout.addLayout(board_row, 1)

        self._reload_template_list()
        self._reload_sam2_list()
        # 프로파일 불러오기는 여기서 하지 않는다 — init_profiles() 를 MainWindow 가
        # 두 페이지를 이어 준 **뒤에** 부른다. 여기서 하면 첫 실행에 만들어지는
        # 기본 프로파일에 포즈 키(normal_source/normal_points/approach_offset)가
        # 빠진다. 그때는 Pose Setting 이 아직 안 붙어서 값을 물어볼 데가 없다.

    def bind_window(self, window):
        """MainWindow가 페이지 등록 시 자신을 주입한다(PythonUI._load_pages).

        stack.addWidget()가 페이지를 재부모화하므로 self.parent()로는
        MainWindow에 닿을 수 없다. 그래서 이 훅으로 참조를 받아 둔다.
        """
        self._window = window

    def bind_pose_page(self, page):
        """Pose Setting 페이지를 주입받는다(PythonUI._link_pages).

        프로파일 파일 하나를 둘이 나눠 쓰므로, 저장할 때 저쪽 값을 받아 오고
        불러올 때 저쪽에 넘겨 준다. 스택을 isinstance 로 훑지 않고 명시적으로
        받는 이유: Pose Setting 이 이 모듈을 import 하므로 반대로 import 하면
        순환이 된다. MainWindow 는 이미 둘 다 알고 있어 거기서 잇는 편이 싸다.

        아직 이어지지 않았어도(테스트/오퍼레이터 단독 경로) 동작해야 하므로,
        쓰는 자리마다 None 을 견딘다.

        여기서 프로파일을 읽지 않는다. 잇는 일과 읽는 일을 갈라 두면 순서가
        분명해진다 — MainWindow 가 둘을 다 이은 뒤 init_profiles() 를 부른다.
        """
        self._pose_page = page

    # ===== Pose Setting 페이지가 쓰는 창구 =====
    def camera_page(self):
        """Camera Capture 페이지. 없으면 None (카메라 페이지가 빠진 구성)."""
        return self._find_camera_page()

    def profile_names(self):
        """지금 콤보에 들어 있는 프로파일 이름 목록."""
        return [self.profile_combo.itemData(i) for i in range(self.profile_combo.count())]

    def select_profile(self, name):
        """프로파일을 고른다. Pose Setting 콤보에서 바꿨을 때 불린다."""
        index = self.profile_combo.findData(name)
        if index >= 0 and index != self.profile_combo.currentIndex():
            self.profile_combo.setCurrentIndex(index)

    def save_current_profile(self):
        """지금 세팅을 저장한다. → (성공 여부, 화면에 띄울 문구)

        Pose Setting 의 '저장'도 이리로 온다. 검출 값과 포즈 값이 한 파일에
        들어가므로 저장하는 자리도 하나여야 한다.
        """
        name = self.current_profile()
        if name is None:
            return False, "저장할 세팅이 선택되어 있지 않습니다."
        missing = self._missing_detection_input()
        if missing:
            return False, missing
        ok, result = save_profile(name, self._collect_settings())
        if ok:
            return True, f"'{name}' 저장됨 — {display_path(result)}"
        return False, f"세팅 저장 실패: {result}"

    def hole_positions(self):
        """마지막 분석의 홀 픽셀 좌표 목록 [(x, y, radius), ...].

        이 페이지의 답은 여기까지다. 여기에 방향과 높이를 붙여 (x,y,z,rx,ry,rz)
        로 만드는 것은 Pose Setting 의 hole_poses() 다.
        """
        return list(self._last_holes)

    def capture_sequence(self):
        """지금까지 몇 번 찍었는지. 한 번도 안 찍었으면 0.

        Pose Setting 이 탭으로 들어올 때 '그새 새로 찍혔나'를 이 값으로 가린다.
        """
        return self._capture_seq

    def capture_context(self):
        """마지막 캡쳐와 검출 결과를 통째로 넘긴다. 아직 없으면 None.

        Pose Setting 이 '검출 결과 가져오기'로 이어받는 자리다. 뎁스와 내부
        파라미터를 함께 넘기는 이유: 포즈를 세우려면 셋이 **같은 시점**의 것이라야
        한다. 저쪽이 카메라에 다시 물어보면 그새 장면이 바뀔 수 있다.
        """
        if self._last_captured_image is None:
            return None
        return {
            "image": self._last_captured_image,
            "depth": self._last_depth,
            "intrinsics": self._last_intrinsics,
            "depth_scale_mm": self._depth_scale_mm,
            "holes": list(self._last_holes),
            "hole_masks": list(self._last_hole_masks),
            "sequence": self._capture_seq,
        }

    def detect_holes(self, bgr_image):
        """지금 화면의 검출 방식으로 홀을 찾는다. → (홀 목록, debug_info)

        Pose Setting 이 직접 찍었을 때 검출만 이리로 위임하는 자리다. 검출
        방식과 파라미터는 이 페이지가 정본이므로, 저쪽이 복사해 가면 두 벌이
        어긋난다.
        """
        method = self.method_combo.currentData()
        if method == DETECTION_TEMPLATE:
            template = load_template(self.template_combo.currentData())
            if template is None:
                raise ValueError("사용할 템플릿을 선택하거나 새로 지정하세요.")
            return TemplateMatcher(
                template, self.template_threshold.value()
            ).analyze(bgr_image, roi=roi_state.get())
        if method in SAM2_METHODS:
            return self._sam2_detector().analyze(bgr_image, roi=roi_state.get())
        if method == DETECTION_WIRE:
            result = WireDetector(self._wire_config(), tracker=self._wire_tracker).analyze(
                bgr_image,
                depth=self._last_depth,
                intrinsics=self._last_intrinsics,
                depth_scale_mm=self._depth_scale_mm,
                roi=roi_state.get(),
            )
            return result.holes, _wire_debug_info(result)
        return self._analyzer.analyze(bgr_image)

    def visualize_for_operator(
        self,
        bgr_image,
        profile_name,
        depth=None,
        intrinsics=None,
        depth_scale_mm=1.0,
        previous_poses=None,
    ):
        """지정한 블랍 세팅으로 오퍼레이터 촬영 이미지를 시각화한다.

        Returns:
            (홀이 표시된 BGR 이미지, 검출된 홀 목록, 포즈 목록, 포즈 목록,
             법선 문구)

            셋째와 넷째는 같은 목록이다. 260910 에 '포즈 조정'이 사라져 '조정
            전/후' 구분이 없어졌지만, 부르는 쪽(오퍼레이터) 계약을 그대로 두려고
            자리는 남겨 두었다.

            다섯째는 법선을 어디서 받았는지와 세 점이 얼마나 벌어져 있는지를
            담은 문구다. 오퍼레이터에는 뎁스 보드가 없어 눈으로 볼 방법이
            없으므로, 이것을 로그로 남겨야 자세가 튈 때 원인을 짚을 수 있다.
        """
        data = load_profile(profile_name)
        if not isinstance(data, dict):
            raise RuntimeError(f"'{profile_name}' 비전 세팅을 읽을 수 없습니다.")

        method = data.get("detection_method")
        if method == DETECTION_SAM2_VIDEO:
            holes, debug_info = sam2_video_detector(data.get("sam2")).analyze(
                bgr_image, roi=data.get("roi")
            )
        elif method == DETECTION_TEMPLATE:
            template_settings = data.get("template_matching") or {}
            template_name = template_settings.get("name")
            template = load_template(template_name)
            if template is None:
                raise RuntimeError(f"'{template_name or ''}' 템플릿을 읽을 수 없습니다.")
            matcher = TemplateMatcher(
                template,
                as_float(
                    template_settings.get("threshold"),
                    DEFAULT_MATCH_THRESHOLD,
                    0.0,
                    1.0,
                ),
            )
            holes, debug_info = matcher.analyze(bgr_image, roi=data.get("roi"))
        elif method == DETECTION_WIRE:
            result = WireDetector(_wire_config_from_settings(data)).analyze(
                bgr_image,
                depth=depth,
                intrinsics=intrinsics,
                depth_scale_mm=depth_scale_mm,
                roi=data.get("roi"),
            )
            holes, debug_info = result.holes, _wire_debug_info(result)
        else:
            analyzer = BlobAnalyzer()
            threshold = data.get("threshold")
            if isinstance(threshold, dict):
                analyzer.set_thresholds(
                    as_int(threshold.get("min"), MIN_THRESHOLD),
                    as_int(threshold.get("max"), MAX_THRESHOLD),
                    max(1, as_int(threshold.get("step"), THRESHOLD_STEP)),
                )
            analyzer.set_min_dist(
                as_float(
                    data.get("min_dist_between_blobs"),
                    MIN_DIST_BETWEEN_BLOBS,
                    MIN_DIST_MIN,
                    MIN_DIST_MAX,
                )
            )
            filters = data.get("filters")
            if isinstance(filters, dict):
                for spec in FILTER_SPECS:
                    saved = filters.get(spec.key)
                    if not isinstance(saved, dict):
                        continue
                    analyzer.set_filter(
                        spec.key,
                        bool(saved.get("enabled", spec.default_enabled)),
                        as_float(
                            saved.get("value"),
                            spec.default_value,
                            spec.minimum,
                            spec.maximum,
                        ),
                    )
            holes, debug_info = analyzer.analyze(bgr_image, roi=data.get("roi"))
        camera_matrix, dist_coeffs = intrinsics or (None, None)
        color_size = (bgr_image.shape[1], bgr_image.shape[0])

        # 자세를 어디서 받을지는 **이 프로파일**이 정한다. 촬영 보드마다 다른
        # 프로파일을 쓰므로 보드마다 방식이 달라도 된다.
        source = normal_source_from_settings(data)
        mode = source["mode"]

        masks, note, taught_boxes = [], "", []
        reference = None
        if mode == NORMAL_MODE_POINTS:
            # 매칭점은 촬영 한 장에 한 번만 찾는다. 여기서 나오는 것은 '법선을
            # 어디서 받을지'이고, 홀 검출과 높이 계산은 건드리지 않는다.
            masks, note, taught_boxes = normal_points_masks(
                normal_points_from_settings(data), bgr_image
            )
            reference, spread = normal_plane_from_masks(
                masks,
                depth,
                camera_matrix,
                dist_coeffs,
                color_size=color_size,
                depth_scale_mm=depth_scale_mm,
            )
            if reference is not None:
                # 세팅 화면의 상태문구와 같은 말을 쓴다(pose_pipeline.spread_note).
                tail = spread_note(spread)
                if tail:
                    note = f"{note} · {tail}" if note else tail
        elif mode == NORMAL_MODE_FLANGE:
            note = "자세: 플랜지 고정 (x·y·z만)"
        else:
            note = f"자세: 홀 주변 {source['ring_radius_mm']:.0f}mm 영역"

        poses = estimate_hole_poses(
            holes,
            depth,
            camera_matrix,
            dist_coeffs,
            color_size=color_size,
            depth_scale_mm=depth_scale_mm,
            surface_plane=reference,
            mode=mode,
            ring_radius_mm=source["ring_radius_mm"],
        )
        # 이전 촬영과 섞지 않는다(260908 사용자 결정). previous_poses 는 받기만
        # 하고 쓰지 않는다 — 부르는 쪽 계약을 깨지 않으려고 남겨 둔다.
        #
        # 260910 에 '포즈 조정'이 사라졌다. 검출한 자세를 눈으로 보며 더 트는
        # 값이었는데, 로봇으로 나가는 보정은 '진입 각도 오프셋' 하나로 충분하고
        # 두 오프셋이 겹치면 어느 쪽이 얼마를 돌렸는지 가릴 수 없었다.
        raw_poses = poses

        # 고른 방식이 실제로 보는 자리만 그린다.
        overlaid = bgr_image
        if mode == NORMAL_MODE_POINTS:
            # 매칭점을 먼저 깔고 홀을 그 위에 그린다. 순서를 바꾸면 옅은 덮개가
            # 홀 원과 좌표축을 가린다.
            overlaid = draw_normal_points(bgr_image, masks, taught_boxes)
        elif mode == NORMAL_MODE_RING:
            overlaid = draw_ring_regions(
                bgr_image, holes, poses, camera_matrix, source["ring_radius_mm"]
            )
        visualized = BlobResultVisualizer.draw_holes(
            overlaid,
            holes,
            poses,
            camera_matrix,
            dist_coeffs,
            hole_masks=debug_info.get("hole_masks"),
        )
        wires = debug_info.get("wire_candidates")
        if wires:
            visualized = draw_wires(visualized, wires)
        return visualized, holes, poses, raw_poses, note

    # ===== UI 구성 =====
    def _build_controls(self):
        row = QHBoxLayout()
        row.setSpacing(CONTENT_SPACING)

        self.method_combo = QComboBox()
        self.method_combo.setObjectName("viewCombo")
        self.method_combo.addItem("블랍 검출", DETECTION_BLOB)
        self.method_combo.addItem("템플릿 매칭", DETECTION_TEMPLATE)
        self.method_combo.addItem("SAM2 비디오 추적", DETECTION_SAM2_VIDEO)
        self.method_combo.addItem("Wire Detection", DETECTION_WIRE)
        # 항목은 남겨 둔다. 왜 못 쓰는지 보이는 편이 항목이 사라지는 것보다 낫다.
        for method, reason in (
            (DETECTION_SAM2_VIDEO, sam2_video_tracker.unavailable_reason()),
        ):
            if not reason:
                continue
            index = self.method_combo.findData(method)
            self.method_combo.model().item(index).setEnabled(False)
            self.method_combo.setItemData(index, reason, Qt.ToolTipRole)
        self.method_combo.currentIndexChanged.connect(self._on_method_changed)

        self.btn_capture = QPushButton("이미지 캡쳐")
        self.btn_capture.setObjectName("primaryBtn")
        self.btn_capture.clicked.connect(self._on_capture_clicked)

        self.btn_roi = QPushButton("ROI 설정")
        self.btn_roi.setObjectName("roiToggle")
        self.btn_roi.setCheckable(True)
        self.btn_roi.setEnabled(False)   # 이미지가 있어야 좌표를 잡을 수 있다
        self.btn_roi.toggled.connect(self._on_roi_mode_toggled)

        self.btn_template_region = QPushButton("템플릿 지정")
        self.btn_template_region.setObjectName("roiToggle")
        self.btn_template_region.setCheckable(True)
        self.btn_template_region.setEnabled(False)
        self.btn_template_region.toggled.connect(self._on_template_mode_toggled)

        # 이미지 보드에 어떤 화면을 띄울지 고르는 콤보.
        self.view_combo = QComboBox()
        self.view_combo.setObjectName("viewCombo")
        for label, key in VIEW_CHOICES:
            self.view_combo.addItem(label, key)
        self.view_combo.setToolTip(
            "디버그를 고르면 검출 중간 단계 이미지를 한 장으로 모아 봅니다."
        )
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)

        # 현재 세팅을 Settings/<이름>.json 으로 남긴다(다음 실행에서 복원).
        self.btn_save = QPushButton("저장")
        self.btn_save.setObjectName("secondaryBtn")
        self.btn_save.clicked.connect(self._on_save_clicked)

        # 체결 위치마다 세팅이 다르므로 프로파일을 골라 쓴다.
        self.profile_combo = QComboBox()
        self.profile_combo.setObjectName("profileCombo")
        self.profile_combo.currentIndexChanged.connect(self._on_profile_changed)

        self.btn_add_profile = QPushButton("+")
        self.btn_add_profile.setObjectName("iconBtn")
        self.btn_add_profile.setToolTip("현재 값으로 새 세팅 만들기")
        self.btn_add_profile.clicked.connect(self._on_add_profile_clicked)

        self.btn_remove_profile = QPushButton("−")
        self.btn_remove_profile.setObjectName("iconBtn")
        self.btn_remove_profile.setToolTip("선택한 세팅 삭제")
        self.btn_remove_profile.clicked.connect(self._on_remove_profile_clicked)

        row.addWidget(self.method_combo)
        row.addWidget(self.btn_capture)
        row.addWidget(self.btn_roi)
        row.addWidget(self.btn_template_region)
        row.addWidget(self.view_combo)
        row.addStretch()
        # 세팅을 고르고(+/-로 관리하고) 마지막에 저장하는 흐름대로 왼쪽부터 배치
        row.addWidget(self.profile_combo)
        row.addWidget(self.btn_add_profile)
        row.addWidget(self.btn_remove_profile)
        row.addWidget(self.btn_save)
        return row

    def _build_setting_panel(self):
        """이미지 보드 오른쪽 패널: 방식별 파라미터 + 플랜지 포즈.

        검출 방식을 바꿔도 **보드 크기가 변하지 않게** 하려고 파라미터를 보드
        위가 아니라 여기에 세운다(setting_panel 모듈 설명 참고). 패널 폭이
        고정이므로 어느 방식을 골라도 보드가 차지하는 자리는 그대로다.
        """
        panel = SettingPanel()
        # 고른 방식에 해당하는 그룹만 보인다(_update_detection_controls).
        self._blob_group = panel.add_group(self._build_blob_group())
        self._template_group = panel.add_group(self._build_template_group())
        self._sam2_group = panel.add_group(self._build_sam2_group())
        self._wire_group = panel.add_group(self._build_wire_group())
        panel.add_group(self._build_flange_group())
        return panel

    def _build_sam2_group(self):
        """SAM2 그룹: [모델 ▼] [홀 지정] 기준 모양 [▼] [삭제].

        '홀 지정'을 누르고 이미지에서 홀을 네모로 드래그하면 그 홀을 티칭한다.
        티칭 결과(홀 주변을 자른 기준 모양)는 템플릿과 같은 자리에 저장한다.
        둘 다 '이 모양을 다시 찾아라'는 같은 뜻이라 창고를 나눌 이유가 없다.
        """
        group = SettingGroup("Parameter Setting")

        model_label = QLabel("SAM2 모델")
        model_label.setObjectName("paramLabel")
        # 이 줄은 두 SAM2 방식이 함께 쓴다. 도움말은 고른 방식에 맞게
        # _update_method_rows 가 바꿔 준다.
        self._sam2_model_label = model_label

        self.sam2_model_combo = QComboBox()
        self.sam2_model_combo.setObjectName("sam2ModelCombo")
        for name in sam2_detector.list_models():
            self.sam2_model_combo.addItem(sam2_detector.model_label(name), name)
        # 어느 장치에서 도는지 여기에 띄운다. 휠에 이 GPU 커널이 없으면 CPU 로
        # 물러서는데, 그때 무엇을 깔아야 하는지가 이 문구에 그대로 나온다.
        self.sam2_model_combo.setToolTip(
            "클수록 정확하지만 느리다.\n"
            f"지금 연산 장치: {sam2_detector.device_note()}"
        )
        self.sam2_model_combo.currentIndexChanged.connect(self._on_sam2_model_changed)

        self.btn_sam2_teach = QPushButton("홀 지정")
        self.btn_sam2_teach.setObjectName("roiToggle")
        self.btn_sam2_teach.setCheckable(True)
        self.btn_sam2_teach.setEnabled(False)   # 이미지가 있어야 드래그할 수 있다
        self.btn_sam2_teach.setToolTip(SAM2_HINT)
        self.btn_sam2_teach.toggled.connect(self._on_sam2_teach_toggled)

        reference_label = QLabel("기준 모양")
        reference_label.setObjectName("paramLabel")

        self.sam2_reference_combo = QComboBox()
        self.sam2_reference_combo.setObjectName("profileCombo")
        self.sam2_reference_combo.setToolTip(
            "티칭해 둔 홀. 오퍼레이터 모드는 이 모양으로 같은 홀을 찾는다."
        )
        self.sam2_reference_combo.currentIndexChanged.connect(
            self._on_sam2_reference_changed
        )

        self.btn_sam2_remove = QPushButton("기준 삭제")
        self.btn_sam2_remove.setObjectName("secondaryBtn")
        self.btn_sam2_remove.clicked.connect(self._on_remove_template_clicked)

        # 좁은 패널이라 라벨 아래에 입력을 놓는다(가로로 붙이면 잘린다).
        group.add_widget(model_label)
        group.add_widget(self.sam2_model_combo)
        group.add_widget(self.btn_sam2_teach)
        group.add_widget(reference_label)
        group.add_widget(self.sam2_reference_combo)
        group.add_widget(self.btn_sam2_remove)
        return group

    def _build_template_group(self):
        group = SettingGroup("Parameter Setting")

        template_label = QLabel("Template")
        template_label.setObjectName("paramLabel")

        self.template_combo = QComboBox()
        self.template_combo.setObjectName("profileCombo")
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)

        self.btn_remove_template = QPushButton("템플릿 삭제")
        self.btn_remove_template.setObjectName("secondaryBtn")
        self.btn_remove_template.clicked.connect(self._on_remove_template_clicked)

        threshold_label = QLabel("Match")
        threshold_label.setObjectName("paramLabel")

        self.template_threshold = QDoubleSpinBox()
        self.template_threshold.setObjectName("paramSpin")
        self.template_threshold.setRange(0.0, 1.0)
        self.template_threshold.setDecimals(2)
        self.template_threshold.setSingleStep(0.05)
        self.template_threshold.setValue(DEFAULT_MATCH_THRESHOLD)
        self.template_threshold.setAlignment(Qt.AlignCenter)
        self.template_threshold.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.template_threshold.valueChanged.connect(self._schedule_apply)

        group.add_widget(template_label)
        group.add_widget(self.template_combo)
        group.add_widget(self.btn_remove_template)
        group.row("Match", self.template_threshold)
        return group

    def _build_wire_group(self):
        group = SettingGroup("Wire Detection")

        self.wire_foreground_mode = QComboBox()
        self.wire_foreground_mode.setObjectName("profileCombo")
        self.wire_foreground_mode.addItem("Bin top layer", WIRE_MODE_BIN_TOP)
        self.wire_foreground_mode.addItem("Plane height", WIRE_MODE_PLANE)
        self.wire_foreground_mode.addItem("RGB edge only", WIRE_MODE_EDGE)
        self.wire_foreground_mode.setMinimumWidth(220)
        self.wire_foreground_mode.setToolTip(
            "Bin picking default: use depth to keep the nearest/top wire layer first."
        )
        self.wire_foreground_mode.currentIndexChanged.connect(self._schedule_apply)

        self.wire_min_length = self._make_param_spin(10, 5000, 80)
        self.wire_min_length.valueChanged.connect(self._schedule_apply)

        self.wire_min_area = self._make_param_spin(1, 50000, 120)
        self.wire_min_area.valueChanged.connect(self._schedule_apply)

        self.wire_min_thickness = self._make_param_double(0.5, 200.0, 2.0, 1, " px")
        self.wire_min_thickness.valueChanged.connect(self._schedule_apply)

        self.wire_max_thickness = self._make_param_double(1.0, 300.0, 80.0, 1, " px")
        self.wire_max_thickness.valueChanged.connect(self._schedule_apply)

        self.wire_plane_threshold = self._make_param_double(0.5, 200.0, 18.0, 1, " mm")
        self.wire_plane_threshold.valueChanged.connect(self._schedule_apply)

        self.wire_top_percentile = self._make_param_double(0.1, 50.0, 1.0, 1, " %")
        self.wire_top_percentile.setToolTip(
            "Lower values make the top-layer reference ignore most lower wires."
        )
        self.wire_top_percentile.valueChanged.connect(self._schedule_apply)

        self.wire_depth_min = self._make_param_double(0.0, 3000.0, 70.0, 0, " mm")
        self.wire_depth_min.setToolTip("Ignore depth values closer than this. 0 disables the limit.")
        self.wire_depth_min.valueChanged.connect(self._schedule_apply)

        self.wire_depth_max = self._make_param_double(0.0, 5000.0, 0.0, 0, " mm")
        self.wire_depth_max.setToolTip("Ignore depth values farther than this. 0 disables the limit.")
        self.wire_depth_max.valueChanged.connect(self._schedule_apply)

        self.wire_merge_distance = self._make_param_double(1.0, 200.0, 18.0, 1, " px")
        self.wire_merge_distance.valueChanged.connect(self._schedule_apply)

        self.wire_merge_angle = self._make_param_double(1.0, 90.0, 35.0, 1, "°")
        self.wire_merge_angle.valueChanged.connect(self._schedule_apply)

        self.wire_confidence_min = self._make_param_double(0.0, 1.0, 0.45, 2, "")
        self.wire_confidence_min.setSingleStep(0.05)
        self.wire_confidence_min.valueChanged.connect(self._schedule_apply)

        self.wire_sort_axis = QComboBox()
        self.wire_sort_axis.setObjectName("compactCombo")
        self.wire_sort_axis.setMinimumWidth(110)
        for label, key in (("X", "x"), ("Y", "y"), ("Z", "z"), ("Custom", SORT_CUSTOM)):
            self.wire_sort_axis.addItem(label, key)
        self.wire_sort_axis.currentIndexChanged.connect(self._schedule_apply)

        self.wire_sort_order = QComboBox()
        self.wire_sort_order.setObjectName("compactCombo")
        self.wire_sort_order.setMinimumWidth(110)
        self.wire_sort_order.addItem("오름차순", False)
        self.wire_sort_order.addItem("내림차순", True)
        self.wire_sort_order.currentIndexChanged.connect(self._schedule_apply)

        self.wire_representative = QComboBox()
        self.wire_representative.setObjectName("profileCombo")
        for label, key in (
            ("중심", REP_CENTER),
            ("위 끝점", REP_TOP),
            ("아래 끝점", REP_BOTTOM),
            ("왼쪽 끝점", REP_LEFT),
            ("오른쪽 끝점", REP_RIGHT),
            ("정렬방향 첫점", REP_FIRST_SORT),
        ):
            self.wire_representative.addItem(label, key)
        self.wire_representative.setMinimumWidth(220)
        self.wire_representative.currentIndexChanged.connect(self._schedule_apply)

        self.wire_sort_vec_x = self._make_param_double(-1.0, 1.0, 1.0, 2, "")
        self.wire_sort_vec_y = self._make_param_double(-1.0, 1.0, 0.0, 2, "")
        self.wire_sort_vec_z = self._make_param_double(-1.0, 1.0, 0.0, 2, "")
        for box in (self.wire_sort_vec_x, self.wire_sort_vec_y, self.wire_sort_vec_z):
            box.setMinimumWidth(70)
            box.setSingleStep(0.1)
            box.valueChanged.connect(self._schedule_apply)

        self.wire_tracking_switch = ToggleSwitch()
        self.wire_tracking_switch.setChecked(True)
        self.wire_tracking_switch.toggled.connect(self._schedule_apply)

        self.wire_debug_save_switch = ToggleSwitch()
        self.wire_debug_save_switch.setToolTip("디버그 contact sheet와 wire 메타데이터를 파일로 저장합니다.")
        self.wire_debug_save_switch.toggled.connect(self._schedule_apply)

        self.wire_depth_foreground_switch = ToggleSwitch()
        self.wire_depth_foreground_switch.setChecked(True)
        self.wire_depth_foreground_switch.setToolTip(
            "Use D405 depth foreground. In Bin top mode this keeps the nearest/top layer."
        )
        self.wire_depth_foreground_switch.toggled.connect(self._schedule_apply)

        self.wire_edge_fallback_switch = ToggleSwitch()
        self.wire_edge_fallback_switch.setChecked(True)
        self.wire_edge_fallback_switch.setToolTip(
            "depth foreground를 만들 수 없을 때 컬러 이미지 edge 기반 후보를 사용합니다."
        )
        self.wire_edge_fallback_switch.toggled.connect(self._schedule_apply)

        self.wire_morph_open_switch = ToggleSwitch()
        self.wire_morph_open_switch.setChecked(True)
        self.wire_morph_open_switch.setToolTip("작은 노이즈를 줄이는 morphology open 단계입니다.")
        self.wire_morph_open_switch.toggled.connect(self._schedule_apply)

        self.wire_morph_close_switch = ToggleSwitch()
        self.wire_morph_close_switch.setChecked(True)
        self.wire_morph_close_switch.setToolTip("끊긴 후보 영역을 이어주는 morphology close 단계입니다.")
        self.wire_morph_close_switch.toggled.connect(self._schedule_apply)

        self.wire_area_filter_switch = ToggleSwitch()
        self.wire_area_filter_switch.setChecked(True)
        self.wire_area_filter_switch.setToolTip("Min area보다 작은 연결 성분을 제거합니다.")
        self.wire_area_filter_switch.toggled.connect(self._schedule_apply)

        self.wire_skeleton_switch = ToggleSwitch()
        self.wire_skeleton_switch.setChecked(True)
        self.wire_skeleton_switch.setToolTip("후보 영역을 centerline 형태로 얇게 줄입니다.")
        self.wire_skeleton_switch.toggled.connect(self._schedule_apply)

        group.note(
            "Top layer + wire cluster + XYZ sort."
        )
        group.add_widget(QLabel("Mode"))
        group.add_widget(self.wire_foreground_mode)
        self._add_two_column_grid(
            group,
            (
                ("Min length", self.wire_min_length),
                ("Min area", self.wire_min_area),
                ("Min thick", self.wire_min_thickness),
                ("Max thick", self.wire_max_thickness),
                ("Top band", self.wire_plane_threshold),
                ("Top %", self.wire_top_percentile),
                ("Depth min", self.wire_depth_min),
                ("Depth max", self.wire_depth_max),
                ("Min conf", self.wire_confidence_min),
                ("Merge dist", self.wire_merge_distance),
                ("Merge angle", self.wire_merge_angle),
            ),
        )
        group.add_widget(QLabel("Sort"))
        group.row("", self.wire_sort_axis, self.wire_sort_order)
        group.add_widget(QLabel("Sort vec"))
        group.row("", self.wire_sort_vec_x, self.wire_sort_vec_y, self.wire_sort_vec_z)
        label = QLabel("대표점")
        label.setObjectName("paramLabel")
        group.add_widget(label)
        group.add_widget(self.wire_representative)
        preprocess_label = QLabel("Preprocess")
        preprocess_label.setObjectName("paramLabel")
        group.add_widget(preprocess_label)
        self._add_toggle_grid(
            group,
            (
                ("Top", self.wire_depth_foreground_switch),
                ("Edge", self.wire_edge_fallback_switch),
                ("Open", self.wire_morph_open_switch),
                ("Close", self.wire_morph_close_switch),
                ("Area", self.wire_area_filter_switch),
                ("Skel", self.wire_skeleton_switch),
                ("Track", self.wire_tracking_switch),
                ("Debug", self.wire_debug_save_switch),
            ),
            columns=4,
        )
        return group

    @staticmethod
    def _add_two_column_grid(group, items):
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for index, (label_text, widget) in enumerate(items):
            row = index // 2
            col = index % 2
            label = QLabel(label_text)
            label.setObjectName("paramLabel")
            cell = QVBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(3)
            cell.addWidget(label)
            cell.addWidget(widget)
            grid.addLayout(cell, row, col)
        group.add_layout(grid)
        return grid

    @staticmethod
    def _add_toggle_grid(group, items, columns=4):
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        for col in range(columns):
            grid.setColumnStretch(col, 1)
        for index, (label_text, widget) in enumerate(items):
            row = index // columns
            col = index % columns
            label = QLabel(label_text)
            label.setObjectName("paramLabel")
            label.setAlignment(Qt.AlignCenter)
            if widget.toolTip():
                label.setToolTip(widget.toolTip())
            cell = QVBoxLayout()
            cell.setContentsMargins(0, 0, 0, 0)
            cell.setSpacing(2)
            cell.addWidget(label)
            cell.addWidget(widget, 0, Qt.AlignCenter)
            grid.addLayout(cell, row, col)
        group.add_layout(grid)
        return grid

    def _build_blob_group(self):
        """블롭 검출기 파라미터: Threshold(구간) · Step · MinDist · 타겟 필터링.

        예전에는 두 행(파라미터 / 필터링)으로 보드 위에 있었다. 260910 에 오른쪽
        패널로 옮기면서 한 그룹으로 합쳤다 — 둘 다 '블랍을 어떻게 걸러낼지'라서
        나눠 둘 이유가 없고, 세로로 쌓으면 좁은 폭에 더 잘 들어간다.
        """
        group = SettingGroup("Parameter Setting")

        label = QLabel("Threshold")
        label.setObjectName("paramLabel")

        self.spin_min_threshold = self._make_param_spin(
            THRESHOLD_ABS_MIN, THRESHOLD_ABS_MAX, MIN_THRESHOLD
        )
        self.spin_max_threshold = self._make_param_spin(
            THRESHOLD_ABS_MIN, THRESHOLD_ABS_MAX, MAX_THRESHOLD
        )

        self.threshold_slider = RangeSlider(THRESHOLD_ABS_MIN, THRESHOLD_ABS_MAX)
        self.threshold_slider.set_values(MIN_THRESHOLD, MAX_THRESHOLD)

        # 패널 폭을 넘지 않게 상한을 둔다(패널이 넓어져 보드를 밀지 않도록).
        self.threshold_slider.setMaximumWidth(RANGE_SLIDER_MAX_WIDTH)

        self.spin_threshold_step = self._make_param_spin(
            THRESHOLD_STEP_MIN, THRESHOLD_STEP_MAX, THRESHOLD_STEP
        )

        self.spin_min_dist = self._make_param_spin(
            MIN_DIST_MIN, MIN_DIST_MAX, MIN_DIST_BETWEEN_BLOBS
        )
        self.spin_min_dist.valueChanged.connect(self._schedule_apply)

        # 슬라이더 -> 입력박스 (드래그 중에는 숫자만 따라가고, 놓으면 재분석)
        self.threshold_slider.range_changed.connect(self._on_slider_range_changed)
        self.threshold_slider.range_committed.connect(self._on_slider_range_changed)
        # 입력박스 -> 슬라이더
        self.spin_min_threshold.valueChanged.connect(self._on_threshold_box_changed)
        self.spin_max_threshold.valueChanged.connect(self._on_threshold_box_changed)
        self.spin_threshold_step.valueChanged.connect(self._schedule_apply)

        self._sync_threshold_bounds()

        group.add_widget(label)
        group.row("", self.spin_min_threshold, self.spin_max_threshold)
        group.add_widget(self.threshold_slider)
        group.row("Step", self.spin_threshold_step, stretch_label=True)
        group.row("MinDist", self.spin_min_dist, stretch_label=True)

        # ── 타겟 필터링 ──
        # 스위치는 '지금 고른 필터'의 켜짐 여부다. 필터마다 켜짐/값을 따로
        # 기억하므로 여러 필터를 동시에 켜 둘 수 있다.
        self._build_filter_controls(group)
        return group


    @staticmethod
    def _make_param_double(minimum, maximum, value, decimals, suffix):
        """클러스터링 파라미터용 소수 입력칸 (_make_param_spin의 소수 버전)."""
        box = QDoubleSpinBox()
        box.setObjectName("paramSpin")
        box.setRange(minimum, maximum)
        box.setDecimals(decimals)
        box.setSuffix(suffix)
        box.setValue(value)
        box.setAlignment(Qt.AlignCenter)
        box.setButtonSymbols(QDoubleSpinBox.NoButtons)
        box.setFixedWidth(PARAM_BOX_WIDTH)
        box.lineEdit().setFixedWidth(PARAM_BOX_WIDTH)
        return box

    def _build_status_row(self):
        """이미지 보드 위 우측: 상태등 + 탐지 결과 문구 (처리 시간 바로 위)."""
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

        # 단계별 소요([홀 · 법선 · 포즈])가 붙어 길어진다. 상태문구와 같은 이유로
        # 폭을 요구하지 않는 라벨을 쓴다. 뒤쪽(합계)보다 앞쪽(캡쳐/분석)이 먼저
        # 읽혀야 하므로 오른쪽부터 줄인다.
        self.elapsed_label = ElidedLabel("", elide=Qt.ElideRight)
        self.elapsed_label.setObjectName("elapsedLabel")
        self.elapsed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        row.addStretch()
        row.addWidget(self.elapsed_label)
        return row

    def _build_flange_group(self):
        """우측 패널의 아래 그룹: 로봇 플랜지 포즈.

        진입 각도 오프셋은 여기 없다 — Pose Setting 페이지로 갔다. 이 값은
        '로봇이 어디 서 있는가'라서 카메라 좌표를 로봇 좌표로 옮길 때만 쓰이고,
        비전 세팅과 함께 있는 편이 티칭 순서에 맞는다.
        """
        group = SettingGroup("Flange Pose")

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        self.flange_inputs = {}
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
            self.flange_inputs[key] = box

        group.add_layout(grid)
        return group

    def current_flange_pose(self):
        """현재 입력된 플랜지 포즈 (x, y, z, rx, ry, rz)를 반환한다."""
        return tuple(self.flange_inputs[key].value() for key, *_ in POSE_FIELDS)

    def _wire_config(self):
        sort_axis = self.wire_sort_axis.currentData()
        if sort_axis not in SORT_CHOICES:
            sort_axis = SORT_X
        representative = self.wire_representative.currentData()
        if representative not in REP_CHOICES:
            representative = REP_CENTER
        foreground_mode = self.wire_foreground_mode.currentData()
        if foreground_mode not in WIRE_MODE_CHOICES:
            foreground_mode = WIRE_MODE_BIN_TOP
        return WireDetectionConfig(
            min_length_px=self.wire_min_length.value(),
            min_area_px=self.wire_min_area.value(),
            min_thickness_px=self.wire_min_thickness.value(),
            max_thickness_px=max(
                self.wire_min_thickness.value(), self.wire_max_thickness.value()
            ),
            foreground_mode=foreground_mode,
            top_band_mm=self.wire_plane_threshold.value(),
            top_percentile=self.wire_top_percentile.value(),
            depth_min_mm=self.wire_depth_min.value(),
            depth_max_mm=self.wire_depth_max.value(),
            plane_threshold_mm=self.wire_plane_threshold.value(),
            merge_distance_px=self.wire_merge_distance.value(),
            merge_angle_deg=self.wire_merge_angle.value(),
            sort_axis=sort_axis,
            sort_desc=bool(self.wire_sort_order.currentData()),
            sort_vector_xyz=(
                self.wire_sort_vec_x.value(),
                self.wire_sort_vec_y.value(),
                self.wire_sort_vec_z.value(),
            ),
            sort_vector_uv=(self.wire_sort_vec_x.value(), self.wire_sort_vec_y.value()),
            representative_mode=representative,
            tracking_enabled=self.wire_tracking_switch.isChecked(),
            confidence_minimum=self.wire_confidence_min.value(),
            use_depth_foreground=self.wire_depth_foreground_switch.isChecked(),
            use_edge_fallback=self.wire_edge_fallback_switch.isChecked(),
            apply_morph_open=self.wire_morph_open_switch.isChecked(),
            apply_morph_close=self.wire_morph_close_switch.isChecked(),
            filter_components=self.wire_area_filter_switch.isChecked(),
            skeleton_enabled=self.wire_skeleton_switch.isChecked(),
            save_debug_images=self.wire_debug_save_switch.isChecked(),
            debug_enabled=True,
        )

    def _build_filter_controls(self, group):
        """타겟 필터링: [Filtering 스위치] [필터 종류 ▼] [기준값].

        블롭 그룹 아래쪽에 이어 붙인다(_build_blob_group 이 부른다).
        """
        self.filter_switch = ToggleSwitch()
        self.filter_switch.toggled.connect(self._on_filter_switch_toggled)

        self.filter_combo = QComboBox()
        self.filter_combo.setObjectName("filterCombo")
        for spec in FILTER_SPECS:
            self.filter_combo.addItem(spec.label, spec.key)
        self.filter_combo.currentIndexChanged.connect(self._on_filter_selected)

        self.filter_value = QDoubleSpinBox()
        self.filter_value.setObjectName("paramSpin")
        self.filter_value.setAlignment(Qt.AlignCenter)
        self.filter_value.setButtonSymbols(QDoubleSpinBox.NoButtons)
        self.filter_value.valueChanged.connect(self._on_filter_value_changed)

        group.row("Filtering", self.filter_switch, stretch_label=True)
        group.add_widget(self.filter_combo)
        group.row("기준값", self.filter_value, stretch_label=True)

        # 첫 항목의 상태를 위젯에 실어 둔다.
        self._load_selected_filter()

    @staticmethod
    def _make_param_spin(minimum, maximum, value):
        spin = QSpinBox()
        spin.setObjectName("paramSpin")
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setAlignment(Qt.AlignCenter)
        # 위/아래 화살표 없이 숫자 입력에 집중 (좁은 행에 여러 개가 들어간다)
        spin.setButtonSymbols(QSpinBox.NoButtons)
        spin.setFixedWidth(PARAM_BOX_WIDTH)
        spin.lineEdit().setFixedWidth(PARAM_BOX_WIDTH)
        return spin

    # ===== 블롭 파라미터 =====
    def _on_slider_range_changed(self, low, high):
        """슬라이더가 움직이면 입력박스를 맞춰 준다."""
        if self._syncing:
            return
        self._syncing = True
        self.spin_min_threshold.setValue(low)
        self.spin_max_threshold.setValue(high)
        self._syncing = False
        self._sync_threshold_bounds()
        self._schedule_apply()

    def _on_threshold_box_changed(self):
        """입력박스를 고치면 슬라이더를 맞춰 준다."""
        if self._syncing:
            return
        self._syncing = True
        self.threshold_slider.set_values(
            self.spin_min_threshold.value(), self.spin_max_threshold.value()
        )
        self._syncing = False
        self._sync_threshold_bounds()
        self._schedule_apply()

    def _sync_threshold_bounds(self):
        """min <= max 가 깨지지 않도록 두 입력박스의 허용 범위를 좁혀 둔다."""
        low, high = self.threshold_slider.values()
        was_syncing = self._syncing
        self._syncing = True
        # 값을 먼저 넣으려면 범위가 넉넉해야 한다. 좁은 범위가 남아 있으면
        # setValue 가 잘려서 슬라이더와 어긋나므로, 넓혔다가 다시 좁힌다.
        self.spin_min_threshold.setMaximum(THRESHOLD_ABS_MAX)
        self.spin_max_threshold.setMinimum(THRESHOLD_ABS_MIN)
        self.spin_min_threshold.setValue(low)
        self.spin_max_threshold.setValue(high)
        self.spin_min_threshold.setMaximum(high - 1)
        self.spin_max_threshold.setMinimum(low + 1)
        self._syncing = was_syncing

    # ===== 타겟 필터링 =====
    def _current_filter_spec(self):
        return FILTER_SPEC_BY_KEY[self.filter_combo.currentData()]

    def _load_selected_filter(self):
        """고른 필터의 켜짐/기준값을 스위치와 입력박스에 실어 준다."""
        spec = self._current_filter_spec()
        state = self._filter_states[spec.key]

        was_syncing = self._syncing
        self._syncing = True
        # 값 범위/자릿수는 필터마다 다르다(크기는 정수, 진원도는 0~1 소수).
        self.filter_value.setDecimals(spec.decimals)
        self.filter_value.setRange(spec.minimum, spec.maximum)
        self.filter_value.setSingleStep(spec.step)
        self.filter_value.setValue(state["value"])
        self.filter_switch.set_checked_silently(state["enabled"])
        self._syncing = was_syncing

        self._update_filter_enabled_style()

    def _update_filter_enabled_style(self):
        """스위치가 꺼져 있으면 기준값 입력을 회색으로 비활성화한다."""
        self.filter_value.setEnabled(self.filter_switch.isChecked())

    def _on_filter_selected(self):
        self._load_selected_filter()

    def _on_filter_switch_toggled(self, checked):
        if self._syncing:
            return
        self._filter_states[self._current_filter_spec().key]["enabled"] = checked
        self._update_filter_enabled_style()
        self._schedule_apply()

    def _on_filter_value_changed(self, value):
        if self._syncing:
            return
        self._filter_states[self._current_filter_spec().key]["value"] = value
        self._schedule_apply()

    # ===== 검출 방식 / 템플릿 관리 =====
    def _on_method_changed(self):
        if self._syncing:
            return
        self._update_detection_controls()
        if self.method_combo.currentData() == DETECTION_WIRE:
            index = self.view_combo.findData(VIEW_DEBUG)
            if index >= 0:
                self.view_combo.setCurrentIndex(index)
        if self._last_captured_image is not None:
            self.analyze_image()

    def _update_detection_controls(self):
        """고른 검출 방식에 맞는 그룹만 남긴다.

        방식이 셋이 되었으므로 '템플릿이냐 아니냐'로 가르면 안 된다. 그렇게
        하면 SAM2를 골랐을 때 블랍 파라미터가 딸려 나온다.

        세 그룹은 모두 오른쪽 패널에 있고 패널 폭이 고정이라, 무엇을 골라도
        이미지 보드 크기는 변하지 않는다(260910 요구).
        """
        method = self.method_combo.currentData()
        self._blob_group.setVisible(method == DETECTION_BLOB)
        self._template_group.setVisible(method == DETECTION_TEMPLATE)
        self._sam2_group.setVisible(method in SAM2_METHODS)
        self._wire_group.setVisible(method == DETECTION_WIRE)
        # 같은 줄을 두 방식이 쓰므로, 어느 쪽을 골랐는지에 맞는 설명을 붙인다.
        hint = SAM2_VIDEO_HINT if method == DETECTION_SAM2_VIDEO else SAM2_HINT
        self._sam2_model_label.setToolTip(hint)
        self.btn_sam2_teach.setToolTip(hint)

        self.btn_template_region.setVisible(method == DETECTION_TEMPLATE)
        if method != DETECTION_TEMPLATE and self.btn_template_region.isChecked():
            self.btn_template_region.setChecked(False)
        if method not in SAM2_METHODS and self.btn_sam2_teach.isChecked():
            self.btn_sam2_teach.setChecked(False)
        if method == DETECTION_WIRE:
            index = self.view_combo.findData(VIEW_DEBUG)
            if index >= 0:
                self.view_combo.setCurrentIndex(index)

    def _reload_template_list(self, select=None):
        """드래그로 만든 템플릿 목록 (템플릿 매칭 전용)."""
        return self._fill_asset_combo(
            self.template_combo, self.btn_remove_template, list_templates(), select
        )

    def _reload_sam2_list(self, select=None):
        """SAM2 로 티칭한 홀 목록 (비디오 추적 전용).

        260908 에 템플릿 목록에서 갈라졌다. 그전에는 한 목록을 두 콤보가 함께
        봤는데, 그때는 'SAM2 홀 지정'이 실제로 template.png 를 써서 같은 자산을
        가리키는 것이 맞았다. 이제 홀은 scene.png + hole_box 이고 템플릿은
        크롭이라 종류가 다르므로, 섞어 보여 주면 고를 수 없는 것이 콤보에 뜬다.
        """
        return self._fill_asset_combo(
            self.sam2_reference_combo, self.btn_sam2_remove, list_holes(), select
        )

    @staticmethod
    def _fill_asset_combo(combo, remove_button, names, select):
        """자산 콤보 하나를 다시 채운다.

        **고르지 않았으면 선택을 그대로 지킨다.** clear() 하면 Qt 가 첫 항목을
        자동으로 고르는데, 그러면 저장과 무관한 콤보가 알파벳순 첫 항목으로
        튄다. 매칭점을 티칭했더니 홀 기준이 그 영역으로 바뀌어 컨투어가 홀 원
        으로 그려진 사고가 이것이었다.
        """
        keep = select if select in names else combo.currentData()
        blocked = combo.blockSignals(True)
        combo.clear()
        for name in names:
            combo.addItem(name, name)
        combo.setCurrentIndex(combo.findData(keep))
        combo.blockSignals(blocked)
        remove_button.setEnabled(bool(names))
        return names

    def _on_template_changed(self):
        if not self._syncing and self._last_captured_image is not None:
            self.analyze_image()

    def _on_template_mode_toggled(self, checked):
        if checked:
            self._release_drag_buttons(keep=self.btn_template_region)
            self._drag_target = "template"
            self._switch_to_color_for_selection()
            self.result_board.set_selection_mode("region")
            self._set_status("현재 이미지에서 템플릿으로 저장할 영역을 드래그하세요.")
        elif self._drag_target == "template":
            self._drag_target = None
            self.result_board.set_selection_mode(None)

    def _release_drag_buttons(self, keep):
        """네모 드래그를 쓰는 다른 버튼을 내린다.

        ROI·템플릿 지정·홀 지정이 **같은 보드의 같은 드래그**를 쓴다. 둘이 함께
        켜져 있으면 그린 네모가 어디로 갈지 알 수 없다(법선 매칭점 세 칸도
        Pose Setting 에서 같은 규칙으로 서로를 내린다).

        먼저 내리고 나서 모드를 세워야 한다 — 내릴 때 그 버튼의 처리기가
        set_selection_mode(None) 을 부르므로, 순서를 바꾸면 방금 세운 모드가
        지워진다.
        """
        for button in (self.btn_roi, self.btn_template_region, self.btn_sam2_teach):
            if button is not keep and button.isChecked():
                button.setChecked(False)

    def _on_region_dragged(self, rect):
        """보드에서 네모 드래그가 끝났을 때. 기다리던 쪽으로 넘긴다.

        템플릿 저장과 SAM2 홀 티칭이 같은 드래그를 쓴다(260914 에 홀 티칭이
        클릭에서 네모로 바뀌면서 둘이 됐다). 법선 매칭점도 네모 드래그지만
        그것은 Pose Setting 페이지에 있다.
        """
        target = self._drag_target
        self._drag_target = None
        self.result_board.set_selection_mode(None)
        if self._last_captured_image is None:
            return
        if target == "hole":
            self._on_sam2_region_taught(rect)
        elif target == "template":
            self._on_template_selected(rect)

    def _on_result_board_clicked(self, point):
        if self.view_combo.currentData() != VIEW_DEBUG:
            return
        if self._last_captured_image is None:
            return
        height, width = self._last_captured_image.shape[:2]
        frame = debug_frame_at(
            self._last_debug_frames,
            point,
            output_size=(width, height),
        )
        if frame is None:
            return
        self._show_debug_frame(frame)

    def _show_debug_frame(self, frame):
        image = to_bgr(frame.image)
        if image is None:
            return
        title = frame.title if not frame.note else f"{frame.title} ({frame.note})"
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(980, 620)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        board = ImageBoard(title)
        board.setMinimumSize(860, 480)
        layout.addWidget(board, 1)
        board.set_image_bgr(image)
        self._debug_dialogs.append(dialog)

        def forget(_result, item=dialog):
            if item in self._debug_dialogs:
                self._debug_dialogs.remove(item)

        dialog.finished.connect(forget)
        dialog.show()

    # ===== SAM2 =====
    def _on_sam2_model_changed(self):
        if not self._syncing and self._last_captured_image is not None:
            self.analyze_image()

    def _on_sam2_reference_changed(self):
        if not self._syncing and self._last_captured_image is not None:
            self.analyze_image()

    def _on_sam2_teach_toggled(self, checked):
        if checked:
            self._release_drag_buttons(keep=self.btn_sam2_teach)
            self._drag_target = "hole"
            self._switch_to_color_for_selection()
            self.result_board.set_selection_mode("region")
            # '클릭이 아니라'를 적어 둔다 — 260914 까지는 클릭이었고, 짧은
            # 드래그(사실상 클릭)는 보드가 흘려버려 아무 일도 일어나지 않는다.
            self._set_status(
                "찾을 홀을 네모로 드래그하세요 — 클릭이 아닙니다 "
                "(홀 하나만 네모에 들어오게)."
            )
        elif self._drag_target == "hole":
            self._drag_target = None
            self.result_board.set_selection_mode(None)

    def _on_sam2_region_taught(self, rect):
        """드래그한 네모 안의 홀을 SAM2로 티칭한다.

        260914 까지는 클릭이었다. 점 하나는 SAM2 가 '홀인지 판재인지'를 스스로
        골라야 해서 **뜻한 것보다 큰 물체가 잡히는** 일이 있었다(사용자 보고).
        네모는 범위를 함께 알려 주고, 네모 밖으로 번진 후보는 아예 뺄 수 있다
        (sam2_detector._hole_picker). 법선 매칭점과 같은 방식이 됐다.

        저장하는 것은 사람이 그린 네모가 아니라 SAM2 가 잡은 경계의 외접
        사각형이다(예전 클릭 시절에도 저장은 네모였다 — 그 구조는 그대로다).
        """
        self.btn_sam2_teach.setChecked(False)
        if self._last_captured_image is None:
            return

        left, top, width, height = rect
        box = (left, top, left + width, top + height)
        model = self.sam2_model_combo.currentData()
        self._set_status("SAM2로 홀을 찾는 중입니다...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            hole, _mask, taught_box = sam2_detector.teach(
                self._last_captured_image, box, model
            )
            # 재검출이 쓸 기준 넓이를 **재검출과 같은 방법으로** 재 둔다.
            # 티칭은 이미지 예측기, 재검출은 비디오 예측기라 마스크 넓이가
            # 달라서, 티칭 쪽 값을 기준 삼으면 문턱이 엉뚱하게 잡힌다.
            #
            # 사람이 그린 네모가 아니라 taught_box 로 잰다 — 저장하는 것도
            # 재검출이 프롬프트로 쓰는 것도 그 네모다.
            area = sam2_video_tracker.measure_hole_area(
                self._last_captured_image, taught_box, model
            )
        except Exception as exc:
            self._set_status(f"홀 지정 실패: {exc}", "error")
            return
        finally:
            QApplication.restoreOverrideCursor()

        if not area:
            # 비디오 추적이 그 자리를 못 잡는다. 그대로 저장하면 촬영마다
            # 실패하므로 지금 알린다(법선 매칭점과 같은 판단).
            self._set_status(
                "이 홀은 재검출이 다시 찾지 못합니다 — 더 또렷한 홀을 "
                "고르거나 네모를 조금 다르게 그려 보세요.",
                "error",
            )
            return

        name = TextInputDialog.get_text(
            self,
            "기준 홀 저장",
            f"지정한 홀 (반지름 {hole[2]:.1f} px)의 이름을 입력하세요.",
            ok_text="저장",
            cancel_text="취소",
            validator=validate_template_name,
        )
        if name is None:
            return
        ok, result = save_hole(name, self._last_captured_image, taught_box, area)
        if not ok:
            self._set_status(f"기준 홀 저장 실패: {result}", "error")
            return
        self._reload_sam2_list(select=name)
        # 분석을 먼저 돌리고 나서 문구를 세운다. 순서를 바꾸면 분석 결과가
        # 티칭 확인 문구를 덮어써, 2초를 기다린 작업이 성공했는지 알 수 없다.
        self.analyze_image()
        self._set_status(
            f"'{name}' 홀을 지정했습니다 — 중심 ({hole[0]}, {hole[1]}) "
            f"반지름 {hole[2]:.1f} px",
            "success",
        )

    def _sam2_detector(self):
        """지금 세팅으로 만든 SAM2 검출기. 준비가 안 됐으면 사유와 함께 예외.

        SAM2 경로는 비디오 추적 하나뿐이다(260908). 오퍼레이터와 사유 문구가
        어긋나지 않도록 모듈 함수(sam2_video_detector)를 함께 쓴다.
        """
        reason = sam2_video_tracker.unavailable_reason()
        if reason:
            raise RuntimeError(reason)
        if not self.sam2_reference_combo.currentData():
            raise RuntimeError("먼저 '홀 지정'으로 찾을 홀을 드래그해 두세요.")
        return sam2_video_detector(
            {
                "reference": self.sam2_reference_combo.currentData(),
                "model": self.sam2_model_combo.currentData(),
            }
        )

    def _on_template_selected(self, rect):
        self.btn_template_region.setChecked(False)
        if self._last_captured_image is None:
            return
        x, y, width, height = rect
        template = self._last_captured_image[y:y + height, x:x + width].copy()
        name = TextInputDialog.get_text(
            self,
            "템플릿 저장",
            "선택한 템플릿의 이름을 입력하세요.",
            ok_text="저장",
            cancel_text="취소",
            validator=validate_template_name,
        )
        if name is None:
            return
        ok, result = save_template(name, template)
        if not ok:
            self._set_status(f"템플릿 저장 실패: {result}", "error")
            return
        self._reload_template_list(select=name)
        self._set_status(f"'{name}' 템플릿을 저장했습니다.", "success")
        self.analyze_image()

    def _on_remove_template_clicked(self):
        # 두 콤보가 서로 다른 종류를 보므로, 지금 화면에 보이는 쪽을 지운다.
        # 자산 폴더는 하나라 지우는 함수는 같다(delete_template).
        sam2_mode = self.method_combo.currentData() in SAM2_METHODS
        source = self.sam2_reference_combo if sam2_mode else self.template_combo
        kind = "기준 홀" if sam2_mode else "템플릿"
        name = source.currentData()
        if not name:
            return
        box = QMessageBox(self)
        box.setWindowTitle(f"{kind} 삭제")
        box.setIcon(QMessageBox.Warning)
        box.setText(f"'{name}' {kind}을(를) 삭제할까요?")
        box.setInformativeText("이 템플릿을 사용하는 세팅에서는 더 이상 검출할 수 없습니다.")
        delete_button = box.addButton("삭제", QMessageBox.DestructiveRole)
        cancel_button = box.addButton("취소", QMessageBox.RejectRole)
        box.setDefaultButton(cancel_button)
        box.exec_()
        if box.clickedButton() is not delete_button:
            return
        ok, error = delete_template(name)
        if not ok:
            self._set_status(f"{kind} 삭제 실패: {error}", "error")
            return
        self._reload_template_list()
        self._reload_sam2_list()
        self._set_status(f"'{name}' {kind}을(를) 삭제했습니다.", "success")
        if self._last_captured_image is not None:
            self.analyze_image()

    # ===== 세팅 저장 / 불러오기 =====
    def _collect_settings(self):
        """지금 UI에 잡힌 값들을 저장용 dict로 모은다.

        검출 키는 이 화면에서, 포즈 키는 Pose Setting 화면에서 온다. 프로파일
        파일은 하나이므로 저장하는 자리도 하나여야 한다.
        """
        roi = roi_state.get()
        settings = {
            "detection_method": self.method_combo.currentData(),
            "template_matching": {
                "name": self.template_combo.currentData(),
                "threshold": self.template_threshold.value(),
            },
            # SAM2로 티칭해 둔 홀. reference 는 템플릿 창고의 이름이다.
            "sam2": {
                "model": self.sam2_model_combo.currentData(),
                "reference": self.sam2_reference_combo.currentData(),
            },
            "wire_detection": {
                "min_length_px": self.wire_min_length.value(),
                "min_area_px": self.wire_min_area.value(),
                "min_thickness_px": self.wire_min_thickness.value(),
                "max_thickness_px": self.wire_max_thickness.value(),
                "foreground_mode": self.wire_foreground_mode.currentData(),
                "top_band_mm": self.wire_plane_threshold.value(),
                "top_percentile": self.wire_top_percentile.value(),
                "depth_min_mm": self.wire_depth_min.value(),
                "depth_max_mm": self.wire_depth_max.value(),
                "plane_threshold_mm": self.wire_plane_threshold.value(),
                "merge_distance_px": self.wire_merge_distance.value(),
                "merge_angle_deg": self.wire_merge_angle.value(),
                "confidence_minimum": self.wire_confidence_min.value(),
                "sort_axis": self.wire_sort_axis.currentData(),
                "sort_desc": bool(self.wire_sort_order.currentData()),
                "sort_vector_xyz": [
                    self.wire_sort_vec_x.value(),
                    self.wire_sort_vec_y.value(),
                    self.wire_sort_vec_z.value(),
                ],
                "representative": self.wire_representative.currentData(),
                "tracking_enabled": self.wire_tracking_switch.isChecked(),
                "use_depth_foreground": self.wire_depth_foreground_switch.isChecked(),
                "use_edge_fallback": self.wire_edge_fallback_switch.isChecked(),
                "apply_morph_open": self.wire_morph_open_switch.isChecked(),
                "apply_morph_close": self.wire_morph_close_switch.isChecked(),
                "filter_components": self.wire_area_filter_switch.isChecked(),
                "skeleton_enabled": self.wire_skeleton_switch.isChecked(),
                "save_debug_images": self.wire_debug_save_switch.isChecked(),
            },
            "threshold": {
                "min": self.spin_min_threshold.value(),
                "max": self.spin_max_threshold.value(),
                "step": self.spin_threshold_step.value(),
            },
            "min_dist_between_blobs": self.spin_min_dist.value(),
            "filters": {
                key: {"enabled": state["enabled"], "value": state["value"]}
                for key, state in self._filter_states.items()
            },
            # 원본 이미지 좌표 기준 [x, y, w, h]. 지정 안 했으면 null.
            "roi": list(roi) if roi is not None else None,
            # 로봇 플랜지 포즈 (베이스 기준)
            "flange_pose": {
                key: self.flange_inputs[key].value() for key, *_ in POSE_FIELDS
            },
        }
        # 포즈 키(normal_source / normal_points / approach_offset)는 Pose Setting
        # 화면이 들고 있다. 파일은 하나이므로 저장 직전에 받아 합친다.
        #
        # 저쪽이 아직 안 붙었으면(테스트 등) 이미 저장돼 있던 값을 그대로 살린다.
        # 안 그러면 화면에 없는 값을 '비어 있음'으로 읽어 조용히 지워 버린다.
        if self._pose_page is not None:
            settings.update(self._pose_page.collect_pose_settings())
        else:
            saved = load_profile(self.current_profile())
            for key in ("normal_source", "normal_points", "approach_offset"):
                if isinstance(saved, dict) and key in saved:
                    settings[key] = saved[key]
        return settings

    @staticmethod
    def _apply_saved_roi(saved):
        """저장된 ROI를 복원한다.

        값이 없거나 상해 있으면 ROI 없음(전체 영역)으로 둔다. 잘못된 ROI가
        조용히 적용되면 분석 범위가 엉뚱해져 원인을 찾기 어렵다.

        Returns:
            bool: 복원했는지
        """
        if not isinstance(saved, (list, tuple)) or len(saved) != 4:
            roi_state.clear()
            return False
        try:
            x, y, width, height = (int(value) for value in saved)
        except (TypeError, ValueError):
            roi_state.clear()
            return False
        # roi_state.set 이 너무 작은 영역은 알아서 거른다.
        if x < 0 or y < 0 or not roi_state.set(x, y, width, height):
            roi_state.clear()
            return False
        return True

    def _on_save_clicked(self):
        ok, message = self.save_current_profile()
        self._set_status(message, "success" if ok else "error")

    # ===== 세팅 프로파일 (체결 위치별) =====
    def current_profile(self):
        """지금 고른 프로파일 이름. 목록이 비어 있으면 None."""
        return self.profile_combo.currentData()

    def _reload_profile_list(self, select=None):
        """Settings 폴더를 다시 훑어 콤보를 채운다.

        Args:
            select: 채운 뒤 고를 이름. 없거나 사라졌으면 첫 항목을 고른다.
        """
        names = list_profiles()

        was_syncing = self._syncing
        self._syncing = True   # 목록을 갈아 끼우는 동안 불러오기가 돌지 않게
        self.profile_combo.clear()
        for name in names:
            self.profile_combo.addItem(name, name)
        if select in names:
            self.profile_combo.setCurrentIndex(names.index(select))
        self._syncing = was_syncing

        # 마지막 하나까지 지우면 저장할 곳이 없어지므로 그때는 삭제를 막는다.
        self.btn_remove_profile.setEnabled(len(names) > 1)
        self.btn_save.setEnabled(bool(names))
        return names

    def init_profiles(self):
        """프로파일 목록을 준비하고 마지막에 쓰던 것을 불러온다.

        **두 페이지가 이어진 뒤에** 불러야 한다(MainWindow._link_pages). 첫
        실행에 만드는 기본 프로파일은 _collect_settings() 로 만드는데, 그때
        Pose Setting 이 안 붙어 있으면 포즈 키가 통째로 빠진 파일이 된다.
        읽는 쪽이 '없으면 꺼짐/0' 으로 버티므로 고장은 아니지만, 파일만 보고
        무엇이 저장되는지 알 수 없어진다.
        """
        names = list_profiles()
        if not names:
            # 첫 실행: 지금 기본값으로 기본 프로파일을 하나 만들어 둔다.
            # 하나도 없으면 콤보가 비어 저장할 대상이 없어진다.
            save_profile(DEFAULT_PROFILE_NAME, self._collect_settings())
            names = list_profiles()

        wanted = active_profile() or (
            DEFAULT_PROFILE_NAME if DEFAULT_PROFILE_NAME in names else names[0]
        )
        self._reload_profile_list(select=wanted)
        self._load_profile(self.current_profile())

    def _on_profile_changed(self):
        if self._syncing:
            return
        name = self.current_profile()
        if name is None:
            return
        set_active_profile(name)
        self._load_profile(name)

    def _missing_detection_input(self):
        """지금 검출 방식이 저장할 준비가 안 됐다면 그 사유. 준비됐으면 빈 문자열.

        고른 모양 없이 저장하면 오퍼레이터가 그 세팅으로 아무것도 못 찾는데,
        그 사실이 현장에서 촬영해 볼 때야 드러난다. 저장 단계에서 막는다.
        """
        method = self.method_combo.currentData()
        if method == DETECTION_TEMPLATE and not self.template_combo.currentData():
            return "템플릿을 먼저 선택하거나 새로 지정하세요."
        if method in SAM2_METHODS and not self.sam2_reference_combo.currentData():
            return "'홀 지정'으로 찾을 홀을 먼저 드래그해 두세요."
        return ""

    def _on_add_profile_clicked(self):
        """현재 값을 새 이름으로 저장한다(다른 이름으로 저장)."""
        missing = self._missing_detection_input()
        if missing:
            self._set_status(missing, "error")
            return
        name = TextInputDialog.get_text(
            self,
            "세팅 추가",
            "새 세팅 이름을 입력하세요. 지금 화면의 값이 그대로 저장됩니다.",
            ok_text="생성",
            cancel_text="취소",
            validator=validate_profile_name,
        )
        if name is None:
            return   # 취소

        ok, result = save_profile(name, self._collect_settings())
        if not ok:
            self._set_status(f"세팅 생성 실패: {result}", "error")
            return
        set_active_profile(name)
        self._reload_profile_list(select=name)
        self._set_status(f"'{name}' 세팅을 만들었습니다.", "success")

    def _on_remove_profile_clicked(self):
        name = self.current_profile()
        if name is None:
            return

        box = QMessageBox(self)
        box.setWindowTitle("세팅 삭제")
        box.setIcon(QMessageBox.Warning)
        box.setText(f"'{name}' 세팅을 삭제할까요?")
        box.setInformativeText("삭제하면 되돌릴 수 없습니다.")
        delete_button = box.addButton("삭제", QMessageBox.DestructiveRole)
        cancel_button = box.addButton("취소", QMessageBox.RejectRole)
        box.setDefaultButton(cancel_button)   # 실수로 엔터를 쳐도 지워지지 않게
        box.exec_()
        if box.clickedButton() is not delete_button:
            return

        ok, error = delete_profile(name)
        if not ok:
            self._set_status(f"세팅 삭제 실패: {error}", "error")
            return

        remaining = self._reload_profile_list()
        next_name = self.current_profile()
        if next_name is not None:
            set_active_profile(next_name)
            self._load_profile(next_name)
        self._set_status(
            f"'{name}' 세팅을 삭제했습니다."
            + (f" 이제 '{next_name}' 세팅을 씁니다." if remaining else ""),
            "success",
        )

    def _load_profile(self, name):
        """프로파일 하나를 읽어 위젯과 검출기에 적용한다."""
        if name is None:
            return
        data = load_profile(name)
        if data is None:
            # 파일이 깨졌어도 프로그램은 떠야 하므로 기본값으로 되돌린다.
            self._reset_to_defaults()
            self._set_status(
                f"'{name}' 세팅을 읽을 수 없어 기본값으로 시작합니다.", "error"
            )
            return
        self._apply_settings(data, profile_name=name)

    def _reset_to_defaults(self):
        """세팅을 초기값으로 되돌린다(깨진 파일을 읽었을 때)."""
        self._syncing = True
        try:
            self.method_combo.setCurrentIndex(0)
            self.template_threshold.setValue(DEFAULT_MATCH_THRESHOLD)
            self.threshold_slider.set_values(MIN_THRESHOLD, MAX_THRESHOLD)
            self.spin_threshold_step.setValue(THRESHOLD_STEP)
            self.spin_min_dist.setValue(MIN_DIST_BETWEEN_BLOBS)
            self._filter_states = default_filter_states()
            if self.sam2_model_combo.count():
                self.sam2_model_combo.setCurrentIndex(0)
            self.sam2_reference_combo.setCurrentIndex(-1)
            self.wire_min_length.setValue(80)
            self.wire_min_area.setValue(120)
            self.wire_min_thickness.setValue(2.0)
            self.wire_max_thickness.setValue(80.0)
            self.wire_foreground_mode.setCurrentIndex(0)
            self.wire_plane_threshold.setValue(18.0)
            self.wire_top_percentile.setValue(1.0)
            self.wire_depth_min.setValue(70.0)
            self.wire_depth_max.setValue(0.0)
            self.wire_merge_distance.setValue(18.0)
            self.wire_merge_angle.setValue(35.0)
            self.wire_confidence_min.setValue(0.45)
            self.wire_sort_axis.setCurrentIndex(0)
            self.wire_sort_order.setCurrentIndex(0)
            self.wire_sort_vec_x.setValue(1.0)
            self.wire_sort_vec_y.setValue(0.0)
            self.wire_sort_vec_z.setValue(0.0)
            self.wire_representative.setCurrentIndex(0)
            self.wire_tracking_switch.set_checked_silently(True)
            self.wire_depth_foreground_switch.set_checked_silently(True)
            self.wire_edge_fallback_switch.set_checked_silently(True)
            self.wire_morph_open_switch.set_checked_silently(True)
            self.wire_morph_close_switch.set_checked_silently(True)
            self.wire_area_filter_switch.set_checked_silently(True)
            self.wire_skeleton_switch.set_checked_silently(True)
            self.wire_debug_save_switch.set_checked_silently(False)
            roi_state.clear()
        finally:
            self._syncing = False
        self._sync_threshold_bounds()
        self._load_selected_filter()
        self._update_detection_controls()
        self._apply_blob_params()
        # 포즈 값은 저쪽이 들고 있으므로 되돌리는 것도 저쪽이 한다.
        if self._pose_page is not None:
            self._pose_page.reset_pose_settings()

    def _apply_settings(self, data, profile_name=None):
        """읽어 들인 세팅 dict를 위젯과 검출기에 적용한다."""
        # 저장 파일을 손으로 고쳤을 수도 있으므로 값 하나하나를 방어적으로 읽는다.
        self._syncing = True
        try:
            method = data.get("detection_method", DETECTION_BLOB)
            method_index = self.method_combo.findData(method)
            self.method_combo.setCurrentIndex(max(0, method_index))

            template_settings = data.get("template_matching")
            if isinstance(template_settings, dict):
                template_name = template_settings.get("name")
                template_index = self.template_combo.findData(template_name)
                self.template_combo.setCurrentIndex(template_index)
                self.template_threshold.setValue(
                    as_float(
                        template_settings.get("threshold"),
                        DEFAULT_MATCH_THRESHOLD,
                        0.0,
                        1.0,
                    )
                )

            threshold = data.get("threshold")
            if isinstance(threshold, dict):
                low = as_int(threshold.get("min"), MIN_THRESHOLD)
                high = as_int(threshold.get("max"), MAX_THRESHOLD)
                self.threshold_slider.set_values(low, high)
                self.spin_threshold_step.setValue(
                    as_int(threshold.get("step"), THRESHOLD_STEP)
                )

            self.spin_min_dist.setValue(
                as_int(data.get("min_dist_between_blobs"), MIN_DIST_BETWEEN_BLOBS)
            )

            filters = data.get("filters")
            if isinstance(filters, dict):
                for key, state in self._filter_states.items():
                    saved = filters.get(key)
                    if not isinstance(saved, dict):
                        continue
                    spec = FILTER_SPEC_BY_KEY[key]
                    state["enabled"] = bool(saved.get("enabled", spec.default_enabled))
                    state["value"] = as_float(
                        saved.get("value"), spec.default_value, spec.minimum, spec.maximum
                    )

            # SAM2 복원. 저장된 모델이 이 PC에 없으면(체크포인트를 안 받았거나
            # 다른 PC에서 만든 세팅이면) 건드리지 않고 기본 모델을 그대로 둔다.
            sam2_settings = data.get("sam2")
            if isinstance(sam2_settings, dict):
                model_index = self.sam2_model_combo.findData(sam2_settings.get("model"))
                if model_index >= 0:
                    self.sam2_model_combo.setCurrentIndex(model_index)
                reference_index = self.sam2_reference_combo.findData(
                    sam2_settings.get("reference")
                )
                self.sam2_reference_combo.setCurrentIndex(reference_index)

            wire_settings = data.get("wire_detection")
            if isinstance(wire_settings, dict):
                self.wire_min_length.setValue(
                    as_int(wire_settings.get("min_length_px"), 80)
                )
                self.wire_min_area.setValue(
                    as_int(wire_settings.get("min_area_px"), 120)
                )
                self.wire_min_thickness.setValue(
                    as_float(wire_settings.get("min_thickness_px"), 2.0, 0.5, 200.0)
                )
                self.wire_max_thickness.setValue(
                    as_float(wire_settings.get("max_thickness_px"), 80.0, 1.0, 300.0)
                )
                mode_index = self.wire_foreground_mode.findData(
                    wire_settings.get("foreground_mode")
                )
                self.wire_foreground_mode.setCurrentIndex(max(0, mode_index))
                depth_band = as_float(
                    wire_settings.get("top_band_mm", wire_settings.get("plane_threshold_mm")),
                    18.0,
                    0.5,
                    200.0,
                )
                self.wire_plane_threshold.setValue(
                    depth_band
                )
                self.wire_top_percentile.setValue(
                    as_float(wire_settings.get("top_percentile"), 1.0, 0.1, 50.0)
                )
                self.wire_depth_min.setValue(
                    as_float(wire_settings.get("depth_min_mm"), 70.0, 0.0, 3000.0)
                )
                self.wire_depth_max.setValue(
                    as_float(wire_settings.get("depth_max_mm"), 0.0, 0.0, 5000.0)
                )
                self.wire_merge_distance.setValue(
                    as_float(wire_settings.get("merge_distance_px"), 18.0, 1.0, 200.0)
                )
                self.wire_merge_angle.setValue(
                    as_float(wire_settings.get("merge_angle_deg"), 35.0, 1.0, 90.0)
                )
                self.wire_confidence_min.setValue(
                    as_float(wire_settings.get("confidence_minimum"), 0.45, 0.0, 1.0)
                )
                sort_index = self.wire_sort_axis.findData(wire_settings.get("sort_axis"))
                self.wire_sort_axis.setCurrentIndex(max(0, sort_index))
                self.wire_sort_order.setCurrentIndex(
                    1 if bool(wire_settings.get("sort_desc", False)) else 0
                )
                sort_vector = _wire_sort_vector(wire_settings)
                self.wire_sort_vec_x.setValue(sort_vector[0])
                self.wire_sort_vec_y.setValue(sort_vector[1])
                self.wire_sort_vec_z.setValue(sort_vector[2])
                rep_index = self.wire_representative.findData(
                    wire_settings.get("representative")
                )
                self.wire_representative.setCurrentIndex(max(0, rep_index))
                self.wire_tracking_switch.set_checked_silently(
                    bool(wire_settings.get("tracking_enabled", True))
                )
                self.wire_depth_foreground_switch.set_checked_silently(
                    bool(wire_settings.get("use_depth_foreground", True))
                )
                self.wire_edge_fallback_switch.set_checked_silently(
                    bool(wire_settings.get("use_edge_fallback", True))
                )
                self.wire_morph_open_switch.set_checked_silently(
                    bool(wire_settings.get("apply_morph_open", True))
                )
                self.wire_morph_close_switch.set_checked_silently(
                    bool(wire_settings.get("apply_morph_close", True))
                )
                self.wire_area_filter_switch.set_checked_silently(
                    bool(wire_settings.get("filter_components", True))
                )
                self.wire_skeleton_switch.set_checked_silently(
                    bool(wire_settings.get("skeleton_enabled", True))
                )
                self.wire_debug_save_switch.set_checked_silently(
                    bool(wire_settings.get("save_debug_images", False))
                )

            roi_restored = self._apply_saved_roi(data.get("roi"))

            # 플랜지 포즈 복원
            flange_pose = data.get("flange_pose")
            if isinstance(flange_pose, dict):
                for key, *_ in POSE_FIELDS:
                    if key in flange_pose:
                        self.flange_inputs[key].setValue(
                            as_float(flange_pose.get(key), 0.0)
                        )

        finally:
            self._syncing = False

        # 슬라이더가 보정한 값을 입력박스에 반영하고, 고른 필터를 다시 실어 준다.
        self._sync_threshold_bounds()
        self._load_selected_filter()
        self._push_blob_params()
        self._update_detection_controls()
        # 법선 매칭점·오프셋은 저쪽 화면이 들고 있다. 같은 파일에서 왔으므로
        # 여기서 한 번에 넘겨 준다 — 두 화면이 다른 프로파일을 보면 안 된다.
        if self._pose_page is not None:
            self._pose_page.apply_pose_settings(data)

        # 어떤 세팅이 적용됐는지 알려 준다. ROI가 살아 있으면 분석 범위가
        # 제한되므로 특히 밝혀 둔다("왜 일부만 탐지되지?" 하고 헤매지 않도록).
        roi_note = ""
        if roi_restored:
            _x, _y, width, height = roi_state.get()
            roi_note = f" · ROI {width}×{height}"
        if profile_name is not None:
            self._set_status(f"'{profile_name}' 세팅 적용됨{roi_note}")
        elif roi_restored:
            self._set_status(f"저장된 ROI 복원됨 ({width}×{height}) — 이미지를 캡쳐하세요.")

        # 파라미터가 그대로여도 ROI가 바뀌었을 수 있으므로 조건 없이 다시 돌린다.
        # (위 상태문구는 탐지 결과 문구로 덮이는데, 그쪽이 더 많은 정보를 준다.)
        if self._last_captured_image is not None:
            self.analyze_image()

    # ===== 파라미터 적용 =====
    def _schedule_apply(self):
        """연속 조작이 끝난 뒤 한 번만 재분석하도록 타이머를 다시 건다."""
        if self._syncing:
            return   # 위젯끼리 값을 맞추는 중이면 재분석 대상이 아니다
        self._apply_timer.start()

    def _apply_active_params(self):
        # 블랍만 '검출기에 값을 밀어 넣고 바뀐 게 있을 때' 다시 분석한다.
        # 나머지 방식은 검출기를 그때그때 새로 만드므로 바로 다시 분석하면 된다.
        if self.method_combo.currentData() == DETECTION_BLOB:
            self._apply_blob_params()
        elif self._last_captured_image is not None:
            self.analyze_image()

    def _push_blob_params(self):
        """UI 값을 검출기에 반영한다.

        Returns:
            bool: 실제로 달라진 값이 있었는지
        """
        changed = self._analyzer.set_thresholds(
            self.spin_min_threshold.value(),
            self.spin_max_threshold.value(),
            self.spin_threshold_step.value(),
        )
        changed |= self._analyzer.set_min_dist(self.spin_min_dist.value())
        for key, state in self._filter_states.items():
            changed |= self._analyzer.set_filter(key, state["enabled"], state["value"])
        return changed

    def _apply_blob_params(self):
        """UI 값을 검출기에 반영하고, 바뀌었으면 다시 분석한다."""
        if self._push_blob_params() and self._last_captured_image is not None:
            self.analyze_image()

    # ===== 카메라 =====
    def _find_camera_page(self):
        """페이지 스택에서 CameraCapturePage 인스턴스를 찾는다.

        PAGES 순서가 바뀌어도 동작하도록 index를 고정하지 않고 훑는다.
        없으면 None(= 카메라 페이지가 제거된 구성).
        """
        stack = getattr(self._window, "stack", None)
        if stack is None:
            return None
        for i in range(stack.count()):
            page = stack.widget(i)
            if isinstance(page, CameraCapturePage):
                return page
        return None

    def _on_capture_clicked(self):
        """Camera Capture 페이지의 카메라로 한 장 찍고 곧바로 분석까지 수행한다."""
        camera_page = self._find_camera_page()
        if camera_page is None:
            self._set_status("Camera Capture 페이지를 찾을 수 없습니다.", "error")
            return

        capture_started = time.perf_counter()
        try:
            rgb, depth = camera_page.grab_frames()
        except Exception as exc:
            self._set_status(f"캡쳐 실패: {exc}", "error")
            self._clear_elapsed()
            return

        # 이후 OpenCV 처리를 위해 BGR로 보관한다. 뎁스는 뎁스맵 화면과
        # 이후 3D 좌표 계산에 쓰려고 같은 시점의 것을 함께 들고 있는다.
        self._last_captured_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._last_depth = depth
        # 포즈는 Pose Setting 이 세우지만, 내부 파라미터와 뎁스 단위는 **캡쳐
        # 시점의 것**이라야 한다. 그래서 여기서 붙잡아 두고 capture_context() 로
        # 이미지·뎁스와 한 묶음으로 넘긴다(카메라 연결이 끊긴 뒤에도 유효하다).
        self._last_intrinsics = camera_page.color_intrinsics()
        self._depth_scale_mm = camera_page.depth_scale_mm()
        self._last_capture_save_note = camera_page.auto_save_frame_log(rgb, depth)
        # Pose Setting 이 탭으로 들어올 때 이 번호가 올라 있으면 새로 이어받는다.
        self._capture_seq += 1
        capture_seconds = time.perf_counter() - capture_started

        self.btn_roi.setEnabled(True)
        self.btn_template_region.setEnabled(True)
        self.btn_sam2_teach.setEnabled(sam2_detector.available())
        # 새 장면이므로 확대/이동을 되돌린다. 파라미터를 고쳐 다시 그릴 때는
        # 그대로 두지만(보던 자리를 매번 다시 찾지 않게), 장면이 바뀌면 확대해 둔
        # 자리가 무엇을 가리키는지 알 수 없다.
        self.result_board.reset_view()
        self.analyze_image(capture_seconds)

    # ===== ROI =====
    def _on_roi_mode_toggled(self, checked):
        if checked:
            self._release_drag_buttons(keep=self.btn_roi)
            self._switch_to_color_for_selection()
            self.result_board.set_selection_mode("roi")
            self._set_status("결과 이미지를 드래그해 ROI를 지정하세요 (작게 클릭하면 해제).")
        else:
            self.result_board.set_selection_mode(None)

    def _switch_to_color_for_selection(self):
        index = self.view_combo.findData(VIEW_COLOR)
        if index >= 0 and self.view_combo.currentData() == VIEW_DEBUG:
            self.view_combo.setCurrentIndex(index)

    def _on_roi_changed(self):
        """드래그가 끝나면 새 ROI로 다시 분석한다."""
        self.analyze_image()

    # ===== 화면 전환 =====
    def _on_view_changed(self):
        """콤보를 바꾸면 다시 분석하지 않고 화면만 갈아 끼운다."""
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
            # 뎁스와 컬러 해상도가 다르면 ROI/홀 좌표가 어긋나므로 컬러 기준으로 맞춘다.
            height, width = self._last_captured_image.shape[:2]
            if bgr.shape[:2] != (height, width):
                bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_NEAREST)
            return bgr

        if view == VIEW_BINARY:
            if self._last_binary is None:
                return None
            return cv2.cvtColor(self._last_binary, cv2.COLOR_GRAY2BGR)

        if view == VIEW_DEBUG:
            frames = list(self._last_debug_frames)
            if not frames and self._last_captured_image is not None:
                frames = [
                    DebugFrame("Color", self._last_captured_image),
                    DebugFrame("Mask", self._last_binary),
                ]
            height, width = self._last_captured_image.shape[:2]
            return make_contact_sheet(frames, output_size=(width, height))

        return None

    def _render_current_view(self):
        """현재 고른 화면에 마지막 분석 결과를 덧그려 보드에 표시한다.

        표시할 수 없으면 사유를 상태문구로 알리고 False를 반환한다(호출한 쪽이
        그 위에 다른 문구를 덮어쓰지 않도록).
        """
        view = self.view_combo.currentData()
        if view == VIEW_DEBUG and (
            self.btn_roi.isChecked()
            or self.btn_template_region.isChecked()
            or self.btn_sam2_teach.isChecked()
        ):
            self._release_drag_buttons(keep=None)
        base = self._build_view_image(view)
        if base is None:
            self.result_board.clear_image()
            reason = {
                VIEW_DEPTH: "뎁스 데이터가 없습니다 (가상 카메라 로그에는 depth_raw가 필요합니다).",
                VIEW_BINARY: "이진화 이미지가 없습니다. 먼저 이미지를 캡쳐하세요.",
                VIEW_DEBUG: "디버그 이미지가 없습니다. 먼저 이미지를 캡쳐하세요.",
            }.get(view, "표시할 이미지가 없습니다.")
            self._set_status(reason, "error")
            return False

        if view == VIEW_DEBUG:
            self.result_board.set_roi_overlay_enabled(False)
            self.result_board.set_image_bgr(base)
            return True

        self.result_board.set_roi_overlay_enabled(True)
        # 좌표축은 그리지 않는다. 자세를 세우는 것은 Pose Setting 이고, 여기서
        # 포즈 없이 그린 축은 실제로 로봇에 나갈 값과 다르다(법선 매칭점과
        # 진입 각도 오프셋이 저쪽에 있다). 없는 편이 거짓말하지 않는다.
        visualized = BlobResultVisualizer.draw_holes(
            base,
            self._last_holes,
            hole_masks=self._last_hole_masks,
        )
        wires = self._last_debug_info.get("wire_candidates") if self._last_debug_info else None
        if wires:
            visualized = draw_wires(visualized, wires)
        self.result_board.set_image_bgr(visualized)
        return True

    # ===== 분석 =====
    def analyze_image(self, capture_seconds=None):
        """캡쳐된 이미지에 대해 블롭 분석을 수행하고 결과를 보드에 그린다.

        Args:
            capture_seconds: 방금 캡쳐해서 넘어온 경우 그 캡쳐에 걸린 시간(초).
                ROI/파라미터 변경으로 다시 분석할 때는 None(분석 시간만 표시).
        """
        if self._last_captured_image is None:
            self._set_status("먼저 이미지를 캡쳐하세요.", "error")
            return

        analysis_started = time.perf_counter()
        # 단계별 소요 시간. 합계만 보면 어디가 느린지 알 수 없다 — 같은 세팅에서
        # 소스 실행과 exe 의 분석 시간이 두 배 차이 났을 때 짚을 단서가 없었다.
        self._stage_seconds = {}
        try:
            stage_started = time.perf_counter()
            holes, debug_info = self.detect_holes(self._last_captured_image)
            self._stage_seconds["홀"] = time.perf_counter() - stage_started
        except Exception as exc:
            self._set_status(f"분석 실패: {exc}", "error")
            self._clear_elapsed()
            return
        analysis_seconds = time.perf_counter() - analysis_started

        self._last_holes = holes
        self._last_binary = debug_info["binary"]
        self._last_hole_masks = debug_info.get("hole_masks") or []
        self._last_debug_info = dict(debug_info)
        self._last_debug_frames = list(debug_info.get("debug_frames") or [])
        if not self._last_debug_frames:
            final = BlobResultVisualizer.draw_holes(
                self._last_captured_image,
                holes,
                hole_masks=self._last_hole_masks,
            )
            self._last_debug_frames = [
                DebugFrame("Original", self._last_captured_image),
                DebugFrame("Detection mask", self._last_binary),
                DebugFrame("Final result", final),
            ]
        # 화면에 그리는 시간은 '블롭 계산' 시간이 아니므로 측정에서 뺀다.
        self._set_elapsed(capture_seconds, analysis_seconds)

        if not self._render_current_view():
            return   # 표시 실패 사유가 이미 상태문구에 있으므로 덮어쓰지 않는다
        self._set_result_status()

    # ===== 처리 시간 표시 =====
    def _set_elapsed(self, capture_seconds, analysis_seconds):
        """이미지 보드 우측 상단에 걸린 시간을 띄운다.

        캡쳐를 거쳐 온 경우에는 캡쳐/분석을 나눠 보여 준다. 카메라 대기 때문에
        느린 건지 블롭 계산이 무거운 건지 구분되어야 튜닝에 쓸 수 있다.

        분석은 다시 단계별로 쪼갠다. 합계만으로는 어디가 느린지 알 수 없어서,
        같은 세팅에서 소스 실행과 exe 의 분석 시간이 두 배 차이 났을 때 원인을
        짚을 단서가 없었다.
        """
        analysis_text = f"분석 {analysis_seconds * 1000:.1f} ms"
        stages = getattr(self, "_stage_seconds", None)
        if stages:
            # 0.1ms 미만 단계는 뺀다(꺼진 기능까지 0.0 으로 늘어놓으면 읽기 나쁘다).
            parts = [
                f"{name} {seconds * 1000:.0f}"
                for name, seconds in stages.items()
                if seconds >= 0.0001
            ]
            if parts:
                analysis_text += "  [" + " · ".join(parts) + "]"
        if capture_seconds is None:
            self.elapsed_label.setText(analysis_text)
            return
        total = capture_seconds + analysis_seconds
        self.elapsed_label.setText(
            f"캡쳐 {capture_seconds * 1000:.1f} ms + {analysis_text}"
            f"  =  {total * 1000:.1f} ms"
        )

    def _clear_elapsed(self):
        """실패했을 때 이전 측정값이 남아 오해를 주지 않도록 지운다."""
        self.elapsed_label.setText("")

    # ===== 상태 표시 =====
    def _set_result_status(self):
        """마지막 분석 결과(분석 범위 + 홀 개수)를 상태문구에 띄운다."""
        if self._last_captured_image is None:
            return
        height, width = self._last_captured_image.shape[:2]
        roi = roi_state.clamped_to(width, height)
        scope = f"ROI {roi[2]}×{roi[3]}" if roi else f"전체 {width}×{height}"
        count = len(self._last_holes)
        target_name = "wire" if self.method_combo.currentData() == DETECTION_WIRE else "홀"
        tail = ""
        if self.method_combo.currentData() == DETECTION_WIRE and self._last_debug_info:
            wire_note = self._last_debug_info.get("wire_note")
            debug_path = self._last_debug_info.get("debug_path")
            if wire_note:
                tail += f" · {wire_note}"
            if debug_path:
                tail += " · debug 저장됨"
        if self._last_capture_save_note:
            tail += self._last_capture_save_note
        self._set_status(
            f"{scope} — {target_name} {count}개 탐지{tail}",
            "success" if count else "error",
        )

    def _set_status(self, text, status_type="info"):
        self.status_text.setText(text)
        self.status_led.setProperty(
            "connected", "true" if status_type == "success" else "false"
        )
        self.status_led.style().unpolish(self.status_led)
        self.status_led.style().polish(self.status_led)
