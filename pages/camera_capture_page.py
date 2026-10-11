"""카메라 선택 페이지 — RealSense D405 한 장 촬영 + 뎁스 품질 조정.

이 페이지가 '카메라에서 뎁스가 어떻게 나올지'를 혼자 정한다. 여기서 고른
프리셋과 후처리는 grab_frames() 를 쓰는 **모든 페이지**(Hole Detection,
EIH Calibration, 오퍼레이터)에 그대로 실린다.

    프리셋 (realsense_preset)     센서가 뎁스를 만드는 방식. Preset/*.json
    후처리 (realsense_postprocess) 만들어진 뎁스를 다듬는 방식. 취득 스레드에서 건다

구성:

    [카메라 ▼] [탐색] [연결] [이미지 캡쳐] [컬러맵 ▼]      ● 상태문구
    ┌───────────────────────────────────┬──────────────┐
    │                                   │ Preset  [▼]  │
    │   컬러맵 또는 뎁스맵 (콤보로 전환)  │ 뎁스 후처리   │
    │                                   │  ▼ 데시메이션 │
    └───────────────────────────────────┴──────────────┘

촬영 방식 (260910)
    라이브 스트림(재생/정지)을 걷어내고 다른 페이지와 같은 '한 장 촬영'으로
    바꿨다. 초당 수십 장을 화면에 옮기는 동안에는 뎁스 프로파일을 재거나
    후처리 값을 비교할 수가 없었고(화면이 계속 갈린다), 정작 다른 페이지와
    오퍼레이터는 모두 한 장씩 찍어 쓴다 — 화면에서 보던 것과 실제로 쓰는 것이
    같아야 뎁스 품질을 여기서 맞출 수 있다.

    카메라 소스는 연결되는 순간부터 전용 스레드에서 프레임을 받아 캐시한다
    (GUI 스레드가 카메라를 기다리지 않게 하려는 것). '이미지 캡쳐'는 그 캐시의
    최신 한 장을 화면으로 옮긴다.

보드 하나 + 전환 콤보 (260910)
    컬러맵과 뎁스맵을 좌우에 나란히 띄우던 것을 보드 하나로 합쳤다. 둘로 나누면
    보드마다 폭이 절반이라 홀 하나를 들여다볼 수 없었다. 이제 콤보로 갈아 끼우고,
    보드에서는 휠 확대와 우클릭 드래그 이동이 된다(zoom_pan).

가상 카메라 모드
    카메라 선택 콤보의 첫 항목. 고르면 이미지 폴더 선택창이 떠서, 실제 카메라
    없이도 날짜 폴더 아래의 color/depth 로그를 순서대로 넘겨 볼 수 있다.
    '이미지 캡쳐'가 다음 이미지로 넘어가는 구실을 한다. 저장된 뎁스는 촬영
    당시 이미 후처리를 거친 값이라 여기서 다시 걸지 않는다.

디자인 값은 전부 theme.py 에서 가져오며, 이 파일에는 색/크기 상수가 없다.
"""

import json
import os
import threading

import cv2
import numpy as np
from PyQt5.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWIDGETSIZE_MAX,
)

from elided_label import ElidedLabel
import realsense_postprocess
import realsense_preset
from depth_view import colorize_depth
from postprocess_panel import PostProcessPanel
from depth_profile_dialog import DepthProfileDialog, sample_depth_line
from folder_dialog import select_folder
from image_saver import (
    default_log_root,
    display_path as display_log_path,
    save_frames,
)
from pages.base_page import BasePage
from theme import (
    CONTENT_SPACING,
    DEPTH_PROFILE_LINE,
    VIDEO_MIN_HEIGHT,
    VIDEO_MIN_WIDTH,
    ZOOM_RESET_HINT,
)
from toggle_switch import ToggleSwitch
from zoom_pan import ZoomPanMixin

try:  # pyrealsense2가 없어도 가상 카메라 모드로 UI는 그대로 동작한다.
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - 런타임 환경 의존
    rs = None

# ===== 기능 파라미터 (디자인이 아닌 동작 값이므로 이 파일에 둔다) =====
COLOR_WIDTH = 1280         # D405 컬러 스트림 해상도
COLOR_HEIGHT = 720
DEPTH_WIDTH = 1280         # D405 뎁스 스트림 해상도
DEPTH_HEIGHT = 720
COLOR_FPS = 30
FIRST_FRAME_TIMEOUT_MS = 5000   # 연결 직후 첫 프레임 대기 한도
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

VIRTUAL_LABEL = "가상 카메라 모드 (이미지 폴더)"

# ===== 이미지 보드에 띄울 화면 종류 =====
# 보드가 하나라 콤보로 갈아 끼운다. 다른 페이지의 VIEW_CHOICES 와 같은 꼴이다.
VIEW_COLOR = "color"
VIEW_DEPTH = "depth"
VIEW_CHOICES = [
    ("컬러맵", VIEW_COLOR),
    ("뎁스맵", VIEW_DEPTH),
]

# D405 뎁스 작동 범위 (mm)
DEPTH_MIN_MM = 70
DEPTH_MAX_MM = 500

# 뎁스를 컬러 시점으로 정렬할지. 홀의 3D 포즈는 '컬러에서 찾은 픽셀의 뎁스'를
# 읽어 계산하므로, 두 스트림의 픽셀이 서로 같은 지점을 가리켜야 한다. D405는
# 컬러와 뎁스가 같은 이미저에서 나와 사실상 이미 정렬돼 있지만, 다른 기종
# (컬러 이미저가 따로 있는 D435 등)까지 그대로 맞도록 정렬을 걸어 둔다.
ALIGN_DEPTH_TO_COLOR = True

# 뎁스 값 1당 mm. 장치에서 못 읽으면 이 값(= 값이 곧 mm)으로 본다.
DEFAULT_DEPTH_SCALE_MM = 1.0

# D405 카메라 내부 파라미터 기본값 (기종 고정)
# D405는 1280×720 해상도에서 이 값을 갖는다.
D405_DEFAULT_CAMERA_MATRIX = np.array(
    [[614.0, 0.0, 325.0], [0.0, 614.0, 238.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
D405_DEFAULT_DIST_COEFFS = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)


# ======================================================
# 카메라 소스
# ======================================================
class _FolderSource:
    """가상 카메라: 날짜 폴더의 color/depth 로그를 순서대로 내보낸다."""

    def __init__(self, folder):
        self.folder = folder
        self.frames = self._list_frames(folder)
        self._index = -1
        self.depth_preview = None
        self.raw_depth = None
        self._depth_scale_mm = self._load_depth_scale(folder)
        self.last_name = ""

    @staticmethod
    def _list_images(folder):
        try:
            names = os.listdir(folder)
        except OSError:
            return []
        return sorted(
            os.path.join(folder, n)
            for n in names
            if n.lower().endswith(IMAGE_EXTENSIONS)
        )

    @classmethod
    def _list_frames(cls, folder):
        color_folder = os.path.join(folder, "color")
        depth_folder = os.path.join(folder, "depth")
        raw_depth_folder = os.path.join(folder, "depth_raw")

        if not os.path.isdir(color_folder):
            return [(path, None, None) for path in cls._list_images(folder)]

        depth_by_key = {
            cls._frame_key(path, "depth_"): path
            for path in cls._list_images(depth_folder)
        }
        raw_depth_by_key = {
            cls._frame_key(path, "depth_raw_"): path
            for path in cls._list_images(raw_depth_folder)
        }
        return [
            (
                color_path,
                depth_by_key.get(cls._frame_key(color_path, "color_")),
                raw_depth_by_key.get(cls._frame_key(color_path, "color_")),
            )
            for color_path in cls._list_images(color_folder)
        ]

    @staticmethod
    def _frame_key(path, prefix):
        name = os.path.basename(path)
        return name[len(prefix):] if name.lower().startswith(prefix) else name

    @staticmethod
    def _read_rgb(path):
        with open(path, "rb") as fp:
            buffer = np.frombuffer(fp.read(), dtype=np.uint8)
        bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"이미지를 읽을 수 없습니다: {os.path.basename(path)}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_raw_depth(path):
        with open(path, "rb") as fp:
            buffer = np.frombuffer(fp.read(), dtype=np.uint8)
        depth = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            raise RuntimeError(
                f"원본 뎁스 데이터를 읽을 수 없습니다: {os.path.basename(path)}"
            )
        return depth

    @staticmethod
    def _load_depth_scale(folder):
        path = os.path.join(folder, "depth_scale.json")
        try:
            with open(path, "r", encoding="utf-8") as fp:
                scale = float(json.load(fp).get("depth_scale_mm", 1.0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return DEFAULT_DEPTH_SCALE_MM
        return scale if scale > 0.0 else DEFAULT_DEPTH_SCALE_MM

    def depth_scale_mm(self):
        return self._depth_scale_mm

    def intrinsics(self):
        """가상 카메라의 D405 기본 내부 파라미터를 반환한다.

        가상 모드에서도 홀 3D 포즈 계산 시 같은 카메라 기본값을 쓸 수 있게 한다.
        실제 카메라처럼 (camera_matrix, dist_coeffs) 튜플을 반환한다.
        """
        return (D405_DEFAULT_CAMERA_MATRIX.copy(), D405_DEFAULT_DIST_COEFFS.copy())

    @property
    def description(self):
        previews = sum(depth_path is not None for _, depth_path, _ in self.frames)
        raw_depths = sum(raw_path is not None for _, _, raw_path in self.frames)
        return (
            f"{os.path.basename(self.folder) or self.folder} "
            f"(컬러 {len(self.frames)}장, 뎁스 {previews}장, 원본 {raw_depths}장)"
        )

    def open(self):
        if not self.frames:
            raise RuntimeError("선택한 날짜 폴더의 color 폴더에 이미지가 없습니다.")

    def close(self):
        pass

    def names(self):
        """폴더 안 이미지 파일 이름들(콤보에 채울 목록)."""
        return [os.path.basename(color_path) for color_path, _, _ in self.frames]

    def current_index(self):
        """가장 마지막에 내보낸 이미지의 자리(0부터). 아직 없으면 -1."""
        return self._index

    def select(self, index):
        """다음 grab() 이 이 자리의 이미지를 내보내게 한다.

        grab() 이 한 칸 앞으로 나아가면서 읽으므로 한 칸 뒤에 세워 둔다.
        """
        if not 0 <= index < len(self.frames):
            return False
        self._index = index - 1
        return True

    def grab(self):
        """다음 이미지를 RGB numpy 배열로 반환. 끝에 도달하면 처음으로 돌아간다."""
        self._index = (self._index + 1) % len(self.frames)
        color_path, depth_path, raw_depth_path = self.frames[self._index]
        self.last_name = (
            f"{self._index + 1}/{len(self.frames)}  {os.path.basename(color_path)}"
        )
        self.depth_preview = self._read_rgb(depth_path) if depth_path else None
        self.raw_depth = (
            self._read_raw_depth(raw_depth_path) if raw_depth_path else None
        )
        return self._read_rgb(color_path)


class _RealSenseSource:
    """RealSense 카메라의 컬러 및 뎁스 스트림에서 최신 프레임을 얻는다.

    RealSense 호출은 전용 스레드에서 실행한다. 장치 분리 시 pipeline.stop()이
    지연되거나 COM 상태가 바뀌어도 GUI 스레드와 폴더 선택창은 영향을 받지 않는다.
    """

    def __init__(self, serial, name, preset_name=None):
        self.serial = serial
        self.name = name
        self.preset_name = preset_name
        self._pipeline = None
        self._align = None
        self._latest_color = None
        self._latest_depth = None
        self._intrinsics = None
        self._depth_scale_mm = DEFAULT_DEPTH_SCALE_MM
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._started_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._thread = None
        self._open_error = None
        self._stream_error = None
        self.last_name = ""
        # 프리셋 적용 결과. 연결 스레드에서 채우고 화면 쪽에서 상태줄에 붙인다.
        self.preset_note = ""

    @property
    def description(self):
        return f"{self.name} ({self.serial})"

    def open(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._started_event.wait(FIRST_FRAME_TIMEOUT_MS / 1000.0):
            self._stop_event.set()
            raise RuntimeError("카메라 연결 시간이 초과되었습니다.")
        if self._open_error is not None:
            raise RuntimeError(str(self._open_error))

    def close(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._thread = None
        with self._lock:
            self._latest_color = None
            self._latest_depth = None

    def _run(self):
        pipeline = None
        try:
            # 프리셋은 스트림을 열기 전에 싣는다. param-depthunits 가 뎁스 배율을
            # 바꾸는데, 뒤에 실으면 _cache_camera_info() 가 이미 읽어 둔 옛 배율이
            # 남아 좌표가 통째로 어긋난다.
            self._apply_preset()

            config = rs.config()
            config.enable_device(self.serial)
            config.enable_stream(
                rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.rgb8, COLOR_FPS
            )
            config.enable_stream(
                rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, COLOR_FPS
            )
            pipeline = rs.pipeline()
            pipeline.start(config)
            self._pipeline = pipeline
            self._align = rs.align(rs.stream.color) if ALIGN_DEPTH_TO_COLOR else None
            self._cache_camera_info()
        except Exception as exc:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            self._pipeline = None
            self._align = None
            self._open_error = exc
            self._started_event.set()
            return

        self._started_event.set()
        # 필터는 상태를 들고 있고 스레드 안전하지 않다. 이 스레드 전용으로 하나
        # 만들어 계속 쓰고, 설정이 바뀌면 Chain 이 스스로 다시 만든다.
        chain = realsense_postprocess.Chain()
        try:
            while not self._stop_event.is_set():
                frames = pipeline.poll_for_frames()
                if not frames:
                    self._stop_event.wait(0.01)
                    continue
                # 후처리를 align 보다 먼저 건다. decimation 이 뎁스 해상도를
                # 바꾸므로, 정렬을 뒤에 걸어야 컬러와 짝이 다시 맞는다.
                frames = chain.process(frames)
                frames = self._aligned(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame and depth_frame:
                    with self._lock:
                        self._latest_color = np.asanyarray(
                            color_frame.get_data()
                        ).copy()
                        self._latest_depth = np.asanyarray(
                            depth_frame.get_data()
                        ).copy()
                    self._first_frame_event.set()
        except Exception as exc:
            self._stream_error = exc
            self._first_frame_event.set()
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
            self._align = None

    def _apply_preset(self):
        """고른 프리셋(Preset/<이름>.json)을 카메라에 싣는다.

        실패해도 연결은 계속한다. 프리셋을 못 실었다고 카메라까지 못 쓰게 되면
        현장에서 손발이 묶인다. 사유는 상태줄에 남겨 조용히 넘어가지 않게 한다.
        """
        applied, note = realsense_preset.apply_to_device(self.serial, self.preset_name)
        stream_note = (
            realsense_preset.stream_note(
                self.preset_name, (COLOR_WIDTH, COLOR_HEIGHT), COLOR_FPS
            )
            if applied
            else None
        )
        self.preset_note = " · ".join(part for part in (note, stream_note) if part)

    def _cache_camera_info(self):
        try:
            profile = self._pipeline.get_active_profile()
            sensor = profile.get_device().first_depth_sensor()
            scale_mm = float(sensor.get_depth_scale()) * 1000.0
            if scale_mm > 0.0:
                self._depth_scale_mm = scale_mm

            stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = stream.get_intrinsics()
            camera_matrix = np.array(
                [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]]
            )
            self._intrinsics = (
                camera_matrix,
                np.array(intr.coeffs, dtype=float).reshape(-1),
            )
        except Exception:
            pass

    def depth_scale_mm(self):
        """뎁스 값 1당 mm. 기종/펌웨어마다 다르므로 장치에서 직접 읽는다.

        (D405는 0.1mm 단위인 경우가 있어, 1mm로 가정하면 z가 10배로 나온다.)
        읽지 못하면 기본값을 돌려주므로 호출한 쪽은 늘 숫자를 받는다.
        """
        return self._depth_scale_mm

    def _aligned(self, frames):
        """뎁스를 컬러 시점으로 맞춘 프레임셋. 정렬이 꺼졌거나 실패하면 원본."""
        if self._align is None:
            return frames
        try:
            return self._align.process(frames)
        except Exception:   # 정렬 실패로 스트림 전체를 잃지는 않게 한다
            return frames

    def intrinsics(self):
        """컬러 스트림의 카메라 내부 파라미터를 (camera_matrix, dist_coeffs)로.

        캘리브레이션 페이지가 공장 보정값을 그대로 쓰기 위해 부른다.
        미연결이거나 못 읽으면 None.
        """
        return self._intrinsics

    def poll(self):
        """호환용 메서드. 프레임 폴링은 전용 스레드가 담당한다."""

    def grab(self):
        """(color_rgb, depth) 튜플을 반환한다.

        color_rgb: (H, W, 3) RGB 배열
        depth: (H, W) uint16 뎁스 배열 (mm 단위)
        """
        if not self._first_frame_event.wait(FIRST_FRAME_TIMEOUT_MS / 1000.0):
            raise RuntimeError("컬러 또는 뎁스 프레임을 받지 못했습니다.")
        if self._stream_error is not None:
            raise RuntimeError(str(self._stream_error))
        with self._lock:
            if self._latest_color is None or self._latest_depth is None:
                raise RuntimeError("카메라가 연결되어 있지 않습니다.")
            return self._latest_color.copy(), self._latest_depth.copy()


# ======================================================
# 이미지 보드
# ======================================================
# 이미지가 아직 없을 때 보드가 취할 비율. D405 컬러 1280x720, 뎁스 848x480 모두
# 16:9 에 가깝다. 연결 전에 보드가 세로를 다 차지하다가 연결 뒤 줄어들면 화면이 한 번
# 크게 움직이므로 처음부터 같은 비율로 둔다.
BOARD_DEFAULT_ASPECT = 16.0 / 9.0


def _limit_height_to_aspect(board, pixmap):
    """보드 **높이**를 이미지 비율에 맞춰 제한한다.

    이 페이지는 컬러·뎁스 보드가 나란히 있고 옆에 후처리 패널까지 있어 **가로가
    제한**된다. 보드가 세로를 다 채우면 16:9 이미지 위아래에 큰 검은 띠가 남는다.
    그래서 '지금 너비에서 비율을 지킬 수 있는 최대 높이'를 상한으로 걸고, 남는
    세로는 보드를 감싼 열의 위아래 스트레치가 나눠 흡수한다(가운데 정렬).

    image_board.ImageBoard._limit_to_aspect 와 축만 다르다 — 거기는 세로가
    제한되어 너비를 잘랐다.

    너비가 아직 안 자랐으면 상한을 걸지 않는다. 레이아웃 전에 계산한 값에 갇히는
    것을 막기 위해서다. 너비는 가로 레이아웃이 정하고 우리는 높이만 건드리므로
    되먹임(진동)은 없다.
    """
    if board.width() < VIDEO_MIN_WIDTH:
        if board.maximumHeight() != QWIDGETSIZE_MAX:
            board.setMaximumHeight(QWIDGETSIZE_MAX)
        return
    if pixmap is None or pixmap.width() <= 0 or pixmap.height() <= 0:
        ratio = BOARD_DEFAULT_ASPECT
    else:
        ratio = pixmap.width() / pixmap.height()
    limit = max(VIDEO_MIN_HEIGHT, int(round(board.width() / ratio)))
    if limit != board.maximumHeight():
        board.setMaximumHeight(limit)


def _centered_column(board):
    """보드를 위아래 스트레치 사이에 끼워 세로 가운데에 놓는 열.

    레이아웃 항목에 Qt.AlignVCenter 를 주는 방법은 쓰지 않는다. 정렬 플래그가
    붙은 항목은 Qt 가 sizeHint 크기로 줄여 버려서, Ignored 정책인 이 보드는 최소
    크기까지 쪼그라든다. 스트레치 샌드위치는 그 함정이 없다.
    보드 쪽 스트레치를 크게 둬야 상한 아래에서는 세로를 꽉 채운다.
    """
    column = QVBoxLayout()
    column.setContentsMargins(0, 0, 0, 0)
    column.setSpacing(0)
    column.addStretch(1)
    column.addWidget(board, 100)
    column.addStretch(1)
    return column


class _CaptureBoard(ZoomPanMixin, QLabel):
    """찍은 한 장을 보여 주는 보드. 컬러맵 / 뎁스맵을 콤보로 갈아 끼운다.

    두 화면을 한 보드가 맡는 이유(260910): 좌우로 나누면 보드마다 폭이 절반이라
    홀 하나를 들여다볼 수 없었다. 어차피 두 화면을 나란히 견주는 일보다 한쪽을
    크게 보는 일이 많다.

    뎁스맵일 때는 **좌클릭 드래그**로 직선 프로파일을 잰다. 확대/이동은 휠과
    우클릭이라(zoom_pan) 서로 부딪히지 않는다.

    그림은 paintEvent 에서 직접 그린다 — 확대·이동한 자리와 프로파일 선, 좌표
    변환이 같은 사각형(_display_rect) 하나를 보게 하려는 것이다.
    """

    profile_selected = pyqtSignal(object, object)
    profile_unavailable = pyqtSignal()

    def __init__(self, placeholder, parent=None):
        super().__init__(placeholder, parent)
        self.setObjectName("videoView")       # 스타일은 theme.py의 QLabel#videoView
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(VIDEO_MIN_WIDTH, VIDEO_MIN_HEIGHT)
        # sizeHint를 무시해 픽스맵 크기와 무관하게 자유롭게 늘고 줄어들게 한다.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setToolTip(ZOOM_RESET_HINT)
        self._placeholder = placeholder
        self._source_pixmap = None
        self._color_rgb = None
        self._depth = None            # 원본 uint16 뎁스 (프로파일과 색칠에 쓴다)
        self._depth_preview = None    # 로그에 저장된 컬러맵 뎁스 이미지 (RGB)
        self._depth_scale_mm = 1.0
        self._view = VIEW_COLOR
        self._line_start = None
        self._line_end = None
        self._init_zoom_pan()

    # ===== 프레임 받기 =====
    def set_frames(self, rgb, depth, depth_preview=None, depth_scale_mm=1.0):
        """찍은 한 장을 넣는다. → 지금 고른 화면을 그릴 수 있었는지

        컬러와 뎁스를 함께 들고 있다가 콤보에 따라 갈아 끼운다. 갈아 끼울 때
        카메라에 다시 묻지 않으므로 두 화면이 늘 **같은 시점**의 것이다.
        """
        self._color_rgb = None if rgb is None else np.ascontiguousarray(rgb)
        self._depth = None if depth is None else np.asarray(depth).copy()
        self._depth_preview = (
            None if depth_preview is None else np.ascontiguousarray(depth_preview)
        )
        self._depth_scale_mm = float(depth_scale_mm)
        self._line_start = None
        self._line_end = None
        self.reset_view()
        return self._render()

    def clear_image(self):
        self._source_pixmap = None
        self._color_rgb = None
        self._depth = None
        self._depth_preview = None
        self._line_start = None
        self._line_end = None
        self.reset_view()
        self.setCursor(Qt.ArrowCursor)
        self.setText(self._placeholder)
        _limit_height_to_aspect(self, None)   # 이미지가 없어도 기본 비율은 지킨다
        self.update()

    def set_view(self, view):
        """띄울 화면을 바꾼다. → 그릴 수 있었는지

        확대는 되돌린다. 컬러맵에서 확대해 둔 자리가 뎁스맵의 같은 자리라는
        보장이 없다(해상도가 다를 수 있다).
        """
        self._view = view
        self._line_start = None
        self._line_end = None
        self.reset_view()
        return self._render()

    def has_image(self):
        return self._source_pixmap is not None

    # ===== 그리기 =====
    def _view_rgb(self):
        """지금 고른 화면의 RGB 배열. 만들 수 없으면 None."""
        if self._view == VIEW_COLOR:
            return self._color_rgb
        if self._depth is not None:
            # 뎁스는 이미 카메라 스레드에서 후처리를 거쳐 들어온다. 여기서 더
            # 손대면 화면과 좌표 계산이 다른 뎁스를 보게 되므로 색만 입힌다.
            bgr = colorize_depth(self._depth)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # 원본 뎁스가 없는 로그 폴더는 저장된 컬러맵 이미지라도 보여 준다.
        return self._depth_preview

    def _render(self):
        rgb = self._view_rgb()
        if rgb is None:
            self._source_pixmap = None
            self.setText(self._placeholder)
            self.setCursor(Qt.ArrowCursor)
            _limit_height_to_aspect(self, None)
            self.update()
            return False

        height, width = rgb.shape[:2]
        image = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888
        ).copy()   # numpy 버퍼와 분리해 두어야 배열이 해제돼도 안전하다.
        self._source_pixmap = QPixmap.fromImage(image)
        self.setText("")
        self.setCursor(
            Qt.CrossCursor if self._profile_ready() else Qt.ArrowCursor
        )
        _limit_height_to_aspect(self, self._source_pixmap)
        self.update()
        return True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        _limit_height_to_aspect(self, self._source_pixmap)
        self._clamp_pan()

    def paintEvent(self, event):
        # 바탕/테두리(그리고 이미지가 없을 때의 안내 글자)는 QLabel 이 그린다.
        super().paintEvent(event)
        if self._source_pixmap is None:
            return
        target = self._display_rect()
        if target is None:
            return
        painter = QPainter(self)
        # 확대해서 보드 밖으로 나간 부분은 잘라 낸다.
        painter.setClipRect(self.contentsRect())
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self._source_pixmap)

        if self._line_start is not None and self._line_end is not None:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(QPen(QColor(DEPTH_PROFILE_LINE), 2))
            painter.drawLine(
                self._image_to_widget(self._line_start),
                self._image_to_widget(self._line_end),
            )
        painter.end()

    # ===== 좌표 변환 =====
    def _image_to_widget(self, point):
        """원본 이미지 좌표를 위젯 좌표로 (확대/이동 반영)."""
        display = self._display_rect()
        height, width = self._source_pixmap.height(), self._source_pixmap.width()
        x_scale = display.width() / max(1, width)
        y_scale = display.height() / max(1, height)
        return QPointF(
            display.x() + (point[0] + 0.5) * x_scale,
            display.y() + (point[1] + 0.5) * y_scale,
        )

    def _widget_to_image(self, position, clamp=False):
        display = self._display_rect()
        if display is None or self._depth is None:
            return None
        if not clamp and not display.contains(position):
            return None
        image_h, image_w = self._depth.shape[:2]
        x = (position.x() - display.x()) / max(1, display.width()) * image_w
        y = (position.y() - display.y()) / max(1, display.height()) * image_h
        return (
            int(min(max(x, 0), image_w - 1)),
            int(min(max(y, 0), image_h - 1)),
        )

    # ===== 뎁스 직선 프로파일 (좌클릭 드래그) =====
    def _profile_ready(self):
        """지금 화면에서 직선 프로파일을 잴 수 있는가.

        뎁스맵 화면이고 원본 뎁스가 있을 때만이다. 컬러맵에서 재게 두면 무엇을
        잰 값인지 알 수 없다(컬러 픽셀에는 거리가 없다).
        """
        return self._view == VIEW_DEPTH and self._depth is not None

    def _restore_cursor(self):
        """우클릭 드래그가 끝난 뒤 커서를 되돌린다(zoom_pan 이 부른다).

        뎁스맵에서는 십자여야 한다 — 화살표로 남으면 직선을 잴 수 있는 화면
        인지 눈으로 알 수 없다.
        """
        self.setCursor(Qt.CrossCursor if self._profile_ready() else Qt.ArrowCursor)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)   # 확대/이동은 믹스인이 맡는다
            return
        if self._view == VIEW_DEPTH and self._depth is None:
            self.profile_unavailable.emit()
            return
        if not self._profile_ready():
            super().mousePressEvent(event)
            return
        point = self._widget_to_image(event.pos())
        if point is None:
            return
        self._line_start = point
        self._line_end = point
        self.update()
        event.accept()

    def mouseMoveEvent(self, event):
        if self._line_start is None or not (event.buttons() & Qt.LeftButton):
            super().mouseMoveEvent(event)
            return
        self._line_end = self._widget_to_image(event.pos(), clamp=True)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self._line_start is None:
            super().mouseReleaseEvent(event)
            return
        self._line_end = self._widget_to_image(event.pos(), clamp=True)
        self.update()
        self.profile_selected.emit(self._line_start, self._line_end)
        event.accept()

    def sample_profile(self, start, end):
        if self._depth is None:
            return None, None
        return sample_depth_line(self._depth, start, end, self._depth_scale_mm)


# ======================================================
# 페이지
# ======================================================
class CameraCapturePage(BasePage):
    title = "Camera Selection"
    nav_icon = "camera"
    expand_content = True   # 이미지 보드가 남는 공간을 모두 채우도록

    def __init__(self, parent=None):
        super().__init__(parent)

        self._source = None          # 현재 연결된 카메라 소스 (None이면 미연결)
        self._virtual_folder = None  # 가상 카메라 모드로 선택한 폴더
        self._depth_profile_dialogs = []

        self.content_layout.addLayout(self._build_controls())

        # 이미지 보드 하나 | 프리셋 + 후처리 패널
        boards_layout = QHBoxLayout()
        boards_layout.setSpacing(CONTENT_SPACING)

        self.board = _CaptureBoard("이미지 캡쳐 버튼을 누르면 여기에 표시됩니다.")
        self.board.profile_selected.connect(self._show_depth_profile)
        self.board.profile_unavailable.connect(
            lambda: self._set_status(
                "직선 그래프에는 원본 depth 데이터가 필요합니다."
            )
        )

        # 보드는 이미지 비율까지만 자라고(_limit_height_to_aspect) 남는 세로는
        # 위아래 스트레치가 나눠 가져 가운데에 놓인다. 오른쪽 패널은 그대로
        # 세로를 다 쓴다 — 항목이 많아 스크롤이 있는 쪽이라 높이가 아쉽다.
        boards_layout.addLayout(_centered_column(self.board), 1)
        boards_layout.addLayout(self._build_side_panel())

        self.content_layout.addLayout(boards_layout, 1)

        self.scan_cameras()
        self._update_controls()

        # 앱 종료 시 카메라를 반드시 닫는다(페이지 밖 코드 수정 없이 자체 정리).
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._disconnect)

    # ===== UI 구성 =====
    def _build_controls(self):
        row = QHBoxLayout()
        row.setSpacing(CONTENT_SPACING)

        self.camera_combo = QComboBox()
        self.camera_combo.setObjectName("cameraCombo")
        # activated: 사용자가 직접 고를 때만 발생(프로그램이 목록을 바꿀 때는 안 뜬다)
        self.camera_combo.activated.connect(self._on_camera_selected)

        self.btn_scan = QPushButton("카메라 탐색")
        self.btn_scan.setObjectName("secondaryBtn")
        self.btn_scan.clicked.connect(self.scan_cameras)

        self.btn_connect = QPushButton("카메라 연결")
        self.btn_connect.setObjectName("connectToggle")
        self.btn_connect.setCheckable(True)
        self.btn_connect.clicked.connect(self._on_connect_clicked)

        # 한 장 촬영. 다른 페이지와 같은 이름/같은 방식이다(라이브 재생은 260910
        # 에 걷어냈다 — 이 파일 맨 위 설명 참고).
        self.btn_capture = QPushButton("이미지 캡쳐")
        self.btn_capture.setObjectName("primaryBtn")
        self.btn_capture.clicked.connect(self.capture_image)

        self.auto_save_label = QLabel("자동 저장")
        self.auto_save_label.setObjectName("sectionLabel")
        self.auto_save_switch = ToggleSwitch()
        self.auto_save_switch.setChecked(True)
        self.auto_save_switch.setToolTip(
            "캡쳐한 컬러 이미지, depth preview, raw depth를 Logs 폴더에 저장합니다."
        )

        # 가상 카메라 모드에서만 쓰는 이미지 선택 콤보. 실제 카메라에는
        # 고를 이미지가 없으므로 그때는 숨긴다(_update_controls).
        self.virtual_combo = QComboBox()
        self.virtual_combo.setObjectName("profileCombo")
        self.virtual_combo.setToolTip("가상 카메라로 내보낼 이미지")
        self.virtual_combo.activated.connect(self._on_virtual_image_selected)
        self.virtual_combo.hide()

        # 보드에 띄울 화면. 컬러맵과 뎁스맵이 한 보드를 나눠 쓴다.
        self.view_combo = QComboBox()
        self.view_combo.setObjectName("viewCombo")
        for label, key in VIEW_CHOICES:
            self.view_combo.addItem(label, key)
        self.view_combo.setToolTip(
            "보드에 띄울 화면. 둘 다 같은 촬영에서 나온 것이라 다시 찍지 않는다."
        )
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)

        self.led = QLabel()
        self.led.setObjectName("statusLed")

        # 연결 메시지는 길다(장치·시리얼·프리셋·해상도 경고, 약 1800px). 보통
        # QLabel 이면 그 길이가 페이지 최소 폭이 되어 창이 모니터를 넘는다
        # (실측: 1576 -> 3212px). 안 들어가면 가운데를 줄이고 원문은 툴팁에.
        self.status = ElidedLabel()
        self.status.setObjectName("cameraStatus")

        row.addWidget(self.camera_combo)
        row.addWidget(self.btn_scan)
        row.addWidget(self.btn_connect)
        row.addWidget(self.btn_capture)
        row.addWidget(self.auto_save_label)
        row.addWidget(self.auto_save_switch)
        row.addWidget(self.virtual_combo)
        row.addWidget(self.view_combo)
        row.addStretch()
        row.addWidget(self.led)
        row.addWidget(self.status)
        return row

    def _build_side_panel(self):
        """오른쪽 세로 칸: 프리셋(위) + 뎁스 후처리(아래).

        둘 다 '뎁스를 어떻게 만들고 다듬을지'라서 한 칸에 모았다. 프리셋은
        센서가 뎁스를 만드는 방식이고 후처리는 만들어진 뎁스를 다듬는 방식이라,
        위에서 아래로 그 순서대로 놓인다.
        """
        column = QVBoxLayout()
        column.setSpacing(CONTENT_SPACING)

        # 뎁스 엔진 프리셋. 뷰어에서 맞춘 값을 Preset/ 에 넣어 두고 고른다.
        preset_row = QHBoxLayout()
        preset_row.setSpacing(CONTENT_SPACING)
        self.preset_label = QLabel("Preset")
        self.preset_label.setObjectName("sectionLabel")
        self.preset_combo = QComboBox()
        self.preset_combo.setObjectName("profileCombo")
        self.preset_combo.activated.connect(self._on_preset_selected)
        self._reload_presets()
        preset_row.addWidget(self.preset_label)
        preset_row.addWidget(self.preset_combo, 1)
        column.addLayout(preset_row)

        # 후처리는 뎁스맵을 보며 맞추는 값이라 뎁스 화면 바로 옆에 둔다.
        self.postprocess_panel = PostProcessPanel()
        self.postprocess_panel.changed.connect(self._on_postprocess_changed)
        column.addWidget(self.postprocess_panel, 1)
        return column

    # ===== 뎁스 엔진 프리셋 =====
    def _reload_presets(self):
        """Preset/ 폴더를 다시 훑어 콤보를 채운다. 고른 것은 되도록 이어 간다.

        파일을 넣거나 빼는 일이 프로그램 밖에서 일어나므로, 카메라를 다시 훑을
        때마다 함께 갱신한다. 고르지 않음(None)이 늘 첫 항목이다.
        """
        previous = self.current_preset()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("사용 안 함", None)
        for name in realsense_preset.preset_names():
            self.preset_combo.addItem(name, name)
        index = self.preset_combo.findData(previous)
        self.preset_combo.setCurrentIndex(max(0, index))
        self.preset_combo.blockSignals(False)
        self._update_preset_tooltip()

    def _update_preset_tooltip(self):
        self.preset_combo.setToolTip(
            realsense_preset.describe(self.current_preset())
            + "\n\nPreset/ 폴더의 JSON 을 읽는다 (RealSense Viewer 의 Save Settings).\n"
            "뎁스 단위까지 바꿀 수 있어 카메라를 열 때 실린다 —\n"
            "연결 중에 바꾸면 다시 연결해야 적용된다."
        )

    def current_preset(self):
        """고른 프리셋 이름. 안 골랐으면 None."""
        combo = getattr(self, "preset_combo", None)
        return None if combo is None else combo.currentData()

    def _on_preset_selected(self, _index):
        self._update_preset_tooltip()
        if not isinstance(self._source, _RealSenseSource):
            return
        # 프리셋은 pipeline 을 열기 전에만 실을 수 있다(뎁스 단위가 바뀌므로).
        # 연결된 채로 콤보만 바꾸면 화면과 실제가 어긋나므로 바로 다시 연결한다.
        self._set_status("프리셋을 바꿔 카메라를 다시 연결합니다...")
        self._disconnect()
        self._connect()
        self._update_controls()

    # ===== 화면 전환 =====
    def _on_view_changed(self):
        """콤보를 바꾸면 다시 찍지 않고 화면만 갈아 끼운다."""
        if not self.board.set_view(self.view_combo.currentData()):
            self._set_status(self._missing_view_reason())

    def _missing_view_reason(self):
        """고른 화면을 못 그리는 사유. 화면마다 원인이 다르다."""
        if self.view_combo.currentData() == VIEW_DEPTH:
            return (
                "뎁스 데이터가 없습니다 "
                "(가상 카메라 로그에는 depth 또는 depth_raw 가 필요합니다)."
            )
        return "표시할 이미지가 없습니다. 먼저 이미지를 캡쳐하세요."

    # ===== 후처리 =====
    def _on_postprocess_changed(self):
        """후처리 값이 바뀌었을 때 한 장 다시 받아 결과를 보여 준다.

        화면에 남은 것은 바꾸기 전의 프레임이라 그대로 두면 값이 먹었는지 알 수
        없다. 후처리는 취득 스레드에서 걸리므로 다음 한 장부터 반영된다.
        """
        if self._source is not None:
            self._refresh_once()
        self._set_status(realsense_postprocess.summary())

    # ===== 카메라 목록 =====
    def scan_cameras(self):
        """연결된 RealSense 장치를 다시 조회해 콤보를 채운다.

        index 0은 항상 가상 카메라 모드. 그 뒤로 발견된 장치가 붙는다.
        각 항목의 userData에는 (kind, serial, name)을 담아 둔다.

        프리셋 목록도 함께 갱신한다. 파일을 넣고 빼는 일은 프로그램 밖에서
        일어나므로, 사용자가 '다시 훑어라'라고 누르는 이 자리가 알맞다.
        """
        previous = self.camera_combo.currentData()
        self._reload_presets()

        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItem(VIRTUAL_LABEL, ("virtual", None, VIRTUAL_LABEL))

        devices, error = self._query_devices()
        for serial, name in devices:
            self.camera_combo.addItem(f"{name} ({serial})", ("realsense", serial, name))

        # 탐색 전에 고른 장치가 여전히 있으면 선택을 유지한다.
        if previous is not None:
            for i in range(self.camera_combo.count()):
                if self.camera_combo.itemData(i) == previous:
                    self.camera_combo.setCurrentIndex(i)
                    break
        self.camera_combo.blockSignals(False)

        if error:
            self._set_status(f"카메라 탐색 실패: {error}")
        elif devices:
            self._set_status(f"카메라 {len(devices)}대 발견")
        else:
            self._set_status("카메라를 찾지 못했습니다. 가상 카메라 모드를 사용하세요.")

    @staticmethod
    def _query_devices():
        if rs is None:
            return [], "pyrealsense2 미설치"
        result = {"devices": [], "error": None}

        def query():
            try:
                for device in rs.context().query_devices():
                    result["devices"].append(
                        (
                            device.get_info(rs.camera_info.serial_number),
                            device.get_info(rs.camera_info.name),
                        )
                    )
            except Exception as exc:
                result["error"] = str(exc)

        thread = threading.Thread(target=query, daemon=True)
        thread.start()
        thread.join(timeout=3.0)
        if thread.is_alive():
            return [], "카메라 탐색 시간이 초과되었습니다."
        return result["devices"], result["error"]

    def _on_camera_selected(self, index):
        """가상 카메라 모드를 고르면 곧바로 이미지 폴더를 묻는다.

        폴더 대화상자는 이 신호가 끝난 뒤에 띄운다. activated 는 콤보 목록이
        아직 닫히는 중에 오는데, 그때 모달 대화상자를 띄우면 닫히던 목록이
        마우스를 붙잡고 있어 대화상자가 입력을 못 받는다(멈춘 것처럼 보인다).
        """
        kind, _serial, _name = self.camera_combo.itemData(index)
        if kind == "virtual":
            QTimer.singleShot(0, self._choose_virtual_folder)
        self._update_controls()

    def _choose_virtual_folder(self):
        folder = select_folder(
            self,
            "가상 카메라로 사용할 이미지 폴더 선택",
            self._virtual_folder or "",
        )
        if not folder:
            if self._virtual_folder is None:
                self._set_status("가상 카메라 모드: 폴더가 선택되지 않았습니다.")
            return
        self._virtual_folder = folder
        source = _FolderSource(folder)
        count = len(source.frames)
        if count:
            previews = sum(
                depth_path is not None for _, depth_path, _ in source.frames
            )
            raw_depths = sum(raw_path is not None for _, _, raw_path in source.frames)
            self._set_status(
                f"가상 카메라 폴더: {os.path.basename(folder)} "
                f"(컬러 {count}장, 뎁스 {previews}장, 원본 {raw_depths}장)"
            )
        else:
            self._set_status(
                f"선택한 날짜 폴더의 color 폴더에 이미지가 없습니다: "
                f"{os.path.basename(folder)}"
            )

    # ===== 연결 / 해제 =====
    def _on_connect_clicked(self, checked):
        if checked:
            self._connect()
        else:
            self._disconnect()
            self._set_status("연결 해제됨")
        self._update_controls()

    def _connect(self):
        kind, serial, name = self.camera_combo.currentData()
        if kind == "virtual":
            if self._virtual_folder is None:
                self._choose_virtual_folder()
            if self._virtual_folder is None:
                self.btn_connect.setChecked(False)
                return
            source = _FolderSource(self._virtual_folder)
        else:
            if rs is None:
                self.btn_connect.setChecked(False)
                self._set_status("pyrealsense2가 설치되어 있지 않습니다.")
                return
            source = _RealSenseSource(serial, name, self.current_preset())

        try:
            source.open()
        except Exception as exc:
            source.close()
            self.btn_connect.setChecked(False)
            self._set_status(f"연결 실패: {exc}")
            return

        self._source = source
        if isinstance(source, _FolderSource):
            self._fill_virtual_combo(source)
        else:
            # 연결하면 한 장 찍어 보여 준다. 연결해 놓고 아무것도 안 보이면
            # 카메라가 살아 있는지부터 의심하게 된다.
            self._refresh_once()
        status = f"연결됨 — {source.description}"
        # 뎁스 배율까지 바꿀 수 있는 값이라, 실렸는지 아닌지가 늘 보여야 한다.
        preset_note = getattr(source, "preset_note", "")
        if preset_note:
            status = f"{status} · {preset_note}"
        self._set_status(status)

    def _disconnect(self):
        if self._source is not None:
            self._source.close()
            self._source = None
        # 연결이 끊긴 뒤에도 남아 있는 캡쳐 이미지는 현재 상태를 오해하게 만들므로 지운다.
        self.board.clear_image()

    # ===== 재시작 상태 =====
    def session_state(self):
        data = self.camera_combo.currentData()
        kind, serial, name = data if data else ("virtual", None, VIRTUAL_LABEL)
        return {
            "kind": kind,
            "serial": serial,
            "name": name,
            "virtual_folder": self._virtual_folder,
            "connected": self._source is not None,
            "preset": self.current_preset(),
            "auto_save": self.auto_save_enabled(),
        }

    def restore_session_state(self, state):
        if not isinstance(state, dict):
            return

        folder = state.get("virtual_folder")
        self._virtual_folder = folder if isinstance(folder, str) and folder else None
        auto_save = state.get("auto_save")
        if isinstance(auto_save, bool):
            self.auto_save_switch.setChecked(auto_save)

        # 프리셋을 카메라 연결보다 먼저 되살린다. 아래에서 이어지는 자동 재연결이
        # 이 값을 읽어 가므로, 순서가 뒤바뀌면 프리셋 없이 열린다.
        self._reload_presets()
        saved_preset = state.get("preset")
        if isinstance(saved_preset, str) and saved_preset:
            index = self.preset_combo.findData(saved_preset)
            if index >= 0:
                self.preset_combo.setCurrentIndex(index)
            else:
                # 파일이 사라졌으면 '사용 안 함'으로 두되, 조용히 넘어가지 않는다.
                self._set_status(f"프리셋 '{saved_preset}' 을 찾을 수 없어 해제했습니다.")
            self._update_preset_tooltip()

        kind = state.get("kind")
        target_index = 0
        if kind == "realsense":
            serial = state.get("serial")
            name = state.get("name")
            for index in range(self.camera_combo.count()):
                data = self.camera_combo.itemData(index)
                if data and data[0] == "realsense" and data[1] == serial:
                    target_index = index
                    break
            else:
                if isinstance(serial, str) and serial:
                    display_name = name if isinstance(name, str) and name else "RealSense"
                    self.camera_combo.addItem(
                        f"{display_name} ({serial}) [미발견]",
                        ("realsense", serial, display_name),
                    )
                    target_index = self.camera_combo.count() - 1

        self.camera_combo.setCurrentIndex(target_index)
        self._update_controls()
        if bool(state.get("connected", False)):
            QTimer.singleShot(0, self._restore_saved_connection)

    def _restore_saved_connection(self):
        if self._source is not None:
            return
        self._connect()
        self._update_controls()

    # ===== 캡쳐 =====
    def grab_frames(self):
        """현재 프레임을 (rgb, depth) 로 반환한다.

        다른 페이지(예: Hole Detection)가 이 페이지의 카메라로 한 장 찍을 때
        쓰는 공개 API다. 가상 카메라는 depth_raw 로그가 있을 때 원본 뎁스를 준다.
        미연결이거나 프레임을 못 받으면 RuntimeError를 올린다.
        """
        if self._source is None:
            raise RuntimeError("카메라가 연결되어 있지 않습니다.")
        if isinstance(self._source, _FolderSource):
            rgb = self._source.grab()
            return rgb, self._source.raw_depth
        return self._source.grab()

    def camera_status(self):
        """지금 카메라의 (이름, 연결 여부).

        오퍼레이터 화면의 상태 카드가 부르는 공개 API다. 연결되지 않았어도
        **고른 장치 이름**은 돌려준다 — '무엇을 쓰려는데 안 붙었다'와 '아무것도
        고르지 않았다'는 현장에서 손볼 곳이 다르다.
        """
        if self._source is not None:
            return self._source.description, True
        data = self.camera_combo.currentData()
        name = data[2] if data else ""
        return name, False

    def color_intrinsics(self):
        """연결된 카메라의 내부 파라미터 (camera_matrix, dist_coeffs).

        다른 페이지(Hole Detection, EIH Calibration 등)가 쓰는 공개 API다.
        실제 카메라에서 읽은 값이 있으면 그것, 가상 카메라 모드이면 D405 기본값을,
        미연결이면 None을 반환한다.
        """
        if isinstance(self._source, _RealSenseSource):
            return self._source.intrinsics()
        if isinstance(self._source, _FolderSource):
            return self._source.intrinsics()
        return None

    def depth_scale_mm(self):
        """뎁스 값 1당 mm.

        다른 페이지(Hole Detection)가 뎁스를 실제 길이로 바꿀 때 쓰는 공개 API다.
        가상 카메라 모드이거나 미연결이면 기본값(1.0)을 준다.
        """
        if isinstance(self._source, _RealSenseSource):
            return self._source.depth_scale_mm()
        if isinstance(self._source, _FolderSource):
            return self._source.depth_scale_mm()
        return DEFAULT_DEPTH_SCALE_MM

    def auto_save_enabled(self):
        return bool(self.auto_save_switch.isChecked())

    def save_frame_log(self, rgb, depth):
        """Save a captured RGB/depth pair in the virtual-camera log format."""
        color_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb is not None else None
        depth_preview = colorize_depth(depth) if depth is not None else None
        return save_frames(
            default_log_root(),
            color_image=color_bgr,
            depth_image=depth_preview,
            raw_depth_image=depth,
            depth_scale_mm=self.depth_scale_mm(),
        )

    def auto_save_frame_log(self, rgb, depth):
        return self._auto_save_frame_log(rgb, depth)

    def _auto_save_frame_log(self, rgb, depth):
        if not self.auto_save_enabled():
            return ""
        ok, result = self.save_frame_log(rgb, depth)
        if not ok:
            return f" · 저장 실패: {result}"
        path = (
            result.get("color")
            or result.get("depth_raw")
            or result.get("depth")
            or next(iter(result.values()), None)
        )
        return f" · 저장: {display_log_path(path)}" if path else " · 저장 완료"

    def _refresh_once(self):
        """지금 프레임 한 장을 받아 보드에 그린다. 실패는 조용히 넘어간다.

        연결 직후와 후처리 값을 바꿨을 때 쓴다. 그 자리에서는 실패 사유보다
        방금 한 조작의 결과가 상태줄에 남아야 한다(진짜 실패는 '이미지 캡쳐'를
        눌러 보면 사유까지 나온다).
        """
        try:
            self._show_frames(*self.grab_frames())
        except Exception:
            pass

    def capture_image(self):
        if self._source is None:
            self._set_status("카메라가 연결되어 있지 않습니다.")
            return
        try:
            rgb, depth = self.grab_frames()
        except Exception as exc:
            self._set_status(f"캡쳐 실패: {exc}")
            return
        save_note = self._auto_save_frame_log(rgb, depth)
        self._show_frames(rgb, depth, save_note=save_note)

    def _show_frames(self, rgb, depth, save_note=""):
        # 컬러와 뎁스를 함께 넘긴다. 보드가 둘을 들고 있다가 콤보에 따라 갈아
        # 끼우므로, 화면을 바꿔도 같은 시점의 것을 본다.
        preview = (
            self._source.depth_preview
            if isinstance(self._source, _FolderSource)
            else None
        )
        drawn = self.board.set_frames(
            rgb, depth, depth_preview=preview, depth_scale_mm=self.depth_scale_mm()
        )

        # 가상 카메라는 캡쳐할 때마다 다음 이미지로 넘어가므로, 콤보도 지금
        # 나온 이미지를 가리키게 맞춘다.
        self._sync_virtual_combo()

        if not drawn:
            self._set_status(self._missing_view_reason())
            return
        height, width = rgb.shape[:2]
        detail = f"  {self._source.last_name}" if self._source.last_name else ""
        self._set_status(f"캡쳐 완료 {width}×{height}{detail}{save_note}")

    def _show_depth_profile(self, start, end):
        pixels, z_values = self.board.sample_profile(start, end)
        if pixels is None:
            self._set_status("직선 그래프를 만들 원본 depth 데이터가 없습니다.")
            return

        dialog = DepthProfileDialog(pixels, z_values, start, end, self)
        self._depth_profile_dialogs.append(dialog)
        dialog.finished.connect(
            lambda _result, item=dialog: self._depth_profile_dialogs.remove(item)
            if item in self._depth_profile_dialogs else None
        )
        dialog.show()
        self._set_status(
            f"Depth 직선 프로파일: {start} → {end} ({len(pixels)} pixels)"
        )

    # ===== 가상 카메라 이미지 선택 =====
    def _fill_virtual_combo(self, source):
        self.virtual_combo.blockSignals(True)
        self.virtual_combo.clear()
        self.virtual_combo.addItems(source.names())
        self.virtual_combo.blockSignals(False)

    def _sync_virtual_combo(self):
        """콤보가 지금 내보낸 이미지를 가리키게 한다."""
        if not isinstance(self._source, _FolderSource):
            return
        index = self._source.current_index()
        if index < 0:
            return
        self.virtual_combo.blockSignals(True)
        self.virtual_combo.setCurrentIndex(index)
        self.virtual_combo.blockSignals(False)

    def _on_virtual_image_selected(self, index):
        """콤보에서 고른 이미지를 바로 띄운다."""
        if not isinstance(self._source, _FolderSource):
            return
        if self._source.select(index):
            self.capture_image()

    # ===== 상태 표시 =====
    def _set_status(self, text):
        self.status.setText(text)

    def _update_controls(self):
        connected = self._source is not None
        self.btn_connect.setChecked(connected)
        self.btn_connect.setText("연결 해제" if connected else "카메라 연결")
        # 가상 카메라는 캡쳐가 곧 '폴더의 다음 이미지'라, 무엇이 일어나는지
        # 버튼 이름에 적어 둔다.
        self.btn_capture.setEnabled(connected)
        self.btn_capture.setText(
            "다음 이미지" if isinstance(self._source, _FolderSource) else "이미지 캡쳐"
        )
        # 연결 중에는 장치를 바꾸거나 다시 탐색하지 못하게 잠근다.
        self.camera_combo.setEnabled(not connected)
        self.btn_scan.setEnabled(not connected)

        # 이미지 선택 콤보는 가상 카메라 모드에서만 쓴다.
        data = self.camera_combo.currentData()
        is_virtual = bool(data) and data[0] == "virtual"
        self.virtual_combo.setVisible(is_virtual)
        self.virtual_combo.setEnabled(
            is_virtual and isinstance(self._source, _FolderSource)
        )

        self.led.setProperty("connected", "true" if connected else "false")
        self.led.style().unpolish(self.led)
        self.led.style().polish(self.led)
