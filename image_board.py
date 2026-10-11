"""이미지를 비율 유지로 띄우고, 그 위에서 드래그를 받는 보드.

Hole Detection(홀을 찾는다)과 Pose Setting(법선 매칭점을 짚는다)이 같은 보드를
쓴다. 두 화면에서 같은 픽셀을 가리켜야 하므로 좌표 변환을 한 군데 둔다.

  ImageBoard   비율 유지 표시 + 휠 확대/우클릭 이동 + 위젯↔원본 이미지 좌표 변환
  ResultBoard  거기에 드래그(ROI / 네모 / 점 찍기)를 얹은 것

그림은 setPixmap 이 아니라 paintEvent 에서 직접 그린다. 확대/이동을 얹으면
'스케일해서 라벨에 넣기'로는 자리를 정할 수 없기 때문이다(라벨은 늘 가운데에
놓는다). 직접 그리면 확대·이동한 자리와 좌표 변환이 같은 사각형(_display_rect)
하나를 보게 되어, 확대한 상태에서 지정한 ROI 도 제자리에 찍힌다.
"""

import cv2
from PyQt5.QtCore import QRect, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QWIDGETSIZE_MAX, QLabel, QSizePolicy

from roi_state import roi_state
from theme import (
    ROI_BORDER,
    ROI_BORDER_WIDTH,
    ROI_DIM_ALPHA,
    ROI_DRAG_BORDER,
    VIDEO_MIN_HEIGHT,
    VIDEO_MIN_WIDTH,
    ZOOM_RESET_HINT,
)
from zoom_pan import ZoomPanMixin

# 이미지가 아직 없을 때 보드가 취할 비율. D405 컬러 스트림(1280x720)과 같다.
# 캡쳐 전에 보드가 창 폭을 다 차지하다가 캡쳐 뒤 16:9 로 줄어들면 화면이 한 번
# 크게 움직인다. 처음부터 같은 비율로 두면 캡쳐해도 크기가 그대로다.
BOARD_DEFAULT_ASPECT = 16.0 / 9.0


class ImageBoard(ZoomPanMixin, QLabel):
    """이미지를 비율 유지로 보여주고 위젯 <-> 이미지 좌표를 변환하는 라벨.

    창 크기가 바뀌면 원본 픽스맵에서 다시 그리므로 화질이 누적 열화되지 않는다.
    휠로 확대하고 우클릭 드래그로 옮길 수 있다(zoom_pan 참고).
    """

    def __init__(self, placeholder, parent=None):
        super().__init__(placeholder, parent)
        self.setObjectName("videoView")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(VIDEO_MIN_WIDTH, VIDEO_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setToolTip(ZOOM_RESET_HINT)
        self._placeholder = placeholder
        self._source_pixmap = None
        self._init_zoom_pan()

    # ===== 이미지 =====
    def set_image_bgr(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888
        ).copy()   # numpy 버퍼와 분리해 두어야 배열이 해제돼도 안전하다.
        previous = self._source_pixmap
        self._source_pixmap = QPixmap.fromImage(image)
        # 해상도가 달라졌으면 확대를 되돌린다. 같은 크기면 그대로 둔다 — 파라미터를
        # 고칠 때마다 다시 그려지는데, 그때 확대가 풀리면 보고 있던 홀을 매번
        # 다시 찾아야 한다(값을 조금씩 바꿔 가며 비교하는 것이 이 화면의 일이다).
        if previous is None or previous.size() != self._source_pixmap.size():
            self.reset_view()
        self.setText("")
        self._limit_to_aspect()
        self.update()

    def clear_image(self):
        self._source_pixmap = None
        self.reset_view()
        self.setText(self._placeholder)
        self._limit_to_aspect()   # 이미지가 없어도 기본 비율(16:9)은 지킨다
        self.update()

    def has_image(self):
        return self._source_pixmap is not None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._limit_to_aspect()
        self._clamp_pan()

    def _limit_to_aspect(self):
        """보드 너비를 이미지 비율에 맞춰 제한한다.

        이 보드는 창 너비를 거의 다 쓴다. 이미지는 비율을 지켜 그려지므로,
        보드가 이미지보다 납작하면 남는 폭이 그대로 빈 여백이 된다 —
        1280x720 이미지에 보드가 2:1 이면 양옆이 비어 보인다. 그래서 '지금
        높이에서 비율을 지킬 수 있는 최대 너비'를 상한으로 걸고, 남는 폭은
        보드를 감싼 행의 좌우 스트레치가 나눠 흡수한다.

        높이는 건드리지 않는다. 너비만 줄이므로 이 위젯의 높이가 다시 바뀌지
        않고, 따라서 resizeEvent 가 되풀이되며 진동하지 않는다.

        **높이가 아직 안 자랐으면 상한을 걸지 않는다.** 레이아웃이 안정되기
        전(창을 띄우기 전 분석이 먼저 도는 경우)에 계산하면 상한이 최소 너비로
        잡히고, 그 뒤 높이가 변하지 않으면 resizeEvent 가 오지 않아 **그 값에
        갇힌다.** 보드가 좁은 띠로 남던 것이 이 때문이었다.
        """
        if self.height() < VIDEO_MIN_HEIGHT:
            if self.maximumWidth() != QWIDGETSIZE_MAX:
                self.setMaximumWidth(QWIDGETSIZE_MAX)
            return
        if self._source_pixmap is None:
            # 이미지가 없어도 카메라 비율로 자리를 잡아 둔다. 캡쳐 전후로
            # 보드 크기가 튀지 않게 하려는 것이다.
            ratio = BOARD_DEFAULT_ASPECT
        else:
            source_w = self._source_pixmap.width()
            source_h = self._source_pixmap.height()
            if min(source_w, source_h) <= 0:
                return
            ratio = source_w / source_h
        limit = max(VIDEO_MIN_WIDTH, int(round(self.height() * ratio)))
        if limit != self.maximumWidth():
            self.setMaximumWidth(limit)

    # ===== 그리기 =====
    def paintEvent(self, event):
        # 바탕과 테두리(그리고 이미지가 없을 때의 안내 글자)는 QLabel 이 그린다.
        super().paintEvent(event)
        if self._source_pixmap is None:
            return
        target = self._display_rect()
        if target is None:
            return
        painter = QPainter(self)
        # 확대해서 보드 밖으로 나간 부분은 잘라 낸다. 안 자르면 옆 위젯 위에 그려진다.
        painter.setClipRect(self.contentsRect())
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(target, self._source_pixmap)
        painter.end()

    # ===== 좌표 변환 (위젯 <-> 원본 이미지) =====
    # _display_rect() 는 ZoomPanMixin 이 준다. 확대/이동이 얹힌 자리이므로
    # 아래 두 변환은 확대한 상태에서도 그대로 맞는다.
    def _widget_to_image(self, pos):
        """위젯 좌표를 원본 이미지 좌표로. 이미지 밖이면 경계로 잘라 준다."""
        display = self._display_rect()
        if display is None:
            return None
        image_w = self._source_pixmap.width()
        image_h = self._source_pixmap.height()
        x = (pos.x() - display.x()) * image_w / display.width()
        y = (pos.y() - display.y()) * image_h / display.height()
        return (
            int(min(max(x, 0), image_w - 1)),
            int(min(max(y, 0), image_h - 1)),
        )

    def _image_rect_to_widget(self, rect):
        """원본 이미지 좌표의 (x, y, w, h)를 위젯 좌표 QRect로."""
        display = self._display_rect()
        if display is None:
            return None
        x, y, width, height = rect
        scale_x = display.width() / self._source_pixmap.width()
        scale_y = display.height() / self._source_pixmap.height()
        return QRect(
            display.x() + int(x * scale_x),
            display.y() + int(y * scale_y),
            max(1, int(width * scale_x)),
            max(1, int(height * scale_y)),
        )


class ResultBoard(ImageBoard):
    """결과를 보여주고, ROI/네모 영역을 드래그로 지정하게 하는 보드.

    ROI는 이미지 위에 굽지 않고 위젯에 덧그리므로, 원본 이미지는 그대로 남는다.
    지정은 **좌클릭** 드래그다 — 우클릭 드래그는 확대한 화면을 옮기는 데 쓴다.
    """

    roi_changed = pyqtSignal()
    region_selected = pyqtSignal(object)
    image_clicked = pyqtSignal(object)
    # SAM2 티칭도 네모 드래그다(260914). 홀을 점으로 찍던 "prompt" 모드는
    # 지웠다 — 점 하나로는 SAM2 가 홀인지 판재인지 스스로 골라야 해서 뜻한
    # 것보다 큰 물체가 잡히곤 했다. 남겨 두면 그 길로 되돌아가기 쉽다.

    def __init__(self, placeholder, parent=None):
        super().__init__(placeholder, parent)
        self._selection_mode = None
        self._drag_start = None   # 드래그 시작점 (이미지 좌표)
        self._drag_end = None     # 드래그 현재점 (이미지 좌표)
        self._roi_overlay_enabled = True

    # ===== ROI 모드 =====
    def set_roi_overlay_enabled(self, enabled):
        self._roi_overlay_enabled = bool(enabled)
        self.update()

    def set_roi_mode(self, enabled):
        self.set_selection_mode("roi" if enabled else None)

    def set_selection_mode(self, mode):
        self._selection_mode = mode
        self._restore_cursor()
        self._drag_start = None
        self._drag_end = None
        self.update()

    def _restore_cursor(self):
        """우클릭 드래그가 끝난 뒤 지정 모드에 맞는 커서로 되돌린다."""
        self.setCursor(Qt.CrossCursor if self._selection_mode else Qt.ArrowCursor)

    def _normalized_drag(self):
        """드래그 방향(좌상→우하가 아니어도)에 상관없이 (x, y, w, h)로 정규화."""
        if self._drag_start is None or self._drag_end is None:
            return None
        x0, y0 = self._drag_start
        x1, y1 = self._drag_end
        return (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))

    # ===== 마우스 드래그로 ROI 지정 =====
    def mousePressEvent(self, event):
        if (
            self._selection_mode is None
            or event.button() != Qt.LeftButton
            or self._source_pixmap is None
        ):
            # 좌클릭이 아니면(= 확대/이동이면) 위쪽 믹스인이 처리한다.
            super().mousePressEvent(event)
            return
        point = self._widget_to_image(event.pos())
        if point is None:
            return
        self._drag_start = point
        self._drag_end = point
        self.update()

    def mouseMoveEvent(self, event):
        if self._drag_start is None:
            super().mouseMoveEvent(event)
            return
        point = self._widget_to_image(event.pos())
        if point is not None:
            self._drag_end = point
            self.update()

    def mouseReleaseEvent(self, event):
        if (
            self._selection_mode is None
            and event.button() == Qt.LeftButton
            and self._source_pixmap is not None
        ):
            point = self._widget_to_image(event.pos())
            if point is not None:
                self.image_clicked.emit(point)
                event.accept()
                return
        if self._drag_start is None or event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        rect = self._normalized_drag()
        self._drag_start = None
        self._drag_end = None
        if self._selection_mode == "region":
            if rect is not None and rect[2] >= 3 and rect[3] >= 3:
                self.region_selected.emit(rect)
        else:
            # 너무 작은 드래그(사실상 클릭)는 ROI 해제로 처리한다.
            if rect is None or not roi_state.set(*rect):
                roi_state.clear()
            self.roi_changed.emit()
        self.update()

    def mouseDoubleClickEvent(self, event):
        # 지정 모드에서는 더블클릭이 '작게 클릭 = 해제'와 겹치므로 확대만 되돌린다.
        if self._selection_mode is None:
            super().mouseDoubleClickEvent(event)

    # ===== ROI 오버레이 =====
    def paintEvent(self, event):
        super().paintEvent(event)
        if self._source_pixmap is None:
            return

        dragging = self._drag_start is not None
        rect = self._normalized_drag() if dragging else roi_state.get()
        if not dragging and not self._roi_overlay_enabled:
            return
        if rect is None:
            return
        box = self._image_rect_to_widget(rect)
        display = self._display_rect()
        if box is None or display is None:
            return

        painter = QPainter(self)
        # 확대해서 보드 밖으로 나간 부분까지 덮지 않게 잘라 낸다.
        painter.setClipRect(self.contentsRect())
        # 확정된 ROI일 때만 바깥을 어둡게 덮어 분석 범위를 분명히 보여준다.
        if not dragging:
            self._dim_outside(painter, display, box)
        painter.setPen(
            QPen(QColor(ROI_DRAG_BORDER if dragging else ROI_BORDER), ROI_BORDER_WIDTH)
        )
        painter.drawRect(box)

    @staticmethod
    def _dim_outside(painter, display, box):
        """ROI 바깥(이미지가 그려진 영역 안)을 반투명하게 덮는다."""
        dim = QColor(0, 0, 0, ROI_DIM_ALPHA)
        painter.fillRect(
            QRect(display.left(), display.top(), display.width(), box.top() - display.top()),
            dim,
        )
        painter.fillRect(
            QRect(
                display.left(),
                box.bottom() + 1,
                display.width(),
                display.bottom() - box.bottom(),
            ),
            dim,
        )
        painter.fillRect(
            QRect(display.left(), box.top(), box.left() - display.left(), box.height()),
            dim,
        )
        painter.fillRect(
            QRect(box.right() + 1, box.top(), display.right() - box.right(), box.height()),
            dim,
        )
