"""이미지 보드의 휠 확대/축소 + 우클릭 드래그 이동.

    휠 위/아래       커서를 둔 자리를 그대로 두고 확대/축소한다
    우클릭 드래그    확대된 이미지를 끌어 옮긴다
    가운데 클릭      원래 크기(화면에 꽉 맞춘 상태)로 되돌린다

세팅 UI 의 모든 이미지 보드가 이 한 가지를 함께 쓴다(Camera Selection,
Hole Detection, Pose Setting, EIH Calibration). **오퍼레이터 화면은 쓰지
않는다** — 현장에서 실수로 휠을 굴려 확대된 채로 남으면, 다음 촬영 결과를
잘린 화면으로 보게 되고 그것을 알아채기 어렵다.

왜 좌클릭이 아니라 우클릭으로 옮기는가
──────────────────────────────────────
좌클릭 드래그는 이미 쓰이는 자리가 많다 — ROI 지정, 템플릿 영역, 법선 매칭점,
뎁스 직선 프로파일. 이동을 좌클릭에 얹으면 그 기능들과 부딪힌다.

좌표 변환에 대해
────────────────
ROI 나 매칭점은 **원본 이미지 좌표**로 저장된다. 확대/이동해도 그 값이
달라지면 안 되므로, 위젯↔이미지 변환은 반드시 지금 그려진 자리(display_rect)
를 기준으로 계산한다. 그래서 확대한 상태에서 드래그해도 ROI 는 제자리에
찍힌다.
"""

from PyQt5.QtCore import QPoint, QRect, Qt

from theme import ZOOM_MAX, ZOOM_MIN, ZOOM_STEP


class ZoomPanMixin:
    """휠 확대 + 우클릭 이동을 얹는 믹스인.

    쓰는 쪽에 필요한 것:
      - self._source_pixmap : 원본 QPixmap (없으면 None)
      - QWidget 의 contentsRect() / update()

    믹스인이라 QLabel 상속과 함께 쓴다. 이벤트 처리기는 super() 로 이어 주므로,
    보드가 좌클릭으로 하던 일(ROI 등)은 그대로 남는다.
    """

    # 클래스 기본값을 둔다. 위젯을 만드는 도중에도 resizeEvent 가 올 수 있어서,
    # _init_zoom_pan() 전에 _pan 을 읽는 일이 생긴다. 값을 제자리에서 고치지
    # 않고 늘 새 QPoint 를 넣으므로 클래스 속성을 공유해도 섞이지 않는다.
    _zoom = 1.0            # 1.0 = 화면에 꽉 맞춘 상태
    _pan = QPoint(0, 0)    # 꽉 맞춘 자리에서 얼마나 밀렸는지 (위젯 픽셀)
    _pan_origin = None     # 우클릭 드래그 시작점
    _pan_start = None      # 드래그 시작 시점의 _pan

    def _init_zoom_pan(self):
        self._zoom = 1.0
        self._pan = QPoint(0, 0)
        self._pan_origin = None
        self._pan_start = None

    # ===== 상태 =====
    def reset_view(self):
        """확대와 이동을 되돌린다. 새 이미지를 넣을 때마다 부른다.

        새 사진에 이전 확대를 물려주지 않는 이유: 장면이 바뀌면 확대해 둔 자리가
        무엇을 가리키는지 알 수 없다. 특히 오퍼레이터로 넘길 세팅을 맞추는
        중이라면, 화면이 잘린 것을 모르고 '홀이 하나만 잡혔다'고 볼 수 있다.
        """
        if self._zoom == 1.0 and self._pan.isNull():
            return
        self._zoom = 1.0
        self._pan = QPoint(0, 0)
        self.update()

    def zoom_percent(self):
        """지금 배율(%)."""
        return self._zoom * 100.0

    # ===== 자리 계산 =====
    def _fit_rect(self):
        """확대 없이 비율을 지켜 꽉 맞췄을 때 이미지가 놓이는 자리."""
        if self._source_pixmap is None:
            return None
        area = self.contentsRect()
        pixmap_w = self._source_pixmap.width()
        pixmap_h = self._source_pixmap.height()
        if min(pixmap_w, pixmap_h, area.width(), area.height()) <= 0:
            return None
        scale = min(area.width() / pixmap_w, area.height() / pixmap_h)
        width = max(1, int(round(pixmap_w * scale)))
        height = max(1, int(round(pixmap_h * scale)))
        return QRect(
            area.x() + (area.width() - width) // 2,
            area.y() + (area.height() - height) // 2,
            width,
            height,
        )

    def _display_rect(self):
        """확대와 이동을 얹은 뒤 이미지가 실제로 그려지는 자리."""
        fit = self._fit_rect()
        if fit is None:
            return None
        zoom = self._zoom
        if zoom == 1.0 and self._pan.isNull():
            return fit
        width = max(1, int(round(fit.width() * zoom)))
        height = max(1, int(round(fit.height() * zoom)))
        # 확대는 '꽉 맞춘 자리의 중심'을 기준으로 한다. 거기서 _pan 만큼 밀린다.
        x = fit.center().x() - width // 2 + self._pan.x()
        y = fit.center().y() - height // 2 + self._pan.y()
        return QRect(x, y, width, height)

    def _clamp_pan(self):
        """이미지가 화면 밖으로 완전히 빠져나가지 않게 이동량을 묶는다.

        확대하지 않았으면 이동을 아예 허용하지 않는다 — 꽉 맞춘 이미지를 옆으로
        밀 이유가 없고, 밀리면 되돌리는 방법을 찾아야 한다.
        """
        fit = self._fit_rect()
        if fit is None:
            return
        zoom = self._zoom
        if zoom <= 1.0:
            self._pan = QPoint(0, 0)
            return
        # 확대된 이미지가 보이는 영역을 덮고 있는 한에서만 움직인다.
        limit_x = int(round(fit.width() * (zoom - 1.0) / 2.0))
        limit_y = int(round(fit.height() * (zoom - 1.0) / 2.0))
        self._pan = QPoint(
            min(max(self._pan.x(), -limit_x), limit_x),
            min(max(self._pan.y(), -limit_y), limit_y),
        )

    # ===== 휠 확대 =====
    def wheelEvent(self, event):
        if self._source_pixmap is None:
            super().wheelEvent(event)
            return
        steps = event.angleDelta().y() / 120.0
        if not steps:
            super().wheelEvent(event)
            return

        old_zoom = self._zoom
        new_zoom = old_zoom * (ZOOM_STEP ** steps)
        new_zoom = min(max(new_zoom, ZOOM_MIN), ZOOM_MAX)
        if new_zoom == old_zoom:
            event.accept()
            return

        # 커서 아래의 픽셀이 제자리에 남게 이동량을 고쳐 준다. 이 보정이 없으면
        # 확대할 때마다 보고 있던 자리가 중심으로 끌려가 홀을 다시 찾아야 한다.
        fit = self._fit_rect()
        if fit is not None:
            center = fit.center()
            cursor = event.pos()
            ratio = new_zoom / old_zoom
            offset_x = cursor.x() - (center.x() + self._pan.x())
            offset_y = cursor.y() - (center.y() + self._pan.y())
            self._pan = QPoint(
                self._pan.x() + int(round(offset_x * (1.0 - ratio))),
                self._pan.y() + int(round(offset_y * (1.0 - ratio))),
            )

        self._zoom = new_zoom
        self._clamp_pan()
        self.update()
        event.accept()

    # ===== 우클릭 드래그 이동 =====
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton and self._source_pixmap is not None:
            self._pan_origin = event.pos()
            self._pan_start = QPoint(self._pan)
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MiddleButton:
            self.reset_view()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan_origin is not None:
            delta = event.pos() - self._pan_origin
            self._pan = self._pan_start + delta
            self._clamp_pan()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.RightButton and self._pan_origin is not None:
            self._pan_origin = None
            self._pan_start = None
            self.unsetCursor()
            self._restore_cursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _restore_cursor(self):
        """드래그가 끝난 뒤 커서를 원래 모양으로. 보드가 필요하면 덮어쓴다."""

    def mouseDoubleClickEvent(self, event):
        """더블클릭으로도 되돌린다(가운데 버튼이 없는 마우스를 위해).

        왼쪽 더블클릭만 받는다. 오른쪽 더블클릭은 이동 중에 잘못 눌리기 쉽다.
        """
        if event.button() == Qt.LeftButton and self._zoom != 1.0:
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)
