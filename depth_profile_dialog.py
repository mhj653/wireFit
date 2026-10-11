"""뎁스맵의 직선 구간을 샘플링하고 z 거리 그래프로 표시한다."""

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QWidget

from theme import ACCENT, CONTENT_TEXT, DEPTH_PROFILE_GRID


def sample_depth_line(depth, start, end, depth_scale_mm=1.0):
    """두 점을 잇는 선 위의 픽셀 인덱스와 z 거리(mm)를 반환한다."""
    x0, y0 = start
    x1, y1 = end
    count = max(abs(x1 - x0), abs(y1 - y0)) + 1
    xs = np.rint(np.linspace(x0, x1, count)).astype(np.intp)
    ys = np.rint(np.linspace(y0, y1, count)).astype(np.intp)
    values = np.asarray(depth)[ys, xs].astype(np.float64) * float(depth_scale_mm)
    values[values <= 0.0] = np.nan
    return np.arange(count, dtype=np.float64), values


class _DepthProfilePlot(QWidget):
    """외부 그래프 패키지 없이 Qt로 그리는 직선 뎁스 프로파일."""

    def __init__(self, pixels, z_values, parent=None):
        super().__init__(parent)
        self._pixels = np.asarray(pixels, dtype=np.float64)
        self._z_values = np.asarray(z_values, dtype=np.float64)
        self.setMinimumSize(640, 360)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), Qt.white)

        plot = QRectF(70, 20, max(1, self.width() - 95), max(1, self.height() - 75))
        valid = np.isfinite(self._z_values)

        painter.setPen(QPen(QColor(CONTENT_TEXT), 1))
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        painter.drawLine(plot.bottomLeft(), plot.topLeft())

        if not valid.any():
            painter.drawText(plot, Qt.AlignCenter, "유효한 뎁스 값이 없습니다.")
            return

        z_min = float(np.nanmin(self._z_values))
        z_max = float(np.nanmax(self._z_values))
        if z_max <= z_min:
            z_min -= 0.5
            z_max += 0.5
        else:
            padding = (z_max - z_min) * 0.05
            z_min -= padding
            z_max += padding

        x_max = max(1.0, float(len(self._pixels) - 1))
        self._draw_ticks(painter, plot, x_max, z_min, z_max)

        path = QPainterPath()
        drawing = False
        for pixel, z_value in zip(self._pixels, self._z_values):
            if not np.isfinite(z_value):
                drawing = False
                continue
            point = QPointF(
                plot.left() + (float(pixel) / x_max) * plot.width(),
                plot.bottom() - ((float(z_value) - z_min) / (z_max - z_min)) * plot.height(),
            )
            if drawing:
                path.lineTo(point)
            else:
                path.moveTo(point)
                drawing = True

        painter.setPen(QPen(QColor(ACCENT), 2))
        painter.drawPath(path)

        painter.setPen(QColor(CONTENT_TEXT))
        painter.drawText(
            QRectF(plot.left(), plot.bottom() + 28, plot.width(), 20),
            Qt.AlignCenter,
            "Pixel",
        )
        painter.save()
        painter.translate(18, plot.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-plot.height() / 2, -10, plot.height(), 20), Qt.AlignCenter, "Z distance (mm)")
        painter.restore()

    @staticmethod
    def _draw_ticks(painter, plot, x_max, z_min, z_max):
        for index in range(5):
            ratio = index / 4.0
            x = plot.left() + ratio * plot.width()
            y = plot.bottom() - ratio * plot.height()
            painter.setPen(QPen(QColor(DEPTH_PROFILE_GRID), 1))
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))

            painter.setPen(QColor(CONTENT_TEXT))
            painter.drawText(
                QRectF(x - 30, plot.bottom() + 4, 60, 18),
                Qt.AlignHCenter | Qt.AlignTop,
                str(int(round(ratio * x_max))),
            )
            z_value = z_min + ratio * (z_max - z_min)
            painter.drawText(
                QRectF(2, y - 9, plot.left() - 8, 18),
                Qt.AlignRight | Qt.AlignVCenter,
                f"{z_value:.1f}",
            )


class DepthProfileDialog(QDialog):
    def __init__(self, pixels, z_values, start, end, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Depth 직선 프로파일")
        self.resize(760, 460)
        self.setAttribute(Qt.WA_DeleteOnClose)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                f"시작 ({start[0]}, {start[1]})  →  끝 ({end[0]}, {end[1]})"
                f"  ·  {len(pixels)} pixels"
            )
        )
        layout.addWidget(_DepthProfilePlot(pixels, z_values), 1)
