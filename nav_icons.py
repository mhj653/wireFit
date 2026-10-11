"""좌측 사이드바 탭 아이콘.

아이콘을 그림 파일로 두지 않고 **코드로 그린다.** 이유가 셋이다.

  - 파일이 늘면 PyInstaller spec 의 datas 도 같이 손봐야 하고, 빠뜨리면 exe
    에서만 아이콘이 사라진다(그러고도 앱은 뜨므로 알아채기 늦다).
  - 화면 배율(125%, 200%)마다 다른 크기를 미리 만들어 둘 수 없다. 여기서는
    부르는 쪽이 원하는 크기로 그때 그린다.
  - 색을 인자로 받으므로 고른 탭/안 고른 탭에서 같은 모양을 다른 색으로 쓸 수 있다.

새 페이지를 추가할 때는 페이지 클래스에 nav_icon = "이름" 을 적고, 아래
_PAINTERS 에 그 이름의 그리는 함수를 하나 더하면 된다. 이름이 표에 없으면
동그라미(기본 아이콘)로 그려지므로 빠뜨려도 화면이 비지는 않는다.

좌표는 모두 0~1 로 적는다(한 변이 1 인 정사각형 기준). 그래야 크기가 바뀌어도
같은 모양이 나온다.
"""

from PyQt5.QtCore import QPointF, QRectF, Qt
from PyQt5.QtGui import QColor, QPainter, QPainterPath, QPen, QPixmap

# 선 두께도 크기에 비례해야 한다(작게 그렸을 때만 두꺼워 보이는 일이 없게).
STROKE_RATIO = 0.075
MIN_STROKE = 1.4


def _pen(painter, size, color, ratio=1.0):
    pen = QPen(QColor(color))
    pen.setWidthF(max(MIN_STROKE, size * STROKE_RATIO * ratio))
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    return pen


def _rect(size, x, y, width, height):
    return QRectF(x * size, y * size, width * size, height * size)


# ======================================================
# 페이지별 아이콘
# ======================================================
def _camera(painter, size, color):
    """Camera Selection — 카메라 몸통 + 렌즈."""
    _pen(painter, size, color)
    painter.drawRoundedRect(
        _rect(size, 0.06, 0.28, 0.88, 0.56), size * 0.08, size * 0.08
    )
    # 위쪽 뷰파인더 혹
    painter.drawRoundedRect(
        _rect(size, 0.30, 0.14, 0.30, 0.16), size * 0.04, size * 0.04
    )
    painter.drawEllipse(_rect(size, 0.34, 0.40, 0.32, 0.32))
    _pen(painter, size, color, 0.7)
    painter.drawEllipse(_rect(size, 0.43, 0.49, 0.14, 0.14))


def _holes(painter, size, color):
    """Hole Detection — 판재 위의 홀 셋 + 조준선."""
    _pen(painter, size, color)
    painter.drawRect(_rect(size, 0.08, 0.14, 0.84, 0.72))
    _pen(painter, size, color, 0.8)
    for cx, cy in ((0.30, 0.36), (0.68, 0.36), (0.30, 0.66)):
        painter.drawEllipse(
            QRectF((cx - 0.10) * size, (cy - 0.10) * size, 0.20 * size, 0.20 * size)
        )
    # 오른쪽 아래 홀만 조준선을 달아 '찾은 홀'을 뜻하게 한다.
    cx, cy = 0.68, 0.66
    painter.drawEllipse(
        QRectF((cx - 0.10) * size, (cy - 0.10) * size, 0.20 * size, 0.20 * size)
    )
    _pen(painter, size, color, 0.6)
    painter.drawLine(
        QPointF((cx - 0.17) * size, cy * size), QPointF((cx - 0.02) * size, cy * size)
    )
    painter.drawLine(
        QPointF((cx + 0.02) * size, cy * size), QPointF((cx + 0.17) * size, cy * size)
    )
    painter.drawLine(
        QPointF(cx * size, (cy - 0.17) * size), QPointF(cx * size, (cy - 0.02) * size)
    )
    painter.drawLine(
        QPointF(cx * size, (cy + 0.02) * size), QPointF(cx * size, (cy + 0.17) * size)
    )


def _arrow(painter, size, start, end, head=0.10):
    """화살표 하나. 좌표축 아이콘이 세 번 쓴다."""
    sx, sy = start
    ex, ey = end
    painter.drawLine(
        QPointF(sx * size, sy * size), QPointF(ex * size, ey * size)
    )
    dx, dy = ex - sx, ey - sy
    length = max(1e-6, (dx * dx + dy * dy) ** 0.5)
    dx, dy = dx / length, dy / length
    # 진행 방향에 수직인 벡터로 날개를 벌린다.
    for side in (1, -1):
        wing_x = ex - dx * head + (-dy) * head * 0.6 * side
        wing_y = ey - dy * head + dx * head * 0.6 * side
        painter.drawLine(
            QPointF(ex * size, ey * size), QPointF(wing_x * size, wing_y * size)
        )


def _pose(painter, size, color):
    """Pose Setting — 좌표축 세 방향(자세)."""
    _pen(painter, size, color)
    origin = (0.24, 0.76)
    _arrow(painter, size, origin, (0.88, 0.76))   # x
    _arrow(painter, size, origin, (0.24, 0.12))   # y
    _arrow(painter, size, origin, (0.74, 0.34))   # z (비스듬히)


def _checkerboard(painter, size, color):
    """EIH Calibration — 체커보드."""
    _pen(painter, size, color, 0.8)
    painter.drawRect(_rect(size, 0.10, 0.10, 0.80, 0.80))
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    cell = 0.80 / 4.0
    for row in range(4):
        for column in range(4):
            if (row + column) % 2:
                continue
            painter.drawRect(
                _rect(
                    size,
                    0.10 + column * cell,
                    0.10 + row * cell,
                    cell,
                    cell,
                )
            )


def _operator(painter, size, color):
    """Operator Mode — 화면 + 실행(재생) 표시."""
    _pen(painter, size, color)
    painter.drawRoundedRect(
        _rect(size, 0.08, 0.16, 0.84, 0.56), size * 0.06, size * 0.06
    )
    painter.drawLine(
        QPointF(0.34 * size, 0.90 * size), QPointF(0.66 * size, 0.90 * size)
    )
    painter.drawLine(
        QPointF(0.50 * size, 0.72 * size), QPointF(0.50 * size, 0.90 * size)
    )
    path = QPainterPath()
    path.moveTo(0.42 * size, 0.31 * size)
    path.lineTo(0.66 * size, 0.44 * size)
    path.lineTo(0.42 * size, 0.57 * size)
    path.closeSubpath()
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    painter.drawPath(path)


def _network(painter, size, color):
    """TCP 서버 — 위쪽 한 점에서 아래 두 점으로 갈라지는 연결선."""
    _pen(painter, size, color)
    painter.drawLine(
        QPointF(0.50 * size, 0.26 * size), QPointF(0.50 * size, 0.46 * size)
    )
    painter.drawLine(
        QPointF(0.22 * size, 0.46 * size), QPointF(0.78 * size, 0.46 * size)
    )
    painter.drawLine(
        QPointF(0.22 * size, 0.46 * size), QPointF(0.22 * size, 0.60 * size)
    )
    painter.drawLine(
        QPointF(0.78 * size, 0.46 * size), QPointF(0.78 * size, 0.60 * size)
    )
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    # 위 한 개(서버) + 아래 두 개(클라이언트)
    painter.drawEllipse(_rect(size, 0.38, 0.06, 0.24, 0.20))
    painter.drawEllipse(_rect(size, 0.10, 0.60, 0.24, 0.24))
    painter.drawEllipse(_rect(size, 0.66, 0.60, 0.24, 0.24))


def _save(painter, size, color):
    """데이터 저장 — 아래로 향하는 화살표 + 받는 접시."""
    _pen(painter, size, color)
    _arrow(painter, size, (0.50, 0.10), (0.50, 0.58), head=0.14)
    painter.drawLine(
        QPointF(0.16 * size, 0.72 * size), QPointF(0.16 * size, 0.88 * size)
    )
    painter.drawLine(
        QPointF(0.16 * size, 0.88 * size), QPointF(0.84 * size, 0.88 * size)
    )
    painter.drawLine(
        QPointF(0.84 * size, 0.72 * size), QPointF(0.84 * size, 0.88 * size)
    )


def _home(painter, size, color):
    """Home — 집."""
    _pen(painter, size, color)
    path = QPainterPath()
    path.moveTo(0.10 * size, 0.48 * size)
    path.lineTo(0.50 * size, 0.14 * size)
    path.lineTo(0.90 * size, 0.48 * size)
    painter.drawPath(path)
    painter.drawRect(_rect(size, 0.20, 0.48, 0.60, 0.38))
    painter.drawRect(_rect(size, 0.42, 0.62, 0.16, 0.24))


def _default(painter, size, color):
    """표에 없는 이름. 빈 자리로 남기지 않으려고 동그라미를 둔다."""
    _pen(painter, size, color)
    painter.drawEllipse(_rect(size, 0.16, 0.16, 0.68, 0.68))


_PAINTERS = {
    "camera": _camera,
    "holes": _holes,
    "pose": _pose,
    "calibration": _checkerboard,
    "operator": _operator,
    "home": _home,
    # 사이드바가 아니라 오퍼레이터 상태 카드가 쓰는 것들(operator_panels).
    # 아이콘을 그리는 자리는 한 군데뿐이라 같은 표에 둔다.
    "network": _network,
    "save": _save,
}


def render(name, size, color):
    """아이콘 하나를 그려 QPixmap 으로 돌려준다.

    Args:
        name: _PAINTERS 의 키. 모르는 이름이면 기본 아이콘을 그린다.
        size: 한 변 픽셀 수
        color: "#rrggbb"

    투명 배경이므로 탭 배경색이 그대로 비친다.
    """
    size = max(8, int(size))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    try:
        _PAINTERS.get(name, _default)(painter, float(size), color)
    finally:
        painter.end()
    return pixmap
