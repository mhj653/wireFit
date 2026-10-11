"""TCP/IP 서버 설정 줄.

    통신 설정 [ ●]  IP [0.0.0.0    ]  Port [5000 ]
              프로토콜 [Little Endian ▼]  회전 순서 [Rz → Ry → Rx ▼]

이 프로그램이 서버다. IP 는 '어느 랜 카드로 접속을 받을지'를 정하는 값이고,
0.0.0.0 은 모든 랜 카드로 받는다는 뜻이다. 장비 네트워크가 여러 개 물려 있어
한쪽에서만 받아야 할 때 그 랜 카드의 IP 를 적는다.

프로토콜 (260910)
    ASCII / Little Endian / Big Endian 을 화면에서 고른다. 코드에는 세 가지가
    다 있었지만 고를 자리가 없어 기본값(리틀 엔디안)으로 고정돼 있었다. 로봇
    컨트롤러마다 다르므로 현장에서 바꿀 수 있어야 한다.

회전 순서 (260910)
    로봇이 회전 세 값을 받는 순서다. 쿠카는 A·B·C = rz·ry·rx 인데 rx 부터
    받는 로봇도 있다. 각도 값을 바꾸는 것이 아니라 **패킷에 실리는 자리**를
    바꾼다(protocol.rotation_values). 바이너리 모드에서는 그 자리가 곧 바이트
    순서다.

값은 tcp_server.FasteningServerConfig 로 넘긴다. 서버를 켜고 오퍼레이터
모드로 들어가면 지정한 IP와 포트에서 실제 통신(bind/listen)을 시작한다.
"""

from PyQt5.QtCore import QRegExp, pyqtSignal
from PyQt5.QtGui import QIntValidator, QRegExpValidator
from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from tcp_server import FasteningServerConfig
from theme import CONTENT_SPACING
from toggle_switch import ToggleSwitch
import protocol

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000
DEFAULT_MODE = protocol.MODE_LITTLE
DEFAULT_ROTATION_ORDER = protocol.ROTATION_ZYX


IP_INPUT_WIDTH = 150
PORT_INPUT_WIDTH = 90

# 0~255 네 덩이. 입력 중에는 중간 상태도 허용되므로(Intermediate)
# 확정된 값은 is_valid() 로 따로 확인한다.
_OCTET = r"(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_REGEX = QRegExp(r"^%s\.%s\.%s\.%s$" % (_OCTET, _OCTET, _OCTET, _OCTET))

PORT_MIN = 1
PORT_MAX = 65535


class TcpSettingsPanel(QWidget):
    """서버 on/off + IP + 포트 + 프로토콜 + 회전 순서."""

    server_enabled_changed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        outer = QVBoxLayout(self)

        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(CONTENT_SPACING // 2)

        caption = QLabel("통신 설정")
        caption.setObjectName("sectionLabel")
        outer.addWidget(caption)

        row = QHBoxLayout()
        row.setSpacing(CONTENT_SPACING)

        # ── 서버 on/off ──
        row.addWidget(QLabel("서버"))
        self.toggle_server = ToggleSwitch()
        self.toggle_server.setToolTip(
            "ON 상태로 오퍼레이터 모드에 들어가면 TCP 서버를 시작합니다."
        )
        self.toggle_server.toggled.connect(self.server_enabled_changed.emit)
        row.addWidget(self.toggle_server)

        # ── IP ──
        row.addSpacing(CONTENT_SPACING)
        row.addWidget(QLabel("IP"))
        self.input_host = QLineEdit(DEFAULT_HOST)
        self.input_host.setObjectName("ipInput")
        self.input_host.setFixedWidth(IP_INPUT_WIDTH)
        self.input_host.setValidator(QRegExpValidator(IPV4_REGEX, self))
        self.input_host.setToolTip(
            "0.0.0.0 = 모든 네트워크에서 접속을 받습니다.\n"
            "특정 랜 카드로만 받으려면 그 IP를 적습니다."
        )
        row.addWidget(self.input_host)

        # ── 포트 ──
        row.addSpacing(CONTENT_SPACING)
        row.addWidget(QLabel("Port"))
        self.input_port = QLineEdit(str(DEFAULT_PORT))
        self.input_port.setObjectName("portInput")
        self.input_port.setFixedWidth(PORT_INPUT_WIDTH)
        self.input_port.setValidator(QIntValidator(PORT_MIN, PORT_MAX, self))
        row.addWidget(self.input_port)

        # ── 프로토콜 ──
        row.addSpacing(CONTENT_SPACING)
        row.addWidget(QLabel("프로토콜"))
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("profileCombo")
        for label, value in protocol.MODE_CHOICES:
            self.mode_combo.addItem(label, value)
        self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(DEFAULT_MODE)))
        self.mode_combo.setToolTip(
            "패킷 형식입니다.\n"
            "ASCII: 쉼표로 끊은 글자 한 줄 (테스트 툴로 확인하기 쉽다).\n"
            "Little / Big Endian: 고정 길이 바이너리 (로봇 컨트롤러가 쓰는 쪽).\n"
            "\n"
            "오퍼레이터 모드에 들어갈 때 반영됩니다."
        )
        row.addWidget(self.mode_combo)

        # ── 회전 순서 ──
        row.addSpacing(CONTENT_SPACING)
        row.addWidget(QLabel("회전 순서"))
        self.rotation_combo = QComboBox()
        self.rotation_combo.setObjectName("profileCombo")
        for label, value in protocol.ROTATION_ORDER_CHOICES:
            self.rotation_combo.addItem(label, value)
        self.rotation_combo.setCurrentIndex(
            max(0, self.rotation_combo.findData(DEFAULT_ROTATION_ORDER))
        )
        self.rotation_combo.setToolTip(
            "회전 세 값을 패킷에 실을 순서입니다.\n"
            "Rz → Ry → Rx: 쿠카의 A·B·C 순서 (기본).\n"
            "Rx → Ry → Rz: rx 부터 받는 로봇용.\n"
            "\n"
            "각도 값은 그대로이고 나가는 자리(바이너리에서는 바이트 순서)만\n"
            "바뀝니다."
        )
        row.addWidget(self.rotation_combo)

        row.addStretch()
        outer.addLayout(row)

    # ===== 읽기 =====
    def is_server_enabled(self):
        return self.toggle_server.isChecked()

    def get_host(self):
        return self.input_host.text().strip()

    def get_port(self):
        text = self.input_port.text().strip()
        return int(text) if text.isdigit() else DEFAULT_PORT

    def is_valid(self):
        """서버를 띄울 수 있는 값인지.

        Returns:
            (True, "") 또는 (False, 사유)
        """
        host = self.get_host()
        if not IPV4_REGEX.exactMatch(host):
            return False, "IP를 0.0.0.0 형식으로 적어 주세요."
        port_text = self.input_port.text().strip()
        if not port_text.isdigit():
            return False, "포트를 숫자로 적어 주세요."
        port = int(port_text)
        if not PORT_MIN <= port <= PORT_MAX:
            return False, f"포트는 {PORT_MIN}~{PORT_MAX} 사이여야 합니다."
        return True, ""

    def get_server_config(self):
        """tcp_server 가 쓰는 설정 객체로 만들어 준다."""
        return FasteningServerConfig(
            host=self.get_host(),
            port=self.get_port(),
            mode=self.get_mode(),
            rotation_order=self.get_rotation_order(),
        )

    # ===== 모드 / 회전 순서 =====
    def get_mode(self):
        """현재 선택된 통신 모드. "ascii" | "big" | "little" """
        return self.mode_combo.currentData() or DEFAULT_MODE

    def set_mode(self, mode):
        """통신 모드를 설정한다. 목록에 없는 값은 무시한다."""
        index = self.mode_combo.findData(mode)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)

    def get_rotation_order(self):
        """현재 선택된 회전 출력 순서. "zyx" | "xyz" """
        return self.rotation_combo.currentData() or DEFAULT_ROTATION_ORDER

    def set_rotation_order(self, order):
        index = self.rotation_combo.findData(order)
        if index >= 0:
            self.rotation_combo.setCurrentIndex(index)

    # ===== 쓰기 =====
    def set_server_enabled(self, enabled):
        # 프로그램이 상태를 맞추는 경우이므로 toggled 를 다시 쏘지 않는다.
        self.toggle_server.set_checked_silently(bool(enabled))

    def set_host(self, host):
        self.input_host.setText(host)

    def set_port(self, port):
        self.input_port.setText(str(port))
