"""프레임리스 창의 가장자리 리사이즈 지원 (Windows 네이티브).

QMainWindow 등에 함께 상속시키면 OS의 히트테스트(WM_NCHITTEST)를 가로채
가장자리/모서리 드래그 리사이즈를 활성화한다. 커서 모양 변경, 드래그,
에어로 스냅까지 OS가 처리하므로 별도 마우스 이벤트 구현이 필요 없다.
"""

import ctypes
from ctypes.wintypes import MSG

from PyQt5.QtCore import QPoint

from theme import RESIZE_MARGIN

# WM_NCHITTEST 반환 코드
_WM_NCHITTEST = 0x0084
_HTLEFT = 10
_HTRIGHT = 11
_HTTOP = 12
_HTTOPLEFT = 13
_HTTOPRIGHT = 14
_HTBOTTOM = 15
_HTBOTTOMLEFT = 16
_HTBOTTOMRIGHT = 17


class NativeResizeMixin:
    #: 가장자리 감지 두께(px)
    resize_margin = RESIZE_MARGIN

    def nativeEvent(self, eventType, message):
        if eventType == b"windows_generic_MSG":
            msg = MSG.from_address(int(message))
            # 전체화면에서도 막는다. 오퍼레이터 모드는 전체화면으로 들어가는데,
            # 그때 화면 가장자리를 끌면 창이 전체화면에서 빠져나와 작아진다.
            if (
                msg.message == _WM_NCHITTEST
                and not self.isMaximized()
                and not self.isFullScreen()
            ):
                # lParam: 하위 16비트 = 화면 x, 상위 16비트 = 화면 y (부호 있음)
                x = ctypes.c_short(msg.lParam & 0xFFFF).value
                y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value
                pos = self.mapFromGlobal(QPoint(x, y))

                w, h = self.width(), self.height()
                m = self.resize_margin
                left = pos.x() < m
                right = pos.x() >= w - m
                top = pos.y() < m
                bottom = pos.y() >= h - m

                if top and left:
                    return True, _HTTOPLEFT
                if top and right:
                    return True, _HTTOPRIGHT
                if bottom and left:
                    return True, _HTBOTTOMLEFT
                if bottom and right:
                    return True, _HTBOTTOMRIGHT
                if left:
                    return True, _HTLEFT
                if right:
                    return True, _HTRIGHT
                if top:
                    return True, _HTTOP
                if bottom:
                    return True, _HTBOTTOM

        return super().nativeEvent(eventType, message)
