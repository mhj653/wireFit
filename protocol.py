"""TCP 통신 패킷 규약.

패킷은 쉼표로 끊은 아스키 한 줄이다. 맨 앞이 명령 번호이고, 서버가 돌려주는
패킷은 명령 번호 다음에 상태 코드가 온다.

    클라이언트 -> 서버 : 101,1,0,0
    서버 -> 클라이언트 : 101,1102                      (성공)
                        101,1003                      (실패)

    클라이언트 -> 서버 : 102,1
    서버 -> 클라이언트 : 102,1100,0,0,0,x,y,z,rx,ry,rz (성공)
                        102,1003                      (실패)

    클라이언트 -> 서버 : 103,1,2
    서버 -> 클라이언트 : 103,1107                      (성공)
                        103,1003                      (실패)


규약을 고치는 방법
──────────────────
아래 COMMANDS 표 한 곳만 고치면 파싱·응답 만들기·문서가 같이 따라온다.

* 명령 추가          COMMANDS 에 Command(...) 를 하나 더 적는다.
* 명령 삭제          그 Command(...) 를 지운다.
* 요청 항목 추가     그 명령의 request 에 Field(...) 를 더한다.
                     (클라이언트가 보내는 칸 수가 늘어난다)
* 응답 항목 추가     그 명령의 reply 에 Field(...) 를 더한다.
                     (성공 응답에 붙는 칸 수가 늘어난다)
* 상태 코드 변경     STATUS_OK / STATUS_FAIL 값을 바꾼다.
* 실패 종류 구분     STATUS_FAIL 대신 새 코드를 만들고
                     build_fail(code, status=새코드) 로 넘긴다.
* 끝 문자 변경       TERMINATOR 를 바꾼다("\\r\\n" 이나 "" 로).
OK
Field 의 cast 는 '받은 글자를 무엇으로 바꿀지'다. 순서 번호는 int,
좌표는 float 이다. 칸 수가 표와 다르면 parse_request 가 ProtocolError 를 낸다.

Field 의 fixed 는 '이 칸에는 이 값만 온다'는 뜻이다(_const 로 적는다). 다른
값이 오면 parse_request 가 ProtocolError 를 내고, 서버는 실패 응답을 돌려준다.
103 의 둘째 칸처럼 규약이 값을 못 박아 둔 자리를 처리기마다 다시 검사하지
않으려고 표에 적어 둔다.
"""

import struct
from collections import namedtuple

# ======================================================
# 상태 코드
# ======================================================
STATUS_OK = 1100 # return OK
STATUS_VISION_OK = 1102 # vision OK
STATUS_SWITCH_OK = 1107 # switch OK
STATUS_FAIL = 1003      # 실패는 아직 한 가지로만 돌려준다(종류 구분은 나중에)

# ======================================================
# 패킷 형식
# ======================================================
SEPARATOR = ","
TERMINATOR = "\n"       # 한 패킷의 끝
COORD_DECIMALS = 3      # 좌표를 소수 몇 자리로 적을지

# ======================================================
# 명령 번호
# ======================================================
CMD_CAPTURE = 101       # 비전 촬영 트리거
CMD_RESULT = 102        # 비전 결과(좌표) 반환 트리거
CMD_SWITCH_MODEL=103    # 모델 변경 트리거

# fixed 는 '규약이 못 박아 둔 값'이다. None 이면 아무 값이나 받는다.
Field = namedtuple("Field", "key note cast fixed")
Field.__new__.__defaults__ = (None,)


def _int(key, note):
    return Field(key, note, int)


def _const(key, note, value):
    """항상 같은 값만 오는 칸 (예: 103 의 둘째 칸은 늘 1)."""
    return Field(key, note, int, int(value))


def _coord(key, note):
    return Field(key, note, float)


class Command:
    """명령 하나의 요청/응답 칸 구성."""

    def __init__(self, code, name, request=(), reply=(), ok_status=STATUS_OK):
        self.code = code
        self.name = name
        self.request = tuple(request)   # 명령 번호 뒤에 오는 요청 칸
        self.reply = tuple(reply)       # 상태 코드 뒤에 오는 성공 응답 칸
        self.ok_status = ok_status      # 명령별 성공 상태 코드


# ======================================================
# 명령 표  ← 규약을 고칠 곳은 여기다
# ======================================================
COMMANDS = (
    Command(
        code=CMD_CAPTURE,
        name="비전 촬영 트리거",
        request=(
            _int("sequence", "체결 세팅의 순서 (1부터)"),
            _int("reserved1", "아직 뜻 없음. 클라이언트가 0으로 채워 보낸다"),
            _int("reserved2", "아직 뜻 없음. 클라이언트가 0으로 채워 보낸다"),
        ),
        reply=(),   # 성공이면 101,1102 로 끝난다
        ok_status=STATUS_VISION_OK,
    ),
    Command(
        code=CMD_RESULT,
        name="비전 결과 반환 트리거",
        request=(
            _int("sequence", "체결 세팅의 순서 (1부터)"),
        ),
        reply=(
            _int("reserved1", "아직 뜻 없음. 1"),
            _int("reserved2", "아직 뜻 없음. 1"),
            _int("reserved3", "아직 뜻 없음. 0"),
            _coord("x", "현재 플랜지 기준 체결 위치 X (mm)"),
            _coord("y", "현재 플랜지 기준 체결 위치 Y (mm)"),
            _coord("z", "현재 플랜지 기준 체결 위치 Z (mm)"),
            # 세 칸에 무엇을 넣을지는 '회전 순서' 설정이 정한다
            # (rotation_values 참고). 기본은 쿠카 A·B·C = rz·ry·rx 다.
            _coord("a", "회전 1 (회전 순서 설정에 따라 rz 또는 rx, deg)"),
            _coord("b", "회전 2 (ry, deg)"),
            _coord("c", "회전 3 (회전 순서 설정에 따라 rx 또는 rz, deg)"),
            _int("reserved4", "아직 뜻 없음. 0"),
            _int("reserved5", "아직 뜻 없음. 0"),
        ),
    ),
    Command(
        code=CMD_SWITCH_MODEL,
        name="모델 변경 트리거",
        request=(
            _const("fixed", "늘 1 이다. 다른 값이 오면 실패로 돌려준다", 1),
            _int("model", "모델 순서 (1부터. '모델 변경' 창에 늘어선 순서다)"),
        ),
        reply=(),   # 성공이면 103,1107 로 끝난다
        ok_status=STATUS_SWITCH_OK,
    ),
)

_BY_CODE = {command.code: command for command in COMMANDS}


class ProtocolError(Exception):
    """패킷이 규약과 다를 때."""


class Request:
    """클라이언트가 보낸 패킷 한 개."""

    def __init__(self, command, values, raw):
        self.command = command
        self.values = values     # {"sequence": 1, "reserved1": 0, ...}
        self.raw = raw

    @property
    def code(self):
        return self.command.code

    @property
    def sequence(self):
        """거의 모든 명령이 쓰는 값이라 따로 꺼내 준다."""
        return self.values.get("sequence")

    def __repr__(self):
        return f"<Request {self.code} {self.values}>"


# ======================================================
# 읽기
# ======================================================
def find_command(code):
    return _BY_CODE.get(code)


def parse_request(text):
    """받은 한 줄을 Request 로 바꾼다.

    Raises:
        ProtocolError: 명령 번호를 모르거나 칸 수가 표와 다를 때.
    """
    if text is None:
        raise ProtocolError("빈 패킷입니다.")
    parts = [part.strip() for part in text.strip().split(SEPARATOR)]
    if not parts or not parts[0]:
        raise ProtocolError("빈 패킷입니다.")

    try:
        code = int(parts[0])
    except ValueError:
        raise ProtocolError(f"명령 번호가 숫자가 아닙니다: {parts[0]!r}")

    command = find_command(code)
    if command is None:
        raise ProtocolError(f"모르는 명령 번호입니다: {code}")

    given = parts[1:]
    if len(given) != len(command.request):
        raise ProtocolError(
            f"{code}({command.name})는 값이 {len(command.request)}개여야 "
            f"하는데 {len(given)}개가 왔습니다: {text.strip()!r}"
        )

    values = {}
    for field, raw_value in zip(command.request, given):
        try:
            values[field.key] = field.cast(raw_value)
        except ValueError:
            raise ProtocolError(
                f"{code}({command.name})의 {field.key} 값이 잘못됐습니다: "
                f"{raw_value!r}"
            )
        if field.fixed is not None and values[field.key] != field.fixed:
            raise ProtocolError(
                f"{code}({command.name})의 {field.key} 는 {field.fixed} 여야 "
                f"하는데 {values[field.key]} 가 왔습니다: {text.strip()!r}"
            )
    return Request(command, values, text.strip())


# ======================================================
# 쓰기
# ======================================================
def _format(field, value):
    if field.cast is float:
        return f"{float(value):.{COORD_DECIMALS}f}"
    return str(field.cast(value))


def build_ok(code, values=None, terminate=True):
    """성공 응답을 만든다.

    Args:
        code: 명령 번호
        values: reply 칸에 채울 값 {"x": 1.0, ...}
        terminate: 끝 문자를 붙일지

    Raises:
        ProtocolError: 모르는 명령이거나 reply 칸 값이 빠졌을 때.
    """
    command = find_command(code)
    if command is None:
        raise ProtocolError(f"모르는 명령 번호입니다: {code}")

    values = values or {}
    parts = [str(code), str(command.ok_status)]
    for field in command.reply:
        if field.key not in values:
            raise ProtocolError(
                f"{code}({command.name}) 응답에 {field.key} 가 빠졌습니다."
            )
        parts.append(_format(field, values[field.key]))

    packet = SEPARATOR.join(parts)
    return packet + TERMINATOR if terminate else packet


def build_fail(code, status=STATUS_FAIL, terminate=True):
    """실패 응답을 만든다 (예: 101,1003).

    실패 종류를 나누게 되면 status 에 새 코드를 넘긴다.
    """
    packet = SEPARATOR.join([str(code), str(status)])
    return packet + TERMINATOR if terminate else packet


# ======================================================
# 바이너리 (빅/리틀 엔디안) 인코딩
# ======================================================
BINARY_PACKET_SIZE = 64   # 최소 패킷 길이 (바이트)
BINARY_SEND_SIZE = 660     # 응답 전송 길이 (쿠카 EKI Size="660" 대응)
INT32_SIZE = 4             # 정수 하나당 4바이트


# 엔디안 모드
MODE_ASCII = "ascii"
MODE_BIG = "big"
MODE_LITTLE = "little"

# 화면(오퍼레이터 모드의 프로토콜 콤보)에 그대로 들어가는 목록.
# (표시 이름, 값) 순서대로 콤보에 들어간다.
MODE_CHOICES = (
    ("ASCII", MODE_ASCII),
    ("Little Endian", MODE_LITTLE),
    ("Big Endian", MODE_BIG),
)

_ENDIAN_FMT = {
    MODE_BIG: ">",
    MODE_LITTLE: "<",
}

# ======================================================
# 회전 출력 순서
# ======================================================
# 로봇마다 회전 세 값을 받는 순서가 다르다. 쿠카는 A, B, C 순서이고 A=rz,
# B=ry, C=rx 다(ZYX 오일러각). 그런데 rx 부터 받는 로봇도 있어서, 어느 순서로
# 실어 보낼지 고를 수 있어야 한다.
#
# **각도 값을 바꾸는 것이 아니라 나가는 자리를 바꾼다.** 회전 자체는 늘 같은
# ZYX 오일러각이고, 패킷의 세 칸(a, b, c)에 rz·ry·rx 를 넣을지 rx·ry·rz 를
# 넣을지만 갈린다. 바이너리 모드에서는 그 세 칸이 그대로 바이트 순서다.
ROTATION_ZYX = "zyx"    # a=rz, b=ry, c=rx  (쿠카 A·B·C)
ROTATION_XYZ = "xyz"    # a=rx, b=ry, c=rz

ROTATION_ORDER_CHOICES = (
    ("Rz → Ry → Rx", ROTATION_ZYX),
    ("Rx → Ry → Rz", ROTATION_XYZ),
)

# 순서마다 (a, b, c) 칸에 넣을 포즈 속성 이름.
_ROTATION_KEYS = {
    ROTATION_ZYX: ("rz", "ry", "rx"),
    ROTATION_XYZ: ("rx", "ry", "rz"),
}


def rotation_values(pose, order=ROTATION_ZYX):
    """포즈의 회전을 고른 순서대로 (a, b, c) 로 돌려준다.

    Args:
        pose: rx / ry / rz 속성을 가진 포즈 (hole_pose 의 포즈)
        order: ROTATION_ZYX 또는 ROTATION_XYZ. 모르는 값이면 ZYX 로 본다
               (예전 세팅 파일에서 읽은 값이 상해 있어도 로봇이 받던 순서를
               유지하는 편이 안전하다).

    Returns:
        (a, b, c) 실수 세 개
    """
    keys = _ROTATION_KEYS.get(order, _ROTATION_KEYS[ROTATION_ZYX])
    return tuple(float(getattr(pose, key)) for key in keys)


def _endian_char(mode):
    """모드 문자열을 struct 포맷 문자로 바꾼다."""
    fmt = _ENDIAN_FMT.get(mode)
    if fmt is None:
        raise ProtocolError(f"알 수 없는 엔디안 모드입니다: {mode!r}")
    return fmt


def pack_binary(values, mode=MODE_BIG, send_size=None):
    """정수 리스트를 고정 길이 바이너리로 변환.

    Args:
        values: [103, 1, 2] 같은 정수 리스트
        mode:   "big" 또는 "little"
        send_size: 패딩할 총 길이 (기본: BINARY_PACKET_SIZE=64)

    Returns:
        send_size 바이트 bytes 객체 (남는 공간은 0x00 으로 채운다)
    """
    fmt = _endian_char(mode)
    size = send_size or BINARY_PACKET_SIZE
    body = b"".join(struct.pack(f"{fmt}i", int(v)) for v in values)
    if len(body) > size:
        raise ProtocolError(
            f"데이터가 {size}바이트를 초과합니다: {len(body)}바이트"
        )
    return body + b"\x00" * (size - len(body))



def unpack_binary(data, count, mode=MODE_BIG):
    """64바이트 바이너리에서 정수 count 개를 꺼낸다.

    Args:
        data:  bytes (최소 count*4 바이트)
        count: 읽을 정수 개수
        mode:  "big" 또는 "little"

    Returns:
        [103, 1, 2] 같은 정수 리스트
    """
    fmt = _endian_char(mode)
    needed = count * INT32_SIZE
    if len(data) < needed:
        raise ProtocolError(
            f"데이터가 짧습니다: {needed}바이트 필요, {len(data)}바이트 수신"
        )
    return [struct.unpack_from(f"{fmt}i", data, i * INT32_SIZE)[0]
            for i in range(count)]


def pack_binary_float(values, mode=MODE_BIG):
    """float 리스트를 64바이트 고정 길이 바이너리로 변환.

    좌표(x,y,z,rx,ry,rz)처럼 float 이 필요할 때 쓴다.
    """
    fmt = _endian_char(mode)
    body = b"".join(struct.pack(f"{fmt}f", float(v)) for v in values)
    if len(body) > BINARY_PACKET_SIZE:
        raise ProtocolError(
            f"데이터가 {BINARY_PACKET_SIZE}바이트를 초과합니다: {len(body)}바이트"
        )
    return body + b"\x00" * (BINARY_PACKET_SIZE - len(body))


def unpack_binary_float(data, count, mode=MODE_BIG):
    """64바이트 바이너리에서 float count 개를 꺼낸다."""
    fmt = _endian_char(mode)
    needed = count * INT32_SIZE
    if len(data) < needed:
        raise ProtocolError(
            f"데이터가 짧습니다: {needed}바이트 필요, {len(data)}바이트 수신"
        )
    return [struct.unpack_from(f"{fmt}f", data, i * INT32_SIZE)[0]
            for i in range(count)]


def pack_binary_mixed(values, types, mode=MODE_BIG, send_size=None):
    """int/float 이 섞인 값을 고정 길이 바이너리로 변환.

    Args:
        values: [102, 1100, 1, 1, 0, -56.0032, ...]
        types:  ["i", "i", "i", "i", "i", "f", ...]
                "i" = int32, "f" = float32
        mode:   "big" 또는 "little"
        send_size: 패딩할 총 길이 (기본: BINARY_PACKET_SIZE=64)

    Returns:
        send_size 바이트 bytes 객체
    """
    fmt = _endian_char(mode)
    size = send_size or BINARY_PACKET_SIZE
    body = b""
    for v, t in zip(values, types):
        if t == "f":
            body += struct.pack(f"{fmt}f", float(v))
        else:
            body += struct.pack(f"{fmt}i", int(v))
    if len(body) > size:
        raise ProtocolError(
            f"데이터가 {size}바이트를 초과합니다: {len(body)}바이트"
        )
    return body + b"\x00" * (size - len(body))



def unpack_binary_mixed(data, count, types, mode=MODE_BIG):
    """int/float 이 섞인 64바이트 바이너리에서 count 개를 꺼낸다.

    Args:
        data:  bytes
        count: 읽을 값 개수
        types: ["i", "i", "i", "i", "i", "f", ...] ("i"=int32, "f"=float32)
        mode:  "big" 또는 "little"

    Returns:
        [102, 1100, 1, 1, 0, -56.0032, ...] (int 는 int, float 는 float)
    """
    fmt = _endian_char(mode)
    needed = count * INT32_SIZE
    if len(data) < needed:
        raise ProtocolError(
            f"데이터가 짧습니다: {needed}바이트 필요, {len(data)}바이트 수신"
        )
    result = []
    for i, t in enumerate(types):
        if t == "f":
            result.append(struct.unpack_from(f"{fmt}f", data, i * INT32_SIZE)[0])
        else:
            result.append(struct.unpack_from(f"{fmt}i", data, i * INT32_SIZE)[0])
    return result


def _reply_types(code):
    """응답의 필드 타입 리스트를 만든다.

    Returns:
        ["i", "i", "i", "i", "i", "f", "f", "f", "f", "f", "f", "i", "i"]
        (명령 번호, 상태 코드, reply 필드들 순서)
    """
    command = find_command(code)
    if command is None:
        return None
    types = ["i", "i"]  # 명령 번호, 상태 코드
    for field in command.reply:
        types.append("f" if field.cast is float else "i")
    return types


def _request_types(code):
    """요청의 필드 타입 리스트를 만든다.

    Returns:
        ["i", "i", "i", "i"] (명령 번호 + request 필드들)
    """
    command = find_command(code)
    if command is None:
        return None
    types = ["i"]  # 명령 번호
    for field in command.request:
        types.append("f" if field.cast is float else "i")
    return types


# ======================================================
# 규약 설명 (문서/로그용)
# ======================================================


def describe():
    """규약을 사람이 읽을 수 있는 글로 만든다."""
    lines = [
        f"상태 코드: 성공(촬영) {STATUS_VISION_OK} / 성공(결과) {STATUS_OK} / "
        f"성공(모델 변경) {STATUS_SWITCH_OK} / 실패 {STATUS_FAIL}",
        "",
    ]
    for command in COMMANDS:
        request_keys = [field.key for field in command.request]
        reply_keys = [field.key for field in command.reply]

        lines.append(f"[{command.code}] {command.name}")
        lines.append("  요청  " + SEPARATOR.join([str(command.code)] + request_keys))
        lines.append(
            "  성공  "
            + SEPARATOR.join([str(command.code), str(command.ok_status)] + reply_keys)
        )
        lines.append(f"  실패  {command.code}{SEPARATOR}{STATUS_FAIL}")
        for field in command.request + command.reply:
            fixed = "" if field.fixed is None else f" (항상 {field.fixed})"
            lines.append(f"        - {field.key}: {field.note}{fixed}")
        lines.append("")
    return "\n".join(lines).rstrip()


if __name__ == "__main__":
    print(describe())
