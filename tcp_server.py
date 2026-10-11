"""TCP 서버 - 오퍼레이터 모드용.

이 프로그램이 서버, 외부 장비가 클라이언트다. 클라이언트가 체결 순서를 실어
트리거를 보내면 서버가 처리하고 결과를 돌려준다.

패킷 형식은 protocol.py 가 정한다. 이 파일은 '받은 줄을 어느 처리기로 보내고
그 결과를 어떤 응답으로 만들지'만 맡는다. 규약을 고칠 일이 있으면
protocol.py 의 COMMANDS 표를 고치면 되고 이 파일은 손대지 않아도 된다.

처리기 붙이는 방법
──────────────────
    server = FasteningTCPServer(config)

    def on_capture(request):
        # request.sequence 번째 체결 위치를 찍는다.
        # 성공이면 dict(비어 있어도 됨)를, 실패면 None 을 돌려준다.
        return {} if 촬영_성공 else None

    def on_result(request):
        # 좌표를 돌려준다. 키는 protocol.py 의 reply 칸 이름과 같아야 한다.
        return {"reserved1": 0, "reserved2": 0,
                "x": 1.0, "y": 2.0, "z": 3.0,
                "a": 0.0, "b": 0.0, "c": 0.0}

    server.set_handler(protocol.CMD_CAPTURE, on_capture)
    server.set_handler(protocol.CMD_RESULT, on_result)

처리기가 None 을 돌려주거나 예외를 내면 실패 응답이 나간다. 그래서 처리기
안에서 실패를 일일이 응답으로 바꿀 필요가 없다.

    ok, reason = server.start()
    ...
    server.stop()

혼자 띄워서 시험하기
────────────────────
GUI 없이도 서버만 띄울 수 있다. TCP 테스트 툴로 붙어 패킷을 주고받아 볼 때 쓴다.

    python tcp_server.py                      (0.0.0.0:5000)
    python tcp_server.py --port 5001
    python tcp_server.py --fail               (일부러 실패 응답만 돌려준다)

스레드에 대해
─────────────
받는 일은 별도 스레드에서 돈다. 그래서 처리기도 **화면 스레드가 아닌 곳에서**
불린다. 처리기 안에서 위젯을 직접 만지면 안 되고, 화면을 고쳐야 하면 Qt 신호
(또는 QTimer.singleShot)로 화면 스레드에 넘겨야 한다.
"""

import argparse
import errno
import logging
import socket
import threading

import protocol

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000

ENCODING = "ascii"
BACKLOG = 5                  # 대기 큐에 둘 접속 수
ACCEPT_TIMEOUT = 0.3         # accept 를 이만큼씩 끊어 stop() 을 확인한다 (초)
IDLE_TIMEOUT = 0.05          # 이만큼 조용하면 남은 버퍼를 살펴본다 (초)
MAX_BUFFER = 4096            # 줄바꿈 없이 쓰레기만 계속 오면 여기서 버린다
JOIN_TIMEOUT = 2.0           # stop() 이 스레드를 기다리는 시간 (초)


class FasteningServerConfig:
    """TCP 서버 설정."""

    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, mode=protocol.MODE_LITTLE,
                 rotation_order=protocol.ROTATION_ZYX):
        self.host = host
        self.port = port
        self.timeout = 30        # 초
        self.buffer_size = 4096  # 바이트
        # 통신 모드: "ascii" | "big" | "little"
        self.mode = mode
        # 회전 세 값을 어느 순서로 실어 보낼지: "zyx"(쿠카 A·B·C) | "xyz".
        # 서버가 직접 쓰지는 않는다 — 응답을 만드는 쪽(OperatorView._on_result)이
        # 이 값을 읽어 a·b·c 칸을 채운다. 설정을 한 군데 모아 두려고 여기 둔다.
        self.rotation_order = rotation_order

    def __repr__(self):
        return (
            f"<FasteningServerConfig {self.host}:{self.port} "
            f"mode={self.mode} rotation={self.rotation_order}>"
        )



class FasteningTCPServer:
    """받은 패킷을 처리기로 넘기고 응답을 만들어 주는 서버."""

    def __init__(self, config=None):
        self.config = config or FasteningServerConfig()
        self._handlers = {}
        self._running = False

        # 받은/보낸 패킷과 접속 상태를 알려 주는 자리. 화면에 상태를 띄울 때
        # 여기에 함수를 걸면 된다. 받는 스레드에서 불리므로 위젯을 직접
        # 만지지 말고 Qt 신호로 넘겨야 한다.
        self.on_event = None

        self._listener = None
        self._accept_thread = None
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._clients = {}       # socket -> thread

    # ===== 처리기 =====
    def set_handler(self, code, handler):
        """명령 번호에 처리기를 붙인다.

        handler(request) 는 성공하면 dict, 실패하면 None 을 돌려준다.
        """
        if protocol.find_command(code) is None:
            raise ValueError(f"protocol.COMMANDS 에 없는 명령 번호입니다: {code}")
        self._handlers[code] = handler

    def clear_handlers(self):
        self._handlers.clear()

    # ===== 패킷 처리 =====
    def handle_packet(self, text):
        """받은 한 줄을 처리하고 돌려줄 한 줄을 만든다.

        소켓과 무관하므로 이 함수만 따로 시험할 수 있다.

        Returns:
            돌려줄 응답 문자열. 명령 번호조차 못 읽으면 None
            (무엇에 대한 실패인지 적을 수 없으므로 응답하지 않는다).
        """
        try:
            request = protocol.parse_request(text)
        except protocol.ProtocolError as exc:
            code = self._peek_code(text)
            logger.warning("패킷을 읽지 못했습니다: %s", exc)
            if code is None:
                return None
            return protocol.build_fail(code)

        handler = self._handlers.get(request.code)
        if handler is None:
            logger.warning("처리기가 없는 명령입니다: %s", request.code)
            return protocol.build_fail(request.code)

        try:
            result = handler(request)
        except Exception:
            # 처리기에서 무엇이 터져도 서버는 실패 응답을 내고 살아 있어야 한다.
            logger.exception("처리기에서 예외가 났습니다: %s", request.code)
            return protocol.build_fail(request.code)

        if result is None:
            return protocol.build_fail(request.code)

        try:
            return protocol.build_ok(request.code, result)
        except protocol.ProtocolError as exc:
            # 처리기가 응답 칸을 덜 채운 경우. 규약을 어긴 응답을 내보내는 대신
            # 실패로 돌려주고 로그에 남긴다.
            logger.error("응답을 만들지 못했습니다: %s", exc)
            return protocol.build_fail(request.code)

    @staticmethod
    def _peek_code(text):
        """규약을 어긴 패킷에서도 명령 번호만은 건져 본다."""
        if not text:
            return None
        head = text.strip().split(protocol.SEPARATOR)[0].strip()
        try:
            code = int(head)
        except ValueError:
            return None
        return code if protocol.find_command(code) else None

    # ===== 켜기 / 끄기 =====
    def start(self):
        """서버를 띄운다.

        Returns:
            (True, "") 또는 (False, 사유). 포트가 이미 쓰이고 있거나 없는 IP를
            적었을 때 예외를 던지지 않고 사유를 돌려주는 이유는, 화면에서
            그대로 사용자에게 보여 주기 위해서다.
        """
        if self._running:
            return False, "서버가 이미 켜져 있습니다."

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # 윈도우에서 SO_REUSEADDR 를 켜면 이미 듣고 있는 포트에도 겹쳐
            # bind 가 된다(포트 가로채기). 그러면 서버가 조용히 두 개 떠서
            # 어느 쪽이 패킷을 받는지 알 수 없다. 겹침을 막는 옵션을 쓴다.
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                listener.setsockopt(
                    socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
                )
            else:
                # 리눅스/맥은 반대로, 껐다 바로 켤 때 TIME_WAIT 로 막히지
                # 않게 이 옵션이 필요하다(겹쳐 bind 되지는 않는다).
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.config.host, self.config.port))
            listener.listen(BACKLOG)
            listener.settimeout(ACCEPT_TIMEOUT)
        except OSError as exc:
            listener.close()
            reason = self._bind_error_text(exc)
            logger.error("서버를 띄우지 못했습니다: %s", reason)
            return False, reason

        self._listener = listener
        self._stopping.clear()
        self._running = True
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="fasten-tcp-accept", daemon=True
        )
        self._accept_thread.start()

        self._notify(f"서버 시작 {self.config.host}:{self.config.port}")
        return True, ""

    def stop(self):
        """서버를 내린다. 켜지지 않았어도 안전하다."""
        if not self._running:
            return

        self._stopping.set()
        self._running = False

        # 듣던 소켓을 닫으면 accept 가 깨어난다.
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass
            self._listener = None

        # 붙어 있는 클라이언트도 깨워서 recv 를 끝낸다.
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            self._shutdown_socket(client)

        if self._accept_thread is not None:
            self._accept_thread.join(timeout=JOIN_TIMEOUT)
            self._accept_thread = None

        with self._lock:
            threads = list(self._clients.values())
            self._clients.clear()
        for thread in threads:
            thread.join(timeout=JOIN_TIMEOUT)

        self._notify("서버 중지")

    def is_running(self):
        return self._running

    def client_count(self):
        with self._lock:
            return len(self._clients)

    # ===== 받기 =====
    def _accept_loop(self):
        while not self._stopping.is_set():
            listener = self._listener
            if listener is None:
                break
            try:
                client, address = listener.accept()
            except socket.timeout:
                continue         # stop() 을 확인하러 한 바퀴 돈다
            except OSError:
                break            # stop() 이 소켓을 닫았다

            thread = threading.Thread(
                target=self._serve_client,
                args=(client, address),
                name=f"fasten-tcp-{address[0]}:{address[1]}",
                daemon=True,
            )
            with self._lock:
                self._clients[client] = thread
            self._notify(f"접속 {address[0]}:{address[1]}")
            thread.start()

    def _serve_client(self, client, address):
        who = f"{address[0]}:{address[1]}"
        if self.config.mode == protocol.MODE_ASCII:
            self._serve_ascii(client, address, who)
        else:
            self._serve_binary(client, address, who)

    def _serve_ascii(self, client, address, who):
        """ASCII 모드: 줄바꿈으로 끊어 처리한다."""
        buffer = ""
        try:
            client.settimeout(IDLE_TIMEOUT)
            while not self._stopping.is_set():
                try:
                    chunk = client.recv(self.config.buffer_size)
                except socket.timeout:
                    # 줄바꿈을 붙이지 않는 테스트 툴을 위해, 조용해지면
                    # 남은 글자가 이미 완성된 패킷인지 살펴본다.
                    buffer = self._flush_idle(client, buffer, who)
                    continue
                except OSError:
                    break

                if not chunk:
                    break        # 클라이언트가 끊었다

                buffer += chunk.decode(ENCODING, errors="replace")
                buffer = self._consume_lines(client, buffer, who)

                if len(buffer) > MAX_BUFFER:
                    logger.warning("버퍼가 너무 커져서 버립니다 (%s)", who)
                    buffer = ""
        finally:
            self._shutdown_socket(client)
            try:
                client.close()
            except OSError:
                pass
            with self._lock:
                self._clients.pop(client, None)
            self._notify(f"접속 종료 {who}")

    def _serve_binary(self, client, address, who):
        """바이너리 모드: 수신은 660바이트 단위, 처리는 첫 64바이트만.

        쿠카 EKI는 660바이트 버퍼 전체를 전송한다.
        실제 데이터는 앞부분에 있고 나머지는 0 패딩이므로
        첫 64바이트만 처리하고 나머지는 버린다.
        수동 테스트 툴(64바이트)도 첫 64바이트만 처리하면 되므로 호환된다.
        """
        buffer = b""
        try:
            client.settimeout(IDLE_TIMEOUT)
            while not self._stopping.is_set():
                try:
                    chunk = client.recv(self.config.buffer_size)
                except socket.timeout:
                    # 조용해지면 64바이트가 모였는지 확인
                    if len(buffer) >= protocol.BINARY_PACKET_SIZE:
                        packet = buffer[:protocol.BINARY_PACKET_SIZE]
                        # 660바이트 단위로 정리 (나머지 0패딩 버림)
                        if len(buffer) >= protocol.BINARY_SEND_SIZE:
                            buffer = buffer[protocol.BINARY_SEND_SIZE:]
                        else:
                            buffer = buffer[protocol.BINARY_PACKET_SIZE:]
                        self._respond_binary(client, packet, who)
                    elif buffer:
                        logger.debug(
                            "바이너리 대기 중 (%s): %d/%d 바이트 수신",
                            who, len(buffer), protocol.BINARY_PACKET_SIZE
                        )
                    continue
                except OSError:
                    break

                if not chunk:
                    break        # 클라이언트가 끊었다

                buffer += chunk
                logger.debug(
                    "바이너리 수신 (%s): +%d바이트, 버퍼 %d바이트",
                    who, len(chunk), len(buffer)
                )
                # 660바이트(쿠카) 또는 64바이트(수동 툴) 단위로 처리
                while len(buffer) >= protocol.BINARY_PACKET_SIZE:
                    packet = buffer[:protocol.BINARY_PACKET_SIZE]
                    self._respond_binary(client, packet, who)
                    # 660바이트가 모였으면 660바이트 소비, 아니면 64바이트 소비
                    if len(buffer) >= protocol.BINARY_SEND_SIZE:
                        buffer = buffer[protocol.BINARY_SEND_SIZE:]
                    else:
                        buffer = buffer[protocol.BINARY_PACKET_SIZE:]
                    # 남은 버퍼가 64바이트 미만이면 대기
                    if len(buffer) < protocol.BINARY_PACKET_SIZE:
                        break

                if len(buffer) > MAX_BUFFER:
                    logger.warning("버퍼가 너무 커져서 버립니다 (%s)", who)
                    buffer = b""


        finally:
            self._shutdown_socket(client)
            try:
                client.close()
            except OSError:
                pass
            with self._lock:
                self._clients.pop(client, None)
            self._notify(f"접속 종료 {who}")


    def _consume_lines(self, client, buffer, who):
        """줄바꿈으로 끊어 완성된 패킷부터 처리한다."""
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.strip()
            if line:
                self._respond(client, line, who)
        return buffer

    def _flush_idle(self, client, buffer, who):
        """줄바꿈 없이 온 패킷 처리.

        조용해진 뒤에도 규약에 맞는 완성된 패킷이면 처리한다. 아직 오는 중일
        수 있으므로(예: 10 의 1 만 도착) 규약에 맞지 않으면 그대로 둔다.
        """
        text = buffer.strip()
        if not text:
            return ""
        try:
            protocol.parse_request(text)
        except protocol.ProtocolError:
            return buffer        # 더 올 수 있으니 기다린다
        self._respond(client, text, who)
        return ""

    def _respond(self, client, text, who):
        """ASCII 모드 응답."""
        self._notify(f"받음 {who} : {text}")
        reply = self.handle_packet(text)
        if reply is None:
            # 무엇에 대한 실패인지 적을 수 없어 응답하지 않는다.
            self._notify(f"보냄 {who} : (응답 없음)")
            return
        try:
            client.sendall(reply.encode(ENCODING))
        except OSError as exc:
            logger.warning("응답을 보내지 못했습니다 (%s): %s", who, exc)
            return
        self._notify(f"보냄 {who} : {reply.strip()}")

    def _send_binary_fail(self, client, code, who, mode):
        """실패 응답을 바이너리로 만들어 전송한다. (660바이트)"""
        fail_values = [code, protocol.STATUS_FAIL]
        fail_types = ["i", "i"]
        try:
            binary = protocol.pack_binary_mixed(
                fail_values, fail_types, mode=mode,
                send_size=protocol.BINARY_SEND_SIZE
            )
            client.sendall(binary)
            self._notify(f"보냄 {who} : {code},{protocol.STATUS_FAIL} (실패)")
        except (OSError, protocol.ProtocolError) as exc:
            logger.warning("실패 응답을 보내지 못했습니다 (%s): %s", who, exc)


    def _respond_binary(self, client, packet, who):
        """바이너리 모드 응답.

        1. 64바이트 바이너리 패킷을 받아 int/float 타입에 맞게 언팩
        2. ASCII 문자열로 변환해서 handle_packet() 으로 처리
        3. 결과를 다시 int/float 혼합 바이너리로 팩해서 전송
        """
        mode = self.config.mode
        logger.debug(
            "_respond_binary (%s): packet %d바이트, hex=%s",
            who, len(packet), packet[:16].hex(' ')
        )

        # --- 1. 바이너리 → 값 리스트 → ASCII 문자열 ---
        try:
            code = protocol.unpack_binary(packet, count=1, mode=mode)[0]
            command = protocol.find_command(code)
            if command is None:
                logger.warning("모르는 명령 번호입니다: %s", code)
                self._send_binary_fail(client, code, who, mode)
                return
            # 요청 필드 타입에 맞게 언팩 (int/float 혼합)
            req_types = protocol._request_types(code)
            req_count = len(req_types)
            values = protocol.unpack_binary_mixed(
                packet, count=req_count, types=req_types, mode=mode
            )
        except protocol.ProtocolError as exc:
            logger.warning("바이너리 패킷을 읽지 못했습니다: %s", exc)
            # 첫 4바이트라도 읽어서 코드 추출 시도
            try:
                code = protocol.unpack_binary(packet, count=1, mode=mode)[0]
            except Exception:
                code = 0
            self._send_binary_fail(client, code, who, mode)
            return

        text = protocol.SEPARATOR.join(str(v) for v in values)
        self._notify(f"받음 {who} : {text}")

        # --- 2. handle_packet 으로 응답 생성 (ASCII) ---
        reply = self.handle_packet(text)
        if reply is None:
            # handle_packet 이 None 을 반환하면 실패 응답을 보낸다
            logger.warning("handle_packet 이 응답을 만들지 못했습니다: %s", text)
            self._send_binary_fail(client, code, who, mode)
            return

        # --- 3. ASCII 응답 → int/float 혼합 바이너리 팩 → 전송 ---
        reply_clean = reply.rstrip(protocol.TERMINATOR)
        reply_parts = [p.strip() for p in reply_clean.split(protocol.SEPARATOR)]
        reply_code = int(reply_parts[0]) if reply_parts else 0
        reply_types = protocol._reply_types(reply_code)
        if reply_types is None:
            # 타입을 모르면 모두 int 로 시도
            reply_types = ["i"] * len(reply_parts)

        # reply_parts(문자열)를 타입에 맞게 숫자로 변환
        reply_values = []
        for part, t in zip(reply_parts, reply_types):
            try:
                reply_values.append(float(part) if t == "f" else int(part))
            except ValueError:
                logger.error("응답 값을 숫자로 변환하지 못했습니다: %r", part)
                self._send_binary_fail(client, code, who, mode)
                return

        try:
            binary = protocol.pack_binary_mixed(
                reply_values, reply_types, mode=mode,
                send_size=protocol.BINARY_SEND_SIZE
            )
        except protocol.ProtocolError as exc:
            logger.error("응답을 바이너리로 만들지 못했습니다: %s", exc)
            self._send_binary_fail(client, code, who, mode)
            return

        try:
            client.sendall(binary)
        except OSError as exc:
            logger.warning("응답을 보내지 못했습니다 (%s): %s", who, exc)
            return
        self._notify(f"보냄 {who} : {reply_clean}")





    # ===== 잡일 =====
    @staticmethod
    def _shutdown_socket(sock):
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def _notify(self, text):
        logger.info("%s", text)
        if self.on_event is None:
            return
        try:
            self.on_event(text)
        except Exception:
            logger.exception("on_event 에서 예외가 났습니다")

    def _bind_error_text(self, exc):
        """bind 실패를 사용자에게 보여 줄 문장으로 바꾼다.

        윈도우는 errno 대신 WSA 번호(10048/10049)로 올 때가 있어 둘 다 본다.
        """
        code = getattr(exc, "errno", None)
        if code in (errno.EADDRINUSE, 10048):
            return (
                f"포트 {self.config.port} 가 이미 사용 중입니다. "
                "다른 포트를 쓰거나 그 프로그램을 종료하세요."
            )
        if code in (errno.EADDRNOTAVAIL, 10049):
            return (
                f"이 PC 에 없는 IP 입니다: {self.config.host}. "
                "0.0.0.0 을 쓰거나 이 PC 의 IP 를 적으세요."
            )
        return f"{self.config.host}:{self.config.port} 를 열지 못했습니다: {exc}"


# ======================================================
# 혼자 띄워 시험하기 (TCP 테스트 툴로 붙어 볼 때)
# ======================================================
def _demo_handlers(server, always_fail=False):
    """비전이 아직 없으므로 가짜 값으로 응답한다."""

    def on_capture(request):
        if always_fail:
            return None
        print(f"  [촬영] {request.sequence}번 체결 위치")
        return {}

    def on_result(request):
        if always_fail:
            return None
        print(f"  [결과] {request.sequence}번 체결 위치 좌표 반환")
        return {
            "reserved1": 0,
            "reserved2": 0,
            "x": 100.0 + request.sequence,
            "y": 200.0,
            "z": 300.0,
            "a": 0.0,
            "b": 0.0,
            "c": 90.0,
        }

    def on_switch_model(request):
        if always_fail:
            return None
        print(f"  [모델] {request.values['model']}번 모델로 변경")
        return {}

    server.set_handler(protocol.CMD_CAPTURE, on_capture)
    server.set_handler(protocol.CMD_RESULT, on_result)
    server.set_handler(protocol.CMD_SWITCH_MODEL, on_switch_model)


def main():
    parser = argparse.ArgumentParser(
        description="체결 자동화 TCP 서버를 혼자 띄운다(시험용)."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind 할 IP")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--fail", action="store_true",
        help="일부러 실패 응답(1003)만 돌려준다",
    )
    parser.add_argument(
        "--mode", default=protocol.MODE_LITTLE,
        choices=[protocol.MODE_ASCII, protocol.MODE_BIG, protocol.MODE_LITTLE],
        help="통신 모드: little(기본) | big(빅 엔디안) | ascii",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    server = FasteningTCPServer(
        FasteningServerConfig(args.host, args.port, mode=args.mode)
    )

    _demo_handlers(server, always_fail=args.fail)

    ok, reason = server.start()
    if not ok:
        print(reason)
        return 1

    print()
    print(protocol.describe())
    print()
    print("Ctrl+C 로 종료합니다.")
    try:
        while True:
            server._stopping.wait(1.0)
    except KeyboardInterrupt:
        print()
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
