"""RealSense Viewer 에서 저장한 Advanced Mode 프리셋을 골라 카메라에 싣는다.

뷰어에서 뎁스 엔진 파라미터를 맞춘 뒤 'Save/Export Settings' 로 받은 JSON 을
Preset/ 폴더에 넣어 두면, Camera Selection 페이지의 콤보에 이름이 뜬다. 고른
프리셋은 카메라를 열 때마다 실린다. 뷰어에서 본 뎁스 품질을 이 프로그램에서도
그대로 얻으려는 것이다.

    Preset/
      실내_기본.json
      광택면.json      <- 파일 이름이 곧 콤보에 뜨는 이름

파일 형식 (schema version 1)
    {
      "device":     { ... }        정보용. 적용 대상 아님
      "parameters": { ... }        ← 실제로 적용되는 것 (뎁스 엔진 파라미터)
      "viewer":     { ... }        뷰어의 스트림 설정. 적용 대상 아님 (아래)
    }

'viewer' 를 적용하지 않는 이유
    해상도/FPS 는 pipeline 을 열 때 정해지고, 바꾸면 내부 파라미터와 손-눈
    캘리브레이션의 전제가 함께 흔들린다. 프리셋 파일이 그것까지 조용히 바꾸면
    사고가 난다. 스트림 설정은 camera_capture_page 의 상수가 계속 정본이고,
    파일 값과 다르면 stream_note() 가 경고 문구만 만들어 준다.

뎁스 단위 주의
    파라미터에 param-depthunits 가 들어 있으면 뎁스 값 1당 mm 가 바뀐다
    (예: 100 = 0.1mm). 그래서 프리셋은 반드시 pipeline.start() **전에** 실어야
    한다. 뒤에 실으면 _cache_camera_info() 가 이미 읽어 둔 옛 배율이 남아
    좌표가 통째로 어긋난다. 같은 이유로, 연결 중에 프리셋을 바꾸면 다시
    연결해야 실제로 적용된다.
"""

import json
import os
import time

from app_paths import user_data_path

try:  # pyrealsense2 가 없어도 가상 카메라 모드로 UI 는 그대로 동작한다.
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - 런타임 환경 의존
    rs = None

PRESET_DIR = user_data_path("Preset")
PRESET_SUFFIX = ".json"

# 고급 모드를 켜면 장치가 리셋되며 USB 에서 잠깐 사라졌다 돌아온다. 그동안
# 기다렸다가 다시 잡는다. 못 잡으면 프리셋 없이 그냥 연결한다.
ADVANCED_MODE_RESET_TIMEOUT_S = 5.0
ADVANCED_MODE_POLL_S = 0.25


def preset_dir():
    """프리셋 폴더 경로. 없으면 만들어 둔다(사용자가 파일을 넣을 자리)."""
    try:
        os.makedirs(PRESET_DIR, exist_ok=True)
    except OSError:
        pass
    return PRESET_DIR


def preset_names():
    """Preset/ 안의 프리셋 이름 목록 (확장자 없는 파일 이름), 가나다순."""
    try:
        entries = os.listdir(preset_dir())
    except OSError:
        return []
    names = [
        os.path.splitext(entry)[0]
        for entry in entries
        if entry.lower().endswith(PRESET_SUFFIX)
    ]
    return sorted(names, key=str.lower)


def preset_path(name):
    return os.path.join(PRESET_DIR, f"{name}{PRESET_SUFFIX}")


def preset_text(name):
    """프리셋 파일 내용. 이름이 비었거나 못 읽으면 None."""
    if not name:
        return None
    try:
        with open(preset_path(name), "r", encoding="utf-8") as fp:
            return fp.read()
    except OSError:
        return None


def _preset_dict(name):
    text = preset_text(name)
    if text is None:
        return None
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def describe(name):
    """콤보 툴팁에 쓸 한 줄 요약. 읽을 수 없으면 사유를 돌려준다."""
    if not name:
        return "프리셋을 쓰지 않습니다 (카메라 기본값)."
    data = _preset_dict(name)
    if data is None:
        return f"'{name}' 을 읽을 수 없습니다 (JSON 형식이 아닐 수 있습니다)."

    device = data.get("device") if isinstance(data.get("device"), dict) else {}
    params = data.get("parameters") if isinstance(data.get("parameters"), dict) else {}
    viewer = data.get("viewer") if isinstance(data.get("viewer"), dict) else {}
    parts = [device.get("name") or "RealSense", f"파라미터 {len(params)}개"]
    if viewer.get("stream-width") and viewer.get("stream-height"):
        parts.append(
            f"뷰어 {viewer['stream-width']}x{viewer['stream-height']}"
            f"@{viewer.get('stream-fps', '?')}"
        )
    return " · ".join(parts)


def stream_note(name, color_size, fps):
    """프리셋을 만든 뷰어의 스트림 설정이 지금과 다르면 알려 줄 문구.

    파라미터 값 자체는 해상도와 무관하지만, 뎁스 엔진이 만들어 내는 결과는
    해상도에 따라 달라진다. 뷰어에서 848x480 으로 보며 맞춘 값을 1280x720 에
    실으면 눈으로 본 것과 다른 그림이 나올 수 있어, 조용히 넘기지 않고 짚는다.

    Returns:
        경고 문구 또는 None (같거나 판단할 수 없으면 None)
    """
    data = _preset_dict(name)
    if data is None:
        return None
    viewer = data.get("viewer")
    if not isinstance(viewer, dict):
        return None
    try:
        width = int(viewer.get("stream-width"))
        height = int(viewer.get("stream-height"))
        preset_fps = int(viewer.get("stream-fps"))
    except (TypeError, ValueError):
        return None

    if (width, height) == tuple(color_size) and preset_fps == fps:
        return None
    return (
        f"프리셋은 {width}x{height}@{preset_fps} 에서 맞춘 값인데 "
        f"지금은 {color_size[0]}x{color_size[1]}@{fps} 로 엽니다. "
        "뎁스 결과가 뷰어에서 본 것과 다를 수 있습니다."
    )


def _advanced_mode(device):
    """장치의 고급 모드 핸들. 고급 모드를 못 쓰는 기종이면 None."""
    try:
        return rs.rs400_advanced_mode(device)
    except Exception:
        return None


def _find_device(serial):
    """일련번호로 장치를 찾는다. 없으면 None."""
    try:
        for device in rs.context().query_devices():
            if device.get_info(rs.camera_info.serial_number) == serial:
                return device
    except Exception:
        return None
    return None


def _enable_advanced_mode(device, serial):
    """고급 모드를 켠다. 장치가 리셋되므로 다시 잡아 돌려준다.

    Returns:
        (device, 안내 문구). 못 켜면 (None, 사유).
    """
    advanced = _advanced_mode(device)
    if advanced is None:
        return None, "이 카메라는 고급 모드를 지원하지 않습니다"
    if advanced.is_enabled():
        return device, ""

    try:
        advanced.toggle_advanced_mode(True)
    except Exception as exc:
        return None, f"고급 모드를 켜지 못했습니다: {exc}"

    # 리셋된 장치가 USB 에 돌아올 때까지 기다린다.
    deadline = time.monotonic() + ADVANCED_MODE_RESET_TIMEOUT_S
    while time.monotonic() < deadline:
        time.sleep(ADVANCED_MODE_POLL_S)
        found = _find_device(serial)
        if found is None:
            continue
        again = _advanced_mode(found)
        if again is not None and again.is_enabled():
            return found, "고급 모드를 켜느라 카메라를 다시 시작했습니다"
    return None, "고급 모드를 켠 뒤 카메라가 돌아오지 않았습니다"


def apply_to_device(serial, name):
    """고른 프리셋을 일련번호로 찾은 장치에 싣는다.

    pipeline.start() **전에** 부른다 (모듈 상단 '뎁스 단위 주의' 참고).

    실패해도 예외를 올리지 않는다. 프리셋을 못 실었다고 카메라 연결까지
    막으면 현장에서 아무것도 못 하게 된다. 대신 사유를 문구로 돌려주어
    상태줄에 남긴다.

    Returns:
        (적용됐는지, 상태 문구)
    """
    if rs is None or not name:
        return False, ""   # 안 고른 것은 정상이므로 조용히 넘어간다
    text = preset_text(name)
    if text is None:
        return False, f"프리셋 '{name}' 을 읽지 못했습니다"

    device = _find_device(serial)
    if device is None:
        return False, "프리셋: 카메라를 찾지 못했습니다"

    device, note = _enable_advanced_mode(device, serial)
    if device is None:
        return False, f"프리셋: {note}"

    advanced = _advanced_mode(device)
    if advanced is None:
        return False, "프리셋: 고급 모드를 열지 못했습니다"

    # 뷰어가 저장한 형식(parameters 로 감싼 형태)을 그대로 넣어 본다. 예전
    # 형식이거나 감싸는 방식이 다른 파일이면 parameters 만 꺼내 다시 시도한다.
    errors = []
    for candidate in _load_candidates(text):
        try:
            advanced.load_json(candidate)
        except Exception as exc:
            errors.append(str(exc))
            continue
        applied = f"프리셋 '{name}' 적용됨"
        return True, f"{applied} ({note})" if note else applied
    reason = errors[-1] if errors else "알 수 없음"
    return False, f"프리셋 '{name}' 을 싣지 못했습니다: {reason}"


def _load_candidates(text):
    """load_json 에 넣어 볼 문자열들. 파일 원본 -> parameters 만."""
    yield text
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return
    if isinstance(data, dict) and isinstance(data.get("parameters"), dict):
        yield json.dumps(data["parameters"])
