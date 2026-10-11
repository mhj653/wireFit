"""RealSense 뎁스 후처리 필터 — 모든 페이지가 공유하는 설정.

RealSense Viewer 의 Post-Processing 패널과 같은 필터를 같은 순서로 건다.
프리셋(realsense_preset)은 '센서가 뎁스를 만드는 방식'을 정하고, 여기는 '만들어진
뎁스를 다듬는 방식'을 정한다. 프리셋 파일에는 후처리가 들어 있지 않아서 따로 둔다.

**취득 단계에서 건다.** _RealSenseSource 가 프레임을 받을 때마다 이 체인을 통과시키므로,
grab_frames() 를 쓰는 모든 페이지(Hole Detection, EIH Calibration, 오퍼레이터)가
자동으로 같은 뎁스를 본다. 소비하는 쪽에서 각자 거는 방식(예전 median filter)은
페이지마다 다른 뎁스를 볼 여지가 있어 버렸다.

순서가 중요하다
    Viewer 와 같은 정규 순서를 따른다. 특히 spatial/temporal 은 disparity 공간에서
    걸어야 원근에 따른 왜곡이 없다 — 그래서 앞뒤로 depth<->disparity 변환이 붙는다.
    이 순서는 고정이고 UI 에서 바꾸지 않는다(순서를 틀리면 조용히 나빠진다).

프레임셋째로 거는 이유
    필터는 frameset 을 받아 frameset 을 돌려준다. 컬러까지 함께 들고 가야 뒤에서
    align 을 걸 수 있다. depth 프레임만 꺼내 거르면 align 할 짝이 사라진다.

주의 — decimation 은 뎁스 해상도를 줄인다
    2 로 두면 1280x720 뎁스가 640x360 이 된다. 홀 주변 고리의 표본 수가 1/4 로
    줄어 평면 맞춤이 나빠지므로 기본은 꺼 둔다. hole_pose.depth_frame_intrinsics()
    가 해상도 차이를 흡수하긴 하지만, 정밀도까지 되돌려 주지는 못한다.
"""

import json
import os
from collections import namedtuple

from app_paths import user_data_path

try:  # pyrealsense2 가 없어도 가상 카메라 모드로 UI 는 그대로 동작한다.
    import pyrealsense2 as rs
except ImportError:  # pragma: no cover - 런타임 환경 의존
    rs = None

SETTINGS_PATH = user_data_path("Settings", "DepthPostProcess.json")

# 값 하나. rs 옵션에 그대로 실린다.
#   scale: UI 값 -> rs 옵션 값 배율. 거리처럼 단위가 다른 것만 1이 아니다.
#          (rs 는 m, 이 프로그램은 mm 를 쓴다)
#   choices: 값이 몇 가지로 정해져 있으면 (표시이름, 값) 목록. 있으면 콤보로 그린다.
Option = namedtuple(
    "Option",
    "key label option minimum maximum step decimals default suffix scale choices",
)
Option.__new__.__defaults__ = ("", 1.0, None)

# 필터 하나. order 는 정규 순서이고 UI 에도 이 차례로 놓인다.
Filter = namedtuple("Filter", "key label factory default_enabled options note")
Filter.__new__.__defaults__ = ((), "")


def _opt(key, label, option, low, high, step, decimals, default, **kw):
    return Option(key, label, option, low, high, step, decimals, default, **kw)


def _filters():
    """필터 목록. pyrealsense2 가 없으면 빈 목록(가상 카메라 모드)."""
    if rs is None:
        return ()
    o = rs.option
    return (
        Filter(
            "decimation",
            "데시메이션",
            rs.decimation_filter,
            False,   # 뎁스 해상도가 줄어 고리 표본이 급감한다 (모듈 상단 주석)
            (
                _opt("magnitude", "배율", o.filter_magnitude, 1, 8, 1, 0, 2),
            ),
            "뎁스 해상도를 1/N 로 줄인다. 표본이 줄어 평면 맞춤이 나빠진다.",
        ),
        Filter(
            "hdr_merge",
            "HDR 병합",
            rs.hdr_merge,
            False,
            (),
            "노출이 다른 프레임을 합친다. 카메라가 HDR 모드일 때만 뜻이 있다.",
        ),
        Filter(
            "threshold",
            "거리 threshold",
            rs.threshold_filter,
            True,
            (
                _opt(
                    "min_mm", "최소 거리", o.min_distance,
                    0, 16000, 10, 0, 50, suffix=" mm", scale=0.001,
                ),
                _opt(
                    "max_mm", "최대 거리", o.max_distance,
                    0, 16000, 10, 0, 600, suffix=" mm", scale=0.001,
                ),
            ),
            "이 범위 밖의 뎁스를 버린다. 배경을 취득 단계에서 지우는 가장 싼 방법.",
        ),
        Filter(
            "spatial",
            "공간 필터",
            rs.spatial_filter,
            True,
            (
                _opt("magnitude", "강도", o.filter_magnitude, 1, 5, 1, 0, 2),
                _opt("alpha", "smooth alpha", o.filter_smooth_alpha, 0.25, 1.0, 0.01, 2, 0.5),
                _opt("delta", "smooth delta", o.filter_smooth_delta, 1, 50, 1, 0, 20),
                _opt(
                    "holes", "구멍 메우기", o.holes_fill, 0, 5, 1, 0, 0,
                    choices=(
                        ("사용 안 함", 0), ("2 px", 1), ("4 px", 2),
                        ("8 px", 3), ("16 px", 4), ("무제한", 5),
                    ),
                ),
            ),
            "가장자리를 지키며 매끈하게 한다. 구멍 메우기는 없는 값을 만들어 내므로 주의.",
        ),
        Filter(
            "temporal",
            "시간 필터",
            rs.temporal_filter,
            True,
            (
                _opt("alpha", "smooth alpha", o.filter_smooth_alpha, 0.0, 1.0, 0.01, 2, 0.4),
                _opt("delta", "smooth delta", o.filter_smooth_delta, 1, 100, 1, 0, 20),
                _opt(
                    "persistence", "지속 방식", o.holes_fill, 0, 8, 1, 0, 3,
                    choices=(
                        ("사용 안 함", 0), ("8/8 유효", 1), ("2/3 유효", 2),
                        ("2/4 유효", 3), ("2/8 유효", 4), ("1/2 유효", 5),
                        ("1/5 유효", 6), ("1/8 유효", 7), ("항상 유지", 8),
                    ),
                ),
            ),
            "이전 프레임과 섞어 흔들림을 줄인다. 라이브에서만 뜻이 있다"
            " (한 장만 찍으면 섞을 이전 프레임이 없다).",
        ),
        Filter(
            "hole_filling",
            "구멍 채우기",
            rs.hole_filling_filter,
            False,   # 없는 값을 만들어 낸다. 법선 계산이 오염되므로 기본은 끔
            (
                _opt(
                    "mode", "채우는 방식", o.holes_fill, 0, 2, 1, 0, 1,
                    choices=(
                        ("왼쪽 값", 0), ("가장 가까운 값", 1), ("가장 먼 값", 2),
                    ),
                ),
            ),
            "빈 픽셀을 이웃 값으로 채운다. 없던 표면을 만들어 내므로"
            " 법선/평면 계산에는 권하지 않는다.",
        ),
    )


FILTERS = _filters()
FILTER_BY_KEY = {item.key: item for item in FILTERS}


def default_settings():
    """모든 필터의 기본 상태 {키: {enabled, values{}}}."""
    return {
        item.key: {
            "enabled": item.default_enabled,
            "values": {opt.key: opt.default for opt in item.options},
        }
        for item in FILTERS
    }


def _sanitize(data):
    """읽어 들인 설정을 스펙에 맞게 다듬는다(손으로 고친 파일도 버티게)."""
    clean = default_settings()
    if not isinstance(data, dict):
        return clean
    for item in FILTERS:
        saved = data.get(item.key)
        if not isinstance(saved, dict):
            continue
        clean[item.key]["enabled"] = bool(saved.get("enabled", item.default_enabled))
        values = saved.get("values")
        if not isinstance(values, dict):
            continue
        for opt in item.options:
            try:
                number = float(values.get(opt.key, opt.default))
            except (TypeError, ValueError):
                number = opt.default
            clean[item.key]["values"][opt.key] = max(
                opt.minimum, min(opt.maximum, number)
            )
    return clean


def _load():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as fp:
            return _sanitize(json.load(fp))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return default_settings()


_settings = _load()
# 설정이 바뀔 때마다 오른다. 소스 스레드가 이 값만 보고 체인을 다시 만든다
# (설정 dict 을 스레드 사이로 넘기지 않으려는 것).
_revision = 0


def settings():
    """현재 설정의 사본."""
    return {
        key: {"enabled": value["enabled"], "values": dict(value["values"])}
        for key, value in _settings.items()
    }


def revision():
    """설정이 바뀐 횟수. 소스 스레드가 체인을 다시 만들 시점을 안다."""
    return _revision


def set_settings(data):
    """설정을 통째로 바꾸고 저장한다."""
    global _settings, _revision
    _settings = _sanitize(data)
    _revision += 1
    try:
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as fp:
            json.dump(_settings, fp, indent=2, ensure_ascii=False)
    except OSError:
        pass
    return settings()


def reset_to_defaults():
    return set_settings(default_settings())


def summary():
    """상태줄에 쓸 한 줄 요약."""
    if not FILTERS:
        return "후처리: 사용할 수 없음"
    on = [item.label for item in FILTERS if _settings[item.key]["enabled"]]
    return "후처리: " + (" · ".join(on) if on else "전부 끔")


# ======================================================
# 프레임 처리 (카메라 스레드에서만 부른다)
# ======================================================
class Chain:
    """설정대로 만들어 둔 필터 체인.

    필터 객체는 상태를 들고 있고(특히 temporal) 스레드 안전하지 않다. 그래서
    카메라 스레드가 자기 것을 하나 만들어 계속 쓰고, 설정이 바뀌면(revision 이
    오르면) 통째로 다시 만든다.
    """

    def __init__(self):
        self.revision = -1
        self._stages = []
        self.rebuild()

    def rebuild(self):
        """현재 설정으로 필터 객체를 새로 만든다."""
        self.revision = _revision
        self._stages = []
        if rs is None:
            return

        active = [item for item in FILTERS if _settings[item.key]["enabled"]]
        for item in active:
            try:
                instance = item.factory()
            except Exception:
                continue   # 이 기종이 못 쓰는 필터는 조용히 건너뛴다
            values = _settings[item.key]["values"]
            for opt in item.options:
                try:
                    instance.set_option(opt.option, float(values[opt.key]) * opt.scale)
                except Exception:
                    pass   # 범위를 벗어난 값 하나 때문에 체인을 통째로 버리지 않는다
            self._stages.append(instance)

        # spatial/temporal 은 disparity 공간에서 걸어야 원근 왜곡이 없다.
        # 둘 중 하나라도 켜져 있으면 앞뒤로 변환을 끼운다.
        needs_disparity = any(
            _settings[key]["enabled"] for key in ("spatial", "temporal")
        )
        if needs_disparity:
            first = next(
                (
                    index
                    for index, item in enumerate(active)
                    if item.key in ("spatial", "temporal")
                ),
                None,
            )
            last = next(
                (
                    index
                    for index in range(len(active) - 1, -1, -1)
                    if active[index].key in ("spatial", "temporal")
                ),
                None,
            )
            if first is not None and last is not None:
                self._stages.insert(first, rs.disparity_transform(True))
                self._stages.insert(last + 2, rs.disparity_transform(False))

    def process(self, frameset):
        """프레임셋에 체인을 건다. 실패하면 원본을 그대로 돌려준다."""
        if _revision != self.revision:
            self.rebuild()
        for stage in self._stages:
            try:
                processed = stage.process(frameset)
            except Exception:
                return frameset
            frameset = processed.as_frameset() or frameset
        return frameset
