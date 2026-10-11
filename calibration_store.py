"""캘리브레이션 결과 저장.

한 번의 캘리브레이션을 폴더 하나에 통째로 남긴다. 나중에 결과를 의심할 때
어떤 사진과 어떤 포즈로 나온 값인지 되짚을 수 있어야 하기 때문이다.

    Fastening_automation/
      └ Calibration/
          └ EIH_20260810_143022/
              ├ image_01.png ... image_08.png   촬영 원본
              ├ poses.txt                       순서별 플랜지 포즈
              └ result.json                     최종 캘리브레이션 값

저장 위치는 실행 위치와 무관하게 항상 이 파일이 있는 폴더 기준이다.
"""

import json
import os
from datetime import datetime

import cv2
import numpy as np

from app_paths import user_data_path

CALIBRATION_DIR_NAME = "Calibration"
SESSION_PREFIX = "EIH"
RESULT_FILE_NAME = "result.json"
POSES_FILE_NAME = "poses.txt"

# calibration_warnings() 가 쓰는 판단 기준
DEFAULT_INTRINSICS_SOURCE = "default_realsense_d405"
MIN_TRUSTWORTHY_SAMPLES = 8


def calibration_dir():
    return user_data_path(CALIBRATION_DIR_NAME)


def list_sessions():
    """저장된 캘리브레이션 세션 폴더 이름을 최신순으로 돌려준다.

    폴더 이름이 EIH_날짜_시각 이라 이름순 역순이 곧 최신순이다.
    """
    try:
        names = os.listdir(calibration_dir())
    except OSError:
        return []
    root = calibration_dir()
    return sorted(
        (
            name for name in names
            if name.startswith(SESSION_PREFIX)
            and os.path.isdir(os.path.join(root, name))
        ),
        reverse=True,
    )


def load_cam2gripper(session_name):
    """저장된 EIH 세션의 camera -> flange 4x4 행렬을 읽는다."""
    if not session_name or os.path.basename(session_name) != session_name:
        raise ValueError("캘리브레이션 세션 이름이 올바르지 않습니다.")

    path = os.path.join(calibration_dir(), session_name, RESULT_FILE_NAME)
    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"캘리브레이션 결과를 읽지 못했습니다: {exc}") from exc

    value = payload.get("cam2gripper")
    if isinstance(value, dict):
        value = value.get("matrix_4x4")
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("캘리브레이션의 cam2gripper 행렬이 올바르지 않습니다.")
    return matrix


def calibration_warnings(session_name):
    """이 캘리브레이션을 믿을 수 없다고 볼 근거들. 믿을 만하면 빈 목록.

    배포본에 들어 있는 기본 캘리브레이션은 실측이 아니라 자리만 채운 값이다
    (cam2gripper 회전 = 단위행렬, 병진 = (0,0,150)). 그대로 두고 쓰면 좌표가
    조용히 틀린 채 로봇으로 나가고, 카메라 장착 각도만큼 접근 방향이 어긋난다.
    사람이 파일을 열어 보지 않으면 알 수 없으므로 프로그램이 짚어 준다.

    Returns:
        경고 문구 목록 (사람이 읽을 한글 문장). 읽지 못하면 그 사유 한 줄.
    """
    path = os.path.join(calibration_dir(), os.path.basename(session_name or ""))
    path = os.path.join(path, RESULT_FILE_NAME)
    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except (OSError, json.JSONDecodeError) as exc:
        return [f"캘리브레이션 결과를 읽지 못했습니다: {exc}"]

    warnings = []
    # 기본값 파일에 일부러 남겨 둔 표식. 있으면 그것만으로 결론이 난다.
    if payload.get("_note"):
        warnings.append(f"실측이 아닌 기본값입니다 — {payload['_note']}")

    quality = payload.get("quality")
    if isinstance(quality, dict):
        error = quality.get("reprojection_error_px")
        # 실제 캘리브레이션에서 재투영 오차가 딱 0이 되는 일은 없다.
        if isinstance(error, (int, float)) and error == 0.0:
            warnings.append("재투영 오차가 0입니다 — 실제로 계산된 값이 아닙니다.")

    if payload.get("intrinsics_source") == DEFAULT_INTRINSICS_SOURCE:
        warnings.append(
            "카메라 내부 파라미터가 기종 기본값입니다 "
            "(3D 좌표에는 장치 공장값을 직접 읽어 쓰므로 영향은 없습니다)."
        )

    z_reference = payload.get("z_reference")
    if (
        payload.get("robot_type") == "4axis"
        and isinstance(z_reference, dict)
        and z_reference.get("source") == "unset"
    ):
        warnings.append(
            "4축 캘리브레이션의 카메라 Z 오프셋이 보정되지 않았습니다 "
            "(보드 높이 없이 계산, Z = 0) — 높이 좌표가 그만큼 어긋납니다."
        )

    count = payload.get("sample_count")
    if isinstance(count, int) and count < MIN_TRUSTWORTHY_SAMPLES:
        warnings.append(
            f"촬영 표본이 {count}장뿐입니다 "
            f"({MIN_TRUSTWORTHY_SAMPLES}장 이상을 권합니다)."
        )
    return warnings


def _unique_session_dir(stamp):
    """같은 초에 두 번 저장해도 폴더가 겹치지 않게 한다."""
    base = os.path.join(calibration_dir(), f"{SESSION_PREFIX}_{stamp}")
    path = base
    suffix = 2
    while os.path.exists(path):
        path = f"{base}_{suffix}"
        suffix += 1
    return path


def _write_image(path, bgr_image):
    """한글 경로에서도 되도록 인코딩 후 바이트로 쓴다(cv2.imwrite는 못 씀)."""
    ok, buffer = cv2.imencode(".png", bgr_image)
    if not ok:
        raise OSError("이미지를 PNG로 변환하지 못했습니다.")
    with open(path, "wb") as fp:
        fp.write(buffer.tobytes())


def _poses_text(samples):
    lines = [
        "# EIH 캘리브레이션 플랜지 포즈",
        "# 위치 단위 mm, 각도 단위 deg, 회전 규약 ZYX 오일러각 (R = Rz·Ry·Rx)",
        "# no\timage\tx\ty\tz\trx\try\trz",
    ]
    for index, sample in enumerate(samples, start=1):
        x, y, z, rx, ry, rz = sample["pose"]
        lines.append(
            f"{index}\t{sample['image_name']}\t"
            f"{x:.3f}\t{y:.3f}\t{z:.3f}\t{rx:.3f}\t{ry:.3f}\t{rz:.3f}"
        )
    return "\n".join(lines) + "\n"


def _jsonable(value):
    """numpy 배열/스칼라를 json 이 받을 수 있는 형태로 바꾼다."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def save_session(samples, result, board):
    """이미지 + 포즈 + 결과를 새 폴더에 저장한다.

    Args:
        samples: [{"image": BGR 이미지, "pose": (x,y,z,rx,ry,rz)}, ...]
        result: hand_eye.calibrate_eye_in_hand 의 반환값
        board: {"cols":, "rows":, "square_size_mm":}

    Returns:
        (성공 여부, 폴더 경로 또는 실패 사유)
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        session_dir = _unique_session_dir(stamp)
        os.makedirs(session_dir, exist_ok=False)

        # 이미지를 먼저 쓰고, 그 파일명을 poses.txt 와 result.json 에서 참조한다.
        stored = []
        for index, sample in enumerate(samples, start=1):
            name = f"image_{index:02d}.png"
            _write_image(os.path.join(session_dir, name), sample["image"])
            stored.append({"image_name": name, "pose": sample["pose"]})

        with open(os.path.join(session_dir, POSES_FILE_NAME), "w",
                  encoding="utf-8") as fp:
            fp.write(_poses_text(stored))

        x, y, z, rx, ry, rz = result["pose"]
        payload = {
            "type": "eye_in_hand",
            "created": datetime.now().isoformat(timespec="seconds"),
            # "6axis" | "4axis". 이 키가 없는 이전 파일은 6축으로 본다.
            "robot_type": result.get("robot_type", "6axis"),
            # 4축 전용: 카메라 Z 오프셋을 어떻게 정했는지 (6축은 null)
            "z_reference": result.get("z_reference"),
            "sample_count": result["sample_count"],
            "units": {"translation": "mm", "rotation": "deg"},
            "rotation_convention": "euler_zyx (R = Rz*Ry*Rx)",
            "checkerboard": board,
            "camera_matrix": result["camera_matrix"],
            "dist_coeffs": result["dist_coeffs"],
            "intrinsics_source": result["intrinsics_source"],
            "intrinsics_rms_px": result["intrinsics_rms_px"],
            # 카메라가 플랜지에 대해 어디에 붙어 있는지 (이게 캘리브레이션 결과다)
            "cam2gripper": {
                "matrix_4x4": result["cam2gripper"],
                "translation_mm": [x, y, z],
                "rpy_deg": [rx, ry, rz],
            },
            # 결과를 얼마나 믿을 수 있는지 (작을수록 좋다)
            "quality": {
                "reprojection_error_px": result["reprojection_error_px"],
                "reprojection_error_max_px": result["reprojection_error_max_px"],
                "target_position_spread_mm": result["consistency"],
            },
            "samples": stored,
        }
        with open(os.path.join(session_dir, RESULT_FILE_NAME), "w",
                  encoding="utf-8") as fp:
            json.dump(_jsonable(payload), fp, ensure_ascii=False, indent=2)

        return True, session_dir
    except OSError as exc:
        return False, str(exc)


def display_path(path):
    """상태문구에 띄울 짧은 경로 (Calibration/EIH_...)."""
    return os.path.join(CALIBRATION_DIR_NAME, os.path.basename(path))
