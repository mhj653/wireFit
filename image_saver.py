"""오퍼레이터 모드 촬영 이미지 저장.

촬영한 컬러/뎁스 이미지를 날짜별로 나눠 남긴다. 나중에 특정 날짜의 불량을
되짚을 때 폴더만 열면 되도록 날짜 → 종류 순으로 계층을 둔다.

    Logs/
      └ 260818/                        촬영 날짜 (YYMMDD)
          ├ color/
          │   ├ color_260818110613.png
          │   └ color_260818110615.png
          ├ depth/
              ├ depth_260818110613.png
              └ depth_260818110615.png
          └ depth_raw/
              ├ depth_raw_260818110613.png
              └ depth_raw_260818110615.png

파일 이름의 숫자는 촬영 시각(YYMMDDHHMMSS)이다. 같은 초에 두 번 찍으면
뒤에 _2, _3 을 붙여 덮어쓰지 않는다.

저장 위치는 기본값이 프로그램 폴더의 Logs 이고, 오퍼레이터 화면에서
다른 곳으로 바꿀 수 있다.
"""

import json
import os
from datetime import datetime

import cv2

from app_paths import user_data_path

LOG_DIR_NAME = "Logs"
COLOR_DIR_NAME = "color"
DEPTH_DIR_NAME = "depth"
RAW_DEPTH_DIR_NAME = "depth_raw"
DEPTH_SCALE_FILE_NAME = "depth_scale.json"

# 폴더 이름 = 날짜, 파일 이름 = 날짜+시각
DATE_FORMAT = "%y%m%d"
STAMP_FORMAT = "%y%m%d%H%M%S"


def default_log_root():
    """저장 경로 기본값 (프로그램 폴더의 Logs).

    exe 로 묶었을 때도 exe 옆 Logs 를 가리킨다(app_paths 참고).
    """
    return user_data_path(LOG_DIR_NAME)


def date_dir(root, when=None):
    """날짜 폴더 경로 (<root>/260818). 만들지는 않는다."""
    when = when or datetime.now()
    return os.path.join(root, when.strftime(DATE_FORMAT))


def _write_image(path, bgr_image):
    """한글 경로에서도 되도록 인코딩 후 바이트로 쓴다(cv2.imwrite는 못 씀)."""
    ok, buffer = cv2.imencode(".png", bgr_image)
    if not ok:
        raise OSError("이미지를 PNG로 변환하지 못했습니다.")
    with open(path, "wb") as fp:
        fp.write(buffer.tobytes())


def _unique_stamp(base, targets, stamp):
    """모든 이미지 종류가 동일한 촬영 키를 쓰도록 중복 suffix를 정한다."""
    candidate = stamp
    suffix = 2
    while any(
        os.path.exists(os.path.join(base, dir_name, f"{prefix}_{candidate}.png"))
        for dir_name, prefix, _image in targets
    ):
        candidate = f"{stamp}_{suffix}"
        suffix += 1
    return candidate


def save_frames(
    root,
    color_image=None,
    depth_image=None,
    raw_depth_image=None,
    when=None,
    depth_scale_mm=None,
):
    """컬러/뎁스 이미지를 날짜 폴더에 저장한다.

    둘 중 하나만 넘겨도 된다(그 종류의 폴더만 만든다).

    Args:
        root: 저장 루트 (보통 default_log_root() 또는 사용자가 고른 폴더)
        color_image: 컬러 BGR 이미지 또는 None
        depth_image: 뎁스를 색칠한 BGR 이미지 또는 None.
            depth_view.colorize_depth() 로 만든 것을 넘긴다 — 화면에서 본 뎁스
            그림과 로그에 쌓인 그림이 같아야 같은 장면인지 알아볼 수 있다.
        raw_depth_image: 카메라에서 받은 원본 uint16 뎁스 배열 또는 None.
            가상 카메라와 Hole Detection에서 거리 데이터로 다시 사용한다.
        depth_scale_mm: 원본 뎁스 값 1당 mm. raw depth와 함께 날짜 폴더에 남긴다.
        when: 촬영 시각 (기본: 지금)

    Returns:
        (성공 여부, 저장한 종류별 경로 dict 또는 실패 사유)
        저장하지 않은 종류는 키가 없다.
    """
    when = when or datetime.now()
    stamp = when.strftime(STAMP_FORMAT)
    base = date_dir(root, when)

    targets = []
    if color_image is not None:
        targets.append((COLOR_DIR_NAME, "color", color_image))
    if depth_image is not None:
        targets.append((DEPTH_DIR_NAME, "depth", depth_image))
    if raw_depth_image is not None:
        targets.append((RAW_DEPTH_DIR_NAME, "depth_raw", raw_depth_image))

    if not targets:
        return False, "저장할 이미지가 없습니다."

    saved = {}
    try:
        unique_stamp = _unique_stamp(base, targets, stamp)
        for dir_name, prefix, image in targets:
            folder = os.path.join(base, dir_name)
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{prefix}_{unique_stamp}.png")
            _write_image(path, image)
            saved[prefix] = path
        if raw_depth_image is not None and depth_scale_mm is not None:
            scale_path = os.path.join(base, DEPTH_SCALE_FILE_NAME)
            with open(scale_path, "w", encoding="utf-8") as fp:
                json.dump({"depth_scale_mm": float(depth_scale_mm)}, fp, indent=2)
        return True, saved
    except OSError as exc:
        return False, str(exc)


def display_path(path):
    """상태문구에 띄울 짧은 경로 (Logs/260818/color/color_....png)."""
    parts = os.path.normpath(path).split(os.sep)
    # 뒤에서 4칸(Logs/날짜/종류/파일)만 보여준다.
    return "/".join(parts[-4:]) if len(parts) >= 4 else path
