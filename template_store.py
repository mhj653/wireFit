"""홀 검출·자세 계산에 쓰는 티칭 자산 저장소.

한 폴더에 세 종류가 섞여 있고, 목록 함수가 info.json 의 키로 서로를 가른다.
종류를 가르는 이유는 예전에 섞여서 홀 기준이 영역으로 바뀌는 사고가 있었기
때문이다. 콤보에 서로 안 보이게 해 둔다.

    템플릿   template.png      사람이 드래그해 만든 크롭. 템플릿 매칭 전용.
    홀       scene.png + hole_box / hole_area
    매칭점   scene.png + normal_boxes / normal_areas

**홀과 매칭점은 같은 구조다** (260908 통일). 둘 다 '티칭 프레임 + 그 안의 네모'
이고, SAM2 비디오 추적이 그 네모를 프롬프트로 삼아 다시 찾는다. 예전에는 홀만
크롭(template.png)을 함께 남겼는데, 그것은 cv2.matchTemplate 으로 대략 위치를
잡던 2단계 방식이 쓰던 것이다. 그 방식은 배율도 회전도 보지 않아 카메라 거리가
20%만 바뀌어도 판재를 물고 그럴듯한 오답을 냈고(260908 실측: 반지름 20px 자리에
257px), 못 찾는 것보다 나빠서 걷어냈다.
"""

import json
import os
from datetime import datetime

import cv2
import numpy as np

from app_paths import user_data_path

TEMPLATES_DIR_NAME = "Templates"
IMAGE_FILE_NAME = "template.png"
# 티칭할 때 본 화면 전체. 크롭(template.png)과 달리 잘라내지 않는다.
# SAM2 비디오 추적은 '티칭 프레임 -> 현재 프레임' 2장을 한 비디오로 묶어 돌리므로
# 크롭이 아니라 이 원본과, 그때 사람이 찍은 점이 그대로 있어야 한다.
SCENE_FILE_NAME = "scene.png"
INFO_FILE_NAME = "info.json"
INVALID_NAME_CHARS = '\\/:*?"<>|'
MAX_NAME_LENGTH = 40


def templates_dir():
    return user_data_path(TEMPLATES_DIR_NAME)


def validate_template_name(name):
    cleaned = (name or "").strip()
    if not cleaned:
        return False, "이름을 입력하세요."
    if len(cleaned) > MAX_NAME_LENGTH:
        return False, f"이름은 {MAX_NAME_LENGTH}자 이내여야 합니다."
    used = sorted({char for char in cleaned if char in INVALID_NAME_CHARS})
    if used:
        return False, "파일 이름에 쓸 수 없는 문자입니다: " + " ".join(used)
    if cleaned.endswith("."):
        return False, "이름 끝에 마침표는 쓸 수 없습니다."
    # 폴더로 직접 본다. 홀 기준 모양과 법선 매칭점이 같은 창고를 쓰지만 목록
    # 함수가 서로를 걸러내므로, 목록으로만 검사하면 다른 종류와 이름이 부딪칠 때
    # 나중에 makedirs 가 OSError 로 터진다.
    try:
        taken = {item.lower() for item in os.listdir(templates_dir())}
    except OSError:
        taken = set()
    if cleaned.lower() in taken:
        return False, f"'{cleaned}' 이름이 이미 있습니다."
    return True, cleaned


def list_templates():
    try:
        names = os.listdir(templates_dir())
    except OSError:
        return []
    root = templates_dir()
    return sorted(
        name for name in names
        if os.path.isdir(os.path.join(root, name))
        and os.path.isfile(os.path.join(root, name, IMAGE_FILE_NAME))
    )


def save_template(name, bgr_image):
    """사람이 드래그해 만든 크롭을 저장한다 (템플릿 매칭 전용).

    이제 크롭만 남긴다. 홀 티칭은 save_hole() 로 갈라졌다 — 둘을 한 함수가
    맡던 시절에는 '크롭도 있고 씬도 있는' 자산이 나와, 검출 방식을 바꾸면
    무엇이 쓰이는지 파일만 보고 알 수 없었다.
    """
    if not isinstance(bgr_image, np.ndarray) or bgr_image.size == 0:
        return False, "저장할 템플릿 이미지가 없습니다."
    height, width = bgr_image.shape[:2]
    return _write_asset(
        name,
        {"name": name, "created": _now(), "width": width, "height": height},
        images={IMAGE_FILE_NAME: bgr_image},
    )


def save_hole(name, scene_bgr, box, area=None):
    """SAM2 로 티칭한 홀을 저장한다 (매칭점과 같은 구조).

    Args:
        scene_bgr: 티칭할 때 본 화면 전체
        box: SAM2 가 잡은 홀 경계의 외접 사각형 (left, top, right, bottom).
            사람이 클릭한 점이 아니라 이것을 남긴다 — 클릭은 홀 안쪽 아무 데나
            찍히지만 네모는 실제 경계에서 나온 값이라 재검출이 더 정확하다.
        area: 티칭 때 **비디오 추적으로 잰** 마스크 넓이. 재검출이 '제대로
            잡혔나'를 이 값에 견준다(매칭점의 normal_areas 와 같은 규약).

    template.png 를 만들지 않는다 — 그래야 list_templates() 가 이것을 드래그
    템플릿으로 오인하지 않는다.
    """
    if not isinstance(scene_bgr, np.ndarray) or scene_bgr.size == 0:
        return False, "저장할 티칭 프레임이 없습니다."
    box = tuple(int(value) for value in (box or ()))
    if len(box) != 4:
        return False, "홀 네모는 (left, top, right, bottom) 네 값이어야 합니다."
    height, width = scene_bgr.shape[:2]
    return _write_asset(
        name,
        {
            "name": name,
            "created": _now(),
            "width": width,
            "height": height,
            "hole_box": list(box),
            "hole_area": None if area is None else int(area),
        },
        images={SCENE_FILE_NAME: scene_bgr},
    )


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _write_asset(name, info, images):
    """자산 폴더 하나를 통째로 쓴다. 실패하면 만든 것을 되돌린다.

    세 종류가 같은 뒷정리를 하므로 한 곳에 둔다. 반쯤 쓰다 만 폴더가 남으면
    목록에는 뜨는데 읽으면 None 이라, 원인을 짚기 어려운 상태가 된다.
    """
    folder = os.path.join(templates_dir(), name)
    written = []
    try:
        os.makedirs(folder, exist_ok=False)
        for filename, image in images.items():
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise OSError(f"{filename} 을 PNG로 변환하지 못했습니다.")
            path = os.path.join(folder, filename)
            with open(path, "wb") as fp:
                fp.write(encoded.tobytes())
            written.append(path)
        path = os.path.join(folder, INFO_FILE_NAME)
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(info, fp, ensure_ascii=False, indent=2)
        written.append(path)
        return True, folder
    except OSError as exc:
        for path in written:
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            os.rmdir(folder)
        except OSError:
            pass
        return False, str(exc)


def load_template(name):
    if not name or os.path.basename(name) != name:
        return None
    path = os.path.join(templates_dir(), name, IMAGE_FILE_NAME)
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image if image is not None and image.size else None


def _read_info(name):
    """info.json 을 dict 로. 없거나 깨졌으면 None."""
    if not name or os.path.basename(name) != name:
        return None
    try:
        with open(
            os.path.join(templates_dir(), name, INFO_FILE_NAME), encoding="utf-8"
        ) as fp:
            info = json.load(fp)
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) else None


def _list_by_key(key):
    """info.json 에 이 키가 든 자산 이름들. 종류를 가르는 유일한 기준이다."""
    try:
        names = os.listdir(templates_dir())
    except OSError:
        return []
    found = []
    for name in names:
        info = _read_info(name)
        if info and info.get(key) is not None:
            found.append(name)
    return sorted(found)


def list_holes():
    """SAM2 로 티칭한 홀 이름들 (info.json 에 hole_box 가 있는 것).

    260908 이전에 티칭한 홀은 hole_box 대신 prompt(점)만 갖고 있어서 여기
    안 잡힌다. 그건 의도한 것이다 — 그때 자산은 template.png 로 대략 위치를
    잡던 2단계 방식을 전제로 만들어졌고, 그 방식을 걷어냈으므로 네모를 다시
    잡아야 한다. 조용히 통과시키면 프롬프트 없이 추적을 돌리게 된다.
    """
    return _list_by_key("hole_box")


def load_hole(name):
    """티칭한 홀. → (티칭 프레임 BGR, 네모, 기준 넓이 또는 None) 또는 None.

    매칭점(load_normal_points)과 같은 모양으로 돌려준다. 재검출 경로가 둘을
    같은 방식으로 다루기 때문이다.
    """
    info = _read_info(name)
    if info is None:
        return None
    box = info.get("hole_box")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        parsed = tuple(int(value) for value in box)
    except (TypeError, ValueError):
        return None
    scene = _load_scene_image(name)
    if scene is None:
        return None
    area = info.get("hole_area")
    try:
        area = None if area is None else int(area)
    except (TypeError, ValueError):
        area = None
    return scene, parsed, area


def _load_scene_image(name):
    """자산 폴더의 scene.png. 없거나 깨졌으면 None."""
    try:
        data = np.fromfile(
            os.path.join(templates_dir(), name, SCENE_FILE_NAME), dtype=np.uint8
        )
    except OSError:
        return None
    scene = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return scene if scene is not None and scene.size else None


def list_normal_points():
    """법선 매칭점으로 저장된 이름들.

    홀·템플릿과 같은 창고를 쓰지만 목록은 갈라 둔다. info.json 의 normal_boxes
    로 골라내고, 홀은 hole_box, 드래그 템플릿은 template.png 로 가른다.

    한 창고에 섞어 두고 목록만 가른 이유는, 예전에 종류가 뒤섞여 홀 기준이
    영역으로 바뀌는 사고가 있었기 때문이다. 목록을 가르면 콤보에 서로 안 보인다.
    """
    return _list_by_key("normal_boxes")


def save_normal_points(name, scene_bgr, boxes, areas=None):
    """법선을 세울 매칭점 세 곳을 저장한다.

    template.png 를 만들지 않는다 — 그래야 list_templates() 가 이것을 홀
    기준 모양이나 템플릿으로 오인하지 않는다.

    Args:
        name: 저장할 이름
        scene_bgr: 티칭할 때 본 화면 전체
        boxes: [(left, top, right, bottom)] 세 개
        areas: 티칭 때 **비디오 추적으로 잰** 넓이 세 개. 재검출이 '제대로
            잡혔나'를 이 값에 견준다(sam2_video_tracker.measure_normal_point_areas).
            네모 크기로 대신할 수 없다 — 네모는 마스크의 외접 사각형일 뿐이고,
            실제 마스크는 그보다 작거나 조각나 있을 수 있다.
    """
    if not isinstance(scene_bgr, np.ndarray) or scene_bgr.size == 0:
        return False, "저장할 티칭 프레임이 없습니다."
    boxes = [tuple(int(value) for value in box) for box in (boxes or [])]
    if len(boxes) != 3:
        return False, f"매칭점은 정확히 3개여야 합니다 (지금 {len(boxes)}개)."
    if areas is not None and len(areas) != len(boxes):
        return False, "기준 넓이 개수가 매칭점 개수와 다릅니다."
    height, width = scene_bgr.shape[:2]
    return _write_asset(
        name,
        {
            "name": name,
            "created": _now(),
            "width": width,
            "height": height,
            "normal_boxes": [list(box) for box in boxes],
            # 없으면 재검출이 절대 하한만 쓴다(옛 자산과 같은 취급).
            "normal_areas": (
                None if areas is None else [int(area) for area in areas]
            ),
        },
        images={SCENE_FILE_NAME: scene_bgr},
    )


def load_normal_points(name):
    """법선 매칭점. → (티칭 프레임 BGR, [네모 세 개], [기준 넓이 세 개] 또는 None).

    셋째 항목이 None 이면 '기준 넓이를 모르는 자산'이다. 이 키가 생기기 전에
    저장한 것들이 여기 해당하고, 그때 재검출은 절대 하한만 쓴다.
    """
    info = _read_info(name)
    if info is None:
        return None
    boxes = info.get("normal_boxes")
    if not isinstance(boxes, list) or len(boxes) != 3:
        return None
    try:
        parsed = [tuple(int(value) for value in box) for box in boxes]
    except (TypeError, ValueError):
        return None
    if any(len(box) != 4 for box in parsed):
        return None

    # 손으로 고쳤을 수도 있으므로 상해 있으면 '모르는 것'으로 물러선다.
    # 자리가 밀린 기준으로 재면 엉뚱한 점이 탈락하는데, 그게 더 나쁘다.
    areas = info.get("normal_areas")
    if isinstance(areas, list) and len(areas) == len(parsed):
        try:
            areas = [int(area) for area in areas]
        except (TypeError, ValueError):
            areas = None
    else:
        areas = None
    scene = _load_scene_image(name)
    if scene is None:
        return None
    return scene, parsed, areas


def delete_template(name):
    if not name or os.path.basename(name) != name:
        return False, "템플릿 이름이 올바르지 않습니다."
    folder = os.path.join(templates_dir(), name)
    try:
        for filename in (IMAGE_FILE_NAME, SCENE_FILE_NAME, INFO_FILE_NAME):
            path = os.path.join(folder, filename)
            if os.path.isfile(path):
                os.remove(path)
        os.rmdir(folder)
        return True, None
    except OSError as exc:
        return False, str(exc)
