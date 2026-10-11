"""포즈를 만드는 데 쓰는 값과 계산 — 페이지에 매이지 않은 부분.

Pose Setting 페이지(세팅 화면)와 오퍼레이터 모드가 **같은 답을 내야** 한다.
두 경로가 각자 계산하므로, 공유해야 하는 것을 여기 한 군데 모아 둔다.

  - 진입 각도 오프셋: 필드 정의, 프로파일에서 읽기
  - 법선 매칭점 세 곳: 프로파일에서 읽기, 이 촬영에서 다시 찾기, 평면 세우기
  - 매칭점을 화면에 그리기

페이지 모듈에 두면 안 되는 이유: Hole Detection 과 Pose Setting 이 서로를
import 하게 되어 순환이 생긴다. 이 모듈은 어느 페이지도 import 하지 않는다.
"""

import cv2
import numpy as np

from hole_pose import (
    NORMAL_MODE_FLANGE,
    NORMAL_MODE_POINTS,
    NORMAL_MODE_RING,
    RING_INNER_SCALE,
    RING_RADIUS_MM_DEFAULT,
    RING_RADIUS_MM_MAX,
    RING_RADIUS_MM_MIN,
    ring_outer_px,
)
from normal_plane import (
    NORMAL_POINT_SPREAD_CAUTION_MM,
    mask_to_depth_size,
    normal_point_spread_mm,
    plane_from_normal_points,
)
import sam2_video_tracker
from settings_store import load_profile
from template_store import load_normal_points

# ===== 진입 각도 오프셋 =====
# 비전이 재는 것은 '대상 전체 면의 법선'이고, 나사가 실제로 앉는 면과 다를 수
# 있다. 그 차이는 부품 형상이 정하는 상수이므로 체결 지점마다 한 번 티칭해 둔다.
APPROACH_OFFSET_FIELDS = [
    ("rx", "RX", -180.0, 180.0, 2, "°"),
    ("ry", "RY", -180.0, 180.0, 2, "°"),
    ("rz", "RZ", -180.0, 180.0, 2, "°"),
]

# 260910 에 '포즈 조정' 오프셋을 걷어냈다. 검출된 홀 자세를 눈으로 보며 직접
# 돌리는 값이었는데(화면 좌표축이 곧바로 따라 돌았다), 로봇으로 나가는 보정은
# 위의 진입 각도 오프셋 하나로 충분했고 둘이 겹치면 어느 쪽이 얼마를 돌렸는지
# 가릴 수 없었다. 옛 프로파일에 남아 있는 pose_adjust 키는 이제 읽지 않는다.

APPROACH_OFFSET_HINT = (
    "비전이 잰 법선에서 공구를 이만큼 더 기울여 로봇에 넘긴다.\n"
    "공구 자신의 축 둘레 회전이므로 오일러 덧셈이 아니다.\n"
    "\n"
    "티칭 방법:\n"
    "  1. 오프셋을 0으로 두고 촬영해 나온 좌표로 로봇을 보낸다.\n"
    "  2. 드라이버가 평평하게 앉을 때까지 조그한다.\n"
    "  3. 그 차이를 여기에 넣고 다시 확인한다.\n"
    "\n"
    "대상 전체 법선은 재현성이 0.11°라, 한 번 맞으면 계속 맞는다."
)

NORMAL_POINTS_HINT = (
    "법선(자세)을 세울 자리 세 곳을 네모로 드래그해 지정한다.\n"
    "\n"
    "**방향만** 이 세 점에서 받는다. 홀은 원래 자리에서 그대로 찾고\n"
    "높이(z)도 홀 주변 고리에서 잡는다 — 세 점 밖의 홀도 좌표가 나온다.\n"
    "\n"
    "왜 고리가 아닌가: 홀은 주변보다 깊어서 고리에 모따기와 단차가 섞이고,\n"
    "고리 폭이 좁아 법선의 지렛대도 짧다(260901 실측 흔들림 ±8도).\n"
    "\n"
    "왜 면 하나가 아니라 점 셋인가: 면을 통째로 잡아 최소제곱으로 맞추면,\n"
    "부품이 놓인 자리가 틀어질 때 면 안에서 표본이 쓸리는 만큼 법선이\n"
    "흔들린다. 같은 자리 세 곳을 각각 추적하면 부품이 움직여도 세 점이\n"
    "함께 따라가므로 평면이 부품에 붙는다.\n"
    "\n"
    "왜 넷 이상이 아닌가: 넷을 최소제곱으로 맞추면 '넷이 모두 같은 평면에\n"
    "있다'를 전제하게 된다. 한 점이 다른 단차에 올라가 있어도 잔차로 흩어\n"
    "놓고 그럴듯한 법선을 내놓아, 틀린 것을 알아챌 수 없다. 셋이면 평면이\n"
    "유일하게 정해지고 어긋남이 기울기로 곧장 드러난다.\n"
    "\n"
    "세 점은 한 화면에서 지정해야 한다(새로 캡쳐하면 초기화된다).\n"
    "거의 일직선이어서 평면이 정해지지 않는 배치만 거절한다.\n"
    "\n"
    "다만 넓게 벌릴수록 정확하다. 세 번째 점이 나머지 둘을 잇는 직선에서\n"
    "벗어난 거리가 법선의 지렛대이고, 뎁스 잡음이 그 길이로 나뉘어 각도\n"
    "오차가 된다 — 20mm 면 약 1.4도, 5mm 면 약 5.7도.\n"
    "\n"
    "재검출은 SAM2 비디오 추적이라 카메라 거리와 회전이 바뀌어도 따라간다.\n"
    "세 곳을 한 번에 추적하므로 CPU 에서도 촬영당 2초 남짓이다."
)

# 뎁스 추출 영역 경계를 그리는 값. 홀 원은 초록(0,255,0)이므로 확실히 다른
# 색을 쓴다 — 둘이 비슷하면 영역이 홀로 잡힌 것으로 오해하게 된다.
NORMAL_POINT_COLOR = (0, 200, 255)      # BGR = 주황. 찾은 매칭점.
# 못 찾은 자리. 티칭 때의 네모를 이 색으로 그려 '여기 있어야 하는데 없다'를
# 보여 준다. 아무것도 안 그리면 셋 중 무엇이 빠졌는지 알 수 없고, 법선이
# 조용히 홀 고리로 물러선 것도 눈에 띄지 않는다.
NORMAL_POINT_MISS_COLOR = (255, 0, 255)  # BGR = 자홍
EXTRACT_CONTOUR_WIDTH = 2
EXTRACT_FILL_ALPHA = 0.18               # 안쪽을 옅게만 덮는다(아래가 보여야 한다)

# ring 모드 영역을 그리는 색. 노랑은 홀 초록·중심 빨강·경계 청록과 안 겹친다.
# 매칭점 주황과는 가깝지만 두 모드가 동시에 보이는 일이 없다.
RING_REGION_COLOR = (0, 255, 255)       # BGR = 노랑
RING_REGION_WIDTH = 1

# ===== 자세를 어디서 받을지 (콤보에 그대로 들어간다) =====
# 새 방식을 넣으려면 여기 한 줄과 hole_pose 의 분기 하나만 더하면 된다.
NORMAL_MODE_CHOICES = [
    ("플랜지 고정 (x·y·z만)", NORMAL_MODE_FLANGE),
    ("홀 주변 영역 평면", NORMAL_MODE_RING),
    ("매칭점 3곳 평면", NORMAL_MODE_POINTS),
]

NORMAL_MODE_HINTS = {
    NORMAL_MODE_FLANGE: (
        "회전을 비전으로 재지 않는다. 로봇은 지금 플랜지 자세를 그대로 두고\n"
        "위치만 옮긴다 — 2D 비전과 같지만 z 를 비전에서 받는다.\n"
        "\n"
        "법선을 안 세우므로 가장 잘 버틴다. 면이 기울어 있으면 공구가\n"
        "비스듬히 앉으므로, 대상이 카메라와 마주 보는 자리에 쓴다."
    ),
    NORMAL_MODE_RING: (
        "홀을 둘러싼 영역의 평면 법선을 z축으로 쓴다.\n"
        "영역 크기를 mm 로 지정한다 — 부품 상의 실제 크기라 카메라 거리가\n"
        "바뀌어도 같은 자리를 본다. 화면의 노란 고리가 실제 계산 범위다.\n"
        "\n"
        "홀마다 그 자리의 면을 보므로 판재가 휘어 있어도 따라간다. 다만\n"
        "고리가 좁으면 법선의 지렛대가 짧아 뎁스 잡음이 각도로 증폭된다\n"
        "(260901 실측: 좁은 고리에서 촬영 간 ±8도). 넓게 잡을수록 안정적이다."
    ),
    NORMAL_MODE_POINTS: (
        "사람이 짚어 둔 매칭점 세 곳의 평면 법선을 쓴다.\n"
        "재현성이 가장 높다(260901 실측 0.11도).\n"
        "\n"
        "세 곳이 다 찾아져야 평면이 선다. 하나라도 놓치면 홀 주변 고리로\n"
        "물러서고 그 사실이 상태문구에 남는다."
    ),
}


# 저장 파일은 손으로 고칠 수 있으므로, 읽을 때 값이 상해 있어도 버티게 한다.
def as_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def as_float(value, fallback, minimum=None, maximum=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if minimum is not None:
        number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


# ======================================================
# 오프셋
# ======================================================
def load_approach_offset(profile_name):
    """프로파일에 티칭해 둔 진입 각도 오프셋 (rx, ry, rz) [도].

    오퍼레이터 모드가 쓴다. 읽지 못하면 (0, 0, 0) = 보정 없음.
    """
    data = load_profile(profile_name)
    offset = data.get("approach_offset") if isinstance(data, dict) else None
    if not isinstance(offset, dict):
        return (0.0, 0.0, 0.0)
    return tuple(
        as_float(offset.get(key), 0.0, low, high)
        for key, _label, low, high, *_rest in APPROACH_OFFSET_FIELDS
    )


# ======================================================
# 자세를 어디서 받을지
# ======================================================
def normal_source_from_settings(data):
    """세팅에 저장된 자세 출처. → {"mode", "ring_radius_mm"}

    세팅 화면과 오퍼레이터가 같은 방식을 쓰도록 두 경로가 이 함수를 함께 쓴다.
    오퍼레이터는 촬영 보드마다 다른 프로파일을 읽으므로, 보드마다 법선을 뽑는
    방식이 달라도 된다.

    모르는 값이 적혀 있으면 **매칭점 방식으로 물러선다.** 이 키가 생기기 전에
    저장한 프로파일이 그렇게 읽히는데, 그때 쓰던 방식이 바로 매칭점이다.
    """
    saved = data.get("normal_source") if isinstance(data, dict) else None
    mode = None
    radius = None
    if isinstance(saved, dict):
        mode = saved.get("mode")
        radius = saved.get("ring_radius_mm")
    valid = {key for _label, key in NORMAL_MODE_CHOICES}
    return {
        "mode": mode if mode in valid else NORMAL_MODE_POINTS,
        "ring_radius_mm": as_float(
            radius, RING_RADIUS_MM_DEFAULT, RING_RADIUS_MM_MIN, RING_RADIUS_MM_MAX
        ),
    }


def draw_ring_regions(image, holes, poses, camera_matrix, ring_radius_mm):
    """ring 모드가 실제로 평면을 맞추는 영역을 그린다 (안쪽/바깥쪽 두 원).

    **표본을 고르는 쪽과 같은 함수(ring_outer_px)로 반지름을 구한다.** 눈에
    보이는 원과 실제 계산 범위가 다르면 영역을 조절하는 일 자체가 뜻을 잃는다.

    바깥 반지름이 지정한 mm 보다 크게 그려질 수 있다. 홀이 커서 지정한 영역이
    홀 안쪽에 들어가면 안쪽 경계 밖으로 밀어내기 때문이다(ring_outer_px). 그
    사실이 화면에 그대로 보이는 것이 맞다 — 값만 보고는 알 수 없다.

    Args:
        holes: [(u, v, radius)] 컬러 이미지 좌표
        poses: holes 와 같은 길이. z 를 알아야 mm 를 px 로 환산할 수 있다.
        camera_matrix: 컬러 내부 파라미터 (초점거리를 꺼낸다)

    Returns:
        영역이 그려진 BGR 이미지. 그릴 것이 없으면 원본을 그대로 돌려준다.
    """
    if not holes or camera_matrix is None:
        return image
    focal = float(np.asarray(camera_matrix, dtype=np.float64)[0, 0])
    if focal <= 0.0:
        return image

    result = None
    poses = poses or []
    for index, (u, v, radius) in enumerate(holes):
        pose = poses[index] if index < len(poses) else None
        if pose is None:
            continue      # z 를 모르면 mm 를 px 로 옮길 수 없다
        outer = ring_outer_px(float(radius), float(pose.z), focal, ring_radius_mm)
        inner = float(radius) * RING_INNER_SCALE
        if result is None:
            result = image.copy()
        center = (int(round(u)), int(round(v)))
        cv2.circle(result, center, int(round(inner)), RING_REGION_COLOR,
                   RING_REGION_WIDTH)
        cv2.circle(result, center, int(round(outer)), RING_REGION_COLOR,
                   RING_REGION_WIDTH)
    return image if result is None else result


# ======================================================
# 법선 매칭점 (세 곳)
# ======================================================
def normal_points_from_settings(data):
    """세팅에 저장된 법선 매칭점 설정. 없으면 꺼진 상태.

    세팅 화면과 오퍼레이터가 같은 값을 읽도록 두 경로가 이 함수를 함께 쓴다.

    네모 좌표는 여기 없다 — 티칭 프레임과 함께 창고에 있고(load_normal_points),
    세팅에는 그 이름만 남는다. 좌표만 옮겨 와도 티칭 프레임이 없으면 비디오
    추적이 기준을 잡을 수 없기 때문이다.
    """
    saved = data.get("normal_points") if isinstance(data, dict) else None
    if not isinstance(saved, dict):
        return {"enabled": False, "reference": None, "model": None}
    return {
        "enabled": bool(saved.get("enabled", False)),
        "reference": saved.get("reference"),
        "model": saved.get("model"),
    }


def normal_points_masks(settings, bgr_image):
    """매칭점 세 곳을 이 촬영에서 다시 찾는다. 마스크는 **컬러 해상도**다.

    못 찾아도 예외를 올리지 않는다. 법선을 못 구했다고 좌표를 아예 안 내는 것보다,
    홀 고리 법선으로 물러서고 그 사실을 문구로 남기는 편이 낫다 — 라인이 멈추는
    것보다 조금 나쁜 값이 낫고, 문구를 보면 바로 알아챌 수 있다.

    **못 찾은 자리를 빈 목록으로 뭉개지 않는다.** 셋 중 하나만 놓쳤을 때도
    나머지 둘은 화면에 보여야 하고, 놓친 자리는 티칭 네모로 표시해야 어디를
    다시 지정할지 알 수 있다. 평면을 세울지 말지는 plane_from_normal_points 가
    None 섞인 목록을 보고 정한다.

    뎁스 해상도로 옮기지 않는 이유: 뎁스가 없어도 화면에는 그려야 한다.
    옮기는 일은 평면을 세우는 쪽에서 한다.

    Returns:
        ([마스크 또는 None] 세 개, 문구, [티칭 네모] 세 개)
        기능이 꺼져 있거나 기준을 못 읽으면 ([], 문구, []).
    """
    if not isinstance(settings, dict) or not settings.get("enabled"):
        return [], "", []
    reference_name = settings.get("reference")
    taught = load_normal_points(reference_name)
    if taught is None:
        return [], (
            f"법선 매칭점 '{reference_name or ''}' 을 읽을 수 없습니다 "
            "(다시 지정하세요)"
        ), []
    scene, boxes, areas = taught

    try:
        masks, note = sam2_video_tracker.Sam2VideoNormalPoints(
            scene, boxes, settings.get("model"), taught_areas=areas
        ).find_masks(bgr_image)
    except Exception as exc:
        return [], f"법선 매칭점을 찾지 못했습니다: {exc}", boxes
    if note:
        return masks, f"법선 기준: {note}", boxes

    total = sum(int(mask.sum()) for mask in masks if mask is not None)
    return masks, f"법선 기준 매칭점 {len(masks)}곳 {total:,} px", boxes


def normal_plane_from_masks(
    masks, depth, camera_matrix, dist_coeffs=None, color_size=None, depth_scale_mm=1.0
):
    """컬러 해상도 매칭점 마스크로 평면을 세운다. → (평면 또는 None, 벌어짐 mm)

    뎁스 해상도로 옮기는 일을 여기서 한다 — 표본을 거르는 자리가 뎁스 배열이다.
    하나라도 None 이면 plane_from_normal_points 가 None 을 준다(여분이 없다).

    벌어진 정도를 함께 돌려주는 이유: 완전 일직선만 거절하므로, 나쁜 배치도
    평면은 나온다. 그게 얼마나 미더운지는 문구로 알려야 한다.
    """
    if not masks or depth is None:
        return None, None
    moved = []
    for mask in masks:
        if mask is None:
            return None, None
        resized = mask_to_depth_size(mask, depth.shape)
        if resized is None:
            return None, None
        moved.append(resized)
    spread = normal_point_spread_mm(
        moved,
        depth,
        camera_matrix,
        dist_coeffs,
        color_size=color_size,
        depth_scale_mm=depth_scale_mm,
    )
    plane = plane_from_normal_points(
        moved,
        depth,
        camera_matrix,
        dist_coeffs,
        color_size=color_size,
        depth_scale_mm=depth_scale_mm,
    )
    return plane, spread


def spread_note(spread):
    """세 점이 벌어진 정도를 사람이 읽을 문구로. 값이 없으면 빈 문자열.

    세팅 화면의 상태문구와 오퍼레이터 로그가 같은 말을 해야 하므로 한 군데 둔다.
    완전 일직선만 거절하므로 좁게 몰린 배치도 평면은 나온다 — 그게 얼마나
    흔들리는지는 문구로 알려야 한다.
    """
    if spread is None:
        return ""
    caution = (
        " (좁다 — 법선이 잡음에 흔들린다)"
        if spread < NORMAL_POINT_SPREAD_CAUTION_MM
        else ""
    )
    return f"벌어짐 {spread:.1f}mm{caution}"


# ======================================================
# 화면 표시
# ======================================================
def _mask_to_image_size(mask, image):
    """뎁스 해상도 마스크를 그릴 이미지 크기로 맞춘다. 못 쓰면 None.

    마스크는 뎁스 해상도로 들고 다니는데(표본을 거르는 자리가 뎁스 배열이다)
    그리는 것은 컬러 이미지다. D405 는 둘이 같은 크기라 대개 그냥 통과한다.
    """
    if mask is None or image is None:
        return None
    return mask_to_depth_size(mask, image.shape[:2])


def draw_normal_points(image, masks, boxes=None):
    """법선 매칭점 세 곳을 그린다. 찾은 것은 경계로, 못 찾은 것은 티칭 네모로.

    **원을 그리지 않는다.** 이것들은 면이고, 원으로 그리면 홀과 구분이 안 된다
    (실제로 홀 원과 헷갈려 잘못 잡힌 것으로 오해한 적이 있다). 경계선만 그리고,
    안쪽은 옅게 덮어 '여기서 법선을 받는다'가 한눈에 보이게 한다.

    번호를 함께 찍는다. 셋 중 어느 자리가 어긋났는지 알아야 그 칸만 다시
    지정할 수 있기 때문이다.

    **못 찾은 자리를 비워 두지 않는다.** 세 곳이 다 있어야 평면이 서는데,
    하나가 빠지면 법선이 조용히 홀 고리로 물러선다. 아무것도 안 그리면 그
    사실이 화면에 드러나지 않아 '왜 자세가 이상한지' 짚을 수 없다. 그래서
    티칭 때의 네모를 자홍색으로 그려 '여기 있어야 하는데 없다'를 보여 준다.

    Args:
        image: BGR 이미지 (원본을 고치지 않는다)
        masks: 마스크 목록. 못 찾은 자리는 None. 뎁스/컬러 해상도 둘 다 받는다.
        boxes: 티칭 때의 (left, top, right, bottom) 목록. 주면 못 찾은 자리를
            이 네모로 표시한다. masks 와 같은 차례여야 한다.

    Returns:
        경계가 그려진 BGR 이미지
    """
    if not masks and not boxes:
        return image

    height, width = image.shape[:2]
    result = None
    count = max(len(masks or []), len(boxes or []))

    for index in range(count):
        mask = masks[index] if masks and index < len(masks) else None
        moved = _mask_to_image_size(mask, image)
        array = None if moved is None else np.asarray(moved)
        found = (
            array is not None
            and array.ndim == 2
            and array.shape == (height, width)
            and array.any()
        )

        if found:
            contours, _hierarchy = cv2.findContours(
                array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                found = False

        if not found:
            # 못 찾았다. 티칭 네모라도 있으면 그 자리를 표시한다.
            box = boxes[index] if boxes and index < len(boxes) else None
            if box is None:
                continue
            if result is None:
                result = image.copy()
            left, top, right, bottom = (int(value) for value in box)
            cv2.rectangle(
                result,
                (left, top),
                (right, bottom),
                NORMAL_POINT_MISS_COLOR,
                EXTRACT_CONTOUR_WIDTH,
            )
            cv2.putText(
                result,
                f"{index + 1}?",
                (left + 2, max(12, top - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                NORMAL_POINT_MISS_COLOR,
                1,
            )
            continue

        if result is None:
            result = image.copy()
        # 안쪽을 옅게 덮는다. 꽉 채우면 그 아래의 컬러가 안 보여 자리가 옳게
        # 잡혔는지 눈으로 확인할 수 없다.
        tint = result.copy()
        cv2.drawContours(tint, contours, -1, NORMAL_POINT_COLOR, cv2.FILLED)
        cv2.addWeighted(
            tint, EXTRACT_FILL_ALPHA, result, 1.0 - EXTRACT_FILL_ALPHA, 0, result
        )
        cv2.drawContours(
            result, contours, -1, NORMAL_POINT_COLOR, EXTRACT_CONTOUR_WIDTH
        )

        rows, cols = np.nonzero(array)
        cv2.putText(
            result,
            str(index + 1),
            (int(cols.min()) + 2, max(12, int(rows.min()) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            NORMAL_POINT_COLOR,
            1,
        )
    return image if result is None else result
