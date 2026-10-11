"""홀 3D 포즈 추정 — 블롭 중심의 위치와 주변 평면에서 세운 자세.

Hole Detection 페이지가 찾아낸 홀(픽셀 원)에 같은 시점의 뎁스 프레임을 겹쳐
홀마다 (x, y, z, rx, ry, rz)를 만든다. 계산만 따로 검증하고 재사용하려고
UI와 분리해 둔다(hand_eye 모듈과 같은 이유).

단위/규약 (hand_eye 모듈에서 정한 규약을 그대로 따른다)
  - 위치 : mm, 카메라 광학 좌표계 (+X 오른쪽, +Y 아래, +Z 피사체 쪽)
  - 회전 : rx, ry, rz 오일러각 [도], ZYX 순서 → R = Rz(rz)·Ry(ry)·Rx(rx)

자세를 세우는 방법
  홀 안쪽은 뚫려 있어서 뎁스가 비거나 한참 뒤의 바닥을 찍는다. 그래서 홀을
  둘러싼 '고리'에서만 표본을 모아 그 판재면에 평면을 맞추고, 평면의 법선 중
  카메라 쪽으로 뻗는 방향을 z축으로 삼는다.

  z값도 홀 중심 픽셀의 뎁스를 그대로 쓰지 않고, 중심을 지나는 시선과 위에서
  맞춘 평면의 교점에서 얻는다. 그래야 뚫린 홀이라 중심 뎁스가 비어 있어도
  '구멍 입구 면'의 높이가 나온다.

  원형 홀은 법선 둘레로 아무리 돌려도 같은 모양이라, z축 둘레의 회전은 영상만
  으로 정해지지 않는다. 대신 매번 같은 값이 나오도록 카메라 +X축을 평면에
  투영해 x축으로 쓴다. 즉 rz는 '홀이 향한 방향'이 아니라 재현 가능한 기준일
  뿐이고, 의미가 있는 것은 z축(법선)이다.

  고리는 폭이 좁아 법선의 지렛대가 짧다(반지름 10px 홀이면 300mm 거리에서 약
  25mm). 법선은 SVD에서 가장 덜 결정되는 방향이라 이 폭으로는 뎁스 노이즈가
  그대로 각도 오차가 된다. 그래서 normal_plane 모듈이 판재 '전체 면'에서
  뽑은 대표 평면을 surface_plane 인자로 넘겨줄 수 있게 해 두었다. 그 경우
  방향(법선)은 면에서 받고, 높이(평면 오프셋)만 홀 주변 고리에서 잡는다.
  판재가 살짝 휘어 있어도 구멍 입구 높이는 그 자리 값이 맞기 때문이다.
"""

import math
from collections import namedtuple

import cv2
import numpy as np

from hand_eye import (
    matrix_to_pose,
    matrix_to_rpy_zyx_deg,
    pose_to_matrix,
    rpy_zyx_deg_to_matrix,
)

# ===== 표본을 모을 고리 (홀 반지름 배수) =====
# 안쪽은 홀 가장자리의 모따기/그림자를 피할 만큼 띄우고, 바깥쪽은 판재가
# 평평하다고 볼 수 있는 범위까지만 넓힌다.
RING_INNER_SCALE = 1.4
RING_OUTER_SCALE = 3.0
RING_MIN_OUTER_PX = 6       # 아주 작은 홀에서도 표본이 남도록 두는 하한
MIN_PLANE_SAMPLES = 30      # 이보다 적으면 평면을 믿을 수 없다고 본다
MAX_PLANE_SAMPLES = 2000    # 큰 홀에서 표본이 수만 개로 늘어나는 것만 막는다

# ===== 이상치 정리 =====
DEPTH_MAD_SIGMA = 3.0       # 중앙값에서 이 배수(로버스트 시그마)를 넘으면 버린다
DEPTH_MIN_SPREAD_MM = 2.0   # 표본이 너무 고르면 허용폭이 0이 되므로 최소치를 둔다
PLANE_INLIER_MM = 1.5       # 평면에서 이만큼 벗어난 표본은 빼고 다시 맞춘다
PLANE_REFIT_ROUNDS = 2

# 대표 법선을 면에서 받았을 때는 평면의 '높이'만 잡으면 되므로 표본이 훨씬
# 적어도 된다. 방향을 세우는 데 필요했던 MIN_PLANE_SAMPLES 만큼 요구하면,
# 면 마스크로 걸러진 좁은 고리에서 멀쩡한 홀이 통째로 버려진다.
MIN_OFFSET_SAMPLES = 10

# ===== 안정화 =====
NORMAL_EMA_ALPHA = 0.2        # 새 법선 반영 비율 (작을수록 안정적, 추종은 느림)
NORMAL_MAX_STEP_DEG = 3.0     # 한 프레임에서 허용할 최대 법선 회전
POSE_MATCH_MAX_PX = 30.0      # 이전 프레임의 같은 홀로 볼 중심 이동 거리

# 시선이 평면과 거의 나란하면 교점이 발산하므로 그런 홀은 포기한다.
MIN_RAY_PLANE_COS = 0.05

# z축 둘레 스핀을 고를 근거가 사라졌다고 볼 문턱(_align_spin_to_flange 참고).
SPIN_CHOICE_EPSILON = 1e-9

# z축(법선)을 카메라 쪽으로 뻗게 할지. 로봇 공구처럼 면을 파고드는 방향이
# 필요하면 False로 두면 부호만 뒤집힌다.
NORMAL_TOWARD_CAMERA = True

# plane_source: 법선을 어디서 받았는지. 정밀도가 크게 다르므로 화면과 로그에서
# 구분해 보여 준다. 로봇 좌표로 옮길 때 "camera"만 다르게 다루므로(아래),
# 이 값은 표시용이 아니라 **동작을 가르는 값**이다.
PLANE_FROM_RING = "ring"          # 홀 주변 영역에서 직접 맞춘 평면
PLANE_FROM_SURFACE = "surface"    # 매칭점 세 곳이 만든 평면
PLANE_FROM_CAMERA = "camera"      # 법선을 재지 않았다 — 위치(x, y, z)만 쓴다

# ===== 자세를 어디서 받을지 (세 가지) =====
# 세팅에 저장되어 오퍼레이터도 같은 방식을 쓴다(pose_pipeline.normal_source_*).
#
#   flange  회전을 비전으로 재지 않는다. 2D 비전처럼 플랜지 자세를 그대로 두고
#           비전에서는 x, y, z 만 받아 이동한다. z 를 쓰므로 2D 와 다르다.
#   ring    홀 주변 영역(고리)의 평면 법선을 z축으로 쓴다. 영역 크기를 mm 로
#           지정한다 — 부품 상의 실제 크기라 카메라 거리가 바뀌어도 뜻이 같다.
#   points  사람이 짚어 둔 매칭점 세 곳의 평면 법선을 쓴다. 가장 재현성이 높다.
NORMAL_MODE_FLANGE = "flange"
NORMAL_MODE_RING = "ring"
NORMAL_MODE_POINTS = "points"

# ring 모드 영역 반지름의 기본값과 허용 범위 [mm]. 하한을 0 으로 두지 않는
# 이유: 너무 작으면 표본이 홀 모따기에 갇혀 법선이 뜻을 잃는다.
RING_RADIUS_MM_DEFAULT = 15.0
RING_RADIUS_MM_MIN = 2.0
RING_RADIUS_MM_MAX = 200.0

HolePose = namedtuple(
    "HolePose",
    "u v x y z rx ry rz normal tilt_deg samples rms_mm plane_source",
)
HolePose.__new__.__defaults__ = (PLANE_FROM_RING,)


# ======================================================
# 공개 API
# ======================================================
def estimate_hole_poses(
    holes,
    depth,
    camera_matrix,
    dist_coeffs=None,
    color_size=None,
    depth_scale_mm=1.0,
    surface_plane=None,
    mode=None,
    ring_radius_mm=None,
):
    """홀 목록의 3D 포즈를 구한다.

    Args:
        holes: [(u, v, radius), ...] 컬러 이미지 픽셀 좌표
        depth: (H, W) 뎁스 배열 (원시 값, mm 환산은 depth_scale_mm으로)
        camera_matrix: 3x3 컬러 카메라 내부 파라미터
        dist_coeffs: 왜곡 계수 (없으면 왜곡 0으로 본다)
        color_size: (width, height) 컬러 이미지 크기. 뎁스와 다르면 그 비율로
            좌표와 내부 파라미터를 맞춘다.
        depth_scale_mm: 뎁스 값 1당 mm
        surface_plane: (normal, centroid) 대상 면의 대표 평면 (카메라 기준, mm).
            주면 z축을 이 법선으로 고정하고 고리에서는 평면 높이만 잡는다.
            normal_plane.plane_from_normal_points()가 이 형태로 만들어 준다.
            mode=NORMAL_MODE_POINTS 일 때만 쓴다.
        mode: NORMAL_MODE_* 중 하나. 자세를 어디서 받을지 정한다.
            None 이면 예전 규약대로 surface_plane 유무로 정한다(있으면 points,
            없으면 ring) — 이 인자가 생기기 전 호출자를 깨지 않기 위해서다.
        ring_radius_mm: ring 모드에서 평면을 맞출 영역의 **바깥 반지름 [mm]**.
            None 이면 예전처럼 홀 반지름의 RING_OUTER_SCALE 배를 쓴다.
            안쪽은 손대지 않는다 — 홀 모따기를 피하는 값이라 홀 크기에 붙어야 한다.

    Returns:
        holes와 같은 길이의 리스트. 구하지 못한 홀 자리에는 None이 들어가므로
        호출한 쪽에서 인덱스로 홀과 짝지을 수 있다.
    """
    if not holes:
        return []
    if depth is None or depth.size == 0 or camera_matrix is None:
        return [None] * len(holes)
    if mode is None:
        mode = NORMAL_MODE_POINTS if surface_plane is not None else NORMAL_MODE_RING

    # 뎁스는 이미 취득 단계에서 후처리(realsense_postprocess)를 거쳐 들어온다.
    # 여기서 또 거르면 페이지마다 다른 뎁스를 보게 되므로 그대로 쓴다.
    filtered_depth = np.asarray(depth)
    matrix, scale_x, scale_y = depth_frame_intrinsics(
        np.asarray(camera_matrix, dtype=np.float64), filtered_depth, color_size
    )
    coeffs = (
        np.zeros(5)
        if dist_coeffs is None
        else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    )
    radius_scale = 0.5 * (scale_x + scale_y)

    # 세 방식이 여기서 갈린다. 아래로는 '법선을 미리 아는가'만 다르므로
    # _estimate_one 은 방식을 몰라도 된다.
    if mode == NORMAL_MODE_FLANGE:
        # 법선을 재지 않는다. 카메라 광축을 평면 법선으로 두면 고리 표본에서
        # 높이만 잡게 되고(_plane_offset), 그 결과가 곧 '홀 입구의 x, y, z' 다.
        # 표본 요구가 30개에서 10개로 내려가 가장 잘 버티는 방식이 된다.
        surface_normal = np.array([0.0, 0.0, -1.0])
        source = PLANE_FROM_CAMERA
    elif mode == NORMAL_MODE_POINTS:
        surface_normal = _surface_normal(surface_plane)
        # 매칭점을 못 찾았으면 surface_normal 이 None 이고, 아래에서 홀 주변
        # 고리로 물러선다. 그때도 'surface' 라고 적으면 **화면과 로그가 거짓말을
        # 한다** — 두 출처는 정밀도가 완전히 다르다(0.11도 대 ±8도). 자세가
        # 튈 때 원인을 짚는 단서가 바로 이 값이다.
        source = PLANE_FROM_SURFACE if surface_normal is not None else PLANE_FROM_RING
    else:
        surface_normal = None
        source = PLANE_FROM_RING

    return [
        _estimate_one(
            filtered_depth,
            matrix,
            coeffs,
            u * scale_x,
            v * scale_y,
            radius * radius_scale,
            float(depth_scale_mm),
            surface_normal,
            source,
            ring_radius_mm,
        )
        for u, v, radius in holes
    ]


def ring_outer_px(radius_px, depth_mm, focal_px, ring_radius_mm):
    """ring 모드 영역의 바깥 반지름을 px 로 환산한다.

    mm 로 받는 이유는 그것이 **부품 상의 크기**라서다. 카메라가 멀어지면 같은
    영역이 더 적은 픽셀에 담기는데, px 로 고정해 두면 그때 실제로 보는 면적이
    달라져 법선의 뜻이 바뀐다.

    화면에 영역을 그리는 쪽과 표본을 고르는 쪽이 **같은 값**을 써야 하므로
    한 군데 둔다. 어긋나면 눈에 보이는 원과 실제 계산 범위가 달라진다.

    환산할 수 없으면(뎁스나 초점거리를 모르면) 예전 규약인 홀 반지름 배수로
    물러선다 — 영역이 조용히 0 이 되어 홀이 통째로 버려지는 것을 막는다.
    """
    if not ring_radius_mm or depth_mm is None or depth_mm <= 0.0 or focal_px <= 0.0:
        return max(radius_px * RING_OUTER_SCALE, float(RING_MIN_OUTER_PX))
    outer = float(ring_radius_mm) * float(focal_px) / float(depth_mm)
    # 안쪽 경계보다 커야 고리에 폭이 남는다.
    return max(outer, radius_px * RING_INNER_SCALE + 2.0, float(RING_MIN_OUTER_PX))


def _surface_normal(surface_plane):
    """(normal, centroid) 에서 카메라 쪽으로 향한 단위 법선만 꺼낸다.

    면에서 온 법선은 이미 같은 규약으로 맞춰져 있지만, 손으로 만든 값이
    들어와도 버티도록 부호를 한 번 더 확인한다. centroid 는 여기서 쓰지 않는다.
    평면의 높이는 홀마다 그 자리 고리에서 다시 잡기 때문이다.
    """
    if surface_plane is None:
        return None
    normal = np.asarray(surface_plane[0], dtype=np.float64).reshape(-1)
    if normal.size != 3:
        return None
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return None
    normal = normal / norm

    centroid = np.asarray(surface_plane[1], dtype=np.float64).reshape(-1)
    if centroid.size == 3 and (float(normal @ centroid) > 0.0) == NORMAL_TOWARD_CAMERA:
        normal = -normal
    return normal


def stabilize_hole_poses(current_poses, previous_poses):
    """같은 위치의 홀 법선을 이전 프레임과 누적해 방향 흔들림을 줄인다.

    법선을 면 전체에서 받은 홀(plane_source == PLANE_FROM_SURFACE)은 건드리지
    않는다. 그 법선은 이미 수만 표본에서 나와 흔들림이 없는데, 여기서 이전
    프레임과 섞으면 오히려 지난 촬영의 값이 80% 남는다. 오퍼레이터 모드는
    트리거마다 한 장씩 찍으므로 '이전 프레임'이 다른 사이클의 것일 수 있다.
    """
    previous = [pose for pose in (previous_poses or []) if pose is not None]
    stabilized = []

    for pose in current_poses:
        if pose is None or pose.plane_source == PLANE_FROM_SURFACE:
            stabilized.append(pose)
            continue

        match = _nearest_previous_pose(pose, previous)
        if match is None:
            stabilized.append(pose)
            continue

        normal = _stabilize_normal(match.normal, pose.normal)
        rx, ry, rz = matrix_to_rpy_zyx_deg(_frame_from_normal(normal))
        stabilized.append(
            pose._replace(
                normal=tuple(float(value) for value in normal),
                rx=rx,
                ry=ry,
                rz=rz,
            )
        )

    return stabilized


def _nearest_previous_pose(pose, previous):
    if not previous:
        return None
    match = min(previous, key=lambda item: (item.u - pose.u) ** 2 + (item.v - pose.v) ** 2)
    distance = math.hypot(match.u - pose.u, match.v - pose.v)
    return match if distance <= POSE_MATCH_MAX_PX else None


def _stabilize_normal(previous_normal, current_normal):
    previous = np.asarray(previous_normal, dtype=np.float64)
    current = np.asarray(current_normal, dtype=np.float64)
    previous /= np.linalg.norm(previous)
    current /= np.linalg.norm(current)

    # SVD 법선은 부호가 양쪽 모두 가능하므로 이전 방향과 같은 반구로 맞춘다.
    if float(previous @ current) < 0.0:
        current = -current

    angle = math.acos(float(np.clip(previous @ current, -1.0, 1.0)))
    if angle <= 1e-9:
        return previous

    max_step = math.radians(NORMAL_MAX_STEP_DEG)
    alpha = min(NORMAL_EMA_ALPHA, max_step / angle)
    blended = (1.0 - alpha) * previous + alpha * current
    return blended / np.linalg.norm(blended)


def pose_matrix(pose):
    """HolePose → 4x4 동차변환행렬 (카메라 기준). 로봇 좌표로 옮길 때 쓴다.

    hand_eye.pose_to_matrix와 같은 규약이므로, 손-눈 결과(cam2gripper)와
    플랜지 포즈를 곱해 베이스 좌표로 바로 넘길 수 있다.
    """
    matrix = np.eye(4)
    matrix[:3, :3] = _frame_from_normal(np.asarray(pose.normal, dtype=np.float64))
    matrix[:3, 3] = (pose.x, pose.y, pose.z)
    return matrix


def flange_target_matrix(pose, flange_from_camera):
    """카메라 기준 홀 → 플랜지 기준 4x4 접촉 자세 (공구 +Z가 작업물을 향한다).

    flip_x(diag(1,-1,-1) = Rx(180°))로 카메라 쪽을 보던 z축을 뒤집어, 공구가
    소재를 파고드는 방향으로 맞춘다.

    행렬을 그대로 돌려주는 이유는, 오일러각으로 한 번 내렸다가 다시 올리면
    그 사이에 각 성분을 손대고 싶어지기 때문이다(그렇게 하면 안 된다 —
    _align_spin_to_flange 의 설명 참고).

    plane_source 가 PLANE_FROM_CAMERA 면(= flange 모드) **회전을 싣지 않는다.**
    플랜지 기준 회전을 단위행렬로 두면 T_base_target 의 자세가 T_base_flange 와
    똑같아진다 — 즉 로봇은 지금 자세를 그대로 유지하고 위치만 옮긴다. 2D 비전과
    같은 동작이면서 z 는 비전에서 받는다.

    모드가 포즈에 실려 오는 것이 요점이다. 부르는 쪽이 모드를 따로 들고 다니면
    포즈와 어긋날 수 있고, 그때 조용히 엉뚱한 자세가 로봇으로 나간다.
    """
    transform = np.asarray(flange_from_camera, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError("좌표 변환 행렬은 4x4여야 합니다.")

    if getattr(pose, "plane_source", None) == PLANE_FROM_CAMERA:
        target = np.eye(4)
        point = np.array([pose.x, pose.y, pose.z, 1.0], dtype=np.float64)
        target[:3, 3] = (transform @ point)[:3]
        return target

    flip_x = np.eye(4)
    flip_x[1, 1] = -1.0
    flip_x[2, 2] = -1.0
    return transform @ pose_matrix(pose) @ flip_x


def apply_approach_offset(target, approach_offset):
    """측정한 자세에 티칭한 진입 각도 오프셋을 얹는다.

    왜 필요한가
      비전이 재는 것은 '대상 전체 면의 법선'이다. 그 값은 아주 재현성이 높지만
      (260901 실측 0.11도), 나사가 실제로 앉는 면과는 다를 수 있다. 260901 대상은
      기계 캐비닛 측면이고 나사는 폭 30mm 짜리 좁은 스트립 위에 있어서, 대상
      전체 평면과 그 스트립의 법선이 20도 차이 났다.

      스트립 법선을 직접 재려면 영역을 그 모양에 맞춰야 하는데 그래도 ±2~3도가
      한계다. 대신 '대상 전체 법선'과 '실제로 나사가 앉는 방향'의 차이는 부품
      형상이 정하는 상수이므로, 한 번 티칭해 두고 매번 더하는 편이 정확하다.

    회전을 **뒤에서** 곱한다. 뒤에서 곱하면 측정한 공구 좌표계 자신의 축 둘레로
    돌아가므로, 오프셋이 '측정값에서 이만큼 더 기울여라'라는 뜻이 된다. 오일러
    각을 그냥 더하면 안 된다(base_target_pose 주석 참고).

    축의 기준은 _align_spin_to_flange 로 스핀을 정한 뒤의 공구 x/y 축이다.
    그래야 같은 체결 지점에서 매번 같은 방향을 가리킨다.

    Args:
        target: 4x4 동차변환행렬 (스핀까지 정해진 타겟 자세)
        approach_offset: (rx, ry, rz) 도. None이나 전부 0이면 그대로 돌려준다.

    Returns:
        오프셋을 얹은 4x4 행렬
    """
    if approach_offset is None:
        return target
    values = [float(value) for value in approach_offset]
    if len(values) != 3 or not any(values):
        return target

    offset = np.eye(4)
    offset[:3, :3] = rpy_zyx_deg_to_matrix(*values)
    return target @ offset


def flange_target_pose(pose, flange_from_camera, approach_offset=None):
    """카메라 기준 홀을 공구 +Z가 작업물을 향하는 플랜지 기준 타겟 포즈로 바꾼다.

    플랜지 포즈를 모를 때(= 베이스 좌표를 만들 수 없을 때) 로봇에 넘기는 값이다.
    z축 둘레 스핀은 base_target_pose 와 같은 기준으로 고르므로, 플랜지 포즈를
    넣기 전과 후에 손목이 엉뚱하게 튀지 않는다.
    """
    target = apply_approach_offset(
        _align_spin_to_flange(flange_target_matrix(pose, flange_from_camera)),
        approach_offset,
    )
    x, y, z, rx, ry, rz = matrix_to_pose(target)
    normal = target[:3, 2]
    return pose._replace(
        x=x, y=y, z=z, rx=rx, ry=ry, rz=rz, normal=normal
    )


def _align_spin_to_flange(t_flange_target):
    """공구 z축 둘레 스핀만 돌려, 플랜지 자세에서 가장 덜 비틀린 자세로 만든다.

    원형 홀은 z축 둘레로 회전 대칭이라 스핀이 영상으로 정해지지 않는다. 남은
    자유도 하나는 아무 값이나 되므로 '손목을 가장 덜 돌리는 값'으로 못 박는다.

    ZYX 오일러의 rz를 0으로 두는 방식은 쓸 수 없다. 그 rz는 공구 자신의 z축이
    아니라 **플랜지 z축** 둘레 회전이라, 0으로 만들면 접근 방향(rx, ry가
    나타내는 법선)까지 함께 돌아간다. 카메라가 플랜지에 90도 돌아 붙어 있으면
    접근 방향이 14도쯤 틀어진다.

    그래서 회전행렬에 Rz를 **뒤에서** 곱한다. 뒤에서 곱하면 공구 자신의 z축
    둘레로만 돌아가므로 접근 방향(3번째 열)이 정확히 보존된다.

        R(θ) = R · Rz(θ)   →   R(θ)[:, 2] = R[:, 2]  (θ와 무관)

    θ는 플랜지 자세에서 돌아간 총 회전각이 가장 작아지는 값으로 고른다. 여기는
    플랜지 좌표계라 '플랜지 자세'가 곧 단위행렬이고, 회전각은 대각합이 클수록
    작으므로 대각합을 최대로 하면 된다.

        trace(R·Rz(θ)) = cos θ (R₀₀ + R₁₁) + sin θ (R₀₁ − R₁₀) + R₂₂
        →  θ = atan2(R₀₁ − R₁₀, R₀₀ + R₁₁)

    공구 x축을 플랜지 x축에 붙이는 방법도 있지만, 그쪽은 총 회전각이 최소보다
    1도쯤 크게 나온다. 원형 홀에서 x축 방향 자체는 아무 뜻이 없으므로 총
    회전각을 줄이는 편이 낫다.

    두 항이 모두 0이면 θ가 무엇이든 회전각이 같다(고를 근거가 없다는 뜻).
    그때는 0으로 두어 같은 입력에 같은 답이 나오게 한다.
    """
    rotation = t_flange_target[:3, :3]
    sin_term = float(rotation[0, 1] - rotation[1, 0])
    cos_term = float(rotation[0, 0] + rotation[1, 1])
    angle = (
        0.0
        if math.hypot(sin_term, cos_term) < SPIN_CHOICE_EPSILON
        else math.degrees(math.atan2(sin_term, cos_term))
    )

    aligned = t_flange_target.copy()
    aligned[:3, :3] = rotation @ rpy_zyx_deg_to_matrix(0.0, 0.0, angle)
    return aligned


def base_target_pose(
    flange_pose, flange_from_camera, hole_pose, approach_offset=None
):
    """로봇 베이스 기준 타겟 좌표를 구한다.

    T_base_target = T_base_flange × T_flange_target

    T_flange_target은 공구 +Z가 홀 법선을 따라 소재를 향하는 플랜지 기준 접촉
    자세이고, T_base_flange는 사용자가 입력한 로봇 플랜지 포즈다.

    원형 홀은 z축 둘레로 회전 대칭이라 스핀이 영상으로 정해지지 않으므로,
    _align_spin_to_flange()가 '손목을 가장 덜 돌리는 값'으로 골라 못 박는다.
    돌려주는 (rx, ry, rz)는 실제 T_base_target을 그대로 분해한 값이라 세 값이
    서로 맞는다. 어느 한 성분만 나중에 덮어쓰면 접근 방향이 조용히 틀어진다.

    Args:
        flange_pose: (x, y, z, rx, ry, rz) 로봇 베이스 기준 플랜지 포즈
        flange_from_camera: 4x4 cam2gripper 행렬 (EIH 캘리브레이션 결과)
        hole_pose: 카메라 기준 HolePose
        approach_offset: (rx, ry, rz) 도. 티칭한 진입 각도 오프셋.
            apply_approach_offset() 설명 참고.

    Returns:
        HolePose (베이스 기준 x, y, z, rx, ry, rz)
    """
    # 1. T_base_flange (플랜지 포즈 → 4x4 행렬)
    t_base_flange = pose_to_matrix(flange_pose)

    # 2. T_flange_target (카메라 기준 홀 → 플랜지 기준 접촉 자세).
    #    남은 자유도 하나(z축 둘레 스핀)를 골라 못 박고, 그 다음에 티칭한
    #    진입 오프셋을 얹는다. 순서를 바꾸면 오프셋의 기준 축이 흔들린다.
    t_flange_target = apply_approach_offset(
        _align_spin_to_flange(
            flange_target_matrix(hole_pose, flange_from_camera)
        ),
        approach_offset,
    )

    # 3. T_base_target = T_base_flange × T_flange_target
    t_base_target = t_base_flange @ t_flange_target

    x, y, z, rx, ry, rz = matrix_to_pose(t_base_target)
    normal = t_base_target[:3, 2]
    return hole_pose._replace(
        x=x, y=y, z=z, rx=rx, ry=ry, rz=rz, normal=normal
    )


# ======================================================
# 내부 구현
# ======================================================
def depth_frame_intrinsics(camera_matrix, depth, color_size):
    """뎁스 해상도가 컬러와 다르면 내부 파라미터를 그 비율로 줄여/늘려 준다.

    D405처럼 두 스트림을 같은 해상도로 열어 두면 그대로 통과한다(비율 1).
    왜곡 계수는 정규화 좌표 기준이라 해상도와 무관하므로 손대지 않는다.

    normal_plane도 같은 환산을 해야 면 마스크와 홀 좌표가 같은 격자에
    올라오므로, 이 함수는 공개해 두고 양쪽이 함께 쓴다.
    """
    depth_h, depth_w = depth.shape[:2]
    if not color_size:
        return camera_matrix, 1.0, 1.0

    color_w, color_h = color_size
    if (color_w, color_h) == (depth_w, depth_h) or min(color_w, color_h) <= 0:
        return camera_matrix, 1.0, 1.0

    scale_x = depth_w / float(color_w)
    scale_y = depth_h / float(color_h)
    scaled = camera_matrix.copy()
    scaled[0, :] *= scale_x
    scaled[1, :] *= scale_y
    return scaled, scale_x, scale_y


def _estimate_one(
    depth,
    camera_matrix,
    dist_coeffs,
    u,
    v,
    radius,
    depth_scale_mm,
    surface_normal=None,
    source=None,
    ring_radius_mm=None,
):
    """홀 하나의 포즈. 표본이 모자라거나 평면이 서지 않으면 None."""
    # 방향을 면에서 받으면 고리는 높이만 재면 되므로 요구 표본이 줄어든다.
    minimum = MIN_PLANE_SAMPLES if surface_normal is None else MIN_OFFSET_SAMPLES

    outer = None
    if ring_radius_mm:
        # mm -> px 환산에 그 자리의 깊이가 필요하다. 기본 고리로 대충 먼저 재고
        # 그 값으로 영역을 다시 잡는다. 두 번 훑지만 둘 다 작은 사각형이다.
        coarse = _coarse_depth_mm(depth, u, v, radius, depth_scale_mm)
        outer = ring_outer_px(radius, coarse, float(camera_matrix[0, 0]), ring_radius_mm)

    pixels, values = _ring_samples(depth, u, v, radius, minimum, outer)
    if pixels is None:
        return None

    depths_mm = values.astype(np.float64) * depth_scale_mm
    # 넓은 영역에서 이 MAD 문턱이 기울어진 면의 정상 표본까지 자르는 것은 맞다.
    # 그래도 빼면 안 된다. 빼 보니 PLANE_INLIER_MM(1.5mm)이 좁아서 평면 재적합이
    # 100표본짜리 조각으로 무너지고 법선이 50도씩 틀어졌다(260901 실측).
    # 이 문턱이 '어느 뎁스 띠를 평면으로 볼지'를 먼저 정해 주는 역할을 한다.
    keep = _depth_inliers(depths_mm)
    if keep.sum() < minimum:
        return None
    pixels, depths_mm = pixels[keep], depths_mm[keep]

    points = _deproject(pixels, depths_mm, camera_matrix, dist_coeffs)
    if surface_normal is None:
        centroid, normal, samples, rms_mm = _fit_plane_robust(points)
        if normal is None:
            return None
        # 요청 규약: 카메라(원점) 쪽으로 뻗는 법선을 z축으로 삼는다. 법선과
        # 중심점이 같은 방향이면 카메라 반대쪽을 보고 있다는 뜻이므로 뒤집는다.
        if (float(normal @ centroid) > 0.0) == NORMAL_TOWARD_CAMERA:
            normal = -normal
    else:
        normal = surface_normal
        centroid, samples, rms_mm = _plane_offset(points, normal, minimum)
        if centroid is None:
            return None
    if source is None:
        source = PLANE_FROM_RING if surface_normal is None else PLANE_FROM_SURFACE

    center = _ray_plane_intersection(
        _deproject_ray(u, v, camera_matrix, dist_coeffs), centroid, normal
    )
    if center is None:
        return None

    rx, ry, rz = matrix_to_rpy_zyx_deg(_frame_from_normal(normal))

    # 면이 카메라 광축에 대해 얼마나 기울었는지. 정면이면 0도이고, 값이 크면
    # 뎁스 품질이 떨어지므로 결과를 얼마나 믿을지 판단하는 눈금이 된다.
    tilt = math.degrees(math.acos(float(np.clip(abs(normal[2]), -1.0, 1.0))))

    return HolePose(
        u=float(u),
        v=float(v),
        x=float(center[0]),
        y=float(center[1]),
        z=float(center[2]),
        rx=rx,
        ry=ry,
        rz=rz,
        normal=(float(normal[0]), float(normal[1]), float(normal[2])),
        tilt_deg=tilt,
        samples=int(samples),
        rms_mm=rms_mm,
        plane_source=source,
    )


def _plane_offset(points, normal, minimum):
    """법선은 그대로 두고 평면의 '높이'만 고리 표본에서 잡는다.

    평면은 {p : p·n = d} 이고, d 는 고리 표본을 법선에 투영한 값의 중앙값이다.
    평균이 아니라 중앙값을 쓰는 이유는, 홀 모따기나 구멍 너머 바닥이 몇 점
    섞여 들어와도 값이 끌려가지 않게 하려는 것이다.

    Returns:
        (centroid, 사용한 표본 수, 잔차 RMS[mm]). 표본이 모자라면 centroid None.
    """
    offsets = points @ normal
    median = float(np.median(offsets))
    deviation = np.abs(offsets - median)
    spread = max(1.4826 * float(np.median(deviation)), DEPTH_MIN_SPREAD_MM)
    keep = deviation <= min(DEPTH_MAD_SIGMA * spread, PLANE_INLIER_MM * 2.0)
    if int(keep.sum()) < minimum:
        return None, 0, 0.0

    inliers = offsets[keep]
    distance = float(np.median(inliers))
    rms = float(np.sqrt(np.mean((inliers - distance) ** 2)))
    # _ray_plane_intersection 은 normal@centroid 만 쓰므로, 법선 위의 한 점을
    # centroid 로 넘겨 주면 그 값이 곧 평면 오프셋 d 가 된다.
    return normal * distance, int(keep.sum()), rms


def _coarse_depth_mm(depth, u, v, radius, depth_scale_mm):
    """기본 고리에서 대충 잰 깊이 [mm]. 표본이 없으면 None.

    ring 모드의 mm -> px 환산에만 쓴다. 정확할 필요가 없다 — 환산 결과가 몇 px
    달라져도 평면은 거의 같고, 어차피 아래에서 제대로 다시 잡는다.
    """
    _pixels, values = _ring_samples(depth, u, v, radius, minimum=1)
    if values is None or values.size == 0:
        return None
    return float(np.median(values.astype(np.float64))) * float(depth_scale_mm)


def _ring_samples(
    depth, u, v, radius, minimum=None, outer=None
):
    """홀을 둘러싼 고리에서 뎁스가 살아 있는 픽셀만 모은다.

    Args:
        outer: 바깥 반지름 [px]. None 이면 홀 반지름의 RING_OUTER_SCALE 배.
            ring 모드가 mm 로 지정한 영역을 여기로 넘긴다.

    Returns:
        (pixels (N,2) float, values (N,) 원시 뎁스) 또는 표본이 모자라면 (None, None)
    """
    if minimum is None:
        minimum = MIN_PLANE_SAMPLES
    height, width = depth.shape[:2]
    if outer is None:
        outer = max(radius * RING_OUTER_SCALE, float(RING_MIN_OUTER_PX))
    inner = radius * RING_INNER_SCALE

    # 고리를 감싸는 최소 사각형만 훑는다(전체 이미지를 도는 것보다 훨씬 싸다).
    left = int(max(0, math.floor(u - outer)))
    right = int(min(width, math.ceil(u + outer) + 1))
    top = int(max(0, math.floor(v - outer)))
    bottom = int(min(height, math.ceil(v + outer) + 1))
    if right - left < 2 or bottom - top < 2:
        return None, None

    patch = depth[top:bottom, left:right]
    ys, xs = np.ogrid[top:bottom, left:right]
    dist_sq = (xs - u) ** 2 + (ys - v) ** 2
    mask = (dist_sq >= inner ** 2) & (dist_sq <= outer ** 2) & (patch > 0)

    rows, cols = np.nonzero(mask)
    if len(rows) < minimum:
        return None, None
    if len(rows) > MAX_PLANE_SAMPLES:
        # 지름이 큰 홀은 고리에 표본이 수만 개씩 들어와 계산만 무거워진다.
        # 고르게 솎아 내되 무작위가 아니라 늘 같은 자리를 고르므로, 같은
        # 영상을 다시 분석하면 값도 그대로 나온다.
        pick = np.linspace(0, len(rows) - 1, MAX_PLANE_SAMPLES).astype(np.intp)
        rows, cols = rows[pick], cols[pick]

    pixels = np.column_stack(
        ((cols + left).astype(np.float64), (rows + top).astype(np.float64))
    )
    return pixels, patch[rows, cols]


def _depth_inliers(depths_mm):
    """중앙값에서 크게 벗어난 뎁스(구멍 너머, 튀는 픽셀)를 걸러낸다."""
    median = float(np.median(depths_mm))
    deviation = np.abs(depths_mm - median)
    # MAD×1.4826 은 정규분포에서 표준편차와 같아지는 로버스트 산포다.
    spread = max(1.4826 * float(np.median(deviation)), DEPTH_MIN_SPREAD_MM)
    return deviation <= DEPTH_MAD_SIGMA * spread


def _deproject(pixels, depths_mm, camera_matrix, dist_coeffs):
    """픽셀 + 뎁스 → 카메라 좌표 3D 점 (N,3). 왜곡 계수가 있으면 함께 편다."""
    normalized = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2), camera_matrix, dist_coeffs
    ).reshape(-1, 2)
    return np.column_stack(
        (normalized[:, 0] * depths_mm, normalized[:, 1] * depths_mm, depths_mm)
    )


def _deproject_ray(u, v, camera_matrix, dist_coeffs):
    """픽셀 하나를 지나는 시선의 단위 방향 벡터."""
    normalized = cv2.undistortPoints(
        np.array([[[float(u), float(v)]]]), camera_matrix, dist_coeffs
    ).reshape(2)
    ray = np.array([normalized[0], normalized[1], 1.0])
    return ray / np.linalg.norm(ray)


def _fit_plane_robust(points):
    """평면을 맞추고, 많이 벗어난 표본을 빼면서 몇 번 다시 맞춘다.

    Returns:
        (centroid, normal, 사용한 표본 수, 잔차 RMS[mm]). 실패하면 normal이 None.
    """
    if len(points) < 3:
        return None, None, 0, 0.0

    centroid, normal = _fit_plane(points)
    for _ in range(PLANE_REFIT_ROUNDS):
        residual = (points - centroid) @ normal
        keep = np.abs(residual) <= PLANE_INLIER_MM
        if keep.all() or int(keep.sum()) < MIN_PLANE_SAMPLES:
            break
        points = points[keep]
        centroid, normal = _fit_plane(points)

    residual = (points - centroid) @ normal
    return centroid, normal, len(points), float(np.sqrt(np.mean(residual ** 2)))


def _fit_plane(points):
    """최소제곱 평면. 중심을 뺀 뒤 가장 작은 특이값 방향이 법선이다."""
    centroid = points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[2]
    return centroid, normal / np.linalg.norm(normal)


def _ray_plane_intersection(ray, centroid, normal):
    """시선과 평면의 교점. 거의 나란하거나 카메라 뒤면 None."""
    denominator = float(normal @ ray)
    if abs(denominator) < MIN_RAY_PLANE_COS:
        return None
    distance = float(normal @ centroid) / denominator
    if distance <= 0.0:
        return None
    return ray * distance


def _frame_from_normal(z_axis):
    """법선을 z축으로 삼는 회전행렬 [x y z] (열벡터).

    x축은 카메라 +X를 평면에 투영해 만든다. 법선이 +X와 거의 나란해서 투영이
    사라질 때만 +Y로 갈아탄다. 어느 쪽이든 같은 입력이면 같은 결과가 나온다.
    """
    z_axis = z_axis / np.linalg.norm(z_axis)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(reference @ z_axis)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    x_axis = reference - float(reference @ z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))
