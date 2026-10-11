"""홀 자세(법선)를 세울 평면을 뎁스에서 뽑는다.

왜 필요한가
  hole_pose 는 홀을 둘러싼 고리(반지름 1.4~3.0배)만 보고 평면을 세운다. 고리
  폭이 좁아 법선의 지렛대가 짧고, 법선은 SVD 에서 가장 덜 결정되는 방향이라
  뎁스 노이즈가 그대로 각도 오차가 된다.

  260901 실측(정지 장면 24장, 홀 픽셀은 24장 모두 완전히 동일)에서
    - 고리 기준 법선 : 평균 방향에서 평균 7.90도, 최대 17.64도 흔들림
    - 넓은 면 기준   : 평균 0.09도, 최대 0.27도
  게다가 고리 기준은 흔들리기만 한 것이 아니라 기울기를 15.7도로 과대평가했다
  (면 기준 4.57도). 좁은 고리에서는 표면 요철이 기울기로 잘못 읽힌다.

  그래서 사람이 짚어 둔 자리에서 방향만 받아 온다. 위치(z)는 여전히 홀 주변
  고리에서 잡는다 — 판재가 살짝 휘어 있어도 구멍 입구 높이는 그 자리 값이
  맞기 때문이다. 즉 방향은 전역, 높이는 국소로 나눠 쓴다.

왜 점 셋인가 (260907)
  전에는 두 가지 방법이 있었다. 뎁스 단차로 대상을 덩어리로 나눠 그 대표
  법선을 쓰거나, 사람이 네모로 짚은 면 하나에 최소제곱 평면을 맞추거나.
  둘 다 걷어냈다.

    - 덩어리 나누기 : 부품이 놓인 자리가 조금 틀어지면 면 선택이 뒤바뀌어
      법선이 통째로 튀었다(260901 대상은 인접 면과 20도 차이).
    - 면 하나 최소제곱 : 부품이 틀어질 때 면 안에서 표본이 쓸리는 만큼
      법선이 흔들렸다.

  지금은 부품 위의 **같은 자리 세 곳**을 각각 추적해 그 대표 점으로 평면을
  세운다. 부품이 움직여도 세 점이 함께 따라가므로 평면이 부품에 붙는다.
  셋에서 늘리지 않는 이유는 plane_from_normal_points 에 적어 두었다.

단위/규약 (hand_eye / hole_pose 와 같다)
  - 위치 : mm, 카메라 광학 좌표계 (+X 오른쪽, +Y 아래, +Z 피사체 쪽)
  - 법선 : 카메라 쪽으로 뻗는 방향 (hole_pose.NORMAL_TOWARD_CAMERA 규약)
"""

import cv2
import numpy as np

from hole_pose import NORMAL_TOWARD_CAMERA, depth_frame_intrinsics

# ===== 연산 격자 =====
# 뎁스 해상도가 얼마든 이 크기로 솎아 계산량을 붙잡아 둔다. 대상은 '큰 덩어리'라
# 격자가 거칠어도 경계가 크게 상하지 않는다.

# ===== 나누는 기준 (UI에서 조정) =====
# 이웃한 격자칸의 뎁스 차이가 이 값을 넘으면 다른 대상으로 본다. 대상 표면의
# 요철과 기울기가 만드는 차이보다는 크게, 대상과 배경의 거리 차이보다는 작게
# 둔다. 대상이 여러 조각으로 갈라지면 올리고, 배경이 대상에 붙으면 내린다.

# 연산 격자 전체 픽셀 수의 몇 %보다 작은 덩어리는 버린다. 픽셀 수가 아니라
# 비율인 이유는 카메라 해상도를 바꿔도 뜻이 같게 하려는 것이다. 화면 면적
# 비율이므로 같은 대상이라도 멀리 있으면 작아진다는 점은 감안해야 한다.

# ===== 거리 밴드 (UI에서 조정) =====
# 이 범위 밖의 뎁스는 클러스터링에서 통째로 뺀다. 0 = 제한 없음(예전 동작).
#
# 단차(depth_step_mm)는 '이웃 간 상대 차이'만 보므로 작업 거리를 몰라도 되지만,
# 그 대신 배경이 늘 후보로 남아 면적 경쟁에 끼어든다. 작업 거리가 정해진
# 현장에서는 그 정보를 쓰는 편이 훨씬 확실하다 — 배경을 아예 지우고 시작하면
# 대상이 유일한 큰 덩어리가 되어 min_area_percent 도 제 뜻을 찾는다.
#
# 단차로는 대신할 수 없다. 단차는 '이웃과 이어지는가'라서 배경이 넓고 매끄러우면
# 그 안에서는 잘 이어져 큰 덩어리로 살아남는다.


# ===== 대표 평면 맞춤 =====


def mask_to_depth_size(mask, depth_shape):
    """컬러 해상도 불리언 마스크를 뎁스 해상도로 옮긴다. 못 쓰면 None.

    SAM2 는 컬러 이미지에서 영역을 따는데, 표본을 거르는 자리는 뎁스 배열이다.
    D405 는 두 스트림이 같은 크기라 대개 그냥 통과하지만, 해상도가 다른 기종이나
    데시메이션을 켠 경우를 위해 한 곳에서 맞춰 둔다.

    최근접으로 옮긴다. 보간하면 경계에 0.5 같은 값이 생겨 어느 쪽인지 모호해진다.
    """
    if mask is None or depth_shape is None:
        return None
    array = np.asarray(mask)
    if array.ndim != 2 or array.size == 0:
        return None
    if array.shape == tuple(depth_shape):
        return array.astype(bool)
    resized = cv2.resize(
        array.astype(np.uint8),
        (int(depth_shape[1]), int(depth_shape[0])),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


# ===== 매칭점 세 곳으로 평면 세우기 =====
# 점 하나의 대표 뎁스를 뽑는 데 필요한 최소 표본. 면 전체(MIN_SURFACE_SAMPLES)
# 보다 훨씬 작게 둔다 — 매칭점은 원래 좁은 자리이기 때문이다.
MIN_POINT_SAMPLES = 10

# ===== 퇴화 판정 — 완전 일직선만 막는다 =====
# 260907: 문턱을 두 번 낮추다 결국 없앴다. 경과를 남겨 둔다.
#
#   1차 sine 0.15 (약 8.6도) : 너무 빡빡했고 **기준 자체가 틀렸다.** sine 은
#     삼각형이 얼마나 얇은지만 보고 실제 크기를 무시한다 — 밑변 200mm/높이 20mm
#     (충분)와 밑변 5mm/높이 0.5mm(못 씀)를 같게 취급한다. 실측에서 수직 높이
#     20mm·오차 0.00도인 배치를 거부하고 있었다.
#   2차 수직 거리 3mm : 기준은 옳아졌지만 여전히 막는 쪽이었다.
#   3차 (지금) 안 막는다 : 사용자 결정. 평면이 **수학적으로 정해지지 않는**
#     경우만 거절한다.
#
# 대신 벌어진 정도를 재서 문구로 알린다(normal_point_spread_mm). 막지 않는다면
# 나쁜 배치인 것을 알려 주기라도 해야 한다 — 잡음이 각도로 증폭된 결과는
# 그럴듯한 숫자로 나오므로 눈으로는 못 걸러낸다.
#
# 왜 벌어진 정도가 품질인가: 세 번째 점이 나머지 둘을 잇는 직선에서 벗어난
# 거리 h 가 법선의 지렛대다. 뎁스 잡음 σ 가 h 로 나뉘어 각도 오차가 된다
# (≈ σ/h 라디안). σ=0.5mm 기준으로 h=20mm → 약 1.4도, h=5mm → 약 5.7도,
# h=3mm → 약 9.5도, h=1mm → 약 29도.
#
# 이 값 아래로는 문구에 주의를 붙인다. **거절하지는 않는다.**
NORMAL_POINT_SPREAD_CAUTION_MM = 5.0

# 여기부터가 유일한 거절 조건이다. '거의 일직선'을 잡아내는 하한이고
# **품질 문턱이 아니다** — 이 위로는 아무리 좁아도 통과시킨다.
#
# 왜 외적에 epsilon 을 걸지 않는가 (260907 실측):
#   point_from_mask 가 x·y·z **좌표별 중앙값**을 쓰므로, 물리적으로 완전한
#   일직선이어도 세 점이 수치적으로는 미세하게 어긋난다. 그 어긋남으로 나온
#   |a×b| 가 1e-9 보다 한참 커서 외적 epsilon 은 절대 걸리지 않고, 어긋난
#   방향으로 법선이 잡힌다. 실측에서 **일직선 배치가 통과하며 법선이 51.56도
#   틀렸다.** 그래서 판정은 mm 로 해야 한다.
#
# 1.0mm 로 둔 근거 (합성 장면, 뎁스 잡음 없음):
#   벌어짐 0.000mm(일직선) → 오차 51.56도   ← 반드시 막아야 한다
#   벌어짐 2.612mm(1px)   → 오차  0.05도   ← 막을 이유가 없다
#   실제 뎁스에는 잡음이 있어 좁은 배치가 이만큼 좋지는 않다. 그래도 막지
#   않는다 — 좁다는 사실은 문구로 알린다(NORMAL_POINT_SPREAD_CAUTION_MM).
NORMAL_POINT_MIN_SPREAD_MM = 1.0


def point_from_mask(mask, depth, matrix, coeffs, depth_scale_mm=1.0):
    """마스크 안 뎁스의 대표 3D 점. 표본이 모자라면 None.

    좌표마다 중앙값을 쓴다. 마스크 경계가 배경으로 조금 새거나 구멍 너머
    바닥이 몇 점 섞여도 값이 끌려가지 않게 하려는 것이다
    (hole_pose._plane_offset 이 같은 이유로 중앙값을 쓴다).
    """
    rows, cols = np.nonzero(mask & (depth > 0))
    if rows.size < MIN_POINT_SAMPLES:
        return None
    depths_mm = depth[rows, cols].astype(np.float64) * float(depth_scale_mm)
    normalized = cv2.undistortPoints(
        np.column_stack((cols, rows)).astype(np.float64).reshape(-1, 1, 2),
        matrix,
        coeffs,
    ).reshape(-1, 2)
    points = np.column_stack(
        (normalized[:, 0] * depths_mm, normalized[:, 1] * depths_mm, depths_mm)
    )
    return np.median(points, axis=0)


def _normal_points_3d(
    masks, depth, camera_matrix, dist_coeffs, color_size, depth_scale_mm
):
    """매칭점 세 곳의 대표 3D 점. 하나라도 못 구하면 None.

    평면 계산과 품질 측정이 **같은 점**을 보게 하려고 한 곳에 둔다. 따로
    구하면 한쪽만 고쳤을 때 문구와 실제 계산이 조용히 어긋난다.
    """
    if depth is None or camera_matrix is None or not masks:
        return None
    if len(masks) != 3 or any(mask is None for mask in masks):
        return None
    filtered = np.asarray(depth)
    if filtered.ndim != 2 or filtered.size == 0:
        return None
    if any(np.asarray(mask).shape != filtered.shape for mask in masks):
        return None

    matrix, _scale_x, _scale_y = depth_frame_intrinsics(
        np.asarray(camera_matrix, dtype=np.float64), filtered, color_size
    )
    coeffs = (
        np.zeros(5)
        if dist_coeffs is None
        else np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    )

    points = []
    for mask in masks:
        point = point_from_mask(
            np.asarray(mask), filtered, matrix, coeffs, depth_scale_mm
        )
        if point is None:
            return None
        points.append(point)
    return points


def plane_from_normal_points(
    masks, depth, camera_matrix, dist_coeffs=None, color_size=None, depth_scale_mm=1.0
):
    """매칭점 세 곳의 대표 3D 점으로 평면을 세운다.

    점이 셋이므로 평면은 **유일하게** 정해진다 — 최소제곱 근사가 끼어들지
    않는다. 그래서 세 점이 실제로 같은 면에 있지 않으면 결과가 조용히
    나빠지는 대신 기울기로 곧장 드러난다(그것이 셋으로 고정한 이유다).

    Args:
        masks: 매칭점마다의 bool 마스크 세 개. **뎁스 해상도**여야 한다
            (부르는 쪽이 mask_to_depth_size 로 옮겨 준다). None 이 섞이면 실패.
        depth: 원시 뎁스, camera_matrix: 컬러 내부 파라미터
        color_size: (w, h) — 뎁스와 다르면 내부 파라미터를 비율로 맞춘다

    Returns:
        (normal, centroid) — 카메라 기준, mm. hole_pose 가 그대로 받는 형태다.
        세울 수 없으면 (None, 사유) 가 아니라 그냥 None 이다 — 사유는
        부르는 쪽이 세 마스크를 보면 알 수 있다.
    """
    points = _normal_points_3d(
        masks, depth, camera_matrix, dist_coeffs, color_size, depth_scale_mm
    )
    if points is None:
        return None

    first, second, third = points
    normal = np.cross(second - first, third - first)
    scale = float(np.linalg.norm(normal))
    if scale <= 0.0 or _spread_from(points) < NORMAL_POINT_MIN_SPREAD_MM:
        # 거의 일직선이다. 법선이 그 직선 둘레로 돌아 방향이 정해지지 않는다.
        # **여기가 유일한 거절 조건이다** — 좁아서 품질이 나쁜 배치는 막지 않고
        # 벌어진 정도를 문구로 알린다(normal_point_spread_mm).
        return None
    normal = normal / scale

    centroid = np.mean(points, axis=0)
    # 카메라 쪽으로 향한 법선이라는 규약을 맞춘다(NORMAL_TOWARD_CAMERA).
    if (float(normal @ centroid) > 0.0) == NORMAL_TOWARD_CAMERA:
        normal = -normal
    return (normal, centroid)


def normal_point_spread_mm(
    masks, depth, camera_matrix, dist_coeffs=None, color_size=None, depth_scale_mm=1.0
):
    """세 점이 얼마나 벌어져 있나 (mm). 못 재면 None.

    삼각형의 **가장 짧은 높이**다 — 가장 긴 변을 밑변으로 잡았을 때의 높이이고,
    그게 법선의 지렛대다. 뎁스 잡음 σ 가 이 길이로 나뉘어 각도 오차가 된다.

    평면을 세울지 말지는 이 값으로 정하지 않는다(완전 일직선만 막는다).
    이 값은 '지금 배치가 얼마나 미더운지'를 사람에게 알리는 데만 쓴다.
    """
    points = _normal_points_3d(
        masks, depth, camera_matrix, dist_coeffs, color_size, depth_scale_mm
    )
    if points is None:
        return None
    return _spread_from(points)


def _spread_from(points):
    """세 3D 점의 가장 짧은 높이 (mm).

    거절 판정과 문구가 **같은 수식**을 쓰게 하려고 한 곳에 둔다. 따로 적으면
    한쪽만 고쳤을 때 '문구는 괜찮다는데 거절당한다'가 된다.
    """
    first, second, third = points
    sides = (
        float(np.linalg.norm(second - first)),
        float(np.linalg.norm(third - first)),
        float(np.linalg.norm(third - second)),
    )
    longest = max(sides)
    if longest <= 0.0:
        return 0.0
    # |a x b| = (밑변) x (그 변에 대한 높이). 가장 긴 변을 밑변으로 잡으면
    # 삼각형의 가장 짧은 높이가 나온다 — 그게 법선의 지렛대다.
    return float(np.linalg.norm(np.cross(second - first, third - first))) / longest




