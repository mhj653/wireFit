"""Eye-in-Hand(EIH) 캘리브레이션 계산.

체커보드를 고정해 두고 로봇을 움직이며 찍은 (플랜지 포즈, 카메라 영상) 짝에서
'카메라가 플랜지에 대해 어디에 어떤 자세로 붙어 있는지'(cam2gripper)를 구한다.

UI와 분리해 둔 이유는 계산만 따로 검증하고 재사용하기 위해서다.

단위/규약 (여기서 한 번만 정하고 전부 이 규약을 따른다)
  - 위치        : mm
  - 회전 입력   : rx, ry, rz 오일러각 [도], ZYX 순서 → R = Rz(rz)·Ry(ry)·Rx(rx)
  - 플랜지 포즈 : 로봇 베이스 기준 (gripper2base)

로봇 종류
  - 6축 : 자세가 (x, y, z, rx, ry, rz) 전부 바뀐다 → _solve_ax_xb
  - 4축 : (x, y, z, rz)만 있고 rx = ry = 0 → _solve_ax_xb_4axis.
          회전축이 베이스 Z 하나뿐이라 카메라의 Z 오프셋(t_z)은 움직임만으로
          원리상 정해지지 않는다. 보드 높이를 받으면 그것으로 정한다.
"""

import math

import cv2
import numpy as np

# 체커보드 코너를 더 정밀하게 잡을 때 쓰는 조건
SUBPIX_WINDOW = (11, 11)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# calibrateHandEye 가 풀리려면 서로 다른 자세가 최소 3개는 있어야 한다.
MIN_SAMPLES = 3
# 이보다 적으면 풀리기는 해도 신뢰하기 어렵다.
RECOMMENDED_SAMPLES = 5

# 로봇 종류 (result.json 의 robot_type 값이기도 하다)
ROBOT_6AXIS = "6axis"
ROBOT_4AXIS = "4axis"

# 4축 카메라 Z 오프셋을 어떻게 정했는가 (result["z_reference"]["source"])
Z_SOURCE_BOARD_HEIGHT = "board_height"
Z_SOURCE_UNSET = "unset"


# ======================================================
# 회전 변환 (ZYX 오일러각 [도] <-> 회전행렬)
# ======================================================
def rpy_zyx_deg_to_matrix(rx_deg, ry_deg, rz_deg):
    """R = Rz(rz)·Ry(ry)·Rx(rx). 각도는 도(deg)."""
    roll = math.radians(rx_deg)
    pitch = math.radians(ry_deg)
    yaw = math.radians(rz_deg)

    cos_r, sin_r = math.cos(roll), math.sin(roll)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)

    rot_x = np.array([[1, 0, 0], [0, cos_r, -sin_r], [0, sin_r, cos_r]])
    rot_y = np.array([[cos_p, 0, sin_p], [0, 1, 0], [-sin_p, 0, cos_p]])
    rot_z = np.array([[cos_y, -sin_y, 0], [sin_y, cos_y, 0], [0, 0, 1]])
    return rot_z @ rot_y @ rot_x


def matrix_to_rpy_zyx_deg(matrix):
    """회전행렬을 ZYX 오일러각 [도] (rx, ry, rz)로 되돌린다."""
    # R[2,0] = -sin(pitch). 수치 오차로 1을 살짝 넘을 수 있어 잘라 준다.
    sin_pitch = -float(np.clip(matrix[2, 0], -1.0, 1.0))
    pitch = math.asin(sin_pitch)

    # pitch 가 ±90도에 가까우면 roll 과 yaw 가 한 축으로 겹쳐(짐벌락) 나뉘지
    # 않는다. 그때는 yaw 를 0으로 두고 roll 에 몰아 준다.
    if abs(matrix[2, 0]) < 1.0 - 1e-9:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = math.atan2(-matrix[1, 2], matrix[1, 1])
        yaw = 0.0

    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def pose_to_matrix(pose):
    """(x, y, z, rx, ry, rz) → 4x4 동차변환행렬. 위치 mm, 각도 도."""
    x, y, z, rx, ry, rz = pose
    matrix = np.eye(4)
    matrix[:3, :3] = rpy_zyx_deg_to_matrix(rx, ry, rz)
    matrix[:3, 3] = (x, y, z)
    return matrix


def matrix_to_pose(matrix):
    """4x4 동차변환행렬 → (x, y, z, rx, ry, rz). 위치 mm, 각도 도."""
    rx, ry, rz = matrix_to_rpy_zyx_deg(matrix[:3, :3])
    x, y, z = matrix[:3, 3]
    return (float(x), float(y), float(z), rx, ry, rz)


# ======================================================
# 체커보드
# ======================================================
def board_object_points(cols, rows, square_size_mm):
    """체커보드 안쪽 코너의 3D 좌표(보드 평면 z=0), 단위 mm."""
    points = np.zeros((rows * cols, 3), dtype=np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return points * float(square_size_mm)


def find_board_corners(bgr_image, cols, rows):
    """체커보드 안쪽 코너를 찾는다.

    Args:
        bgr_image: BGR 이미지
        cols, rows: 안쪽 코너 개수 (가로, 세로)

    Returns:
        (N, 1, 2) float32 코너 좌표, 못 찾으면 None
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        (cols, rows),
        flags=(
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK
        ),
    )
    if not found:
        return None
    # 픽셀 단위보다 더 세밀하게 코너를 다듬는다(정확도에 크게 영향).
    return cv2.cornerSubPix(
        gray, corners, SUBPIX_WINDOW, (-1, -1), SUBPIX_CRITERIA
    )


def draw_board_corners(bgr_image, corners, cols, rows):
    """찾은 코너를 선과 점으로 그린 이미지를 돌려준다(원본은 두고 복사본에)."""
    result = bgr_image.copy()
    cv2.drawChessboardCorners(result, (cols, rows), corners, True)
    return result


# ======================================================
# 결과 품질 판정
# ======================================================
# 재투영 오차: 찾은 코너와 계산으로 되돌린 코너가 몇 픽셀 어긋나는가.
GOOD_REPROJECTION_PX = 0.5
FAIR_REPROJECTION_PX = 1.0
# 체커보드 위치 편차: 고정된 보드가 장면마다 몇 mm씩 다른 곳으로 계산되는가.
# 캘리브레이션이 실제로 맞는지를 가장 직접적으로 보여 주는 값이다.
GOOD_SPREAD_MM = 2.0
FAIR_SPREAD_MM = 5.0


def assess_quality(result):
    """계산 결과를 믿을 만한지 사람 말로 판정한다.

    Returns:
        {"level": "good"|"fair"|"poor", "summary": 한 줄, "notes": [설명, ...]}
    """
    reprojection = result["reprojection_error_px"]
    spread = result["consistency"]["max_mm"]
    samples = result["sample_count"]

    notes = []
    levels = []

    if reprojection <= GOOD_REPROJECTION_PX:
        levels.append("good")
        notes.append(f"코너 재투영 오차 {reprojection:.3f}px — 양호 "
                     f"({GOOD_REPROJECTION_PX}px 이하)")
    elif reprojection <= FAIR_REPROJECTION_PX:
        levels.append("fair")
        notes.append(f"코너 재투영 오차 {reprojection:.3f}px — 보통 "
                     f"({FAIR_REPROJECTION_PX}px 이하). 초점과 조명을 확인하세요.")
    else:
        levels.append("poor")
        notes.append(f"코너 재투영 오차 {reprojection:.3f}px — 큼 "
                     f"({FAIR_REPROJECTION_PX}px 초과). 흔들림·초점·보드 평탄도를 확인하세요.")

    if spread <= GOOD_SPREAD_MM:
        levels.append("good")
        notes.append(f"체커보드 위치 편차 {spread:.2f}mm — 양호 "
                     f"({GOOD_SPREAD_MM}mm 이하)")
    elif spread <= FAIR_SPREAD_MM:
        levels.append("fair")
        notes.append(f"체커보드 위치 편차 {spread:.2f}mm — 보통 "
                     f"({FAIR_SPREAD_MM}mm 이하). 포즈 입력값을 다시 확인해 보세요.")
    else:
        levels.append("poor")
        notes.append(f"체커보드 위치 편차 {spread:.2f}mm — 큼 "
                     f"({FAIR_SPREAD_MM}mm 초과). 플랜지 포즈 입력이나 회전 규약이 "
                     f"틀렸을 수 있습니다.")

    if samples < RECOMMENDED_SAMPLES:
        levels.append("fair")
        notes.append(f"자세 {samples}개 — 권장 {RECOMMENDED_SAMPLES}개보다 적습니다. "
                     f"자세를 더 모으면 안정적입니다.")
    else:
        notes.append(f"자세 {samples}개 — 충분")

    if result.get("robot_type") == ROBOT_4AXIS:
        z_reference = result.get("z_reference") or {}
        if z_reference.get("source") == Z_SOURCE_BOARD_HEIGHT:
            notes.append(
                f"4축 — 카메라 Z 오프셋을 보드 높이 "
                f"{z_reference['board_height_mm']:.3f}mm 로 정했습니다."
            )
        else:
            # 4축 움직임만으로는 원리상 정해지지 않는 값이다. 0으로 둔 채 쓰면
            # 높이 좌표가 그만큼 통째로 어긋나므로 반드시 짚어 준다.
            levels.append("fair")
            notes.append(
                "4축 — 카메라 Z 오프셋이 보정되지 않았습니다(Z = 0으로 둠). "
                "툴 끝을 보드 면에 닿게 했을 때의 Z를 '보드 높이'로 넣고 다시 "
                "계산하세요."
            )
        if result.get("intrinsics_source") == "estimated_from_images":
            # 4축은 보드를 기울여 볼 수 없어 초점거리가 영상만으로 잘 안 잡힌다.
            levels.append("fair")
            notes.append(
                "4축 — 카메라 내부 파라미터를 이미지로 추정했습니다. 보드를 "
                "기울여 볼 수 없어 믿기 어렵습니다. 실제 카메라를 연결해 "
                "내장값을 쓰세요."
            )

    level = "poor" if "poor" in levels else ("fair" if "fair" in levels else "good")
    summary = {
        "good": "양호 — 그대로 사용해도 됩니다.",
        "fair": "보통 — 쓸 수는 있지만 다시 잡으면 더 좋아집니다.",
        "poor": "주의 — 그대로 쓰기 어렵습니다. 아래 항목을 확인하세요.",
    }[level]
    return {"level": level, "summary": summary, "notes": notes}


# ======================================================
# AX = XB 풀이 (hand-eye 본체)
# ======================================================
# 두 자세 사이 회전이 이보다 작으면 방정식에 정보를 거의 주지 못하고
# 잡음만 키우므로 뺀다.
MIN_PAIR_ROTATION_DEG = 3.0


def _solve_ax_xb(gripper_poses, target_poses):
    """AX = XB 를 풀어 X(cam2gripper)를 구한다.

    OpenCV 5의 파이썬 바인딩에는 cv2.calibrateHandEye 가 빠져 있어(상수만 남아
    있다) 직접 푼다. 방식은 널리 쓰이는 Park–Martin 계열이다.

      1) 서로 다른 두 자세 i, j 로 상대 운동을 만든다
             A = inv(g2b_j)·g2b_i ,  B = t2c_j·inv(t2c_i)
         체커보드가 고정이라는 사실에서 A·X = X·B 가 성립한다.
      2) 회전: A, B 의 회전축 벡터를 α, β 라 하면 α = R_x·β 이므로
         모든 쌍을 모아 직교 프로크루스테스(SVD)로 R_x 를 구한다.
      3) 이동: (A_R − I)·t_x = R_x·b_t − a_t 를 모든 쌍에 대해 쌓아
         최소자승으로 t_x 를 구한다.

    Args:
        gripper_poses: 4x4 gripper2base 리스트
        target_poses: 4x4 target2cam 리스트

    Returns:
        4x4 cam2gripper

    Raises:
        RuntimeError: 자세 변화가 모자라 풀 수 없을 때
    """
    axis_a, axis_b, motions = [], [], []
    count = len(gripper_poses)
    min_angle = math.radians(MIN_PAIR_ROTATION_DEG)

    for i in range(count):
        for j in range(i + 1, count):
            motion_a = np.linalg.inv(gripper_poses[j]) @ gripper_poses[i]
            motion_b = target_poses[j] @ np.linalg.inv(target_poses[i])

            vec_a = cv2.Rodrigues(motion_a[:3, :3])[0].reshape(3)
            vec_b = cv2.Rodrigues(motion_b[:3, :3])[0].reshape(3)
            # 회전이 거의 없는 쌍은 방향을 정하지 못해 잡음만 된다.
            if min(np.linalg.norm(vec_a), np.linalg.norm(vec_b)) < min_angle:
                continue

            axis_a.append(vec_a)
            axis_b.append(vec_b)
            motions.append((motion_a, motion_b))

    if len(motions) < 2:
        raise RuntimeError(
            "자세 변화가 부족합니다. 로봇을 서로 다른 방향으로 "
            f"({MIN_PAIR_ROTATION_DEG:.0f}도 이상) 회전시켜 다시 찍어 주세요."
        )

    # --- 회전: α ≈ R_x·β 를 만족하는 R_x (직교 프로크루스테스) ---
    correlation = np.zeros((3, 3))
    for vec_a, vec_b in zip(axis_a, axis_b):
        correlation += np.outer(vec_a, vec_b)

    u_mat, _s, vt_mat = np.linalg.svd(correlation)
    rotation = u_mat @ vt_mat
    if np.linalg.det(rotation) < 0:
        # 거울상(det = -1)이 나오면 회전이 아니므로 마지막 축을 뒤집는다.
        u_mat[:, -1] *= -1
        rotation = u_mat @ vt_mat

    # --- 이동: (A_R − I)·t_x = R_x·b_t − a_t ---
    left = np.zeros((3 * len(motions), 3))
    right = np.zeros(3 * len(motions))
    for index, (motion_a, motion_b) in enumerate(motions):
        block = slice(3 * index, 3 * index + 3)
        left[block] = motion_a[:3, :3] - np.eye(3)
        right[block] = rotation @ motion_b[:3, 3] - motion_a[:3, 3]

    translation, *_ = np.linalg.lstsq(left, right, rcond=None)

    cam2gripper = np.eye(4)
    cam2gripper[:3, :3] = rotation
    cam2gripper[:3, 3] = translation
    return cam2gripper


# 4축 연립식이 이보다 나쁜 조건이면 자세 구성이 모자라 풀리지 않은 것으로 본다.
MIN_4AXIS_CONDITION = 1e-6


def _rotation_onto_z(direction):
    """direction(단위벡터)을 +Z로 보내는 최소 회전행렬."""
    z_axis = np.array([0.0, 0.0, 1.0])
    cosine = float(np.clip(direction @ z_axis, -1.0, 1.0))
    axis = np.cross(direction, z_axis)
    sine = float(np.linalg.norm(axis))
    if sine < 1e-12:
        # 이미 +Z 이거나 정반대(-Z). 정반대면 X축 둘레로 반 바퀴 돌린다.
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    return cv2.Rodrigues((axis / sine * math.atan2(sine, cosine)).reshape(3, 1))[0]


def _solve_ax_xb_4axis(gripper_poses, target_poses):
    """4축 로봇(rx = ry = 0)용 AX = XB 풀이. t_z 는 0으로 둔다.

    4축은 플랜지가 베이스 Z 둘레로만 돈다. 상대 운동 A 의 회전축이 언제나
    플랜지 Z 하나뿐이라 6축 풀이(회전축 쌍으로 R_x 를 맞추는 방식)로는 Z축
    둘레 1자유도가 남는다. 그래서 두 단계로 나눈다.

      1) 축 맞추기: A 의 회전축(= 플랜지 Z)은 B 의 회전축 β 와
         α = R_x·β 관계다. β 의 평균 방향을 +Z로 보내는 R0 를 구하면
         R_x = Rz(φ)·R0 꼴이고, 남은 미지수는 φ 하나다.
      2) φ 와 이동: 이동 식 (A_R − I)·t_x + t_A = R_x·b_t 의 x, y 두 줄은
         u = R0·b_t 로 두면 (t_x, t_y, cos φ, sin φ)에 대해 선형이다.
             [ (A_R−I)₀₀  (A_R−I)₀₁  −u_x   u_y ]   [t_x  ]
             [ (A_R−I)₁₀  (A_R−I)₁₁  −u_y  −u_x ] · [t_y  ] = −t_A(x,y)
                                                    [cos φ]
                                                    [sin φ]
         모든 쌍을 쌓아 최소자승으로 풀고, (cos φ, sin φ)를 단위 길이로 맞춘 뒤
         그 φ 로 t_x, t_y 를 다시 푼다.

    t_z 는 어느 식에도 나오지 않는다(A_R − I 의 세 번째 열이 0). 움직임만으로는
    정해지지 않는 값이라 0으로 두고, 부르는 쪽이 보드 높이로 채운다.

    Args:
        gripper_poses: 4x4 gripper2base 리스트 (회전은 Rz 뿐)
        target_poses: 4x4 target2cam 리스트

    Returns:
        4x4 cam2gripper (t_z = 0)

    Raises:
        RuntimeError: 자세 구성이 모자라 풀 수 없을 때
    """
    count = len(gripper_poses)
    min_angle = math.radians(MIN_PAIR_ROTATION_DEG)
    axis_sum = np.zeros(3)
    rotating_pairs = 0
    motions = []

    for i in range(count):
        for j in range(i + 1, count):
            motion_a = np.linalg.inv(gripper_poses[j]) @ gripper_poses[i]
            motion_b = target_poses[j] @ np.linalg.inv(target_poses[i])
            motions.append((motion_a, motion_b))

            vec_a = cv2.Rodrigues(motion_a[:3, :3])[0].reshape(3)
            vec_b = cv2.Rodrigues(motion_b[:3, :3])[0].reshape(3)
            if min(np.linalg.norm(vec_a), np.linalg.norm(vec_b)) < min_angle:
                continue
            # α 는 ±Z. 부호를 +Z 쪽으로 맞춰 β 를 더한다(크기 = 회전각이라
            # 많이 돈 쌍일수록 축 방향을 더 믿게 된다).
            axis_sum += vec_b if vec_a[2] > 0 else -vec_b
            rotating_pairs += 1

    if rotating_pairs == 0 or np.linalg.norm(axis_sum) < 1e-12:
        raise RuntimeError(
            "RZ 회전이 부족합니다. RZ를 서로 다른 각도로 "
            f"({MIN_PAIR_ROTATION_DEG:.0f}도 이상, 권장 ±30도) 돌린 자세를 넣어 주세요."
        )

    align = _rotation_onto_z(axis_sum / np.linalg.norm(axis_sum))

    left = np.zeros((2 * len(motions), 4))
    right = np.zeros(2 * len(motions))
    for index, (motion_a, motion_b) in enumerate(motions):
        rows = slice(2 * index, 2 * index + 2)
        u_x, u_y, _u_z = align @ motion_b[:3, 3]
        left[rows, :2] = (motion_a[:3, :3] - np.eye(3))[:2, :2]
        left[rows, 2:] = [[-u_x, u_y], [-u_y, -u_x]]
        right[rows] = -motion_a[:2, 3]

    singular = np.linalg.svd(left, compute_uv=False)
    if singular[-1] < MIN_4AXIS_CONDITION * singular[0]:
        raise RuntimeError(
            "자세 구성이 모자라 풀 수 없습니다. RZ를 돌린 자세와, RZ는 그대로 두고 "
            "X·Y로 옮긴 자세를 섞어서 찍어 주세요."
        )

    solution, *_ = np.linalg.lstsq(left, right, rcond=None)
    angle = math.atan2(solution[3], solution[2])
    cos_phi, sin_phi = math.cos(angle), math.sin(angle)

    # φ 를 단위원 위로 고정한 뒤 t_x, t_y 만 다시 푼다(φ 를 맞춘 만큼 이동도 바뀐다).
    fixed = right - left[:, 2:] @ np.array([cos_phi, sin_phi])
    translation_xy, *_ = np.linalg.lstsq(left[:, :2], fixed, rcond=None)

    cam2gripper = np.eye(4)
    cam2gripper[:3, :3] = rpy_zyx_deg_to_matrix(0.0, 0.0, math.degrees(angle)) @ align
    cam2gripper[:2, 3] = translation_xy
    return cam2gripper


def _fit_4axis_z_offset(samples, cam2gripper, target_translations, board_height_mm):
    """보드 면 높이(베이스 Z)가 board_height_mm 가 되도록 t_z 를 구한다.

    4축 플랜지 회전은 Rz 뿐이라 베이스 기준 보드 원점의 Z는
        z_flange + (R_x·t_target)_z + t_z
    로 t_z 가 그대로 더해진다. 모든 장면의 평균이 입력 높이가 되게 맞춘다.
    보드 원점은 인쇄 면 위의 코너이므로 툴 끝을 보드 면에 대고 읽은 Z와 같은 면이다.
    """
    rotation = cam2gripper[:3, :3]
    heights = [
        sample["pose"][2] + float((rotation @ translation.reshape(3))[2])
        for sample, translation in zip(samples, target_translations)
    ]
    return float(board_height_mm) - float(np.mean(heights))


# ======================================================
# 캘리브레이션
# ======================================================
def estimate_intrinsics(samples, cols, rows, square_size_mm, image_size):
    """모은 이미지들로 카메라 내부 파라미터를 추정한다.

    카메라 내장값을 못 얻는 경우(가상 카메라 모드 등)에 쓴다.

    Returns:
        (camera_matrix, dist_coeffs, rms) 또는 실패 시 (None, None, None)
    """
    object_points = board_object_points(cols, rows, square_size_mm)
    obj = [object_points for _ in samples]
    img = [sample["corners"] for sample in samples]
    try:
        rms, camera_matrix, dist, _rvecs, _tvecs = cv2.calibrateCamera(
            obj, img, image_size, None, None
        )
    except cv2.error:
        return None, None, None
    return camera_matrix, dist, float(rms)


def _target_poses_in_camera(samples, cols, rows, square_size_mm, camera_matrix, dist):
    """각 장면에서 '카메라 기준 체커보드 자세'(target2cam)를 푼다.

    Returns:
        (R 리스트, t 리스트, 재투영오차 리스트) — 실패한 장면은 제외되지 않고
        예외로 올라간다(한 장이라도 못 풀면 결과를 믿을 수 없기 때문).
    """
    object_points = board_object_points(cols, rows, square_size_mm)
    rotations, translations, errors = [], [], []

    for sample in samples:
        corners = sample["corners"]
        ok, rvec, tvec = cv2.solvePnP(
            object_points, corners, camera_matrix, dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("체커보드 자세를 풀지 못한 장면이 있습니다.")

        rotations.append(cv2.Rodrigues(rvec)[0])
        translations.append(tvec.reshape(3, 1))

        # 얼마나 잘 맞았는지: 3D 점을 다시 투영해 실제 코너와 비교한다.
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, dist
        )
        errors.append(
            float(np.sqrt(np.mean(np.sum(
                (projected.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2, axis=1
            ))))
        )

    return rotations, translations, errors


def _consistency_check(samples, cam2gripper, target_rotations, target_translations):
    """결과가 얼마나 믿을 만한지 스스로 검산한다.

    체커보드는 고정돼 있으므로, 어느 장면에서 계산하든
    '베이스 기준 체커보드 위치'는 같아야 한다. 장면마다 다르게 나온다면
    그만큼 캘리브레이션이 틀린 것이다.

    Returns:
        {"max_mm": 최대 편차, "std_mm": 표준편차} — 작을수록 좋다.
    """
    positions = []
    for sample, rotation, translation in zip(
        samples, target_rotations, target_translations
    ):
        target2cam = np.eye(4)
        target2cam[:3, :3] = rotation
        target2cam[:3, 3] = translation.reshape(3)

        gripper2base = pose_to_matrix(sample["pose"])
        target2base = gripper2base @ cam2gripper @ target2cam
        positions.append(target2base[:3, 3])

    positions = np.array(positions)
    center = positions.mean(axis=0)
    distances = np.linalg.norm(positions - center, axis=1)
    return {
        "max_mm": float(distances.max()),
        "std_mm": float(distances.std()),
    }


def calibrate_eye_in_hand(
    samples, cols, rows, square_size_mm, camera_matrix=None, dist_coeffs=None,
    image_size=None, robot_type=ROBOT_6AXIS, board_height_mm=None,
):
    """Eye-in-Hand 캘리브레이션을 수행한다.

    Args:
        samples: [{"pose": (x,y,z,rx,ry,rz), "corners": (N,1,2)}, ...]
            4축이면 rx = ry = 0 이어야 한다.
        cols, rows: 체커보드 안쪽 코너 개수
        square_size_mm: 한 칸 크기 (mm)
        camera_matrix, dist_coeffs: 카메라 내부 파라미터. None이면 이미지로 추정.
        image_size: (width, height) — 내부 파라미터를 추정할 때만 필요
        robot_type: ROBOT_6AXIS 또는 ROBOT_4AXIS
        board_height_mm: 4축 전용. 툴 끝을 보드 면에 댔을 때의 베이스 Z.
            None이면 카메라 Z 오프셋을 0으로 둔다(움직임만으로는 정해지지 않는다).

    Returns:
        결과 dict

    Raises:
        ValueError: 표본이 모자라거나 입력이 로봇 종류와 맞지 않을 때
        RuntimeError: 계산이 실패했을 때
    """
    if robot_type not in (ROBOT_6AXIS, ROBOT_4AXIS):
        raise ValueError(f"알 수 없는 로봇 종류입니다: {robot_type}")
    if len(samples) < MIN_SAMPLES:
        raise ValueError(
            f"자세가 최소 {MIN_SAMPLES}개는 있어야 합니다 (현재 {len(samples)}개)."
        )
    if robot_type == ROBOT_4AXIS:
        for number, sample in enumerate(samples, start=1):
            _x, _y, _z, rx, ry, _rz = sample["pose"]
            if abs(rx) > 1e-9 or abs(ry) > 1e-9:
                # 6축으로 모은 자세가 섞였다. 그대로 풀면 조용히 틀린 값이 나온다.
                raise ValueError(
                    f"{number}번 자세에 RX/RY 값이 있습니다. 4축 캘리브레이션에는 "
                    "RX = RY = 0 인 자세만 쓸 수 있습니다."
                )

    intrinsics_source = "provided"
    intrinsics_rms = None
    if camera_matrix is None:
        if image_size is None:
            raise RuntimeError("내부 파라미터를 추정하려면 이미지 크기가 필요합니다.")
        camera_matrix, dist_coeffs, intrinsics_rms = estimate_intrinsics(
            samples, cols, rows, square_size_mm, image_size
        )
        if camera_matrix is None:
            raise RuntimeError("카메라 내부 파라미터를 추정하지 못했습니다.")
        intrinsics_source = "estimated_from_images"

    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1))

    target_rotations, target_translations, reprojection_errors = _target_poses_in_camera(
        samples, cols, rows, square_size_mm, camera_matrix, dist_coeffs
    )

    gripper_poses = [pose_to_matrix(sample["pose"]) for sample in samples]

    target_poses = []
    for rotation, translation in zip(target_rotations, target_translations):
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation.reshape(3)
        target_poses.append(matrix)

    z_reference = None
    if robot_type == ROBOT_4AXIS:
        cam2gripper = _solve_ax_xb_4axis(gripper_poses, target_poses)
        if board_height_mm is None:
            z_reference = {"source": Z_SOURCE_UNSET, "board_height_mm": None}
        else:
            cam2gripper[2, 3] = _fit_4axis_z_offset(
                samples, cam2gripper, target_translations, board_height_mm
            )
            z_reference = {
                "source": Z_SOURCE_BOARD_HEIGHT,
                "board_height_mm": float(board_height_mm),
            }
    else:
        cam2gripper = _solve_ax_xb(gripper_poses, target_poses)

    consistency = _consistency_check(
        samples, cam2gripper, target_rotations, target_translations
    )

    return {
        "cam2gripper": cam2gripper,
        "pose": matrix_to_pose(cam2gripper),
        "camera_matrix": np.asarray(camera_matrix),
        "dist_coeffs": np.asarray(dist_coeffs).reshape(-1),
        "intrinsics_source": intrinsics_source,
        "intrinsics_rms_px": intrinsics_rms,
        "reprojection_error_px": float(np.mean(reprojection_errors)),
        "reprojection_error_max_px": float(np.max(reprojection_errors)),
        "consistency": consistency,
        "sample_count": len(samples),
        "robot_type": robot_type,
        "z_reference": z_reference,
    }
