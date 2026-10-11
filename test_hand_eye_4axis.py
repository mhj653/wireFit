"""4축(rx = ry = 0) EIH 캘리브레이션 풀이 검증.

정답 cam2gripper 를 정해 두고, 4축 자세로 찍었을 때의 체커보드 코너를 투영해
만든 뒤 풀이가 정답을 되찾는지 본다.
"""

import math
import unittest

import cv2
import numpy as np

import hand_eye
from hand_eye import pose_to_matrix

COLS, ROWS, SQUARE_MM = 9, 6, 25.0
IMAGE_SIZE = (1280, 720)
CAMERA_MATRIX = np.array([[450.0, 0.0, 640.0], [0.0, 450.0, 360.0], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)

# 카메라는 아래(-Z)를 본다: X축 둘레 180도에 장착 오차를 조금 얹는다.
TRUE_CAM2GRIPPER = pose_to_matrix((42.0, -27.0, -65.0, 178.2, 1.3, 33.0))
# 보드는 베이스 기준 높이 12.5mm 에 수평으로 놓여 있고, 기준 자세의 카메라
# 바로 아래에 보드 중심이 오게 둔다(그래야 모든 자세에서 화면 안에 든다).
BOARD_HEIGHT_MM = 12.5
NOMINAL_POSE = (420.0, -20.0, 310.0, 0.0, 0.0, 0.0)


def _board_in_base():
    camera = pose_to_matrix(NOMINAL_POSE) @ TRUE_CAM2GRIPPER
    axis = camera[:3, 2]
    reach = (BOARD_HEIGHT_MM - camera[2, 3]) / axis[2]
    center = camera[:3, 3] + reach * axis
    board = pose_to_matrix((0.0, 0.0, BOARD_HEIGHT_MM, 0.0, 0.0, 8.0))
    half = np.array([(COLS - 1) * SQUARE_MM / 2, (ROWS - 1) * SQUARE_MM / 2, 0.0])
    board[:2, 3] = center[:2] - (board[:3, :3] @ half)[:2]
    return board


BOARD_IN_BASE = _board_in_base()


def _scara_poses():
    """같은 RZ에서 X·Y로 옮긴 자세 + RZ를 돌린 자세를 섞는다(현장 권장 구성)."""
    poses = []
    for rz in (-30.0, 0.0, 30.0):
        for dx, dy in ((0.0, 0.0), (25.0, 0.0), (0.0, 25.0)):
            poses.append((420.0 + dx, -20.0 + dy, 310.0, 0.0, 0.0, rz))
    return poses


def _samples(poses, noise_px=0.0, seed=0):
    rng = np.random.default_rng(seed)
    board = hand_eye.board_object_points(COLS, ROWS, SQUARE_MM).astype(np.float64)
    samples = []
    for pose in poses:
        target2cam = (
            np.linalg.inv(pose_to_matrix(pose) @ TRUE_CAM2GRIPPER) @ BOARD_IN_BASE
        )
        in_camera = board @ target2cam[:3, :3].T + target2cam[:3, 3]
        assert (in_camera[:, 2] > 0).all(), "보드가 카메라 뒤에 있다"
        rvec = cv2.Rodrigues(target2cam[:3, :3])[0]
        corners, _ = cv2.projectPoints(
            board, rvec, target2cam[:3, 3], CAMERA_MATRIX, DIST
        )
        corners = corners.reshape(-1, 2)
        assert (corners >= 0).all() and (corners < IMAGE_SIZE).all(), "화면 밖"
        corners = corners + rng.normal(0.0, noise_px, corners.shape)
        samples.append({
            "pose": pose,
            "corners": corners.reshape(-1, 1, 2).astype(np.float32),
        })
    return samples


def _rotation_error_deg(actual, expected):
    relative = actual[:3, :3].T @ expected[:3, :3]
    return math.degrees(np.linalg.norm(cv2.Rodrigues(relative)[0]))


def _calibrate(samples, board_height_mm=None):
    return hand_eye.calibrate_eye_in_hand(
        samples, COLS, ROWS, SQUARE_MM,
        camera_matrix=CAMERA_MATRIX, dist_coeffs=DIST,
        robot_type=hand_eye.ROBOT_4AXIS, board_height_mm=board_height_mm,
    )


class HandEye4AxisTest(unittest.TestCase):
    def test_recovers_mount_with_board_height(self):
        result = _calibrate(_samples(_scara_poses()), BOARD_HEIGHT_MM)
        found = result["cam2gripper"]

        self.assertLess(_rotation_error_deg(found, TRUE_CAM2GRIPPER), 1e-4)
        np.testing.assert_allclose(found[:3, 3], TRUE_CAM2GRIPPER[:3, 3], atol=1e-3)
        self.assertLess(result["consistency"]["max_mm"], 1e-3)
        self.assertEqual(result["robot_type"], hand_eye.ROBOT_4AXIS)
        self.assertEqual(
            result["z_reference"]["source"], hand_eye.Z_SOURCE_BOARD_HEIGHT
        )
        self.assertEqual(hand_eye.assess_quality(result)["level"], "good")

    def test_without_board_height_z_is_left_zero_and_flagged(self):
        result = _calibrate(_samples(_scara_poses()))
        found = result["cam2gripper"]

        # 회전과 X·Y는 그대로 맞고, Z만 0으로 남는다.
        self.assertLess(_rotation_error_deg(found, TRUE_CAM2GRIPPER), 1e-4)
        np.testing.assert_allclose(found[:2, 3], TRUE_CAM2GRIPPER[:2, 3], atol=1e-3)
        self.assertEqual(found[2, 3], 0.0)
        self.assertEqual(result["z_reference"]["source"], hand_eye.Z_SOURCE_UNSET)
        self.assertEqual(hand_eye.assess_quality(result)["level"], "fair")

    def test_rotation_only_poses_are_enough(self):
        # 모든 자세의 RZ가 달라 평행 이동 쌍이 하나도 없어도 풀린다.
        poses = [
            (420.0 + 10.0 * k, -20.0 - 7.0 * k, 310.0, 0.0, 0.0, -30.0 + 15.0 * k)
            for k in range(5)
        ]
        result = _calibrate(_samples(poses), BOARD_HEIGHT_MM)
        self.assertLess(_rotation_error_deg(result["cam2gripper"], TRUE_CAM2GRIPPER), 1e-4)
        np.testing.assert_allclose(
            result["cam2gripper"][:3, 3], TRUE_CAM2GRIPPER[:3, 3], atol=1e-3
        )

    def test_noise_stays_small(self):
        result = _calibrate(_samples(_scara_poses(), noise_px=0.2), BOARD_HEIGHT_MM)
        found = result["cam2gripper"]
        # 0.2px 잡음에서 기존 6축 풀이도 0.1~0.4도 흔들린다(PnP 기울기 잡음).
        self.assertLess(_rotation_error_deg(found, TRUE_CAM2GRIPPER), 1.0)
        self.assertLess(
            np.linalg.norm(found[:3, 3] - TRUE_CAM2GRIPPER[:3, 3]), 1.5
        )

    def test_same_rz_everywhere_is_rejected(self):
        poses = [p for p in _scara_poses() if p[5] == 0.0]
        with self.assertRaises(RuntimeError):
            _calibrate(_samples(poses))

    def test_tilted_pose_is_rejected(self):
        samples = _samples(_scara_poses())
        samples[2]["pose"] = samples[2]["pose"][:3] + (0.5, 0.0) + samples[2]["pose"][5:]
        with self.assertRaises(ValueError):
            _calibrate(samples)

    def test_six_axis_path_unchanged(self):
        # 6축 풀이는 기존대로: 4축 결과 키가 붙는 것 말고는 그대로다.
        poses = [
            (420.0, -20.0, 310.0, 5.0, -4.0, 0.0),
            (440.0, -10.0, 300.0, -6.0, 3.0, 20.0),
            (400.0, -30.0, 320.0, 3.0, 7.0, -25.0),
            (430.0, 0.0, 305.0, -4.0, -6.0, 10.0),
        ]
        result = hand_eye.calibrate_eye_in_hand(
            _samples(poses), COLS, ROWS, SQUARE_MM,
            camera_matrix=CAMERA_MATRIX, dist_coeffs=DIST,
        )
        self.assertLess(_rotation_error_deg(result["cam2gripper"], TRUE_CAM2GRIPPER), 1e-4)
        np.testing.assert_allclose(
            result["cam2gripper"][:3, 3], TRUE_CAM2GRIPPER[:3, 3], atol=1e-3
        )
        self.assertEqual(result["robot_type"], hand_eye.ROBOT_6AXIS)
        self.assertIsNone(result["z_reference"])


if __name__ == "__main__":
    unittest.main()
