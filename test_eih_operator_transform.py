import json
import math
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import calibration_store
import hole_pose as hole_pose_module
from hand_eye import pose_to_matrix, rpy_zyx_deg_to_matrix
from hole_pose import (
    HolePose,
    base_target_pose,
    flange_target_matrix,
    flange_target_pose,
)


def _hole_pose(tilt_deg, azimuth_deg, distance_mm=200.0):
    """카메라 기준 홀 하나. 법선은 카메라 쪽으로 향한다(hole_pose 규약)."""
    tilt = math.radians(tilt_deg)
    azimuth = math.radians(azimuth_deg)
    normal = (
        math.sin(tilt) * math.cos(azimuth),
        math.sin(tilt) * math.sin(azimuth),
        -math.cos(tilt),
    )
    return HolePose(
        u=320.0, v=240.0, x=5.0, y=3.0, z=distance_mm,
        rx=0.0, ry=0.0, rz=0.0,
        normal=normal, tilt_deg=tilt_deg, samples=500, rms_mm=0.3,
    )


class EIHOperatorTransformTest(unittest.TestCase):
    def test_camera_hole_pose_is_transformed_to_flange_frame(self):
        camera_pose = HolePose(
            10, 20, 10.0, 20.0, 30.0, 0.0, 0.0, 0.0,
            np.array([0.0, 0.0, 1.0]), 0.0, 100, 0.1,
        )
        cam2gripper = np.array([
            [0.0, -1.0, 0.0, 100.0],
            [1.0,  0.0, 0.0, 200.0],
            [0.0,  0.0, 1.0, 300.0],
            [0.0,  0.0, 0.0,   1.0],
        ])

        result = flange_target_pose(camera_pose, cam2gripper)

        self.assertEqual((result.x, result.y, result.z), (80.0, 210.0, 330.0))
        self.assertAlmostEqual(abs(result.rx), 180.0)
        self.assertAlmostEqual(result.ry, 0.0)
        # 이 입력은 공구 z축이 플랜지 z축의 정반대(-Z)라, 어떤 스핀을 골라도
        # 총 회전각이 180도로 같다. 고를 근거가 없으므로 스핀은 0으로 두고
        # 날 것의 값이 그대로 남는다(_align_spin_to_flange 의 축퇴 처리).
        self.assertAlmostEqual(result.rz, 90.0)
        np.testing.assert_allclose(result.normal, [0.0, 0.0, -1.0], atol=1e-12)

    def test_approach_direction_survives_the_spin_choice(self):
        """어떤 장착/자세에서도 보고된 (rx,ry,rz)가 참 접근 방향을 그대로 낸다.

        예전 구현은 t_flange_target 의 ZYX 오일러 rz를 0으로 만들고 최종 rz를
        플랜지 rz로 덮어썼다. 그 rz는 공구 z축이 아니라 플랜지 z축 둘레 회전이라
        접근 방향까지 함께 돌아갔다(카메라 90도 회전 장착에서 약 14도).
        캘리브레이션이 단위행렬일 때만 증상이 숨으므로 전수로 못 박아 둔다.
        """
        mounts = [
            (0.0, 0.0, 0.0),        # 회전 없이 장착
            (0.0, 0.0, 90.0),       # 90도 돌려 장착 (예전 구현이 14도 틀어졌던 경우)
            (12.0, -35.0, 110.0),   # 임의로 비틀어 장착
            (0.0, 90.0, 0.0),       # 측면 장착
        ]
        flanges = [
            (1000.0, 200.0, 800.0, 180.0, 0.0, 45.0),
            (900.0, -150.0, 700.0, 165.0, 12.0, -70.0),
        ]
        for mount in mounts:
            cam2gripper = pose_to_matrix((30.0, -20.0, 60.0) + mount)
            for flange in flanges:
                for tilt in (0.0, 5.0, 15.0, 30.0, 50.0):
                    for azimuth in (0.0, 37.0, 90.0, 143.0, 250.0):
                        hole = _hole_pose(tilt, azimuth)
                        result = base_target_pose(flange, cam2gripper, hole)

                        # 스핀을 손대지 않은 완전한 행렬이 참값이다.
                        truth = pose_to_matrix(flange) @ flange_target_matrix(
                            hole, cam2gripper
                        )
                        reported = rpy_zyx_deg_to_matrix(
                            result.rx, result.ry, result.rz
                        )
                        message = f"mount={mount} flange={flange} tilt={tilt} az={azimuth}"
                        # 접근축(3번째 열)이 정확히 보존되어야 한다.
                        np.testing.assert_allclose(
                            reported[:, 2], truth[:3, 2], atol=1e-9, err_msg=message
                        )
                        np.testing.assert_allclose(
                            [result.x, result.y, result.z],
                            truth[:3, 3],
                            atol=1e-9,
                            err_msg=message,
                        )

    def test_spin_minimises_wrist_rotation(self):
        """고른 스핀이 총 회전각을 최소로 만든다(전수조사와 비교)."""
        cam2gripper = pose_to_matrix((30.0, -20.0, 60.0, 12.0, -35.0, 110.0))
        sweep = np.linspace(-180.0, 180.0, 3601)

        for tilt, azimuth in ((0.0, 70.0), (10.0, 45.0), (30.0, 120.0), (50.0, 200.0)):
            raw = flange_target_matrix(_hole_pose(tilt, azimuth), cam2gripper)
            chosen = hole_pose_module._align_spin_to_flange(raw)[:3, :3]

            def total_rotation_deg(matrix):
                return np.degrees(
                    np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
                )

            best = min(
                total_rotation_deg(raw[:3, :3] @ rpy_zyx_deg_to_matrix(0.0, 0.0, angle))
                for angle in sweep
            )
            self.assertAlmostEqual(
                total_rotation_deg(chosen), best, delta=0.01,
                msg=f"tilt={tilt} az={azimuth}",
            )
            # 접근축은 스핀과 무관해야 한다.
            np.testing.assert_allclose(chosen[:, 2], raw[:3, 2], atol=1e-12)

    def test_approach_offset_tilts_by_exactly_the_taught_amount(self):
        """오프셋은 공구 축 둘레 회전이라, 접근축이 정확히 그만큼 기울어진다.

        오일러 각을 더하는 방식이면 기울어지는 양이 자세에 따라 달라진다.
        여기서는 어떤 장착/자세에서도 딱 티칭한 값만큼만 움직여야 한다.
        """
        for mount in ((0.0, 0.0, 0.0), (0.86, 0.56, 171.5), (12.0, -35.0, 110.0)):
            cam2gripper = pose_to_matrix((40.0, 97.0, 168.0) + mount)
            for flange in (
                (500.0, -430.0, 1130.0, 90.0, -8.0, 2.0),
                (1000.0, 200.0, 800.0, 180.0, 0.0, 45.0),
            ):
                for tilt, azimuth in ((0.0, 0.0), (5.0, 40.0), (20.0, 200.0)):
                    hole = _hole_pose(tilt, azimuth)
                    base = base_target_pose(flange, cam2gripper, hole)
                    for offset_rx, offset_ry in ((11.0, 0.0), (0.0, -7.5), (4.0, 3.0)):
                        moved = base_target_pose(
                            flange, cam2gripper, hole, (offset_rx, offset_ry, 0.0)
                        )
                        before = rpy_zyx_deg_to_matrix(base.rx, base.ry, base.rz)
                        after = rpy_zyx_deg_to_matrix(moved.rx, moved.ry, moved.rz)

                        # 접근축이 움직인 각도는 오프셋이 만드는 각도와 같아야 한다.
                        expected = rpy_zyx_deg_to_matrix(offset_rx, offset_ry, 0.0)
                        wanted = math.degrees(
                            math.acos(
                                float(np.clip(expected[2, 2], -1.0, 1.0))
                            )
                        )
                        actual = math.degrees(
                            math.acos(
                                float(np.clip(after[:, 2] @ before[:, 2], -1.0, 1.0))
                            )
                        )
                        self.assertAlmostEqual(
                            actual, wanted, places=6,
                            msg=f"mount={mount} flange={flange} tilt={tilt} "
                                f"offset=({offset_rx},{offset_ry})",
                        )
                        # 위치는 오프셋과 무관하다 (회전만 바꾼다).
                        np.testing.assert_allclose(
                            [moved.x, moved.y, moved.z],
                            [base.x, base.y, base.z],
                            atol=1e-9,
                        )

    def test_zero_offset_changes_nothing(self):
        """오프셋 0 / None 은 예전 결과와 완전히 같다."""
        cam2gripper = pose_to_matrix((40.0, 97.0, 168.0, 0.86, 0.56, 171.5))
        flange = (500.0, -430.0, 1130.0, 90.0, -8.0, 2.0)
        hole = _hole_pose(6.0, 130.0)
        plain = base_target_pose(flange, cam2gripper, hole)
        for offset in (None, (0.0, 0.0, 0.0)):
            same = base_target_pose(flange, cam2gripper, hole, offset)
            for name in ("x", "y", "z", "rx", "ry", "rz"):
                self.assertAlmostEqual(
                    getattr(same, name), getattr(plain, name), places=12,
                    msg=f"offset={offset} field={name}",
                )

    def test_offset_survives_a_changing_measured_normal(self):
        """법선이 흔들려도 오프셋이 주는 '추가 기울기'는 그대로다.

        오일러 덧셈은 이 성질이 없다. 대상 법선이 조금 달라지면 같은 숫자를
        더해도 실제로 기울어지는 양이 달라진다.
        """
        cam2gripper = pose_to_matrix((40.0, 97.0, 168.0, 0.86, 0.56, 171.5))
        flange = (500.0, -430.0, 1130.0, 90.0, -8.0, 2.0)
        offset = (11.0, -3.0, 0.0)
        expected = rpy_zyx_deg_to_matrix(*offset)
        wanted = math.degrees(
            math.acos(float(np.clip(expected[2, 2], -1.0, 1.0)))
        )
        for tilt in (0.0, 3.0, 8.0, 15.0, 25.0):
            hole = _hole_pose(tilt, 47.0)
            base = base_target_pose(flange, cam2gripper, hole)
            moved = base_target_pose(flange, cam2gripper, hole, offset)
            before = rpy_zyx_deg_to_matrix(base.rx, base.ry, base.rz)
            after = rpy_zyx_deg_to_matrix(moved.rx, moved.ry, moved.rz)
            actual = math.degrees(
                math.acos(float(np.clip(after[:, 2] @ before[:, 2], -1.0, 1.0)))
            )
            self.assertAlmostEqual(actual, wanted, places=6, msg=f"tilt={tilt}")

    def test_loads_current_nested_calibration_format(self):
        with tempfile.TemporaryDirectory() as root:
            session = "EIH_test"
            session_dir = os.path.join(root, session)
            os.makedirs(session_dir)
            matrix = np.eye(4)
            with open(os.path.join(session_dir, "result.json"), "w", encoding="utf-8") as fp:
                json.dump({"cam2gripper": {"matrix_4x4": matrix.tolist()}}, fp)

            with patch.object(calibration_store, "calibration_dir", return_value=root):
                loaded = calibration_store.load_cam2gripper(session)

            np.testing.assert_allclose(loaded, matrix)


if __name__ == "__main__":
    unittest.main()
