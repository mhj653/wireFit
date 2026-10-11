"""매칭점 세 곳으로 세운 평면이 옳은가.

대조군은 이 파일 안에 따로 적어 둔 최소제곱 평면(_reference_plane)이다.
같은 모듈의 함수와 견주면 그 모듈이 함께 틀렸을 때 둘이 나란히 틀린 값을 내고
시험은 통과해 버린다. 구현을 갈라 둬야 진짜 대조가 된다.
"""

import io
import json
import math
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from normal_plane import (
    NORMAL_POINT_MIN_SPREAD_MM,
    NORMAL_POINT_SPREAD_CAUTION_MM,
    normal_point_spread_mm,
    plane_from_normal_points,
)
# torch 가 없는 PC 에서도 이 시험은 돌아야 한다. 아래 둘은 모델을 만지지 않는
# 순수 판정/저장 코드라, sam2_video_tracker 가 torch 없이 import 되는 덕을 본다.
from hole_pose import (
    NORMAL_MODE_FLANGE,
    NORMAL_MODE_POINTS,
    NORMAL_MODE_RING,
    PLANE_FROM_CAMERA,
    PLANE_FROM_RING,
    PLANE_FROM_SURFACE,
    base_target_pose,
    estimate_hole_poses,
    ring_outer_px,
)
from pose_pipeline import normal_source_from_settings
from sam2_video_tracker import NORMAL_POINT_MIN_AREA_PX, _min_area_for
from template_store import load_normal_points, save_normal_points

WIDTH, HEIGHT = 160, 120
CAMERA_MATRIX = np.array(
    [[200.0, 0.0, WIDTH / 2.0], [0.0, 200.0, HEIGHT / 2.0], [0.0, 0.0, 1.0]]
)


def plane_depth(normal, through_mm):
    """법선 normal, 점 through_mm 을 지나는 평면을 뎁스맵으로 렌더한다."""
    normal = np.asarray(normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    offset = float(normal @ np.asarray(through_mm, dtype=np.float64))

    cols, rows = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    xn = (cols - CAMERA_MATRIX[0, 2]) / CAMERA_MATRIX[0, 0]
    yn = (rows - CAMERA_MATRIX[1, 2]) / CAMERA_MATRIX[1, 1]
    # 시선 (xn, yn, 1) 과 평면의 교점. z = t 이므로 t 를 그대로 뎁스로 쓴다.
    denominator = xn * normal[0] + yn * normal[1] + normal[2]
    denominator[np.abs(denominator) < 1e-9] = np.nan
    depth = offset / denominator
    depth[~np.isfinite(depth)] = 0.0
    depth[depth <= 0.0] = 0.0
    return depth


def box_mask(left, top, right, bottom):
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


def _reference_plane(mask, depth):
    """대조군: 마스크 안 뎁스 전부에 최소제곱 평면을 맞춘다 (SVD).

    normal_plane 은 점 셋의 외적을 쓴다. 여기는 수천 점의 최소제곱이라
    완전히 다른 길로 같은 답에 닿아야 한다.
    """
    rows, cols = np.nonzero(mask & (depth > 0))
    depths = depth[rows, cols].astype(np.float64)
    normalized = cv2.undistortPoints(
        np.column_stack((cols, rows)).astype(np.float64).reshape(-1, 1, 2),
        CAMERA_MATRIX,
        np.zeros(5),
    ).reshape(-1, 2)
    points = np.column_stack(
        (normalized[:, 0] * depths, normalized[:, 1] * depths, depths)
    )
    centroid = points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[2] / np.linalg.norm(vt[2])
    # normal_plane 과 같은 부호 규약(카메라 쪽으로 향한다)으로 맞춘다.
    if float(normal @ centroid) > 0.0:
        normal = -normal
    return normal, centroid


def angle_between(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    cosine = float(first @ second) / (np.linalg.norm(first) * np.linalg.norm(second))
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


class ThreePointPlaneTest(unittest.TestCase):
    def setUp(self):
        # 눕지도 정면도 아닌 기울기. 정면(0,0,1)이면 부호 실수를 못 잡는다.
        self.normal = np.array([0.25, -0.15, 1.0])
        self.normal /= np.linalg.norm(self.normal)
        self.depth = plane_depth(self.normal, (0.0, 0.0, 500.0))
        # 삼각형으로 넉넉히 벌려 둔 세 자리
        self.masks = [
            box_mask(20, 20, 40, 40),
            box_mask(115, 25, 135, 45),
            box_mask(65, 80, 85, 100),
        ]

    def _plane(self, masks=None):
        return plane_from_normal_points(
            self.masks if masks is None else masks,
            self.depth,
            CAMERA_MATRIX,
            depth_scale_mm=1.0,
        )

    def test_matches_an_independent_least_squares_fit(self):
        """다른 길로 구한 최소제곱 평면과 같은 법선·같은 부호가 나와야 한다."""
        whole = _reference_plane(
            box_mask(10, 10, WIDTH - 10, HEIGHT - 10), self.depth
        )
        three = self._plane()
        self.assertIsNotNone(three)

        # 부호까지 같아야 한다 — 뒤집혀 있으면 각도가 180도로 나온다.
        self.assertLess(angle_between(whole[0], three[0]), 0.5)
        # 세 점의 중심은 면 전체의 중심과 다를 수 있다. 다만 같은 평면 위여야
        # 하므로, 중심의 차이를 법선에 투영하면 0 에 가까워야 한다.
        self.assertLess(abs(float((three[1] - whole[1]) @ three[0])), 1.0)

    def test_recovers_the_true_normal(self):
        three = self._plane()
        self.assertIsNotNone(three)
        angle = angle_between(three[0], self.normal)
        # 규약이 부호를 뒤집을 수 있으니 축만 견준다.
        self.assertLess(min(angle, 180.0 - angle), 0.5)

    def test_exactly_collinear_is_refused(self):
        """완전 일직선이면 법선이 그 직선 둘레로 자유롭게 돌아 정해지지 않는다."""
        collinear = [
            box_mask(20, 55, 40, 65),
            box_mask(70, 55, 90, 65),
            box_mask(120, 55, 140, 65),
        ]
        self.assertIsNone(self._plane(collinear))

    def test_a_very_thin_triangle_is_still_accepted(self):
        """260907 2차 완화. **완전 일직선이 아니면 막지 않는다** (사용자 결정).

        세로로 1px 만 벗어난 배치(높이 약 2.6mm)다. 잡음이 각도로 크게
        증폭되는 자리지만, 평면은 수학적으로 정해지므로 거절하지 않는다.
        대신 벌어짐이 작다는 것을 재서 문구로 알린다.
        """
        thin = [
            box_mask(15, 55, 25, 65),
            box_mask(75, 55, 85, 65),
            box_mask(45, 56, 55, 66),
        ]
        self.assertIsNotNone(self._plane(thin), "일직선이 아닌데 거절했다")

        spread = normal_point_spread_mm(
            thin, self.depth, CAMERA_MATRIX, depth_scale_mm=1.0
        )
        self.assertIsNotNone(spread)
        # 막지 않는 대신 '좁다'는 것이 값으로 드러나야 한다.
        self.assertLess(spread, NORMAL_POINT_SPREAD_CAUTION_MM)

    def test_spread_grows_with_how_far_the_points_are_apart(self):
        """벌어짐이 실제 배치를 따라가는가. 문구가 이 값을 그대로 쓴다."""
        wide = normal_point_spread_mm(
            self.masks, self.depth, CAMERA_MATRIX, depth_scale_mm=1.0
        )
        thin = normal_point_spread_mm(
            [
                box_mask(15, 55, 25, 65),
                box_mask(75, 55, 85, 65),
                box_mask(45, 56, 55, 66),
            ],
            self.depth,
            CAMERA_MATRIX,
            depth_scale_mm=1.0,
        )
        self.assertGreater(wide, NORMAL_POINT_SPREAD_CAUTION_MM)
        self.assertGreater(wide, thin * 10)

    def test_a_thin_but_usable_triangle_is_accepted(self):
        """260907 완화. 예전 sine(8.6도) 기준이 거부하던 배치다.

        수직 높이가 20mm 나 되는데도 거부당했다. 실제로는 오차 0.00도로
        정확하다 — 각도로 재던 기준이 삼각형의 크기를 무시했기 때문이다.
        """
        thin = [
            box_mask(15, 55, 25, 65),
            box_mask(70, 47, 80, 57),
            box_mask(125, 55, 135, 65),
        ]
        plane = self._plane(thin)
        self.assertIsNotNone(plane, "얇지만 쓸 수 있는 배치를 거부했다")
        angle = angle_between(plane[0], self.normal)
        self.assertLess(min(angle, 180.0 - angle), 1.0)

    def test_closely_spaced_points_are_accepted(self):
        """260907 완화. 예전 간격 기준(10mm)이 거부하던 배치다.

        높이 7.7mm 에 오차 0.01도. 좁아도 평면은 제대로 정해진다.
        """
        crowded = [
            box_mask(69, 57, 75, 63),
            box_mask(73, 57, 79, 63),
            box_mask(71, 60, 77, 66),
        ]
        self.assertIsNotNone(self._plane(crowded))

    def test_well_spread_points_are_accepted(self):
        """문턱이 정상 배치를 막지 않는지도 함께 못박아 둔다."""
        self.assertIsNotNone(self._plane())

    def test_a_missing_point_fails_closed(self):
        """여분이 없다. 하나만 빠져도 평면을 내지 않아야 한다."""
        for index in range(3):
            masks = list(self.masks)
            masks[index] = None
            self.assertIsNone(self._plane(masks))

    def test_wrong_count_is_refused(self):
        self.assertIsNone(self._plane(self.masks[:2]))
        self.assertIsNone(self._plane(self.masks + [box_mask(0, 0, 10, 10)]))

    def test_a_point_off_the_plane_tilts_the_result(self):
        """넷 이상을 최소제곱하지 않는 이유. 셋이면 어긋남이 기울기로 드러난다."""
        depth = self.depth.copy()
        # 세 번째 자리만 20mm 들어올린다 (다른 단차에 올라간 상황)
        raised = self.masks[2]
        depth[raised] = depth[raised] - 20.0
        tilted = plane_from_normal_points(
            self.masks, depth, CAMERA_MATRIX, depth_scale_mm=1.0
        )
        self.assertIsNotNone(tilted)
        # 조용히 넘어가지 않고 각도가 분명히 달라져야 한다.
        self.assertGreater(angle_between(tilted[0], self.normal), 5.0)

    def test_only_near_collinearity_is_a_hard_stop(self):
        """거절 조건은 **하나뿐**이라는 것을 못박아 둔다 (사용자 결정).

        1px(약 2.6mm) 만 벗어나도 통과한다. 예전 문턱(3mm/10mm/sine 8.6도)을
        되살리려면 이 시험이 먼저 실패하므로 실수로 되돌아가지 않는다.
        """
        for offset in (1, 2, 4, 8):
            masks = [
                box_mask(15, 55, 25, 65),
                box_mask(75, 55, 85, 65),
                box_mask(45, 55 + offset, 55, 65 + offset),
            ]
            self.assertIsNotNone(
                self._plane(masks), f"{offset}px 벗어난 배치를 거절했다"
            )

    def test_thresholds_are_documented_numbers(self):
        """두 값의 **역할이 다르다**는 것을 고정해 둔다.

        MIN_SPREAD 는 거의 일직선을 잡아내는 유일한 거절 조건이고,
        CAUTION 은 거절하지 않고 문구에 주의만 붙이는 값이다.
        """
        self.assertEqual(NORMAL_POINT_MIN_SPREAD_MM, 1.0)
        self.assertEqual(NORMAL_POINT_SPREAD_CAUTION_MM, 5.0)
        # 거절선이 주의선보다 낮아야 한다 — 아니면 주의 문구가 나올 일이 없다.
        self.assertLess(NORMAL_POINT_MIN_SPREAD_MM, NORMAL_POINT_SPREAD_CAUTION_MM)


class TrackedAreaGateTest(unittest.TestCase):
    """'제대로 잡혔나'를 티칭 넓이에 견주는가 (절대 픽셀 수가 아니라).

    260908 실측이 이 시험을 낳았다. 절대 문턱 200px 을 쓰던 때, 티칭 넓이
    204px 인 점이 추적에서 184px 로 나와 **16px 차이로 매번 탈락**했다.
    추적은 멀쩡히 그 자리를 찾고 있었는데도 그랬고, 세 곳이 다 있어야 평면이
    서므로 그 점 하나가 촬영마다 법선을 통째로 죽였다.
    """

    def test_absolute_floor_is_the_fallback_not_the_rule(self):
        """티칭 넓이를 모르면 절대 하한만 쓴다. 그 값이 예전 200 이면 안 된다."""
        self.assertEqual(_min_area_for(None), NORMAL_POINT_MIN_AREA_PX)
        self.assertEqual(_min_area_for(0), NORMAL_POINT_MIN_AREA_PX)
        # 이 값이 200 으로 되돌아가면 위 실측 사례가 그대로 재발한다.
        self.assertLess(NORMAL_POINT_MIN_AREA_PX, 184)

    def test_gate_scales_with_the_taught_area(self):
        """큰 점은 크게, 작은 점은 작게 — 그게 상대 기준의 요점이다."""
        self.assertEqual(_min_area_for(1000), 250)
        self.assertEqual(_min_area_for(4000), 1000)

    def test_gate_survives_the_camera_backing_off(self):
        """넓이는 거리의 제곱에 반비례한다. 거리 2배까지는 견뎌야 한다.

        틀린 거절은 세 점 중 하나를 죽여 평면을 통째로 무너뜨리고 법선이 홀
        고리로 물러선다(±8도). 마스크가 줄어도 같은 면 위면 뎁스 중앙값은
        멀쩡하므로, 여기서 잡을 것은 파국적 붕괴뿐이다.
        """
        taught = 4000
        for distance in (1.2, 1.5, 2.0):
            shrunk = int(taught / (distance ** 2))
            self.assertGreaterEqual(
                shrunk, _min_area_for(taught),
                f"거리 {distance}배에서 멀쩡한 점을 거절한다",
            )

    def test_tiny_taught_area_still_hits_the_absolute_floor(self):
        """상대 기준이 0 으로 수렴하지는 않게 한다.

        뎁스 표본이 MIN_POINT_SAMPLES 개는 나와야 대표점이 서기 때문이다.
        """
        self.assertEqual(_min_area_for(10), NORMAL_POINT_MIN_AREA_PX)

    def test_the_regression_case_passes(self):
        """티칭 204 / 추적 184 는 통과해야 한다 (예전 절대 200 에서 탈락하던 것)."""
        self.assertLessEqual(_min_area_for(204), 184)

    def test_a_catastrophic_collapse_is_still_refused(self):
        """거의 사라진 것은 여전히 놓친 것으로 본다 (관대해도 바닥은 있다)."""
        self.assertGreater(_min_area_for(4000), 4000 * 0.1)


class NormalModeTest(unittest.TestCase):
    """자세를 어디서 받을지 — 세 방식이 각자 약속한 대로 도는가.

    이 선택은 로봇으로 나가는 자세를 통째로 가른다. 촬영 보드마다 다른
    프로파일을 쓰므로 보드마다 방식이 달라도 되고, 그래서 세팅을 오가며
    값이 살아남는 것까지 못박아 둔다.
    """

    TILT_DEG = 20.0

    def _depth(self):
        """z 에서 TILT_DEG 만큼 기울어진 평면을 뎁스맵으로 렌더한다."""
        normal = np.array([math.sin(math.radians(self.TILT_DEG)), 0.0,
                           -math.cos(math.radians(self.TILT_DEG))])
        offset = float(normal @ np.array([0.0, 0.0, 400.0]))
        cols, rows = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
        den = ((cols - WIDTH / 2) / 900.0) * normal[0] + normal[2]
        depth = np.where(np.abs(den) < 1e-9, 0.0, offset / den)
        depth[~np.isfinite(depth)] = 0.0
        depth[depth <= 0.0] = 0.0
        return depth.astype(np.uint16), normal

    def _pose(self, mode, **kwargs):
        depth, _n = self._depth()
        matrix = np.array([[900.0, 0.0, WIDTH / 2],
                           [0.0, 900.0, HEIGHT / 2], [0.0, 0.0, 1.0]])
        return estimate_hole_poses(
            [(WIDTH // 2, HEIGHT // 2, 5.0)], depth, matrix,
            color_size=(WIDTH, HEIGHT), mode=mode, **kwargs
        )[0]

    def test_flange_mode_does_not_measure_a_normal(self):
        """회전을 재지 않는다 — 기울어진 면에서도 기울기 0 이어야 한다."""
        pose = self._pose(NORMAL_MODE_FLANGE)
        self.assertEqual(pose.plane_source, PLANE_FROM_CAMERA)
        self.assertAlmostEqual(pose.tilt_deg, 0.0, places=6)
        # 위치는 제대로 나와야 한다. 그것이 이 방식의 존재 이유다.
        self.assertGreater(pose.z, 0.0)

    def test_ring_mode_recovers_the_tilt(self):
        pose = self._pose(NORMAL_MODE_RING, ring_radius_mm=15.0)
        self.assertEqual(pose.plane_source, PLANE_FROM_RING)
        self.assertAlmostEqual(pose.tilt_deg, self.TILT_DEG, delta=1.0)

    def test_flange_mode_keeps_the_flange_rotation(self):
        """로봇 자세가 플랜지와 **정확히** 같아야 한다. 이 방식의 정의다."""
        flange = (500.0, -400.0, 1100.0, 90.0, -8.0, 2.0)
        cam2gripper = np.eye(4)
        cam2gripper[:3, 3] = [10.0, 20.0, 30.0]
        target = base_target_pose(flange, cam2gripper,
                                  self._pose(NORMAL_MODE_FLANGE))
        for got, want in zip((target.rx, target.ry, target.rz), flange[3:]):
            self.assertAlmostEqual(got, want, places=6)

    def test_other_modes_do_change_the_rotation(self):
        """대조군 — ring 은 면을 따라가므로 플랜지와 달라야 한다."""
        flange = (500.0, -400.0, 1100.0, 90.0, -8.0, 2.0)
        cam2gripper = np.eye(4)
        target = base_target_pose(
            flange, cam2gripper, self._pose(NORMAL_MODE_RING, ring_radius_mm=15.0)
        )
        self.assertGreater(
            max(abs(target.rx - 90.0), abs(target.ry + 8.0), abs(target.rz - 2.0)),
            1.0,
        )

    def test_points_mode_admits_falling_back_to_the_ring(self):
        """매칭점을 못 찾았는데 'surface' 라고 하면 로그가 거짓말을 한다.

        두 출처는 정밀도가 0.11도 대 ±8도로 완전히 다르다. 자세가 튈 때
        원인을 짚는 단서가 이 값 하나뿐이다.
        """
        self.assertEqual(
            self._pose(NORMAL_MODE_POINTS, surface_plane=None).plane_source,
            PLANE_FROM_RING,
        )
        _depth, normal = self._depth()
        self.assertEqual(
            self._pose(NORMAL_MODE_POINTS,
                       surface_plane=(-normal, np.array([0.0, 0.0, 400.0]))
                       ).plane_source,
            PLANE_FROM_SURFACE,
        )

    def test_region_shrinks_in_pixels_as_the_camera_backs_off(self):
        """mm 로 지정한 영역은 거리가 멀어지면 더 적은 픽셀에 담겨야 한다."""
        near = ring_outer_px(4.0, 200.0, 900.0, 15.0)
        far = ring_outer_px(4.0, 600.0, 900.0, 15.0)
        self.assertGreater(near, far)
        # 실제 크기는 그대로여야 한다: px * z / f ≈ 15mm
        for depth_mm in (200.0, 400.0, 600.0):
            got = ring_outer_px(4.0, depth_mm, 900.0, 15.0) * depth_mm / 900.0
            self.assertAlmostEqual(got, 15.0, delta=0.1)

    def test_unknown_mode_falls_back_to_points(self):
        """이 키가 없는 옛 프로파일은 그때 쓰던 방식으로 읽혀야 한다."""
        self.assertEqual(normal_source_from_settings({})["mode"], NORMAL_MODE_POINTS)
        self.assertEqual(
            normal_source_from_settings({"normal_source": {"mode": "헛소리"}})["mode"],
            NORMAL_MODE_POINTS,
        )

    def test_saved_mode_survives_a_round_trip(self):
        got = normal_source_from_settings(
            {"normal_source": {"mode": NORMAL_MODE_RING, "ring_radius_mm": 23.0}}
        )
        self.assertEqual(got["mode"], NORMAL_MODE_RING)
        self.assertEqual(got["ring_radius_mm"], 23.0)


class CaptureSequenceTest(unittest.TestCase):
    """Pose Setting 이 Hole Detection 의 **새** 촬영을 이어받는가.

    260908에 이 페이지를 가르면서 showEvent 가 '아직 아무것도 없을 때만'
    이어받게 만들었다. 그러면 부품을 옮겨 다시 찍고 탭으로 와도 첫 촬영이
    그대로 남아, 위치 변화에 반응하지 않는 것처럼 보인다. 촬영 번호로 가린다.

    Qt 위젯을 띄우지 않고 번호 규칙만 시험한다 — 규칙이 깨지면 회귀다.
    """

    def test_sequence_rises_with_each_capture(self):
        """번호가 안 오르면 새 촬영을 알아볼 방법이 없다."""
        source = io.open("pages/hole_detection_page.py", encoding="utf-8").read()
        self.assertIn("self._capture_seq += 1", source)
        self.assertIn("def capture_sequence(self)", source)
        # 촬영 번호가 context 에 실려야 저쪽이 자기 것과 견줄 수 있다.
        self.assertIn('"sequence": self._capture_seq', source)

    def test_pose_page_compares_the_sequence(self):
        """'이미지가 없을 때만' 이어받는 옛 조건으로 되돌아가면 회귀다."""
        source = io.open("pages/pose_setting_page.py", encoding="utf-8").read()
        head = source[source.index("def showEvent"):source.index("def _build_controls")]
        self.assertIn("capture_sequence()", head)
        self.assertIn("self._ingested_seq", head)


class NormalAreaStorageTest(unittest.TestCase):
    """기준 넓이가 저장을 오가며 살아남는가. 없는 옛 자산도 읽히는가."""

    def setUp(self):
        self._root = tempfile.mkdtemp()
        self._patch = patch("template_store.templates_dir", return_value=self._root)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        self.addCleanup(shutil.rmtree, self._root, True)
        self._scene = np.full((40, 60, 3), 128, dtype=np.uint8)
        self._boxes = [(1, 1, 9, 9), (20, 5, 30, 15), (40, 20, 55, 35)]

    def test_areas_survive_a_round_trip(self):
        ok, _ = save_normal_points("점셋", self._scene, self._boxes, [111, 222, 333])
        self.assertTrue(ok)
        _scene, boxes, areas = load_normal_points("점셋")
        self.assertEqual(boxes, self._boxes)
        self.assertEqual(areas, [111, 222, 333])

    def test_missing_areas_read_as_unknown(self):
        """이 키가 생기기 전에 저장한 자산도 그대로 읽혀야 한다."""
        save_normal_points("옛것", self._scene, self._boxes)
        _scene, _boxes, areas = load_normal_points("옛것")
        self.assertIsNone(areas)

    def test_wrong_length_is_ignored(self):
        """자리가 밀린 기준으로 재면 엉뚱한 점이 탈락한다 — 아예 안 쓰는 게 낫다."""
        save_normal_points("정상", self._scene, self._boxes, [111, 222, 333])
        info_path = os.path.join(self._root, "정상", "info.json")
        with io.open(info_path, encoding="utf-8") as fp:
            info = json.load(fp)
        info["normal_areas"] = [111, 222]        # 손으로 고쳐 개수가 어긋난 경우
        with io.open(info_path, "w", encoding="utf-8") as fp:
            json.dump(info, fp)
        _scene, _boxes, areas = load_normal_points("정상")
        self.assertIsNone(areas)

    def test_count_mismatch_is_refused_at_save(self):
        ok, message = save_normal_points("나쁨", self._scene, self._boxes, [111, 222])
        self.assertFalse(ok)
        self.assertIn("개수", message)


if __name__ == "__main__":
    unittest.main()
