"""SAM2 비디오 추적이 티칭 자산과 오퍼레이터 경로에 옳게 물리는지.

추적 정확도는 여기서 보지 않는다 — SAM2 체크포인트와 촬영당 2초가 필요해서
단위 시험에 담기에 무겁다. 여기서 지키는 것은 그 앞뒤다.

  - 티칭 프레임과 프롬프트가 실제로 남는가
  - 비디오 추적이 생기기 전에 티칭한 기준 모양(크롭만 있는 것)을 만났을 때
    무엇을 해야 하는지 알려 주고 멈추는가
  - 오퍼레이터가 세팅의 방식대로 검출기를 고르는가
"""

import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import template_store
from pages.hole_detection_page import (
    HOLE_CONTOUR_COLOR,
    BlobResultVisualizer,
    HoleDetectionPage,
    sam2_video_detector,
)
from pose_pipeline import (
    NORMAL_POINT_COLOR,
    NORMAL_POINT_MISS_COLOR,
    draw_normal_points,
)


def _scene(width=64, height=48, cx=32, cy=24):
    image = np.full((height, width, 3), 180, dtype=np.uint8)
    cv2.circle(image, (cx, cy), 5, (30, 30, 30), -1)
    return image


class TaughtSceneStorageTest(unittest.TestCase):
    """홀 티칭 자산이 법선 매칭점과 **같은 구조**인가 (260908 통일).

    예전에는 홀만 크롭(template.png)을 함께 남겼다. 그것은 cv2.matchTemplate
    으로 대략 위치를 잡던 2단계 방식이 쓰던 것인데, 그 방식이 배율·회전에서
    그럴듯한 오답을 내서 걷어냈다. 크롭이 다시 생기면 그 방식도 함께 돌아온다.
    """

    def setUp(self):
        self.crop = np.zeros((12, 12, 3), dtype=np.uint8)
        cv2.circle(self.crop, (6, 6), 4, (255, 255, 255), -1)
        self.scene = _scene()
        self.box = (26, 18, 38, 30)

    def test_hole_saves_scene_and_box_like_normal_points(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(template_store, "templates_dir", return_value=root):
                ok, _path = template_store.save_hole(
                    "홀A", self.scene, self.box, area=321
                )
                loaded = template_store.load_hole("홀A")
                listed = template_store.list_holes()

            self.assertTrue(ok)
            self.assertTrue(os.path.isfile(os.path.join(root, "홀A", "scene.png")))
            # 크롭을 만들면 안 된다. 만들면 드래그 템플릿으로 오인된다.
            self.assertFalse(os.path.isfile(os.path.join(root, "홀A", "template.png")))
            self.assertEqual(listed, ["홀A"])
            scene, box, area = loaded
            np.testing.assert_array_equal(scene, self.scene)
            self.assertEqual(box, self.box)
            self.assertEqual(area, 321)

    def test_hole_and_template_do_not_show_up_in_each_other(self):
        """종류가 섞이면 콤보에 고를 수 없는 것이 뜬다."""
        with tempfile.TemporaryDirectory() as root:
            with patch.object(template_store, "templates_dir", return_value=root):
                template_store.save_hole("홀A", self.scene, self.box, area=321)
                template_store.save_template("드래그템플릿", self.crop)
                self.assertEqual(template_store.list_holes(), ["홀A"])
                self.assertEqual(template_store.list_templates(), ["드래그템플릿"])
                self.assertEqual(template_store.list_normal_points(), [])

    def test_old_point_taught_hole_is_refused(self):
        """260908 이전 자산은 네모가 없다. 조용히 넘기면 프롬프트 없이 추적한다."""
        with tempfile.TemporaryDirectory() as root:
            with patch.object(template_store, "templates_dir", return_value=root):
                os.makedirs(os.path.join(root, "옛것"))
                cv2.imwrite(os.path.join(root, "옛것", "scene.png"), self.scene)
                with io.open(
                    os.path.join(root, "옛것", "info.json"), "w", encoding="utf-8"
                ) as fp:
                    json.dump({"name": "옛것", "prompt": {"x": 32, "y": 24}}, fp)

                self.assertIsNone(template_store.load_hole("옛것"))
                self.assertEqual(template_store.list_holes(), [])

    def test_delete_removes_the_scene_too(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(template_store, "templates_dir", return_value=root):
                template_store.save_hole("홀A", self.scene, self.box, area=321)
                ok, _error = template_store.delete_template("홀A")

            self.assertTrue(ok)
            self.assertFalse(os.path.isdir(os.path.join(root, "홀A")))


class VideoDetectorFromSettingsTest(unittest.TestCase):
    """세팅 -> 검출기. 재티칭이 필요한 상황을 조용히 넘기지 않는가."""

    def test_missing_scene_says_to_teach_again(self):
        with patch("pages.hole_detection_page.load_hole", return_value=None):
            with self.assertRaises(RuntimeError) as caught:
                sam2_video_detector({"reference": "옛것", "model": None})
        # 사유만 있고 할 일이 없으면 현장에서 멈춘다.
        self.assertIn("다시", str(caught.exception))

    def test_missing_reference_says_to_teach_first(self):
        with self.assertRaises(RuntimeError) as caught:
            sam2_video_detector({"reference": None, "model": None})
        self.assertIn("홀 지정", str(caught.exception))

    def test_passes_scene_box_and_area_through(self):
        scene = _scene()
        with patch(
            "pages.hole_detection_page.load_hole",
            return_value=(scene, (26, 18, 38, 30), 321),
        ), patch(
            "pages.hole_detection_page.sam2_video_tracker.Sam2VideoDetector"
        ) as detector:
            sam2_video_detector({"reference": "홀A", "model": "tiny.pt"})

        detector.assert_called_once()
        passed_scene, passed_box, passed_model = detector.call_args[0]
        np.testing.assert_array_equal(passed_scene, scene)
        self.assertEqual(passed_box, (26, 18, 38, 30))
        self.assertEqual(passed_model, "tiny.pt")
        # 기준 넓이가 빠지면 재검출에 하한이 없어진다.
        self.assertEqual(detector.call_args[1]["taught_area"], 321)


class OperatorPathTest(unittest.TestCase):
    """오퍼레이터가 세팅에 적힌 방식대로 검출기를 고르는가."""

    def test_operator_uses_the_video_tracker_for_that_method(self):
        image = _scene()
        profile = {
            "detection_method": "sam2_video",
            "sam2": {"reference": "홀A", "model": None},
            "roi": None,
        }
        with patch("pages.hole_detection_page.load_profile", return_value=profile), \
                patch(
                    "pages.hole_detection_page.load_hole",
                    return_value=(image, (26, 18, 38, 30), 321),
                ), \
                patch(
                    "pages.hole_detection_page.sam2_video_tracker.Sam2VideoDetector"
                ) as detector:
            detector.return_value.analyze.return_value = (
                [(32, 24, 5.0)],
                {"binary": np.zeros(image.shape[:2], np.uint8), "sam2_stage": "ok"},
            )
            (
                _visualized,
                holes,
                poses,
                _raw,
                _choice,
            ) = HoleDetectionPage.visualize_for_operator(None, image, "video_profile")

        self.assertEqual(holes, [(32, 24, 5.0)])
        self.assertEqual(poses, [None])
        detector.return_value.analyze.assert_called_once()

    def test_the_two_step_detector_is_gone(self):
        """'SAM2 홀 지정'(2단계)이 되살아나면 그럴듯한 오답도 함께 돌아온다.

        cv2.matchTemplate 이 배율·회전을 못 봐서, 축소 0.8 에서 반지름 20px 이
        나와야 할 자리에 257px 을 냈다. 넓이 상한 검사까지 통과해서 화면상으로도
        멀쩡해 보였다 — 못 찾는 것보다 나쁘다.
        """
        import sam2_detector

        self.assertFalse(hasattr(sam2_detector, "Sam2Detector"))
        source = io.open("pages/hole_detection_page.py", encoding="utf-8").read()
        self.assertNotIn('DETECTION_SAM2 = "sam2"', source)


if __name__ == "__main__":
    unittest.main()
