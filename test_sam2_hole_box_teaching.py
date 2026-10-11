"""홀 티칭이 **네모 드래그**인가 (260914).

260914 까지 홀 티칭은 클릭이었다. 점 하나는 '이 점이 속한 무엇'이라 SAM2 가
홀인지 판재인지 스스로 골라야 해서, 뜻한 것보다 **큰 물체가 잡히는** 일이
있었다(사용자 보고). 그래서 법선 매칭점과 같은 네모 프롬프트로 바꿨다.

여기서 지키는 것은 셋이다.

  - 네모 밖으로 번진 후보는 점수가 높아도 고르지 않는다 (이 변경의 요점)
  - teach() 가 점이 아니라 네모를 받고, 저장은 SAM2 가 좁힌 네모로 한다
  - 화면에서 '홀 지정'이 클릭이 아니라 드래그를 기다린다

추적 정확도는 여기서 보지 않는다 — 체크포인트와 촬영당 2초가 필요해서 단위
시험에 담기에 무겁다(test_sam2_video_tracking 과 같은 선이다).
"""

import os
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import sam2_detector

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _scene(width=64, height=48):
    image = np.full((height, width, 3), 180, dtype=np.uint8)
    cv2.circle(image, (32, 24), 5, (30, 30, 30), -1)
    return image


def _mask(shape, box):
    """네모를 꽉 채운 마스크."""
    left, top, right, bottom = box
    mask = np.zeros(shape, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask


class HolePickerTest(unittest.TestCase):
    """드래그한 네모가 곧 '이만한 것을 찾아라'는 뜻인가."""

    def setUp(self):
        self.shape = (48, 64)          # 3072 px
        self.drawn = (26, 18, 38, 30)  # 12x12 = 144 px

    def test_the_bigger_object_loses_even_with_a_better_score(self):
        """이 변경의 요점. 판재가 더 그럴듯해도 네모 안의 홀을 고른다."""
        hole = _mask(self.shape, (28, 20, 36, 28))     # 64 px, 네모 안
        plate = _mask(self.shape, (0, 0, 64, 20))      # 1280 px, 네모 밖으로 번짐
        picked = sam2_detector._hole_picker(self.drawn)(
            [plate, hole], [0.99, 0.60], self.shape
        )
        np.testing.assert_array_equal(picked, hole)

    def test_nothing_is_picked_when_every_candidate_spills_out(self):
        """조건을 풀어 큰 것을 돌려주면 뜻한 적 없는 물체가 기준 홀이 된다.

        None 이면 teach() 가 '다시 그려 보라'고 알린다 — 그쪽이 낫다.
        """
        plate = _mask(self.shape, (0, 0, 40, 32))
        picked = sam2_detector._hole_picker(self.drawn)(
            [plate], [0.99], self.shape
        )
        self.assertIsNone(picked)

    def test_a_mask_slightly_over_the_edge_still_counts(self):
        """사람이 바짝 그리면 경계가 조금 삐져나온다. 그것까지 막지는 않는다."""
        hole = _mask(self.shape, (25, 18, 38, 30))   # 왼쪽으로 1px 넘침
        picked = sam2_detector._hole_picker(self.drawn)(
            [hole], [0.80], self.shape
        )
        np.testing.assert_array_equal(picked, hole)

    def test_specks_are_still_refused(self):
        speck = _mask(self.shape, (30, 22, 32, 24))   # 4 px
        picked = sam2_detector._hole_picker(self.drawn)(
            [speck], [0.99], self.shape
        )
        self.assertIsNone(picked)


class _StubSegmenter:
    """segment_box 호출을 받아 적는 가짜 예측기."""

    def __init__(self, mask):
        self.mask = mask
        self.calls = []

    def segment_box(self, bgr_image, box, picker=None):
        self.calls.append((bgr_image, box, picker))
        return self.mask


class TeachTakesABoxTest(unittest.TestCase):
    """teach() 의 프롬프트가 점에서 네모로 바뀌었는가."""

    def test_the_drawn_box_goes_to_sam2_and_the_taught_box_comes_back(self):
        scene = _scene()
        drawn = (26, 18, 38, 30)
        mask = _mask(scene.shape[:2], (28, 20, 36, 28))   # 64 px
        stub = _StubSegmenter(mask)
        with patch.object(sam2_detector, "segmenter", return_value=stub):
            hole, got_mask, taught = sam2_detector.teach(scene, drawn, "tiny.pt")

        self.assertEqual(stub.calls[0][1], drawn)
        self.assertIsNotNone(stub.calls[0][2])   # 홀용 picker 를 넣어 부른다
        np.testing.assert_array_equal(got_mask, mask)
        # 사람이 그린 네모가 아니라 SAM2 가 잡은 경계를 돌려준다. 그 여백이
        # 넓으면 다음 촬영에서 옆 물체까지 프롬프트에 들어온다.
        self.assertEqual(taught, (28, 20, 36, 28))
        self.assertEqual(hole[:2], (32, 24))   # 8x8 마스크의 중심

    def test_it_says_what_to_do_when_nothing_is_found(self):
        scene = _scene()
        stub = _StubSegmenter(None)
        with patch.object(sam2_detector, "segmenter", return_value=stub):
            with self.assertRaises(RuntimeError) as caught:
                sam2_detector.teach(scene, (26, 18, 38, 30))
        self.assertIn("네모", str(caught.exception))

    def test_the_point_prompt_is_gone(self):
        """남겨 두면 '큰 물체가 잡히는' 그 길로 되돌아가기 쉽다."""
        self.assertFalse(hasattr(sam2_detector._Segmenter, "segment_at"))


class TeachButtonWaitsForADragTest(unittest.TestCase):
    """화면 쪽 배선. '홀 지정'이 클릭이 아니라 드래그를 받는가."""

    @classmethod
    def setUpClass(cls):
        from PyQt5.QtWidgets import QApplication
        from pages.hole_detection_page import HoleDetectionPage

        cls.app = QApplication.instance() or QApplication([])
        cls.page = HoleDetectionPage()
        cls.page._last_captured_image = _scene()
        cls.page.btn_sam2_teach.setEnabled(True)
        cls.page.btn_roi.setEnabled(True)

    def tearDown(self):
        self.page.btn_sam2_teach.setChecked(False)
        self.page.btn_roi.setChecked(False)
        self.page._drag_target = None

    def test_turning_it_on_waits_for_a_region_drag(self):
        self.page.btn_sam2_teach.setChecked(True)
        self.assertEqual(self.page.result_board._selection_mode, "region")
        self.assertEqual(self.page._drag_target, "hole")

    def test_roi_and_hole_teaching_do_not_stay_on_together(self):
        """같은 보드의 같은 드래그다. 둘이 켜져 있으면 어디로 갈지 알 수 없다."""
        self.page.btn_sam2_teach.setChecked(True)
        self.page.btn_roi.setChecked(True)
        self.assertFalse(self.page.btn_sam2_teach.isChecked())
        self.assertEqual(self.page.result_board._selection_mode, "roi")

    def test_the_drag_teaches_the_hole_and_saves_the_taught_box(self):
        page = self.page
        taught_box = (27, 19, 37, 29)
        mask = _mask(page._last_captured_image.shape[:2], taught_box)
        page.btn_sam2_teach.setChecked(True)

        with patch(
            "pages.hole_detection_page.sam2_detector.teach",
            return_value=((32, 24, 5.0), mask, taught_box),
        ) as teach, patch(
            "pages.hole_detection_page.sam2_video_tracker.measure_hole_area",
            return_value=321,
        ) as measure, patch(
            "pages.hole_detection_page.TextInputDialog.get_text",
            return_value="홀A",
        ), patch(
            "pages.hole_detection_page.save_hole", return_value=(True, "경로")
        ) as save, patch.object(
            type(page), "_reload_sam2_list"
        ), patch.object(
            type(page), "analyze_image"
        ):
            # 보드가 넘기는 것은 (x, y, 폭, 높이) 다.
            page.result_board.region_selected.emit((26, 18, 12, 12))

        self.assertEqual(teach.call_args[0][1], (26, 18, 38, 30))
        # 재검출이 쓸 넓이도 저장할 네모도 SAM2 가 좁힌 네모로 재야 한다.
        self.assertEqual(measure.call_args[0][1], taught_box)
        self.assertEqual(save.call_args[0][2], taught_box)
        # 한 번 쓰면 대기를 푼다. 안 그러면 다음 드래그가 또 티칭이 된다.
        self.assertIsNone(page._drag_target)
        self.assertIsNone(page.result_board._selection_mode)
        self.assertFalse(page.btn_sam2_teach.isChecked())


if __name__ == "__main__":
    unittest.main()
