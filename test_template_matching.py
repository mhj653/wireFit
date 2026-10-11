import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import template_store
from template_matching import TemplateMatcher
from pages.hole_detection_page import HoleDetectionPage


class TemplateMatchingTest(unittest.TestCase):
    def setUp(self):
        self.template = np.zeros((12, 12, 3), dtype=np.uint8)
        cv2.circle(self.template, (6, 6), 4, (255, 255, 255), -1)

    def test_finds_repeated_templates_and_suppresses_duplicates(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[10:22, 15:27] = self.template
        image[45:57, 70:82] = self.template

        holes, _debug = TemplateMatcher(self.template, 0.95).analyze(image)

        self.assertEqual(len(holes), 2)
        centers = {(round(x), round(y)) for x, y, _radius in holes}
        self.assertEqual(centers, {(21, 16), (76, 51)})

    def test_applies_search_roi_and_returns_image_coordinates(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[10:22, 15:27] = self.template
        image[45:57, 70:82] = self.template

        holes, _debug = TemplateMatcher(self.template, 0.95).analyze(
            image, roi=(50, 30, 45, 40)
        )

        self.assertEqual(len(holes), 1)
        self.assertEqual(tuple(round(value) for value in holes[0][:2]), (76, 51))

    def test_template_is_saved_in_named_folder_and_loaded(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(template_store, "templates_dir", return_value=root):
                ok, _path = template_store.save_template("홀A", self.template)
                loaded = template_store.load_template("홀A")

            self.assertTrue(ok)
            self.assertTrue(os.path.isfile(os.path.join(root, "홀A", "template.png")))
            np.testing.assert_array_equal(loaded, self.template)

    def test_operator_uses_template_profile(self):
        image = np.zeros((50, 60, 3), dtype=np.uint8)
        image[20:32, 30:42] = self.template
        profile = {
            "detection_method": "template",
            "template_matching": {"name": "홀A", "threshold": 0.95},
            "roi": None,
        }
        with patch("pages.hole_detection_page.load_profile", return_value=profile), \
                patch("pages.hole_detection_page.load_template", return_value=self.template):
            (
                _visualized,
                holes,
                poses,
                _raw,
                _choice,
            ) = HoleDetectionPage.visualize_for_operator(
                None, image, "template_profile"
            )

        self.assertEqual(len(holes), 1)
        self.assertEqual(poses, [None])


if __name__ == "__main__":
    unittest.main()
