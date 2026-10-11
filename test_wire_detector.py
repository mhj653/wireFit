import unittest
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from detection_debug import DebugFrame
from detection_debug import debug_frame_at, make_contact_sheet
from wire_debug_tools import save_debug_artifacts
from wire_detector import (
    SORT_CUSTOM,
    SORT_X,
    WIRE_MODE_BIN_TOP,
    WireDetectionConfig,
    WireDetector,
)
from wire_tracker import WireTracker


class WireDetectorTests(unittest.TestCase):
    def test_detects_and_sorts_parallel_wires_from_depth(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8) + 80
        depth = np.zeros((240, 320), dtype=np.uint16) + 1000
        for x in (80, 140, 200):
            cv2.line(image, (x, 20), (x + 30, 220), (20, 20, 20), 8)
            cv2.line(depth, (x, 20), (x + 30, 220), 960, 8)

        intrinsics = np.array(
            [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]]
        )
        config = WireDetectionConfig(
            min_length_px=50,
            min_area_px=40,
            plane_threshold_mm=10.0,
            sort_axis=SORT_X,
        )

        result = WireDetector(config).analyze(
            image, depth=depth, intrinsics=(intrinsics, None), depth_scale_mm=1.0
        )

        self.assertEqual(3, len(result.wires))
        self.assertEqual(3, len(result.holes))
        xs = [wire.representative_xyz[0] for wire in result.wires]
        self.assertEqual(xs, sorted(xs))
        self.assertEqual(8, len(result.debug_frames))

    def test_bin_top_layer_keeps_nearest_depth_layer(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8) + 80
        depth = np.zeros((240, 320), dtype=np.uint16) + 1000
        cv2.line(image, (80, 20), (110, 220), (20, 20, 20), 8)
        cv2.line(depth, (80, 20), (110, 220), 920, 8)
        cv2.line(image, (210, 20), (240, 220), (20, 20, 20), 8)
        cv2.line(depth, (210, 20), (240, 220), 960, 8)

        config = WireDetectionConfig(
            foreground_mode=WIRE_MODE_BIN_TOP,
            top_band_mm=10.0,
            top_percentile=1.0,
            min_length_px=50,
            min_area_px=40,
            use_edge_fallback=False,
        )

        result = WireDetector(config).analyze(image, depth=depth, depth_scale_mm=1.0)

        self.assertEqual(2, len(result.wires))
        self.assertEqual(1, len(result.holes))
        pick_wires = [wire for wire in result.wires if wire.is_pick_candidate]
        self.assertEqual(1, len(pick_wires))
        self.assertLess(pick_wires[0].representative_uv[0], 150)

    def test_bin_top_layer_uses_plane_candidates_before_depth_filtering(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8) + 80
        depth = np.zeros((240, 320), dtype=np.uint16) + 1000
        cv2.circle(image, (30, 30), 3, (20, 20, 20), -1)
        cv2.circle(depth, (30, 30), 3, 900, -1)
        cv2.line(image, (80, 20), (110, 220), (20, 20, 20), 8)
        cv2.line(depth, (80, 20), (110, 220), 940, 8)
        cv2.line(image, (210, 20), (240, 220), (20, 20, 20), 8)
        cv2.line(depth, (210, 20), (240, 220), 960, 8)

        config = WireDetectionConfig(
            foreground_mode=WIRE_MODE_BIN_TOP,
            top_band_mm=10.0,
            top_percentile=1.0,
            min_length_px=50,
            min_area_px=40,
            use_edge_fallback=False,
        )

        result = WireDetector(config).analyze(image, depth=depth, depth_scale_mm=1.0)

        self.assertEqual(2, len(result.wires))
        self.assertEqual(1, len(result.holes))
        pick_wires = [wire for wire in result.wires if wire.is_pick_candidate]
        self.assertEqual(1, len(pick_wires))
        self.assertLess(pick_wires[0].representative_uv[0], 150)
        self.assertIn("Plane candidates", result.debug_frames[1].title)
        self.assertTrue(
            any("Depth layers" in frame.title for frame in result.debug_frames)
        )

    def test_splits_crossing_skeleton_into_continuous_wires(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8) + 80
        depth = np.zeros((240, 320), dtype=np.uint16) + 1000
        for start, end in (((80, 30), (240, 210)), ((240, 30), (80, 210))):
            cv2.line(image, start, end, (20, 20, 20), 8)
            cv2.line(depth, start, end, 960, 8)

        config = WireDetectionConfig(
            min_length_px=50,
            min_area_px=40,
            plane_threshold_mm=10.0,
            merge_distance_px=24.0,
            merge_angle_deg=45.0,
        )

        result = WireDetector(config).analyze(
            image, depth=depth, depth_scale_mm=1.0
        )

        self.assertEqual(2, len(result.wires))
        self.assertTrue(all(wire.segment_count == 2 for wire in result.wires))

    def test_custom_z_sort_and_confidence_parts(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8) + 80
        depth = np.zeros((240, 320), dtype=np.uint16) + 1000
        for x, z in ((80, 940), (140, 960), (200, 980)):
            cv2.line(image, (x, 20), (x + 30, 220), (20, 20, 20), 8)
            cv2.line(depth, (x, 20), (x + 30, 220), z, 8)

        intrinsics = np.array(
            [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]]
        )
        config = WireDetectionConfig(
            min_length_px=50,
            min_area_px=40,
            plane_threshold_mm=10.0,
            sort_axis=SORT_CUSTOM,
            sort_vector_xyz=(0.0, 0.0, 1.0),
        )

        result = WireDetector(config).analyze(
            image, depth=depth, intrinsics=(intrinsics, None), depth_scale_mm=1.0
        )

        zs = [wire.representative_xyz[2] for wire in result.wires]
        self.assertEqual(zs, sorted(zs))
        self.assertTrue(all("depth" in wire.confidence_parts for wire in result.wires))
        self.assertTrue(all(wire.confidence > 0.0 for wire in result.wires))

    def test_tracker_keeps_ids_between_frames(self):
        tracker = WireTracker()
        config = WireDetectionConfig(
            min_length_px=50,
            min_area_px=40,
            plane_threshold_mm=10.0,
            tracking_enabled=True,
        )

        def scene(shift):
            image = np.zeros((240, 320, 3), dtype=np.uint8) + 80
            depth = np.zeros((240, 320), dtype=np.uint16) + 1000
            for x in (80 + shift, 140 + shift, 200 + shift):
                cv2.line(image, (x, 20), (x + 30, 220), (20, 20, 20), 8)
                cv2.line(depth, (x, 20), (x + 30, 220), 960, 8)
            return image, depth

        first_image, first_depth = scene(0)
        second_image, second_depth = scene(4)
        detector = WireDetector(config, tracker=tracker)
        first = detector.analyze(first_image, depth=first_depth, depth_scale_mm=1.0)
        second = detector.analyze(second_image, depth=second_depth, depth_scale_mm=1.0)

        self.assertEqual(
            [wire.tracking_id for wire in first.wires],
            [wire.tracking_id for wire in second.wires],
        )

    def test_debug_artifacts_are_saved_with_metadata(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        frames = [DebugFrame("Original", image)]
        wire = WireDetector(WireDetectionConfig()).analyze(
            np.zeros((120, 160, 3), dtype=np.uint8),
            depth=np.zeros((120, 160), dtype=np.uint16) + 1000,
        ).wires

        with TemporaryDirectory() as folder:
            path = save_debug_artifacts(frames, wire, root=folder)
            self.assertTrue(path.endswith(".png"))
            self.assertTrue(cv2.imread(path) is not None)

    def test_preprocess_toggles_can_disable_sources_safely(self):
        image = np.zeros((240, 320, 3), dtype=np.uint8) + 80
        depth = np.zeros((240, 320), dtype=np.uint16) + 1000
        cv2.line(image, (80, 20), (120, 220), (20, 20, 20), 8)
        cv2.line(depth, (80, 20), (120, 220), 960, 8)

        config = WireDetectionConfig(
            min_length_px=50,
            min_area_px=40,
            use_depth_foreground=False,
            use_edge_fallback=False,
        )
        result = WireDetector(config).analyze(image, depth=depth, depth_scale_mm=1.0)

        self.assertEqual([], result.wires)
        self.assertEqual(0, int(np.count_nonzero(result.binary)))
        self.assertEqual(7, len(result.debug_frames))

    def test_debug_contact_sheet_hit_test_returns_clicked_frame(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        frames = [DebugFrame(f"stage-{index}", image) for index in range(7)]
        sheet = make_contact_sheet(frames, output_size=(1280, 720))

        self.assertEqual((720, 1280, 3), sheet.shape)
        self.assertEqual("stage-0", debug_frame_at(frames, (20, 20), output_size=(1280, 720)).title)


if __name__ == "__main__":
    unittest.main()
