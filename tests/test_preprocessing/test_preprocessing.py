"""Unit tests for video preprocessor and 478-point landmark extractor."""

import os
from pathlib import Path
import unittest
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp"

from facetwin.core.exceptions import VideoProcessingError, LandmarkExtractionError
from facetwin.preprocessing.landmark_extractor import LandmarkExtractor
from facetwin.preprocessing.video_preprocessor import VideoPreprocessor


class TestPreprocessingPipeline(unittest.TestCase):
    """Test suite for image preprocessing and MediaPipe landmark extraction."""

    def setUp(self):
        self.preprocessor = VideoPreprocessor(
            target_width=640,
            target_height=480,
            apply_clahe=True,
        )
        self.model_path = Path("data/external/models/face_landmarker.task")

    def test_preprocessor_shape_and_clahe(self):
        """Verify preprocessor converts dimensions and equalizes luminance."""
        dummy_bgr = np.random.randint(50, 200, (720, 1280, 3), dtype=np.uint8)
        result = self.preprocessor.process_frame(dummy_bgr, frame_index=1, timestamp_ms=33.3)

        self.assertEqual(result.frame_index, 1)
        self.assertEqual(result.original_shape, (720, 1280))
        self.assertEqual(result.processed_shape, (480, 640))
        self.assertEqual(result.image_rgb.shape, (480, 640, 3))
        self.assertAlmostEqual(result.timestamp_ms, 33.3)

    def test_preprocessor_invalid_input(self):
        """Verify preprocessor raises VideoProcessingError on invalid frames."""
        with self.assertRaises(VideoProcessingError):
            self.preprocessor.process_frame(None)

        with self.assertRaises(VideoProcessingError):
            self.preprocessor.process_frame(np.array([1, 2, 3]))

    def test_landmark_extractor_no_face_detected(self):
        """Verify extractor handles blank frame gracefully without exceptions."""
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        with LandmarkExtractor(model_path=self.model_path) as extractor:
            res = extractor.extract_from_frame(black_frame, frame_index=0)

            self.assertFalse(res.face_detected)
            self.assertEqual(res.frame_index, 0)
            self.assertEqual(res.landmarks_normalized.shape, (478, 3))
            self.assertEqual(res.landmarks_pixel.shape, (478, 2))
            self.assertIsNone(res.bounding_box)
            self.assertIsNone(res.left_iris_center)

    def test_landmark_extractor_batch(self):
        """Verify batch extraction across multiple frames."""
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]

        with LandmarkExtractor(model_path=self.model_path) as extractor:
            results = extractor.extract_batch(frames, start_index=10)

            self.assertEqual(len(results), 3)
            self.assertEqual(results[0].frame_index, 10)
            self.assertEqual(results[1].frame_index, 11)
            self.assertEqual(results[2].frame_index, 12)


if __name__ == "__main__":
    unittest.main()
