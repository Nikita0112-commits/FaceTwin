"""Unit tests for YFP Dataset Manager."""

import io
from pathlib import Path
import unittest
import zipfile

from facetwin.core.exceptions import VideoProcessingError
from facetwin.data.yfp_dataset import YFPDatasetManager


class TestYFPDatasetManager(unittest.TestCase):
    """Test suite for YFP in-memory archive management."""

    def setUp(self):
        """Create mock ZIP archives to test indexing and in-memory frame reading."""
        self.test_dir = Path("experiments/results/test_yfp_mock")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.mock_zip1 = self.test_dir / "mock_Image.zip"
        self.mock_zip2 = self.test_dir / "mock_Image2.zip"

        # Populate mock_zip1 with Subject 1 frames
        with zipfile.ZipFile(self.mock_zip1, "w") as zf:
            zf.writestr("Image/1/100.bmp", b"MOCK_BMP_BYTES_FRAME_100")
            zf.writestr("Image/1/105.bmp", b"MOCK_BMP_BYTES_FRAME_105")
            zf.writestr("Image/2/100.bmp", b"MOCK_BMP_BYTES_SUBJ2")

        # Populate mock_zip2 with Subject 9 frames
        with zipfile.ZipFile(self.mock_zip2, "w") as zf:
            zf.writestr("Image2/9/200.bmp", b"MOCK_BMP_BYTES_FRAME_200")

    def tearDown(self):
        """Cleanup test fixtures."""
        for f in self.test_dir.glob("*"):
            f.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_archive_status_detection(self):
        """Verify archive detection, size calculation, and readability check."""
        mgr = YFPDatasetManager(
            archive_1=self.mock_zip1,
            archive_2=self.mock_zip2,
            archive_3=self.test_dir / "non_existent.zip",
        )
        status = mgr.get_archive_status()

        self.assertTrue(status["Image.zip"].exists)
        self.assertTrue(status["Image.zip"].readable)
        self.assertEqual(status["Image.zip"].total_images, 3)
        self.assertEqual(status["Image.zip"].subjects, [1, 2])

        self.assertTrue(status["Image2.zip"].exists)
        self.assertEqual(status["Image2.zip"].total_images, 1)
        self.assertEqual(status["Image2.zip"].subjects, [9])

        self.assertFalse(status["Image3.zip"].exists)
        self.assertEqual(status["Image3.zip"].total_images, 0)

    def test_in_memory_indexing_and_reading(self):
        """Verify reading frame bytes directly from zip without disk extraction."""
        mgr = YFPDatasetManager(
            archive_1=self.mock_zip1,
            archive_2=self.mock_zip2,
            archive_3=self.test_dir / "non_existent.zip",
        )
        mgr.build_index()

        subjects = mgr.list_subjects()
        self.assertEqual(subjects, [1, 2, 9])
        self.assertEqual(mgr.get_subject_frame_count(1), 2)

        # Read specific frame bytes directly
        data = mgr.read_frame_bytes(subject_id=1, frame_number=100)
        self.assertEqual(data, b"MOCK_BMP_BYTES_FRAME_100")

        # Request invalid frame raises VideoProcessingError
        with self.assertRaises(VideoProcessingError):
            mgr.read_frame_bytes(subject_id=1, frame_number=9999)

    def test_batch_generator(self):
        """Verify batch iteration across subject frames."""
        mgr = YFPDatasetManager(
            archive_1=self.mock_zip1,
            archive_2=self.mock_zip2,
            archive_3=self.test_dir / "non_existent.zip",
        )
        batches = list(mgr.iter_subject_batches(subject_id=1, batch_size=1))
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0][0][0], 100)
        self.assertEqual(batches[1][0][0], 105)


if __name__ == "__main__":
    unittest.main()
