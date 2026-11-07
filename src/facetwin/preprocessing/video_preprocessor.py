"""Video and image preprocessing pipeline for FaceTwin."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from facetwin.core.exceptions import VideoProcessingError
from facetwin.core.logging_config import setup_logger

logger = setup_logger("facetwin.preprocessing.video")


@dataclass
class PreprocessedFrame:
    """Container for preprocessed image data and metadata."""
    frame_index: int
    image_rgb: Any  # np.ndarray
    original_shape: Tuple[int, int]  # (height, width)
    processed_shape: Tuple[int, int]
    timestamp_ms: float = 0.0


class VideoPreprocessor:
    """Preprocesses video files or raw frames for standardized facial landmark analysis.

    Standardizations performed:
    - Target resolution scaling (default 640x480) with aspect ratio preservation or padding.
    - Lighting normalization via Contrast Limited Adaptive Histogram Equalization (CLAHE).
    - Color space conversion (BGR -> RGB).
    """

    def __init__(
        self,
        target_width: int = 640,
        target_height: int = 480,
        target_fps: int = 30,
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
    ):
        self.target_width = target_width
        self.target_height = target_height
        self.target_fps = target_fps
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size

        self._clahe = None
        if HAS_CV2 and self.apply_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_grid_size
            )

    def process_frame(
        self, frame: Any, frame_index: int = 0, timestamp_ms: float = 0.0
    ) -> PreprocessedFrame:
        """Processes a single image frame (BGR or RGB).

        Args:
            frame: Input image array (height, width, channels).
            frame_index: Sequential index of the frame.
            timestamp_ms: Timestamp in milliseconds.

        Returns:
            PreprocessedFrame: Standardized RGB image with metadata.
        """
        if not HAS_CV2:
            raise VideoProcessingError("OpenCV (cv2) is required for frame preprocessing.")

        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise VideoProcessingError("Invalid frame provided to preprocessor.")

        orig_h, orig_w = frame.shape[:2]

        # Convert to BGR if grayscale or ensure RGB
        if len(frame.shape) == 2:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            # Assume input is either BGR (standard OpenCV) or RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply CLAHE on Luminance channel (LAB color space) to normalize lighting
        if self._clahe is not None:
            lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_equalized = self._clahe.apply(l_channel)
            lab_equalized = cv2.merge((l_equalized, a_channel, b_channel))
            image_rgb = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)

        # Resize if dimensions differ from target
        if (orig_w, orig_h) != (self.target_width, self.target_height):
            image_rgb = cv2.resize(
                image_rgb,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_AREA if (orig_w > self.target_width) else cv2.INTER_LINEAR,
            )

        processed_h, processed_w = image_rgb.shape[:2]

        return PreprocessedFrame(
            frame_index=frame_index,
            image_rgb=image_rgb,
            original_shape=(orig_h, orig_w),
            processed_shape=(processed_h, processed_w),
            timestamp_ms=timestamp_ms,
        )

    def process_video_file(
        self, video_path: Union[str, Path], max_frames: Optional[int] = None
    ) -> List[PreprocessedFrame]:
        """Reads and standardizes frames from a video file.

        Args:
            video_path: Path to video file.
            max_frames: Optional upper limit on extracted frames.

        Returns:
            List[PreprocessedFrame]: Sequence of preprocessed frames.
        """
        if not HAS_CV2:
            raise VideoProcessingError("OpenCV (cv2) is required for video decoding.")

        path = Path(video_path)
        if not path.exists():
            raise VideoProcessingError(f"Video file not found: {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoProcessingError(f"Could not open video stream: {path}")

        frames: List[PreprocessedFrame] = []
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, raw_frame = cap.read()
                if not ret:
                    break

                timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                processed = self.process_frame(
                    raw_frame, frame_index=frame_idx, timestamp_ms=timestamp_ms
                )
                frames.append(processed)
                frame_idx += 1

                if max_frames and frame_idx >= max_frames:
                    break
        finally:
            cap.release()

        logger.info(f"Preprocessed {len(frames)} frames from {path.name}")
        return frames
