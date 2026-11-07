"""MediaPipe 478-Point Facial Landmark Extractor for FaceTwin."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

from facetwin.core.exceptions import LandmarkExtractionError
from facetwin.core.logging_config import setup_logger

logger = setup_logger("facetwin.preprocessing.landmarks")


@dataclass
class FaceLandmarksResult:
    """Encapsulates extracted 478-point landmark and blendshape data for a single frame."""
    frame_index: int
    face_detected: bool
    landmarks_normalized: Any  # np.ndarray shape (478, 3) normalized [0, 1]
    landmarks_pixel: Any       # np.ndarray shape (478, 2) pixel coordinates
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (xmin, ymin, width, height)
    left_iris_center: Optional[Tuple[float, float, float]] = None
    right_iris_center: Optional[Tuple[float, float, float]] = None
    blendshapes: Dict[str, float] = field(default_factory=dict)


class LandmarkExtractor:
    """Extracts 478 high-density 3D facial landmarks using Google MediaPipe FaceLandmarker Tasks API.

    Features extracted:
    - 468 canonical facial surface mesh vertices.
    - 10 refined iris contour landmarks (indices 468-472 left iris, 473-477 right iris).
    - 52 FACS-aligned facial blendshapes (e.g. eyeBlinkLeft, eyeBlinkRight, mouthSmileLeft, mouthSmileRight).
    """

    NUM_LANDMARKS = 478
    LEFT_IRIS_INDEX = 468
    RIGHT_IRIS_INDEX = 473
    DEFAULT_MODEL_PATH = Path("data/external/models/face_landmarker.task")

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_blendshapes: bool = True,
    ):
        """Initializes MediaPipe FaceLandmarker with the Tasks API."""
        self.model_path = Path(model_path or self.DEFAULT_MODEL_PATH)
        self.num_faces = num_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_presence_confidence = min_presence_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.output_blendshapes = output_blendshapes

        self._detector = None
        if HAS_MEDIAPIPE and self.model_path.exists():
            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=self.num_faces,
                min_face_detection_confidence=self.min_detection_confidence,
                min_face_presence_confidence=self.min_presence_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                output_face_blendshapes=self.output_blendshapes,
            )
            self._detector = vision.FaceLandmarker.create_from_options(options)

    def extract_from_frame(
        self, image_rgb: Any, frame_index: int = 0
    ) -> FaceLandmarksResult:
        """Extracts 478 landmarks and blendshapes from an RGB frame.

        Args:
            image_rgb: RGB image numpy array (H, W, 3).
            frame_index: Sequence frame index.

        Returns:
            FaceLandmarksResult: Structured 478 landmark coordinates and detection status.
        """
        if not HAS_MEDIAPIPE:
            raise LandmarkExtractionError(
                "MediaPipe is not installed. Install 'mediapipe'."
            )

        if self._detector is None:
            if not self.model_path.exists():
                raise LandmarkExtractionError(
                    f"FaceLandmarker model not found at {self.model_path}. "
                    "Ensure face_landmarker.task is present."
                )
            raise LandmarkExtractionError("FaceLandmarker detector failed to initialize.")

        if image_rgb is None or len(image_rgb.shape) != 3:
            raise LandmarkExtractionError("Invalid RGB image passed to landmark extractor.")

        h, w = image_rgb.shape[:2]

        # Convert numpy array to mp.Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self._detector.detect(mp_image)

        if not detection_result.face_landmarks:
            # Face not detected in this frame
            return FaceLandmarksResult(
                frame_index=frame_index,
                face_detected=False,
                landmarks_normalized=np.zeros((self.NUM_LANDMARKS, 3), dtype=np.float32),
                landmarks_pixel=np.zeros((self.NUM_LANDMARKS, 2), dtype=np.int32),
                bounding_box=None,
                left_iris_center=None,
                right_iris_center=None,
                blendshapes={},
            )

        # Primary face mesh (first detected face)
        raw_landmarks = detection_result.face_landmarks[0]
        points_norm = np.zeros((len(raw_landmarks), 3), dtype=np.float32)
        points_px = np.zeros((len(raw_landmarks), 2), dtype=np.int32)

        for i, lm in enumerate(raw_landmarks):
            points_norm[i] = [lm.x, lm.y, lm.z]
            px = int(np.clip(lm.x * w, 0, w - 1))
            py = int(np.clip(lm.y * h, 0, h - 1))
            points_px[i] = [px, py]

        # Compute bounding box
        xmin = int(np.min(points_px[:, 0]))
        ymin = int(np.min(points_px[:, 1]))
        xmax = int(np.max(points_px[:, 0]))
        ymax = int(np.max(points_px[:, 1]))
        bbox = (xmin, ymin, xmax - xmin, ymax - ymin)

        # Extract iris center coordinates (landmarks 468 and 473)
        left_iris = None
        right_iris = None
        if len(points_norm) >= self.NUM_LANDMARKS:
            left_iris = tuple(points_norm[self.LEFT_IRIS_INDEX])
            right_iris = tuple(points_norm[self.RIGHT_IRIS_INDEX])

        # Extract facial blendshapes (AU proxies: smile, blink, brow, jaw)
        blendshape_dict: Dict[str, float] = {}
        if detection_result.face_blendshapes:
            for category in detection_result.face_blendshapes[0]:
                blendshape_dict[category.category_name] = float(category.score)

        return FaceLandmarksResult(
            frame_index=frame_index,
            face_detected=True,
            landmarks_normalized=points_norm,
            landmarks_pixel=points_px,
            bounding_box=bbox,
            left_iris_center=left_iris,
            right_iris_center=right_iris,
            blendshapes=blendshape_dict,
        )

    def extract_batch(
        self, frames_rgb: List[Any], start_index: int = 0
    ) -> List[FaceLandmarksResult]:
        """Extracts landmarks from a batch of RGB frames sequentially.

        Args:
            frames_rgb: List of RGB numpy arrays.
            start_index: Starting frame sequence index.

        Returns:
            List[FaceLandmarksResult]: Landmark results for each frame.
        """
        results: List[FaceLandmarksResult] = []
        for i, frame in enumerate(frames_rgb):
            res = self.extract_from_frame(frame, frame_index=start_index + i)
            results.append(res)
        return results

    def close(self) -> None:
        """Releases detector resources."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
