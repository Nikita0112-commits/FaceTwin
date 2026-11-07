"""Script to extract and save 478 MediaPipe landmarks from YFP frames in memory."""

import argparse
import os
from pathlib import Path
import time
from typing import List

import numpy as np
import pandas as pd

from facetwin.core.logging_config import setup_logger
from facetwin.data.yfp_dataset import YFPDatasetManager
from facetwin.preprocessing.landmark_extractor import LandmarkExtractor
from facetwin.preprocessing.video_preprocessor import VideoPreprocessor

logger = setup_logger("facetwin.scripts.extract_yfp")


def extract_subject_landmarks(
    subject_id: int,
    max_frames: int = 100,
    output_dir: str = "data/features",
    model_path: str = "data/external/models/face_landmarker.task",
) -> Path:
    """Extracts 478 landmarks from frames of a given subject and exports feature matrix.

    Args:
        subject_id: YFP subject ID (1 to 24).
        max_frames: Maximum number of frames to process.
        output_dir: Destination folder for feature file.
        model_path: Path to face_landmarker.task model.

    Returns:
        Path: Output feature CSV/Parquet path.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_file = out_path / f"yfp_subject_{subject_id:02d}_landmarks.csv"

    mgr = YFPDatasetManager()
    total_available = mgr.get_subject_frame_count(subject_id)
    logger.info(f"Subject {subject_id}: {total_available} total frames available.")

    preprocessor = VideoPreprocessor(target_width=640, target_height=480, apply_clahe=True)
    rows: List[dict] = []

    start_time = time.time()
    processed_count = 0

    with LandmarkExtractor(model_path=model_path) as extractor:
        for batch in mgr.iter_subject_batches(subject_id=subject_id, batch_size=16):
            for frame_num, img_rgb in batch:
                prep = preprocessor.process_frame(img_rgb, frame_index=frame_num)
                res = extractor.extract_from_frame(prep.image_rgb, frame_index=frame_num)

                row = {
                    "subject_id": subject_id,
                    "frame_number": frame_num,
                    "face_detected": res.face_detected,
                }

                if res.face_detected:
                    row["bbox_x"] = res.bounding_box[0]
                    row["bbox_y"] = res.bounding_box[1]
                    row["bbox_w"] = res.bounding_box[2]
                    row["bbox_h"] = res.bounding_box[3]

                    # Store key reference landmarks:
                    # Nose tip: 4
                    row["nose_tip_x"] = res.landmarks_normalized[4, 0]
                    row["nose_tip_y"] = res.landmarks_normalized[4, 1]
                    row["nose_tip_z"] = res.landmarks_normalized[4, 2]

                    # Left eye outer corner: 33, Right eye outer corner: 263
                    row["left_eye_x"] = res.landmarks_normalized[33, 0]
                    row["left_eye_y"] = res.landmarks_normalized[33, 1]
                    row["right_eye_x"] = res.landmarks_normalized[263, 0]
                    row["right_eye_y"] = res.landmarks_normalized[263, 1]

                    # Mouth left corner: 61, Mouth right corner: 291
                    row["mouth_left_x"] = res.landmarks_normalized[61, 0]
                    row["mouth_left_y"] = res.landmarks_normalized[61, 1]
                    row["mouth_right_x"] = res.landmarks_normalized[291, 0]
                    row["mouth_right_y"] = res.landmarks_normalized[291, 1]

                    # Upper lip center: 13, Lower lip center: 14
                    row["upper_lip_y"] = res.landmarks_normalized[13, 1]
                    row["lower_lip_y"] = res.landmarks_normalized[14, 1]

                    # Irises
                    if res.left_iris_center:
                        row["left_iris_x"] = res.left_iris_center[0]
                        row["left_iris_y"] = res.left_iris_center[1]
                    if res.right_iris_center:
                        row["right_iris_x"] = res.right_iris_center[0]
                        row["right_iris_y"] = res.right_iris_center[1]

                    # Key blendshapes for facial asymmetry and palsy:
                    for bs_name in [
                        "eyeBlinkLeft", "eyeBlinkRight",
                        "eyeSquintLeft", "eyeSquintRight",
                        "mouthSmileLeft", "mouthSmileRight",
                        "mouthFrownLeft", "mouthFrownRight",
                        "mouthStretchLeft", "mouthStretchRight",
                        "browDownLeft", "browDownRight",
                        "browInnerUp", "jawOpen"
                    ]:
                        row[bs_name] = res.blendshapes.get(bs_name, 0.0)

                rows.append(row)
                processed_count += 1

                if max_frames and processed_count >= max_frames:
                    break

            if max_frames and processed_count >= max_frames:
                break

    elapsed = time.time() - start_time
    fps = processed_count / elapsed if elapsed > 0 else 0.0

    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)

    file_size_kb = csv_file.stat().st_size / 1024
    logger.info(
        f"Completed Subject {subject_id}: {processed_count} frames processed in "
        f"{elapsed:.2f}s ({fps:.1f} FPS). Saved {file_size_kb:.1f} KB -> {csv_file}"
    )

    return csv_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 478 landmarks from YFP dataset.")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1-24)")
    parser.add_argument("--max_frames", type=int, default=50, help="Max frames to process")
    args = parser.parse_args()

    extract_subject_landmarks(subject_id=args.subject, max_frames=args.max_frames)
