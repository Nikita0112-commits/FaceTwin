"""YFP (YouTube Facial Palsy) Dataset Management Module.

Provides storage-efficient access to the YFP dataset directly from ZIP archives
without requiring disk extraction. Supports index caching, in-memory frame decoding,
and batch-based temporal processing.
"""

from dataclasses import dataclass, field
import io
import os
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
import zipfile

from facetwin.core.exceptions import VideoProcessingError
from facetwin.core.logging_config import setup_logger

logger = setup_logger("facetwin.data.yfp")

# Optional image decoding imports
try:
    import cv2
    import numpy as np
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class YFPFrameInfo:
    """Metadata for a single frame inside an archive."""
    subject_id: int
    frame_number: int
    archive_name: str
    archive_path: Path
    internal_path: str
    file_size_bytes: int


@dataclass
class YFPArchiveStatus:
    """Readiness and inventory status of an archive."""
    archive_id: str
    path: Path
    exists: bool
    readable: bool
    file_size_bytes: int
    file_size_gb: float
    total_images: int
    subjects: List[int] = field(default_factory=list)


class YFPDatasetManager:
    """Manages YouTube Facial Palsy (YFP) dataset archives and direct in-memory access.

    This manager implements zero-disk-duplication reading by querying and decoding
    frames directly from the original ZIP archives in memory.
    """

    def __init__(
        self,
        archive_1: Optional[str | Path] = None,
        archive_2: Optional[str | Path] = None,
        archive_3: Optional[str | Path] = None,
        extracted_dir: Optional[str | Path] = None,
    ):
        """Initializes manager with configurable paths via parameters or environment.

        Environment variables:
            YFP_ARCHIVE_1: Path to Image.zip (Subjects 1-8)
            YFP_ARCHIVE_2: Path to Image2.zip (Subjects 9-16)
            YFP_ARCHIVE_3: Path to Image3.zip (Subjects 17-24)
            YFP_EXTRACTED_DIR: Path to partially extracted directory
        """
        # Resolve user home dynamically to prevent hardcoding personal paths
        user_downloads = Path.home() / "Downloads"

        self.archive_paths = {
            "Image.zip": Path(
                archive_1
                or os.environ.get("YFP_ARCHIVE_1")
                or (user_downloads / "Image.zip")
            ),
            "Image2.zip": Path(
                archive_2
                or os.environ.get("YFP_ARCHIVE_2")
                or (user_downloads / "Image2.zip")
            ),
            "Image3.zip": Path(
                archive_3
                or os.environ.get("YFP_ARCHIVE_3")
                or (user_downloads / "Image3.zip")
            ),
        }

        self.extracted_dir = Path(
            extracted_dir
            or os.environ.get("YFP_EXTRACTED_DIR")
            or (user_downloads / "YFP_Extracted_2026")
        )

        self._frame_index: Dict[Tuple[int, int], YFPFrameInfo] = {}
        self._subject_to_frames: Dict[int, List[YFPFrameInfo]] = {}
        self._is_indexed = False

    def get_archive_status(self) -> Dict[str, YFPArchiveStatus]:
        """Checks existence, readability, size, and subject partitions for all archives.

        Returns:
            Dict[str, YFPArchiveStatus]: Status keyed by archive identifier.
        """
        status: Dict[str, YFPArchiveStatus] = {}

        for name, path in self.archive_paths.items():
            exists = path.exists()
            readable = os.access(path, os.R_OK) if exists else False
            size_bytes = path.stat().st_size if exists else 0
            size_gb = size_bytes / (1024 ** 3) if exists else 0.0

            total_images = 0
            subjects: Set[int] = set()

            if exists and readable:
                try:
                    with zipfile.ZipFile(path, "r") as zf:
                        for info in zf.infolist():
                            if not info.is_dir() and info.filename.lower().endswith(".bmp"):
                                total_images += 1
                                parts = Path(info.filename).parts
                                if len(parts) >= 2 and parts[1].isdigit():
                                    subjects.add(int(parts[1]))
                except Exception as e:
                    logger.error(f"Error inspecting archive {path}: {e}")

            status[name] = YFPArchiveStatus(
                archive_id=name,
                path=path,
                exists=exists,
                readable=readable,
                file_size_bytes=size_bytes,
                file_size_gb=size_gb,
                total_images=total_images,
                subjects=sorted(list(subjects)),
            )

        return status

    def get_extracted_dir_status(self) -> Dict[str, Any]:
        """Inspects the partially extracted directory to determine redundancy.

        Returns:
            Dict[str, Any]: Details on file count, storage footprint, and archive source.
        """
        if not self.extracted_dir.exists():
            return {"exists": False, "total_files": 0, "size_gb": 0.0, "subdirs": []}

        try:
            files = [f for f in self.extracted_dir.rglob("*") if f.is_file()]
            total_size = sum(f.stat().st_size for f in files)
            subdirs: Set[str] = set()
            for f in files:
                rel = f.relative_to(self.extracted_dir).as_posix()
                parts = Path(rel).parts
                if len(parts) > 1:
                    subdirs.add(f"{parts[0]}/{parts[1]}")

            return {
                "exists": True,
                "path": str(self.extracted_dir),
                "total_files": len(files),
                "size_gb": total_size / (1024 ** 3),
                "subdirs": sorted(list(subdirs)),
            }
        except PermissionError:
            return {
                "exists": True,
                "path": str(self.extracted_dir),
                "error": "Permission denied without bypass permissions.",
            }

    def build_index(self, force_rebuild: bool = False) -> None:
        """Constructs an in-memory catalog of all available frames across all archives.

        Args:
            force_rebuild: If True, forces re-reading ZIP table of contents.
        """
        if self._is_indexed and not force_rebuild:
            return

        self._frame_index.clear()
        self._subject_to_frames.clear()

        for name, path in self.archive_paths.items():
            if not path.exists() or not os.access(path, os.R_OK):
                logger.warning(f"Archive not readable or missing: {path}")
                continue

            with zipfile.ZipFile(path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".bmp"):
                        continue

                    # Expected internal path format: "Image/1/156.bmp" or "Image2/9/201.bmp"
                    parts = Path(info.filename).parts
                    if len(parts) >= 3 and parts[1].isdigit():
                        subject_id = int(parts[1])
                        frame_stem = Path(parts[2]).stem
                        frame_num = int(frame_stem) if frame_stem.isdigit() else 0

                        frame_info = YFPFrameInfo(
                            subject_id=subject_id,
                            frame_number=frame_num,
                            archive_name=name,
                            archive_path=path,
                            internal_path=info.filename,
                            file_size_bytes=info.file_size,
                        )

                        self._frame_index[(subject_id, frame_num)] = frame_info
                        if subject_id not in self._subject_to_frames:
                            self._subject_to_frames[subject_id] = []
                        self._subject_to_frames[subject_id].append(frame_info)

        # Sort frames per subject sequentially
        for subj in self._subject_to_frames:
            self._subject_to_frames[subj].sort(key=lambda x: x.frame_number)

        self._is_indexed = True
        logger.info(
            f"YFP index built: {len(self._frame_index)} frames across "
            f"{len(self._subject_to_frames)} subjects."
        )

    def list_subjects(self) -> List[int]:
        """Returns sorted list of all indexed subject IDs."""
        if not self._is_indexed:
            self.build_index()
        return sorted(list(self._subject_to_frames.keys()))

    def get_subject_frame_count(self, subject_id: int) -> int:
        """Returns total available frames for a specific subject."""
        if not self._is_indexed:
            self.build_index()
        return len(self._subject_to_frames.get(subject_id, []))

    def read_frame_bytes(self, subject_id: int, frame_number: int) -> bytes:
        """Reads raw bytes of a specified frame directly from the appropriate archive.

        Args:
            subject_id: Subject index (1 to 24)
            frame_number: Sequential frame index

        Returns:
            bytes: Uncompressed image bytes (.bmp)
        """
        if not self._is_indexed:
            self.build_index()

        key = (subject_id, frame_number)
        if key not in self._frame_index:
            raise VideoProcessingError(
                f"Frame ({subject_id}, {frame_number}) not found in any indexed archive."
            )

        info = self._frame_index[key]
        with zipfile.ZipFile(info.archive_path, "r") as zf:
            return zf.read(info.internal_path)

    def read_frame_image(self, subject_id: int, frame_number: int) -> Any:
        """Decodes frame directly into an RGB image array in memory without saving to disk.

        Returns:
            np.ndarray: Decoded image array (RGB format).
        """
        raw_bytes = self.read_frame_bytes(subject_id, frame_number)

        if HAS_OPENCV:
            np_arr = np.frombuffer(raw_bytes, np.uint8)
            bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise VideoProcessingError(
                    f"Failed to decode frame ({subject_id}, {frame_number}) with OpenCV."
                )
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif HAS_PIL:
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            return img
        else:
            raise VideoProcessingError(
                "Neither OpenCV nor PIL is available to decode image bytes."
            )

    def iter_subject_batches(
        self, subject_id: int, batch_size: int = 32
    ) -> Generator[List[Tuple[int, Any]], None, None]:
        """Yields batches of decoded frames for memory-efficient sequential processing.

        Args:
            subject_id: Subject index
            batch_size: Number of frames per batch

        Yields:
            List[Tuple[int, np.ndarray]]: List of (frame_number, rgb_image_array)
        """
        if not self._is_indexed:
            self.build_index()

        frames = self._subject_to_frames.get(subject_id, [])
        if not frames:
            return

        current_batch: List[Tuple[int, Any]] = []
        # Group reads by archive to minimize ZipFile reopen overhead
        archive_path = frames[0].archive_path

        with zipfile.ZipFile(archive_path, "r") as zf:
            for frame_info in frames:
                raw_bytes = zf.read(frame_info.internal_path)
                if HAS_OPENCV:
                    np_arr = np.frombuffer(raw_bytes, np.uint8)
                    bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                elif HAS_PIL:
                    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                else:
                    img = raw_bytes

                current_batch.append((frame_info.frame_number, img))

                if len(current_batch) >= batch_size:
                    yield current_batch
                    current_batch = []

            if current_batch:
                yield current_batch
