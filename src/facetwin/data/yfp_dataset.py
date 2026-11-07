"""YFP (YouTube Facial Palsy) Dataset Management Module.

Provides storage-efficient access to the YFP dataset directly from ZIP archives
or pre-extracted unencrypted directories. Supports password-protected ZIP decoding,
fallback directory serving, and batch-based temporal streaming.
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
    """Metadata for a single frame inside an archive or directory."""
    subject_id: int
    frame_number: int
    archive_name: str
    archive_path: Path
    internal_path: str
    file_size_bytes: int
    is_encrypted: bool = False
    fallback_path: Optional[Path] = None


@dataclass
class YFPArchiveStatus:
    """Readiness and inventory status of an archive."""
    archive_id: str
    path: Path
    exists: bool
    readable: bool
    is_encrypted: bool
    file_size_bytes: int
    file_size_gb: float
    total_images: int
    subjects: List[int] = field(default_factory=list)


class YFPDatasetManager:
    """Manages YouTube Facial Palsy (YFP) dataset archives and direct in-memory access."""

    def __init__(
        self,
        archive_1: Optional[str | Path] = None,
        archive_2: Optional[str | Path] = None,
        archive_3: Optional[str | Path] = None,
        extracted_dir: Optional[str | Path] = None,
        password: Optional[str | bytes] = None,
    ):
        """Initializes manager with configurable paths via parameters or environment."""
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

        # Check for unencrypted fallback directories (YFP_Database or YFP_Extracted_2026)
        self.fallback_dirs = [
            user_downloads / "YFP_Database",
            self.extracted_dir,
        ]

        raw_pwd = password or os.environ.get("YFP_ARCHIVE_PASSWORD")
        self.password: Optional[bytes] = None
        if raw_pwd is not None:
            self.password = raw_pwd.encode("utf-8") if isinstance(raw_pwd, str) else raw_pwd

        self._frame_index: Dict[Tuple[int, int], YFPFrameInfo] = {}
        self._subject_to_frames: Dict[int, List[YFPFrameInfo]] = {}
        self._is_indexed = False

    def get_archive_status(self) -> Dict[str, YFPArchiveStatus]:
        """Checks existence, readability, encryption, and subject partitions."""
        status: Dict[str, YFPArchiveStatus] = {}

        for name, path in self.archive_paths.items():
            exists = path.exists()
            readable = os.access(path, os.R_OK) if exists else False
            size_bytes = path.stat().st_size if exists else 0
            size_gb = size_bytes / (1024 ** 3) if exists else 0.0

            total_images = 0
            subjects: Set[int] = set()
            is_encrypted = False

            if exists and readable:
                try:
                    with zipfile.ZipFile(path, "r") as zf:
                        for info in zf.infolist():
                            if not info.is_dir() and info.filename.lower().endswith(".bmp"):
                                total_images += 1
                                if (info.flag_bits & 0x1) != 0:
                                    is_encrypted = True
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
                is_encrypted=is_encrypted,
                file_size_bytes=size_bytes,
                file_size_gb=size_gb,
                total_images=total_images,
                subjects=sorted(list(subjects)),
            )

        return status

    def build_index(self, force_rebuild: bool = False) -> None:
        """Constructs an in-memory catalog of all available frames."""
        if self._is_indexed and not force_rebuild:
            return

        self._frame_index.clear()
        self._subject_to_frames.clear()

        # Build fallback file lookup if fallback directories exist
        fallback_lookup: Dict[Tuple[int, int], Path] = {}
        for fb_dir in self.fallback_dirs:
            if fb_dir.exists():
                for img_path in fb_dir.rglob("*.bmp"):
                    parts = img_path.parts
                    # e.g., .../Image/1/156.bmp
                    if len(parts) >= 3 and parts[-2].isdigit():
                        s_id = int(parts[-2])
                        f_stem = img_path.stem
                        if f_stem.isdigit():
                            fallback_lookup[(s_id, int(f_stem))] = img_path

        for name, path in self.archive_paths.items():
            if not path.exists() or not os.access(path, os.R_OK):
                continue

            with zipfile.ZipFile(path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir() or not info.filename.lower().endswith(".bmp"):
                        continue

                    parts = Path(info.filename).parts
                    if len(parts) >= 3 and parts[1].isdigit():
                        subject_id = int(parts[1])
                        frame_stem = Path(parts[2]).stem
                        frame_num = int(frame_stem) if frame_stem.isdigit() else 0
                        is_enc = (info.flag_bits & 0x1) != 0

                        fb_path = fallback_lookup.get((subject_id, frame_num))

                        frame_info = YFPFrameInfo(
                            subject_id=subject_id,
                            frame_number=frame_num,
                            archive_name=name,
                            archive_path=path,
                            internal_path=info.filename,
                            file_size_bytes=info.file_size,
                            is_encrypted=is_enc,
                            fallback_path=fb_path,
                        )

                        self._frame_index[(subject_id, frame_num)] = frame_info
                        if subject_id not in self._subject_to_frames:
                            self._subject_to_frames[subject_id] = []
                        self._subject_to_frames[subject_id].append(frame_info)

        # Sort frames per subject
        for subj in self._subject_to_frames:
            self._subject_to_frames[subj].sort(key=lambda x: x.frame_number)

        self._is_indexed = True
        logger.info(
            f"YFP index built: {len(self._frame_index)} frames across "
            f"{len(self._subject_to_frames)} subjects."
        )

    def list_subjects(self) -> List[int]:
        """Returns sorted list of indexed subject IDs."""
        if not self._is_indexed:
            self.build_index()
        return sorted(list(self._subject_to_frames.keys()))

    def get_subject_frame_count(self, subject_id: int) -> int:
        """Returns total frame count for subject."""
        if not self._is_indexed:
            self.build_index()
        return len(self._subject_to_frames.get(subject_id, []))

    def read_frame_bytes(self, subject_id: int, frame_number: int) -> bytes:
        """Reads raw bytes of frame from ZIP (or unencrypted fallback directory)."""
        if not self._is_indexed:
            self.build_index()

        key = (subject_id, frame_number)
        if key not in self._frame_index:
            raise VideoProcessingError(
                f"Frame ({subject_id}, {frame_number}) not found in index."
            )

        info = self._frame_index[key]

        # 1. If fallback unencrypted file exists, read it directly
        if info.fallback_path is not None and info.fallback_path.exists():
            return info.fallback_path.read_bytes()

        # 2. Otherwise read from ZIP using password if available
        with zipfile.ZipFile(info.archive_path, "r") as zf:
            if info.is_encrypted:
                if self.password is None:
                    raise VideoProcessingError(
                        f"Frame ({subject_id}, {frame_number}) in {info.archive_name} is password protected. "
                        "Set YFP_ARCHIVE_PASSWORD in environment or pass password to YFPDatasetManager."
                    )
                return zf.read(info.internal_path, pwd=self.password)
            return zf.read(info.internal_path)

    def read_frame_image(self, subject_id: int, frame_number: int) -> Any:
        """Decodes frame directly into an RGB image array."""
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
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        else:
            raise VideoProcessingError(
                "Neither OpenCV nor PIL is available to decode image bytes."
            )

    def iter_subject_batches(
        self, subject_id: int, batch_size: int = 32
    ) -> Generator[List[Tuple[int, Any]], None, None]:
        """Yields batches of (frame_number, rgb_image_array) for sequential processing."""
        if not self._is_indexed:
            self.build_index()

        frames = self._subject_to_frames.get(subject_id, [])
        if not frames:
            return

        current_batch: List[Tuple[int, Any]] = []

        for frame_info in frames:
            img = self.read_frame_image(subject_id, frame_info.frame_number)
            current_batch.append((frame_info.frame_number, img))

            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []

        if current_batch:
            yield current_batch
