"""Lectura y escritura de video."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: Path
    fps: float
    width: int
    height: int
    total_frames: int

    @property
    def duration_s(self) -> float:
        return self.total_frames / self.fps if self.fps else 0.0


def get_video_info(path: str | Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {path}")
    info = VideoInfo(
        path=Path(path),
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def iter_frames(
    path: str | Path,
    stride: int = 1,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Itera (frame_index, frame BGR) cada `stride` frames en [start_s, end_s]."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps) if end_s is not None else None
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    index = start_frame
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (end_frame is not None and index > end_frame):
                break
            if (index - start_frame) % stride == 0:
                yield index, frame
            index += 1
    finally:
        cap.release()


class VideoWriter:
    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
