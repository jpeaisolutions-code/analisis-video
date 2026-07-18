"""Tracking de jugadores con ByteTrack: asigna IDs consistentes entre frames."""

from dataclasses import dataclass, field

import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from .detection import FrameDetections


@dataclass
class TrackedFrame:
    """Un frame con tracks de personas (con ID) y posición del balón (sin ID)."""

    frame_index: int
    time_s: float
    persons: sv.Detections
    ball_xy: tuple[float, float] | None


@dataclass
class BallSmoother:
    """Suaviza la posición del balón e interpola huecos cortos de detección."""

    max_gap: int = 12
    _last_xy: tuple[float, float] | None = field(default=None, init=False)
    _gap: int = field(default=0, init=False)

    def update(self, ball: sv.Detections) -> tuple[float, float] | None:
        if len(ball) > 0:
            x1, y1, x2, y2 = ball.xyxy[0]
            self._last_xy = (float((x1 + x2) / 2), float((y1 + y2) / 2))
            self._gap = 0
            return self._last_xy
        self._gap += 1
        if self._last_xy is not None and self._gap <= self.max_gap:
            return self._last_xy
        return None


class Tracker:
    def __init__(self, fps: float):
        self.fps = fps
        self.byte_track = ByteTrackTracker(
            frame_rate=fps,
            track_activation_threshold=0.25,
            high_conf_det_threshold=0.5,
        )
        self.ball_smoother = BallSmoother()

    def update(self, detections: FrameDetections) -> TrackedFrame:
        persons = self.byte_track.update(detections.persons)
        if persons.tracker_id is not None:
            # -1 = track aún no confirmado por el tracker
            persons = persons[persons.tracker_id != -1]
        ball_xy = self.ball_smoother.update(detections.ball)
        return TrackedFrame(
            frame_index=detections.frame_index,
            time_s=detections.frame_index / self.fps,
            persons=persons,
            ball_xy=ball_xy,
        )


def bottom_center(xyxy: np.ndarray) -> np.ndarray:
    """Punto de apoyo (pies) de cada caja: centro del borde inferior."""
    xy = np.empty((len(xyxy), 2), dtype=np.float32)
    xy[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    xy[:, 1] = xyxy[:, 3]
    return xy
