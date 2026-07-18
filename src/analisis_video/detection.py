"""Detección de jugadores, árbitros y balón con YOLOv8."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import supervision as sv
from ultralytics import YOLO

# Clases del modelo COCO estándar: 0 = person, 32 = sports ball
PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32

DEFAULT_MODEL = "yolov8m.pt"


@dataclass
class FrameDetections:
    """Detecciones de un frame: personas (jugadores/árbitros) y balón."""

    frame_index: int
    persons: sv.Detections
    ball: sv.Detections


class Detector:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        confidence: float = 0.3,
        ball_confidence: float = 0.15,
        device: str | None = None,
    ):
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        # El balón es pequeño y borroso en tomas amplias; umbral más bajo
        self.ball_confidence = ball_confidence
        self.device = device

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> FrameDetections:
        result = self.model(
            frame,
            classes=[PERSON_CLASS_ID, BALL_CLASS_ID],
            conf=min(self.confidence, self.ball_confidence),
            device=self.device,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)

        persons = detections[
            (detections.class_id == PERSON_CLASS_ID)
            & (detections.confidence >= self.confidence)
        ]
        ball = detections[
            (detections.class_id == BALL_CLASS_ID)
            & (detections.confidence >= self.ball_confidence)
        ]
        if len(ball) > 1:
            ball = ball[[int(np.argmax(ball.confidence))]]

        return FrameDetections(frame_index=frame_index, persons=persons, ball=ball)
