"""Detección de jugadores, porteros, árbitros y balón.

Usa dos modelos YOLO afinados específicamente para fútbol (en vez de un YOLO
genérico entrenado en COCO, que solo distingue "person" y "sports ball" — no
separa árbitro/portero de jugador, y detecta mal un balón pequeño y borroso
en plano amplio):

- `player_model`: jugador/portero/árbitro/balón en una sola pasada.
- `ball_model`: modelo pequeño especializado solo en balón, con mejor
  recall/precisión que la clase "ball" del modelo anterior (según su
  evaluación: recall 0.80 vs 0.68, mAP50-95 0.55 vs 0.34) — es la fuente de
  verdad para el balón; el modelo de jugadores se usa solo para personas.

Los árbitros se descartan aquí mismo: no deben entrar en tracking,
clasificación de equipo ni estadísticas.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import supervision as sv
from ultralytics import YOLO

# Clases de martinjolif/yolo-football-player-detection — confirmadas via
# `model.names`, NO coinciden con el orden listado en su README:
# {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

DEFAULT_PLAYER_MODEL = "data/models/yolo-football-player-detection.pt"
DEFAULT_BALL_MODEL = "data/models/yolo-football-ball-detection.pt"


@dataclass
class FrameDetections:
    """Detecciones de un frame: personas (jugadores+porteros) y balón."""

    frame_index: int
    persons: sv.Detections
    ball: sv.Detections


class Detector:
    def __init__(
        self,
        player_model_path: str | Path = DEFAULT_PLAYER_MODEL,
        ball_model_path: str | Path = DEFAULT_BALL_MODEL,
        confidence: float = 0.3,
        ball_confidence: float = 0.15,
        device: str | None = None,
    ):
        self.player_model = YOLO(str(player_model_path))
        self.ball_model = YOLO(str(ball_model_path))
        self.confidence = confidence
        # El balón sigue siendo pequeño y borroso incluso para el modelo
        # especializado; umbral más bajo que para personas.
        self.ball_confidence = ball_confidence
        self.device = device

    def detect(self, frame: np.ndarray, frame_index: int = 0) -> FrameDetections:
        player_result = self.player_model(
            frame,
            classes=[GOALKEEPER_CLASS_ID, PLAYER_CLASS_ID],
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        persons = sv.Detections.from_ultralytics(player_result)
        persons = persons[persons.confidence >= self.confidence]

        ball_result = self.ball_model(
            frame,
            conf=self.ball_confidence,
            device=self.device,
            verbose=False,
        )[0]
        ball = sv.Detections.from_ultralytics(ball_result)
        ball = ball[ball.confidence >= self.ball_confidence]
        if len(ball) > 1:
            ball = ball[[int(np.argmax(ball.confidence))]]

        return FrameDetections(frame_index=frame_index, persons=persons, ball=ball)
