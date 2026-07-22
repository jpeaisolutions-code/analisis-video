"""Calibración de cancha: homografía de píxeles a coordenadas reales (metros).

Dimensiones estándar FIFA: 105 x 68 m. Origen en la esquina superior izquierda
de la vista cenital, eje X a lo largo de la cancha, eje Y a lo ancho.

v1 soporta calibración manual: un JSON con >= 4 correspondencias
píxel -> cancha para cámara fija. Ejemplo:

    {
      "points": [
        {"pixel": [102, 340], "pitch": [0, 0]},
        {"pixel": [1810, 355], "pitch": [105, 0]},
        {"pixel": [1650, 980], "pitch": [105, 68]},
        {"pixel": [240, 970], "pitch": [0, 68]}
      ]
    }

Sin calibración, el pipeline sigue funcionando pero las distancias/velocidades
se reportan en píxeles y los eventos geométricos (gol, córner) se desactivan.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
GOAL_WIDTH_M = 7.32
GOAL_Y_MIN = (PITCH_WIDTH_M - GOAL_WIDTH_M) / 2
GOAL_Y_MAX = (PITCH_WIDTH_M + GOAL_WIDTH_M) / 2

# Las 4 esquinas de la cancha, en el orden en que se le pide al usuario que
# haga clic sobre el frame: superior-izq, superior-der, inferior-der,
# inferior-izq (recorriendo el contorno en sentido horario).
PITCH_CORNERS = (
    (0.0, 0.0),
    (PITCH_LENGTH_M, 0.0),
    (PITCH_LENGTH_M, PITCH_WIDTH_M),
    (0.0, PITCH_WIDTH_M),
)


@dataclass
class PitchCalibration:
    homography: np.ndarray  # 3x3, píxel -> metros

    @classmethod
    def from_points(
        cls, pixel_points: list | np.ndarray, pitch_points: list | np.ndarray
    ) -> "PitchCalibration":
        if len(pixel_points) < 4 or len(pitch_points) != len(pixel_points):
            raise ValueError("Se necesitan al menos 4 correspondencias píxel->cancha")
        src = np.array(pixel_points, dtype=np.float64)
        dst = np.array(pitch_points, dtype=np.float64)
        homography, _ = cv2.findHomography(src, dst, cv2.RANSAC)
        if homography is None:
            raise ValueError("No se pudo estimar la homografía con esos puntos")
        return cls(homography=homography)

    @classmethod
    def from_pixel_corners(cls, pixel_corners: list | np.ndarray) -> "PitchCalibration":
        """Atajo cuando los puntos clicados son las 4 esquinas de la cancha,
        en el orden de `PITCH_CORNERS` (superior-izq → horario)."""
        return cls.from_points(pixel_corners, PITCH_CORNERS)

    @classmethod
    def from_json(cls, path: str | Path) -> "PitchCalibration":
        data = json.loads(Path(path).read_text())
        points = data["points"]
        return cls.from_points(
            [p["pixel"] for p in points], [p["pitch"] for p in points]
        )

    def to_json(self, path: str | Path, pixel_points, pitch_points) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "points": [
                        {"pixel": list(px), "pitch": list(pt)}
                        for px, pt in zip(pixel_points, pitch_points)
                    ]
                },
                indent=2,
            )
        )

    def to_pitch(self, points_xy: np.ndarray) -> np.ndarray:
        """Transforma puntos (N, 2) de píxeles a metros sobre la cancha."""
        if len(points_xy) == 0:
            return points_xy.reshape(0, 2)
        pts = points_xy.reshape(-1, 1, 2).astype(np.float64)
        out = cv2.perspectiveTransform(pts, self.homography)
        return out.reshape(-1, 2)


def load_calibration(path: str | Path | None) -> PitchCalibration | None:
    if path is None:
        return None
    return PitchCalibration.from_json(path)
