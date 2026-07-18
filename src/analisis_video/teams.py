"""Clasificación de equipos por color de camiseta (clustering HSV)."""

from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv
from scipy.cluster.vq import kmeans2

TEAM_A = 0
TEAM_B = 1
UNKNOWN = -1


def _shirt_color(frame: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
    """Color HSV medio del torso (mitad superior central de la caja)."""
    x1, y1, x2, y2 = xyxy.astype(int)
    h, w = y2 - y1, x2 - x1
    if h < 10 or w < 5:
        return None
    # Torso: recorte central para evitar césped, brazos y piernas
    ty1 = y1 + int(h * 0.2)
    ty2 = y1 + int(h * 0.5)
    tx1 = x1 + int(w * 0.25)
    tx2 = x2 - int(w * 0.25)
    torso = frame[ty1:ty2, tx1:tx2]
    if torso.size == 0:
        return None
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3).mean(axis=0)


@dataclass
class TeamClassifier:
    """Aprende los 2 colores dominantes de camiseta y clasifica cada track.

    Acumula colores de torso durante los primeros frames (fase de calibración),
    hace k-means con k=2, y después asigna cada track al cluster más cercano.
    El voto se acumula por track_id para que un jugador no cambie de equipo
    por ruido de un frame.
    """

    calibration_samples: int = 300
    _samples: list[np.ndarray] = field(default_factory=list, init=False)
    _centroids: np.ndarray | None = field(default=None, init=False)
    _votes: dict[int, np.ndarray] = field(default_factory=dict, init=False)

    @property
    def calibrated(self) -> bool:
        return self._centroids is not None

    def update(self, frame: np.ndarray, persons: sv.Detections) -> np.ndarray:
        """Devuelve un array de equipo (TEAM_A/TEAM_B/UNKNOWN) por detección."""
        colors = [_shirt_color(frame, xyxy) for xyxy in persons.xyxy]

        if not self.calibrated:
            self._samples.extend(c for c in colors if c is not None)
            if len(self._samples) >= self.calibration_samples:
                data = np.array(self._samples, dtype=np.float64)
                self._centroids, _ = kmeans2(data, 2, minit="++", seed=42)
            return np.full(len(persons), UNKNOWN, dtype=int)

        teams = np.full(len(persons), UNKNOWN, dtype=int)
        tracker_ids = (
            persons.tracker_id
            if persons.tracker_id is not None
            else [None] * len(persons)
        )
        for i, (color, tid) in enumerate(zip(colors, tracker_ids)):
            if color is None:
                continue
            dists = np.linalg.norm(self._centroids - color, axis=1)
            team = int(np.argmin(dists))
            if tid is None:
                teams[i] = team
                continue
            votes = self._votes.setdefault(int(tid), np.zeros(2))
            votes[team] += 1
            teams[i] = int(np.argmax(votes))
        return teams
