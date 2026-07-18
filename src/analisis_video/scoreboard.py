"""OCR del marcador en pantalla: señal de refuerzo para detectar goles.

Lee periódicamente la zona del marcador (por defecto, esquina superior
izquierda) y busca un patrón "N-N". Cuando el marcador cambia, se emite un
evento de gol confirmado.
"""

import re
from dataclasses import dataclass, field

import numpy as np

from .events import Event

SCORE_PATTERN = re.compile(r"(\d{1,2})\s*[-–:]\s*(\d{1,2})")


@dataclass
class ScoreboardReader:
    """Lector de marcador. `roi` es (x1, y1, x2, y2) relativo (0-1) al frame."""

    roi: tuple[float, float, float, float] = (0.0, 0.0, 0.35, 0.12)
    interval_s: float = 2.0
    languages: tuple[str, ...] = ("es", "en")
    events: list[Event] = field(default_factory=list)
    _reader: object | None = field(default=None, init=False)
    _last_read_time: float = field(default=-1e9, init=False)
    _score: tuple[int, int] | None = field(default=None, init=False)

    def _get_reader(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(list(self.languages), gpu=False, verbose=False)
        return self._reader

    def update(self, time_s: float, frame: np.ndarray) -> None:
        if time_s - self._last_read_time < self.interval_s:
            return
        self._last_read_time = time_s

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.roi
        crop = frame[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]
        if crop.size == 0:
            return

        texts = self._get_reader().readtext(crop, detail=0)
        match = next(
            (m for t in texts if (m := SCORE_PATTERN.search(t))), None
        )
        if match is None:
            return
        score = (int(match.group(1)), int(match.group(2)))

        if self._score is not None and score != self._score:
            # Solo cambios incrementales de 1 son creíbles como gol
            diff = (score[0] - self._score[0], score[1] - self._score[1])
            if diff in ((1, 0), (0, 1)):
                team = "local" if diff == (1, 0) else "visitante"
                self.events.append(
                    Event(
                        kind="goal",
                        time_s=time_s,
                        detail=f"marcador cambió a {score[0]}-{score[1]} ({team})",
                    )
                )
        self._score = score
