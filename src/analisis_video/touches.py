"""Capa de "toques": posesión discreta por jugador, no por equipo.

`stats.MatchStats` ya sabe qué equipo tiene el balón más cerca en cada frame,
pero solo lleva un contador acumulado por equipo (para el % de posesión) —
no conserva quién fue, ni cuándo empezó y acabó cada contacto. Este módulo
detecta el mismo tipo de proximidad balón-jugador pero a nivel de `track_id`
y agrupa frames consecutivos del mismo jugador en un evento discreto (un
"toque"): con eso, dos toques consecutivos de track_ids distintos son
candidatos a pase o duelo, y toques consecutivos del mismo track_id son la
misma posesión continua. Es la capa base de la Fase 3 (pases, duelos,
regates, disparos), que todavía no existe.

Se calcula para TODOS los jugadores detectados (no solo el elegido) porque
en el momento de procesar el vídeo aún no sabemos con certeza final quién es
el jugador elegido: la cadena de `player_track.json` puede corregirse a mano
más tarde en el wizard de la app, sin volver a analizar el vídeo. Filtrar
los toques del jugador elegido a partir de esta capa (ver `target_touches`)
es una operación barata sobre JSON ya guardado, no requiere reprocesar
el vídeo.
"""

from dataclasses import dataclass, field

import numpy as np

from .pitch import PitchCalibration
from .stats import POSSESSION_RADIUS_M, POSSESSION_RADIUS_PX
from .teams import TEAM_A, TEAM_B
from .tracking import TrackedFrame, bottom_center

# Frames consecutivos sin balón detectado o sin nadie dentro de radio antes
# de dar por cerrado el toque en curso — tolera parpadeos sueltos de la
# detección de balón (o del jugador más cercano por ruido de una detección)
# sin fragmentar en toques distintos lo que en realidad es la misma
# posesión continua.
GAP_TOLERANCE_FRAMES = 3

# Toques de muy pocos frames suelen ser ruido (el jugador más cercano
# cambiando de frame a frame durante un duelo apretado) más que un contacto
# real con el balón.
MIN_TOUCH_FRAMES = 2

_TEAM_LABELS = {TEAM_A: "team_a", TEAM_B: "team_b"}


@dataclass
class Touch:
    track_id: int
    team: int
    start_time: float
    end_time: float
    start_pos: tuple[float, float]
    end_pos: tuple[float, float]
    frames: int = 1

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "team": _TEAM_LABELS.get(self.team, "unknown"),
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "position": [round(self.start_pos[0], 1), round(self.start_pos[1], 1)],
        }


@dataclass
class TouchDetector:
    calibration: PitchCalibration | None
    fps: float
    touches: list[Touch] = field(default_factory=list)
    _current: Touch | None = field(default=None, init=False)
    _gap: int = field(default=0, init=False)

    def update(self, tracked: TrackedFrame, teams: np.ndarray) -> None:
        persons = tracked.persons
        if tracked.ball_xy is None or persons.tracker_id is None or len(persons) == 0:
            self._advance_gap()
            return

        feet_px = bottom_center(persons.xyxy)
        if self.calibration:
            ball = self.calibration.to_pitch(np.array([tracked.ball_xy]))[0]
            feet = self.calibration.to_pitch(feet_px)
            radius = POSSESSION_RADIUS_M
        else:
            ball = np.array(tracked.ball_xy)
            feet = feet_px
            radius = POSSESSION_RADIUS_PX

        dists = np.linalg.norm(feet - ball, axis=1)
        nearest = int(np.argmin(dists))
        if dists[nearest] > radius:
            self._advance_gap()
            return

        tid = int(persons.tracker_id[nearest])
        team = int(teams[nearest])
        pos = (float(feet[nearest][0]), float(feet[nearest][1]))

        if self._current is not None and self._current.track_id == tid:
            self._current.end_time = tracked.time_s
            self._current.end_pos = pos
            self._current.frames += 1
            self._gap = 0
        else:
            self._close_current()
            self._current = Touch(
                track_id=tid,
                team=team,
                start_time=tracked.time_s,
                end_time=tracked.time_s,
                start_pos=pos,
                end_pos=pos,
            )
            self._gap = 0

    def _advance_gap(self) -> None:
        if self._current is None:
            return
        self._gap += 1
        if self._gap > GAP_TOLERANCE_FRAMES:
            self._close_current()

    def _close_current(self) -> None:
        if self._current is not None and self._current.frames >= MIN_TOUCH_FRAMES:
            self.touches.append(self._current)
        self._current = None
        self._gap = 0

    def finish(self) -> None:
        """Cierra el toque en curso al terminar el vídeo — llamar una vez tras
        el bucle de frames, si no el último toque nunca se guardaría."""
        self._close_current()

    def to_dict(self) -> list[dict]:
        return [t.to_dict() for t in self.touches]


def target_touches(touches: list[dict], player_track: dict) -> list[dict]:
    """Filtra, de la capa general de toques, los que pertenecen al jugador
    elegido según la cadena de tramos de `player_track.json`.

    Se puede recalcular en cualquier momento a partir de los dos JSON ya
    guardados — en particular, después de que el usuario corrija tramos en
    el wizard de la app, sin volver a analizar el vídeo.
    """
    segments = player_track.get("segments") or []
    result = []
    for touch in touches:
        for seg in segments:
            if (
                touch["track_id"] == seg["track_id"]
                and seg["start_time"] <= touch["start_time"] <= seg["end_time"]
            ):
                result.append(touch)
                break
    return result
