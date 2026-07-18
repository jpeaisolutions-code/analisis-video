"""Detección de eventos clave por heurísticas geométricas sobre los tracks.

Eventos soportados:
- goal: el balón cruza una línea de gol entre los postes (requiere calibración).
  Se refuerza/confirma con el OCR del marcador si está disponible.
- corner: el balón sale por la línea de fondo fuera de los postes y reaparece
  cerca de una esquina (requiere calibración).
- shot: el balón se mueve rápido hacia el arco desde cerca del área
  (requiere calibración).

Sin calibración de cancha solo se emiten eventos del OCR del marcador (goles).
Las tarjetas NO se detectan en v1 (ver README: limitaciones conocidas).
"""

from dataclasses import dataclass, field

import numpy as np

from .pitch import GOAL_Y_MAX, GOAL_Y_MIN, PITCH_LENGTH_M, PITCH_WIDTH_M, PitchCalibration

SHOT_SPEED_M_S = 15.0
SHOT_MAX_DIST_M = 30.0
CORNER_ZONE_M = 3.0
EVENT_COOLDOWN_S = 8.0


@dataclass
class Event:
    kind: str  # "goal" | "corner" | "shot"
    time_s: float
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "time_s": round(self.time_s, 1),
            "time": f"{int(self.time_s // 60):02d}:{int(self.time_s % 60):02d}",
            "detail": self.detail,
        }


@dataclass
class EventDetector:
    calibration: PitchCalibration | None
    fps: float
    events: list[Event] = field(default_factory=list)
    _ball_history: list[tuple[float, np.ndarray]] = field(default_factory=list)
    _last_event_time: dict[str, float] = field(default_factory=dict)

    def update(self, time_s: float, ball_xy_px: tuple[float, float] | None) -> None:
        if self.calibration is None or ball_xy_px is None:
            return
        ball = self.calibration.to_pitch(np.array([ball_xy_px]))[0]
        self._ball_history.append((time_s, ball))
        # Ventana de ~2 segundos de trayectoria
        cutoff = time_s - 2.0
        self._ball_history = [(t, p) for t, p in self._ball_history if t >= cutoff]

        self._check_goal(time_s, ball)
        self._check_corner(time_s, ball)
        self._check_shot(time_s)

    def _cooled(self, kind: str, time_s: float) -> bool:
        last = self._last_event_time.get(kind)
        return last is None or time_s - last >= EVENT_COOLDOWN_S

    def _emit(self, kind: str, time_s: float, detail: str) -> None:
        if not self._cooled(kind, time_s):
            return
        self._last_event_time[kind] = time_s
        self.events.append(Event(kind=kind, time_s=time_s, detail=detail))

    def _check_goal(self, time_s: float, ball: np.ndarray) -> None:
        if len(self._ball_history) < 2:
            return
        _, prev = self._ball_history[-2]
        for goal_x, side in ((0.0, "left"), (PITCH_LENGTH_M, "right")):
            crossed = (prev[0] - goal_x) * (ball[0] - goal_x) < 0 or (
                abs(ball[0] - goal_x) < 0.5 and abs(prev[0] - goal_x) > 0.5
            )
            between_posts = GOAL_Y_MIN <= ball[1] <= GOAL_Y_MAX
            if crossed and between_posts:
                self._emit("goal", time_s, f"balón cruza línea de gol ({side})")

    def _check_corner(self, time_s: float, ball: np.ndarray) -> None:
        near_end_line = ball[0] < CORNER_ZONE_M or ball[0] > PITCH_LENGTH_M - CORNER_ZONE_M
        near_side_line = ball[1] < CORNER_ZONE_M or ball[1] > PITCH_WIDTH_M - CORNER_ZONE_M
        if near_end_line and near_side_line:
            self._emit("corner", time_s, "balón en zona de esquina")

    def _check_shot(self, time_s: float) -> None:
        if len(self._ball_history) < 3:
            return
        (t0, p0), (t1, p1) = self._ball_history[-3], self._ball_history[-1]
        dt = t1 - t0
        if dt <= 0:
            return
        velocity = (p1 - p0) / dt
        speed = float(np.linalg.norm(velocity))
        if speed < SHOT_SPEED_M_S:
            return
        goal_y = PITCH_WIDTH_M / 2
        for goal_x, side in ((0.0, "left"), (PITCH_LENGTH_M, "right")):
            goal_vec = np.array([goal_x, goal_y]) - p1
            dist = float(np.linalg.norm(goal_vec))
            if dist > SHOT_MAX_DIST_M or dist == 0:
                continue
            heading = float(np.dot(velocity / speed, goal_vec / dist))
            if heading > 0.85:
                self._emit(
                    "shot", time_s, f"balón rápido hacia el arco ({side})"
                )
