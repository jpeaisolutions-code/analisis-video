"""Estadísticas del partido derivadas de los tracks: posesión, distancia,
velocidad y mapas de calor."""

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .pitch import PitchCalibration
from .teams import TEAM_A, TEAM_B, UNKNOWN
from .tracking import TrackedFrame, bottom_center

# Un jugador "posee" el balón si está a menos de esta distancia
POSSESSION_RADIUS_M = 2.5
POSSESSION_RADIUS_PX = 70.0

# Velocidad máxima plausible de un jugador; saltos mayores son errores de track
MAX_SPEED_M_S = 12.0


@dataclass
class PlayerStats:
    track_id: int
    team: int = UNKNOWN
    distance: float = 0.0
    max_speed: float = 0.0
    positions: list[tuple[float, float]] = field(default_factory=list)
    _last_pos: np.ndarray | None = None
    _last_time: float | None = None


@dataclass
class MatchStats:
    calibration: PitchCalibration | None
    fps: float
    players: dict[int, PlayerStats] = field(default_factory=dict)
    possession_frames: dict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    frames_seen: int = 0

    @property
    def units(self) -> str:
        return "m" if self.calibration else "px"

    def update(self, tracked: TrackedFrame, teams: np.ndarray) -> None:
        self.frames_seen += 1
        persons = tracked.persons
        if persons.tracker_id is None or len(persons) == 0:
            return

        feet_px = bottom_center(persons.xyxy)
        feet = (
            self.calibration.to_pitch(feet_px) if self.calibration else feet_px
        )

        for i, tid in enumerate(persons.tracker_id):
            tid = int(tid)
            player = self.players.setdefault(tid, PlayerStats(track_id=tid))
            team = int(teams[i])
            if team != UNKNOWN:
                player.team = team

            pos = feet[i]
            player.positions.append((float(pos[0]), float(pos[1])))
            if player._last_pos is not None and player._last_time is not None:
                dt = tracked.time_s - player._last_time
                if dt > 0:
                    step = float(np.linalg.norm(pos - player._last_pos))
                    speed = step / dt
                    max_speed = (
                        MAX_SPEED_M_S
                        if self.calibration
                        else MAX_SPEED_M_S * 30  # sin calibrar: umbral laxo en px
                    )
                    if speed <= max_speed:
                        player.distance += step
                        player.max_speed = max(player.max_speed, speed)
            player._last_pos = pos
            player._last_time = tracked.time_s

        self._update_possession(tracked, teams, feet_px)

    def _update_possession(
        self, tracked: TrackedFrame, teams: np.ndarray, feet_px: np.ndarray
    ) -> None:
        if tracked.ball_xy is None or len(feet_px) == 0:
            return
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
        if dists[nearest] <= radius and teams[nearest] != UNKNOWN:
            self.possession_frames[int(teams[nearest])] += 1

    def possession_pct(self) -> dict[str, float]:
        total = self.possession_frames[TEAM_A] + self.possession_frames[TEAM_B]
        if total == 0:
            return {"team_a": 0.0, "team_b": 0.0}
        return {
            "team_a": round(100 * self.possession_frames[TEAM_A] / total, 1),
            "team_b": round(100 * self.possession_frames[TEAM_B] / total, 1),
        }

    def to_dict(self) -> dict:
        return {
            "units": self.units,
            "possession_pct": self.possession_pct(),
            "players": [
                {
                    "track_id": p.track_id,
                    "team": {TEAM_A: "team_a", TEAM_B: "team_b"}.get(
                        p.team, "unknown"
                    ),
                    "distance": round(p.distance, 1),
                    "max_speed": round(p.max_speed, 2),
                }
                for p in sorted(
                    self.players.values(), key=lambda p: -p.distance
                )
                # Tracks con poca vida suelen ser falsos positivos o fragmentos
                if len(p.positions) >= self.fps * 2
            ],
        }
