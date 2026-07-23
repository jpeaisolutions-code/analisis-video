"""Seguimiento de UN jugador elegido, encadenando tramos del tracker
multi-objeto ya existente (BoT-SORT).

No se reentrena ni se sustituye el tracker: se apoya en él. Un "tramo" es
la vida de un track_id (desde su primera hasta su última aparición). El
usuario elige al jugador con un clic; a partir de ahí se encadenan tramos
del mismo equipo cuya posición inicial es coherente con la trayectoria
extrapolada del tramo anterior y cuyo color de camiseta es parecido. Si el
mejor candidato no destaca claramente sobre el resto, el tramo se marca
"revisar" para que el usuario lo confirme a mano en la app — el balón y
los demás jugadores no necesitan este tratamiento porque no se les exige
identidad estable en todo el partido, solo al jugador elegido.
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .teams import UNKNOWN, _shirt_color
from .tracking import TrackedFrame, bottom_center

MAX_GAP_S = 8.0
CONFIDENCE_MARGIN = 1.6  # el mejor candidato debe "ganar" al 2º por este factor para autoenlazar
COLOR_WEIGHT = 1.5  # peso relativo de la diferencia de color de camiseta frente a la distancia espacial (px)
THUMB_SIZE = (160, 200)  # ancho, alto
# El plano táctico de Veo deja al jugador en muy pocos píxeles reales; recortar
# justo su caja y ampliarla no recupera detalle que nunca existió. En vez de
# eso se recorta con contexto alrededor (más viable reconocerse por posición
# en el campo/jugadores cercanos que por rasgos a esta resolución) y se
# resalta con un recuadro cuál de los jugadores del recorte es el candidato.
THUMB_PAD_X = 1.2  # contexto horizontal, relativo al ancho del propio jugador
THUMB_PAD_Y = 0.6  # contexto vertical, relativo a su propia altura

# El pool usado para el enlace automático es estricto (mismo equipo, MAX_GAP_S)
# a propósito. Pero cuando el tramo queda "revisar", ese mismo filtro puede
# dejar fuera al jugador real (equipo mal clasificado, o el track real no
# entra en el top-4 por score) sin que el usuario tenga forma de saberlo o
# elegirlo. Para la galería del wizard se usa un pool más amplio (cualquier
# equipo, ventana más larga) con el equipo incorrecto como penalización en
# vez de exclusión, para que la opción correcta esté disponible aunque el
# filtro automático la hubiera descartado.
DISPLAY_CANDIDATES = 8
DISPLAY_GAP_S = MAX_GAP_S * 2
TEAM_MISMATCH_PENALTY = 200.0


@dataclass
class _TrackAccum:
    track_id: int
    team: int
    start_frame: int
    start_time: float
    start_pos: tuple[float, float]
    end_frame: int
    end_time: float
    end_pos: tuple[float, float]
    color_sum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    color_n: int = 0
    recent: list = field(default_factory=list)  # [(time, pos)], últimas muestras

    @property
    def color(self) -> np.ndarray:
        if self.color_n == 0:
            return np.zeros(3)
        return self.color_sum / self.color_n

    @property
    def velocity(self) -> tuple[float, float]:
        if len(self.recent) < 2:
            return (0.0, 0.0)
        t0, p0 = self.recent[0]
        t1, p1 = self.recent[-1]
        dt = t1 - t0
        if dt <= 0:
            return (0.0, 0.0)
        return ((p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt)


class PlayerTrackBuilder:
    """Acumula un resumen por track_id durante el pipeline y, al final,
    encadena los tramos que forman la trayectoria del jugador elegido."""

    def __init__(self, target_time_s: float, target_xy: tuple[float, float]):
        self.target_time_s = target_time_s
        self.target_xy = np.array(target_xy, dtype=float)
        self.target_track_id: int | None = None
        self._tracks: dict[int, _TrackAccum] = {}

    def update(
        self,
        frame: np.ndarray,
        tracked: TrackedFrame,
        teams: np.ndarray,
        thumbs_dir: Path | None = None,
    ) -> None:
        persons = tracked.persons
        if persons.tracker_id is None or len(persons) == 0:
            return
        feet = bottom_center(persons.xyxy)
        centers = (persons.xyxy[:, :2] + persons.xyxy[:, 2:]) / 2

        for i, tid in enumerate(persons.tracker_id):
            tid = int(tid)
            pos = (float(feet[i][0]), float(feet[i][1]))
            acc = self._tracks.get(tid)
            if acc is None:
                acc = _TrackAccum(
                    track_id=tid,
                    team=int(teams[i]),
                    start_frame=tracked.frame_index,
                    start_time=tracked.time_s,
                    start_pos=pos,
                    end_frame=tracked.frame_index,
                    end_time=tracked.time_s,
                    end_pos=pos,
                )
                self._tracks[tid] = acc
                if thumbs_dir is not None:
                    _save_thumb(frame, persons.xyxy[i], thumbs_dir / f"{tid}.jpg")
            elif acc.team == UNKNOWN and int(teams[i]) != UNKNOWN:
                # El clasificador de equipo tarda ~300 muestras en calibrar; si
                # este track nació antes de eso quedó con equipo UNKNOWN. Lo
                # corregimos en cuanto el clasificador ya sepa clasificarlo,
                # si no, nunca podría enlazarse con ningún candidato futuro.
                acc.team = int(teams[i])
            acc.end_frame = tracked.frame_index
            acc.end_time = tracked.time_s
            acc.end_pos = pos
            acc.recent.append((tracked.time_s, pos))
            if len(acc.recent) > 5:
                acc.recent.pop(0)
            color = _shirt_color(frame, persons.xyxy[i])
            if color is not None:
                acc.color_sum += color
                acc.color_n += 1

        if self.target_track_id is None and tracked.time_s >= self.target_time_s:
            dists = np.linalg.norm(centers - self.target_xy, axis=1)
            nearest = int(np.argmin(dists))
            # Umbral relativo al tamaño real de los jugadores en el frame (no
            # un nº de píxeles fijo) para que funcione igual en 480p que en
            # 4K o con la cámara más o menos alejada.
            widths = persons.xyxy[:, 2] - persons.xyxy[:, 0]
            max_click_dist = 2.5 * float(np.median(widths))
            if dists[nearest] < max_click_dist:
                self.target_track_id = int(persons.tracker_id[nearest])

    def _link_score(self, a: _TrackAccum, b: _TrackAccum) -> float:
        dt = max(b.start_time - a.end_time, 1e-3)
        vx, vy = a.velocity
        predicted = (a.end_pos[0] + vx * dt, a.end_pos[1] + vy * dt)
        spatial = float(np.hypot(predicted[0] - b.start_pos[0], predicted[1] - b.start_pos[1]))
        color_dist = float(np.linalg.norm(a.color - b.color)) * COLOR_WEIGHT
        return spatial + color_dist

    def build_chain(self) -> dict:
        if self.target_track_id is None or self.target_track_id not in self._tracks:
            return {"target_track_id": None, "segments": []}

        used = {self.target_track_id}
        current = self._tracks[self.target_track_id]
        segments = [_segment_dict(current, "inicial")]

        while True:
            candidates = [
                t
                for t in self._tracks.values()
                if t.track_id not in used
                and t.team == current.team
                and 0 < (t.start_time - current.end_time) <= MAX_GAP_S
            ]
            if not candidates:
                break
            scored = sorted(candidates, key=lambda t: self._link_score(current, t))
            best = scored[0]
            best_score = self._link_score(current, best)
            confident = len(scored) == 1 or self._link_score(current, scored[1]) > best_score * CONFIDENCE_MARGIN
            status = "confirmado" if confident else "revisar"
            seg = _segment_dict(best, status)
            if not confident:
                seg["candidates"] = self._display_candidates(current, used)
            segments.append(seg)
            used.add(best.track_id)
            current = best

        return {"target_track_id": self.target_track_id, "segments": segments}

    def _display_candidates(self, current: _TrackAccum, used: set) -> list:
        pool = [
            t
            for t in self._tracks.values()
            if t.track_id not in used
            and 0 < (t.start_time - current.end_time) <= DISPLAY_GAP_S
        ]

        def score(t: _TrackAccum) -> float:
            penalty = 0.0 if t.team == current.team else TEAM_MISMATCH_PENALTY
            return self._link_score(current, t) + penalty

        scored = sorted(pool, key=score)
        return [
            {"track_id": t.track_id, "score": round(score(t), 1)}
            for t in scored[:DISPLAY_CANDIDATES]
        ]


def _segment_dict(acc: _TrackAccum, status: str) -> dict:
    return {
        "track_id": acc.track_id,
        "start_time": round(acc.start_time, 2),
        "end_time": round(acc.end_time, 2),
        "status": status,
    }


def _save_thumb(frame: np.ndarray, xyxy: np.ndarray, path: Path) -> None:
    x1, y1, x2, y2 = xyxy.astype(int)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return
    fh, fw = frame.shape[:2]
    pad_x, pad_y = int(w * THUMB_PAD_X), int(h * THUMB_PAD_Y)
    cx1, cy1 = max(x1 - pad_x, 0), max(y1 - pad_y, 0)
    cx2, cy2 = min(x2 + pad_x, fw), min(y2 + pad_y, fh)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return
    crop = crop.copy()
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 215, 255), 2)
    crop = cv2.resize(crop, THUMB_SIZE, interpolation=cv2.INTER_CUBIC)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), crop)
