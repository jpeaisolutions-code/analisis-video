"""Visualización: frames anotados (cajas, IDs, equipos, balón)."""

import cv2
import numpy as np
import supervision as sv

from .teams import TEAM_A, TEAM_B
from .tracking import TrackedFrame

TEAM_COLORS = {
    TEAM_A: sv.Color(255, 80, 80),
    TEAM_B: sv.Color(80, 140, 255),
}
UNKNOWN_COLOR = sv.Color(200, 200, 200)


def annotate_frame(
    frame: np.ndarray, tracked: TrackedFrame, teams: np.ndarray
) -> np.ndarray:
    out = frame.copy()
    persons = tracked.persons

    for i, xyxy in enumerate(persons.xyxy):
        color = TEAM_COLORS.get(int(teams[i]), UNKNOWN_COLOR)
        x1, y1, x2, y2 = xyxy.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), color.as_bgr(), 2)
        if persons.tracker_id is not None:
            cv2.putText(
                out,
                f"#{int(persons.tracker_id[i])}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color.as_bgr(),
                2,
            )

    if tracked.ball_xy is not None:
        bx, by = int(tracked.ball_xy[0]), int(tracked.ball_xy[1])
        cv2.circle(out, (bx, by), 8, (0, 255, 255), 2)

    return out
