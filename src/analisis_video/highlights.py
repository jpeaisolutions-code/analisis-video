"""Generación de video resumen: recorta clips alrededor de eventos y los une."""

import subprocess
import tempfile
from pathlib import Path

from .events import Event

PRE_EVENT_S = 12.0
POST_EVENT_S = 6.0

# Prioridad para ordenar/filtrar si hay demasiados eventos
KIND_PRIORITY = {"goal": 0, "shot": 1, "corner": 2}


def _merge_windows(
    events: list[Event], pre: float, post: float
) -> list[tuple[float, float]]:
    windows = sorted(
        (max(0.0, e.time_s - pre), e.time_s + post) for e in events
    )
    merged: list[tuple[float, float]] = []
    for start, end in windows:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def build_highlights(
    video_path: str | Path,
    events: list[Event],
    output_path: str | Path,
    pre_s: float = PRE_EVENT_S,
    post_s: float = POST_EVENT_S,
) -> Path | None:
    """Genera un video resumen con ffmpeg. Devuelve None si no hay eventos."""
    if not events:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windows = _merge_windows(events, pre_s, post_s)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        clip_paths = []
        for i, (start, end) in enumerate(windows):
            clip = tmp_dir / f"clip_{i:03d}.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-ss", str(start), "-to", str(end),
                    "-i", str(video_path),
                    "-c:v", "libx264", "-preset", "fast", "-c:a", "aac",
                    str(clip),
                ],
                check=True,
            )
            clip_paths.append(clip)

        concat_list = tmp_dir / "list.txt"
        concat_list.write_text(
            "".join(f"file '{p}'\n" for p in clip_paths)
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy",
                str(output_path),
            ],
            check=True,
        )
    return output_path
