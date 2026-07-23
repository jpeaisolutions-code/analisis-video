"""Orquestación end-to-end: video -> tracks -> stats/eventos -> outputs."""

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .detection import DEFAULT_BALL_MODEL, DEFAULT_PLAYER_MODEL, Detector
from .events import EventDetector
from .highlights import build_highlights
from .pitch import PITCH_CORNERS, PitchCalibration, load_calibration
from .player_track import PlayerTrackBuilder
from .scoreboard import ScoreboardReader
from .stats import MatchStats
from .teams import TeamClassifier
from .touches import TouchDetector
from .tracking import Tracker
from .video import VideoWriter, get_video_info, iter_frames
from .visualize import annotate_frame


@dataclass
class PipelineConfig:
    video_path: Path
    output_dir: Path = Path("outputs")
    calibration_path: Path | None = None
    # Alternativa a calibration_path: las 4 esquinas de la cancha (píxeles),
    # en el orden de pitch.PITCH_CORNERS, recogidas por clic en la app.
    calibration_points: list[tuple[float, float]] | None = None
    # Jugador a seguir: (segundo, x_px, y_px) del clic sobre el frame elegido.
    target_click: tuple[float, float, float] | None = None
    player_model_path: str = DEFAULT_PLAYER_MODEL
    ball_model_path: str = DEFAULT_BALL_MODEL
    start_s: float = 0.0
    end_s: float | None = None
    stride: int = 1
    use_scoreboard_ocr: bool = False
    write_annotated_video: bool = True
    device: str | None = None


def _reencode_h264(path: Path) -> None:
    """Re-codifica a H.264 para que el video sea reproducible en navegadores.

    Prueba primero el codificador por hardware de NVIDIA (NVENC) — en una GPU
    tipo T4 codifica un partido completo en minutos. Si no hay GPU o el ffmpeg
    del sistema no lo soporta, cae a libx264 por CPU.
    """
    tmp = path.with_name(path.stem + "_h264.mp4")
    for codec_args in (
        ["-c:v", "h264_nvenc", "-preset", "p4"],
        ["-c:v", "libx264", "-preset", "veryfast"],
    ):
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(path),
                *codec_args, "-pix_fmt", "yuv420p", "-an",
                str(tmp),
            ]
        )
        if result.returncode == 0:
            tmp.replace(path)
            return
    tmp.unlink(missing_ok=True)


def run_pipeline(
    config: PipelineConfig,
    progress_callback: Callable[[int, int], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> dict:
    def status(msg: str) -> None:
        print(msg)
        if status_callback is not None:
            status_callback(msg)

    info = get_video_info(config.video_path)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.calibration_points:
        calibration = PitchCalibration.from_pixel_corners(config.calibration_points)
        calibration.to_json(
            output_dir / "calibration.json", config.calibration_points, PITCH_CORNERS
        )
    else:
        calibration = load_calibration(config.calibration_path)
    effective_fps = info.fps / config.stride

    end_s = min(config.end_s, info.duration_s) if config.end_s else info.duration_s
    total_frames = max(1, int((end_s - config.start_s) * info.fps / config.stride))

    detector = Detector(
        player_model_path=config.player_model_path,
        ball_model_path=config.ball_model_path,
        device=config.device,
    )
    tracker = Tracker(fps=effective_fps, source_fps=info.fps)
    team_classifier = TeamClassifier()
    stats = MatchStats(calibration=calibration, fps=effective_fps)
    event_detector = EventDetector(calibration=calibration, fps=effective_fps)
    touch_detector = TouchDetector(calibration=calibration, fps=effective_fps)
    scoreboard = ScoreboardReader() if config.use_scoreboard_ocr else None

    player_builder = None
    thumbs_dir = None
    if config.target_click:
        target_time_s, target_x, target_y = config.target_click
        player_builder = PlayerTrackBuilder(target_time_s, (target_x, target_y))
        thumbs_dir = output_dir / "player_thumbs"

    writer = None
    if config.write_annotated_video:
        writer = VideoWriter(
            output_dir / "annotated.mp4",
            fps=effective_fps,
            width=info.width,
            height=info.height,
        )

    processed = 0
    try:
        for frame_index, frame in iter_frames(
            config.video_path,
            stride=config.stride,
            start_s=config.start_s,
            end_s=config.end_s,
        ):
            detections = detector.detect(frame, frame_index=frame_index)
            tracked = tracker.update(detections, frame)
            teams = team_classifier.update(frame, tracked.persons)
            stats.update(tracked, teams)
            event_detector.update(tracked.time_s, tracked.ball_xy)
            touch_detector.update(tracked, teams)
            if player_builder is not None:
                player_builder.update(frame, tracked, teams, thumbs_dir)
            if scoreboard is not None:
                scoreboard.update(tracked.time_s, frame)
            if writer is not None:
                writer.write(annotate_frame(frame, tracked, teams))
            processed += 1
            if progress_callback is not None and processed % 10 == 0:
                progress_callback(processed, total_frames)
            if processed % 100 == 0:
                print(f"  {processed} frames procesados (t={tracked.time_s:.0f}s)")
    finally:
        if writer is not None:
            writer.close()

    if progress_callback is not None:
        progress_callback(total_frames, total_frames)
    if config.write_annotated_video:
        status("Convirtiendo el video anotado a H.264…")
        _reencode_h264(output_dir / "annotated.mp4")

    status("Calculando estadísticas…")
    all_events = event_detector.events + (
        scoreboard.events if scoreboard is not None else []
    )
    all_events.sort(key=lambda e: e.time_s)

    touch_detector.finish()

    stats_data = stats.to_dict()
    events_data = [e.to_dict() for e in all_events]
    touches_data = touch_detector.to_dict()
    stats_path = output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats_data, indent=2, ensure_ascii=False))
    events_path = output_dir / "events.json"
    events_path.write_text(json.dumps(events_data, indent=2, ensure_ascii=False))
    touches_path = output_dir / "touches.json"
    touches_path.write_text(json.dumps(touches_data, indent=2, ensure_ascii=False))

    player_track_data = None
    if player_builder is not None:
        player_track_data = player_builder.build_chain()
        (output_dir / "player_track.json").write_text(
            json.dumps(player_track_data, indent=2, ensure_ascii=False)
        )

    status("Generando highlights…")
    highlights_path = build_highlights(
        config.video_path, all_events, output_dir / "highlights.mp4"
    )

    return {
        "frames_processed": processed,
        "stats": str(stats_path),
        "events": str(events_path),
        "events_count": len(all_events),
        "annotated_video": str(output_dir / "annotated.mp4")
        if config.write_annotated_video
        else None,
        "highlights": str(highlights_path) if highlights_path else None,
        "stats_data": stats_data,
        "events_data": events_data,
        "touches": touches_data,
        "player_track": player_track_data,
        "player_thumbs_dir": str(thumbs_dir) if thumbs_dir else None,
        "analyzed_start_s": config.start_s,
        "analyzed_end_s": end_s,
    }
