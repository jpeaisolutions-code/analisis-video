"""Orquestación end-to-end: video -> tracks -> stats/eventos -> outputs."""

import json
from dataclasses import dataclass
from pathlib import Path

from .detection import Detector
from .events import EventDetector
from .highlights import build_highlights
from .pitch import load_calibration
from .scoreboard import ScoreboardReader
from .stats import MatchStats
from .teams import TEAM_A, TEAM_B, TeamClassifier
from .tracking import Tracker
from .video import VideoWriter, get_video_info, iter_frames
from .visualize import annotate_frame, save_heatmap


@dataclass
class PipelineConfig:
    video_path: Path
    output_dir: Path = Path("outputs")
    calibration_path: Path | None = None
    model_path: str = "yolov8m.pt"
    start_s: float = 0.0
    end_s: float | None = None
    stride: int = 1
    use_scoreboard_ocr: bool = False
    write_annotated_video: bool = True
    device: str | None = None


def run_pipeline(config: PipelineConfig) -> dict:
    info = get_video_info(config.video_path)
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = load_calibration(config.calibration_path)
    effective_fps = info.fps / config.stride

    detector = Detector(model_path=config.model_path, device=config.device)
    tracker = Tracker(fps=effective_fps)
    team_classifier = TeamClassifier()
    stats = MatchStats(calibration=calibration, fps=effective_fps)
    event_detector = EventDetector(calibration=calibration, fps=effective_fps)
    scoreboard = ScoreboardReader() if config.use_scoreboard_ocr else None

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
            tracked = tracker.update(detections)
            teams = team_classifier.update(frame, tracked.persons)
            stats.update(tracked, teams)
            event_detector.update(tracked.time_s, tracked.ball_xy)
            if scoreboard is not None:
                scoreboard.update(tracked.time_s, frame)
            if writer is not None:
                writer.write(annotate_frame(frame, tracked, teams))
            processed += 1
            if processed % 100 == 0:
                print(f"  {processed} frames procesados (t={tracked.time_s:.0f}s)")
    finally:
        if writer is not None:
            writer.close()

    all_events = event_detector.events + (
        scoreboard.events if scoreboard is not None else []
    )
    all_events.sort(key=lambda e: e.time_s)

    stats_path = output_dir / "stats.json"
    stats_path.write_text(
        json.dumps(stats.to_dict(), indent=2, ensure_ascii=False)
    )
    events_path = output_dir / "events.json"
    events_path.write_text(
        json.dumps([e.to_dict() for e in all_events], indent=2, ensure_ascii=False)
    )

    for team in (TEAM_A, TEAM_B):
        name = "team_a" if team == TEAM_A else "team_b"
        save_heatmap(stats, team, output_dir / f"heatmap_{name}.png")

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
    }
