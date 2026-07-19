"""CLI del pipeline de análisis de video de fútbol.

Uso:
    python scripts/run_pipeline.py --video data/raw/partido.mp4
    python scripts/run_pipeline.py --video clip.mp4 --start 60 --end 180 --stride 2
    python scripts/run_pipeline.py --video clip.mp4 --calibration cancha.json --ocr
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analisis_video.pipeline import PipelineConfig, run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis de video de fútbol")
    parser.add_argument("--video", required=True, help="Ruta al video de entrada")
    parser.add_argument(
        "--output-dir", default="outputs", help="Directorio de salida"
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="JSON de calibración de cancha (ver src/analisis_video/pitch.py)",
    )
    parser.add_argument(
        "--model", default="yolov8m.pt", help="Modelo YOLO (ej. yolov8n.pt para CPU)"
    )
    parser.add_argument(
        "--start", type=float, default=0.0, help="Segundo inicial a procesar"
    )
    parser.add_argument(
        "--end", type=float, default=None, help="Segundo final a procesar"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Procesar 1 de cada N frames (acelera en CPU)",
    )
    parser.add_argument(
        "--ocr", action="store_true", help="Activar OCR del marcador en pantalla"
    )
    parser.add_argument(
        "--no-annotated-video",
        action="store_true",
        help="No generar el video anotado (más rápido, menos disco)",
    )
    parser.add_argument(
        "--device", default=None, help="Dispositivo de inferencia (ej. cpu, 0)"
    )
    args = parser.parse_args()

    config = PipelineConfig(
        video_path=Path(args.video),
        output_dir=Path(args.output_dir),
        calibration_path=Path(args.calibration) if args.calibration else None,
        model_path=args.model,
        start_s=args.start,
        end_s=args.end,
        stride=args.stride,
        use_scoreboard_ocr=args.ocr,
        write_annotated_video=not args.no_annotated_video,
        device=args.device,
    )

    print(f"Procesando {args.video} ...")
    result = run_pipeline(config)
    print("\nResultados:")
    for key, value in result.items():
        if not key.endswith("_data"):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
