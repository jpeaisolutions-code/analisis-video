"""Descarga los pesos del modelo YOLO a data/models/.

Uso:
    python scripts/download_models.py            # yolov8m (default del pipeline)
    python scripts/download_models.py yolov8n    # variante ligera para CPU
"""

import sys
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "yolov8m"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(f"{name}.pt")
    weights = Path(model.ckpt_path).resolve()
    target = MODELS_DIR / weights.name
    if weights != target:
        target.write_bytes(weights.read_bytes())
    print(f"Modelo disponible en: {target}")


if __name__ == "__main__":
    main()
