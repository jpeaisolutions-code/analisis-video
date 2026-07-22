"""Descarga a data/models/ los pesos YOLO afinados para fútbol que usa el
pipeline por defecto (jugador/portero/árbitro y balón — ver detection.py).

No es estrictamente necesario ejecutarlo a mano: `Detector` descarga los
pesos que falten la primera vez que se usan. Sirve para adelantar la
descarga antes de lanzar un análisis.

Uso:
    python scripts/download_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analisis_video.detection import (
    DEFAULT_BALL_MODEL,
    DEFAULT_PLAYER_MODEL,
    ensure_weights,
)


def main() -> None:
    for model_path in (DEFAULT_PLAYER_MODEL, DEFAULT_BALL_MODEL):
        if Path(model_path).exists():
            print(f"Ya existe: {model_path}")
            continue
        print(f"Descargando {model_path}…")
        ensure_weights(model_path)
        print(f"  guardado en: {model_path}")


if __name__ == "__main__":
    main()
