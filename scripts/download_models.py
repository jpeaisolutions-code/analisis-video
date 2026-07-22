"""Descarga a data/models/ los pesos YOLO afinados para fútbol que usa el
pipeline por defecto (jugador/portero/árbitro y balón — ver detection.py).

Uso:
    python scripts/download_models.py
"""

import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "data" / "models"

# Pesos públicos de martinjolif en HuggingFace (AGPL-3.0), afinados sobre
# YOLO11 específicamente para fútbol. Ver README de cada repo para métricas.
MODELS = {
    "yolo-football-player-detection.pt": (
        "https://huggingface.co/martinjolif/yolo-football-player-detection"
        "/resolve/main/yolo-football-player-detection.pt"
    ),
    "yolo-football-ball-detection.pt": (
        "https://huggingface.co/martinjolif/yolo-football-ball-detection"
        "/resolve/main/yolo-football-ball-detection.pt"
    ),
}


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in MODELS.items():
        target = MODELS_DIR / filename
        if target.exists():
            print(f"Ya existe: {target}")
            continue
        print(f"Descargando {filename}…")
        urllib.request.urlretrieve(url, target)
        print(f"  guardado en: {target}")


if __name__ == "__main__":
    main()
