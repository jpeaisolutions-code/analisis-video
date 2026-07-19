"""Interfaz web del análisis de video de fútbol (Gradio).

Uso:
    python app.py            # local: http://127.0.0.1:7860
    python app.py --share    # además genera un enlace público (para Colab)
"""

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr
import pandas as pd

from analisis_video.pipeline import PipelineConfig, run_pipeline

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DOWNLOAD_DIR = Path(__file__).resolve().parent / "data" / "raw"

VEO_MATCH_RE = re.compile(r"app\.veo\.co/matches/([^/?#]+)")

# Algunos CDN (p.ej. Wikimedia) rechazan el User-Agent por defecto de urllib
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (analisis-video; jpe.aisolutions@gmail.com)"}


def _urlopen(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers=_HTTP_HEADERS))


def _resolve_video_url(url: str) -> tuple[str, str]:
    """Devuelve (url directa, nombre de archivo). Entiende páginas de Veo."""
    match = VEO_MATCH_RE.search(url)
    if not match:
        name = Path(urllib.parse.urlparse(url).path).name or "video.mp4"
        return url, name
    slug = match.group(1)
    api = f"https://app.veo.co/api/app/matches/{slug}/videos"
    try:
        with _urlopen(api) as response:
            videos = json.load(response)
    except Exception as exc:
        raise gr.Error(
            f"No se pudo acceder al partido de Veo (¿el enlace es público?): {exc}"
        ) from exc
    # "standard" es la vista de cámara que sigue el juego; "panorama" es el
    # gran angular sin editar
    for video in videos:
        if video.get("render_type") == "standard" and video.get("url"):
            return video["url"], f"{slug}.mp4"
    for video in videos:
        if video.get("mime_type") == "video/mp4" and video.get("url"):
            return video["url"], f"{slug}.mp4"
    raise gr.Error("No se encontró un video descargable en ese enlace de Veo.")


def _download_video(url: str, name: str, progress) -> Path:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = DOWNLOAD_DIR / name
    with _urlopen(url) as response:
        total = int(response.headers.get("Content-Length") or 0)
        if dest.exists() and total and dest.stat().st_size == total:
            return dest  # ya descargado
        done = 0
        with open(dest, "wb") as f:
            while chunk := response.read(4 * 1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if total:
                    progress(
                        done / total,
                        desc=f"Descargando video… {done / 1e6:.0f}/{total / 1e6:.0f} MB",
                    )
    return dest

KIND_LABELS = {"goal": "⚽ Gol", "shot": "🎯 Remate", "corner": "🚩 Córner"}
TEAM_LABELS = {"team_a": "Equipo A", "team_b": "Equipo B", "unknown": "—"}
MODEL_CHOICES = {
    "Rápido (yolov8n)": "yolov8n.pt",
    "Preciso (yolov8m, recomendado con GPU)": "yolov8m.pt",
}


def analizar(
    video,
    url,
    modelo,
    stride,
    inicio,
    fin,
    usar_ocr,
    generar_video,
    progress=gr.Progress(),
):
    if url and url.strip():
        progress(0, desc="Localizando el video…")
        direct_url, name = _resolve_video_url(url.strip())
        video = str(_download_video(direct_url, name, progress))
    elif not video:
        raise gr.Error("Sube un video o pega un enlace primero.")

    run_dir = OUTPUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    config = PipelineConfig(
        video_path=Path(video),
        output_dir=run_dir,
        model_path=MODEL_CHOICES[modelo],
        start_s=float(inicio or 0),
        end_s=float(fin) if fin else None,
        stride=int(stride),
        use_scoreboard_ocr=usar_ocr,
        write_annotated_video=generar_video,
    )

    progress(0, desc="Preparando análisis…")

    def on_progress(done: int, total: int) -> None:
        progress(done / total, desc=f"Analizando… {done}/{total} frames")

    try:
        result = run_pipeline(config, progress_callback=on_progress)
    except FileNotFoundError as exc:
        raise gr.Error(str(exc)) from exc

    stats = result["stats_data"]
    pos = stats["possession_pct"]
    unidades = "metros" if stats["units"] == "m" else "píxeles (sin calibración de cancha)"
    resumen_md = (
        f"## Posesión\n"
        f"# 🔴 {pos['team_a']}% — 🔵 {pos['team_b']}%\n"
        f"Equipo A — Equipo B\n\n"
        f"*Frames analizados: {result['frames_processed']} · "
        f"Eventos detectados: {result['events_count']} · "
        f"Distancias en {unidades}*"
    )

    players_df = pd.DataFrame(
        [
            {
                "Jugador (ID)": f"#{p['track_id']}",
                "Equipo": TEAM_LABELS.get(p["team"], p["team"]),
                f"Distancia ({stats['units']})": p["distance"],
                f"Vel. máx ({stats['units']}/s)": p["max_speed"],
            }
            for p in stats["players"]
        ]
    )

    events_df = pd.DataFrame(
        [
            {
                "Minuto": e["time"],
                "Evento": KIND_LABELS.get(e["kind"], e["kind"]),
                "Detalle": e["detail"],
            }
            for e in result["events_data"]
        ]
    )
    if events_df.empty:
        events_df = pd.DataFrame(
            [{"Minuto": "—", "Evento": "Sin eventos detectados", "Detalle": ""}]
        )

    heatmaps = [
        (result["heatmaps"]["team_a"], "Equipo A"),
        (result["heatmaps"]["team_b"], "Equipo B"),
    ]

    return (
        resumen_md,
        result["annotated_video"],
        players_df,
        events_df,
        heatmaps,
        result["highlights"],
    )


with gr.Blocks(title="JPE AI Solutions — Análisis de Fútbol") as demo:
    gr.Markdown(
        "# ⚽ Análisis de Video de Fútbol\n"
        "Sube un video del partido, ajusta las opciones si quieres, y pulsa "
        "**Analizar**. Obtendrás el video con jugadores y balón marcados, "
        "estadísticas, eventos y mapas de calor."
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Video del partido")
            url_in = gr.Textbox(
                label="…o pega un enlace (Veo o URL directa de video)",
                placeholder="https://app.veo.co/matches/…",
                info="Si pegas un enlace, se usa el enlace y no el archivo subido.",
            )
            modelo_in = gr.Dropdown(
                choices=list(MODEL_CHOICES),
                value=list(MODEL_CHOICES)[0],
                label="Modelo de detección",
            )
            with gr.Accordion("Opciones avanzadas", open=False):
                stride_in = gr.Slider(
                    1, 10, value=2, step=1,
                    label="Procesar 1 de cada N frames (más alto = más rápido)",
                )
                inicio_in = gr.Number(value=0, label="Segundo inicial")
                fin_in = gr.Number(
                    value=None, label="Segundo final (vacío = hasta el final)"
                )
                ocr_in = gr.Checkbox(
                    value=False,
                    label="Leer marcador en pantalla (OCR, detecta goles)",
                )
                anotado_in = gr.Checkbox(
                    value=True, label="Generar video anotado"
                )
            boton = gr.Button("🔍 Analizar", variant="primary", size="lg")

        with gr.Column(scale=2):
            resumen_out = gr.Markdown()
            with gr.Tabs():
                with gr.Tab("🎥 Video anotado"):
                    video_out = gr.Video(label="Jugadores y balón detectados")
                with gr.Tab("📊 Jugadores"):
                    players_out = gr.Dataframe(label="Estadísticas por jugador")
                with gr.Tab("⚡ Eventos"):
                    events_out = gr.Dataframe(label="Eventos detectados")
                with gr.Tab("🔥 Mapas de calor"):
                    heatmaps_out = gr.Gallery(label="Por equipo", columns=2)
                with gr.Tab("🎬 Highlights"):
                    highlights_out = gr.Video(label="Resumen automático")

    boton.click(
        analizar,
        inputs=[
            video_in, url_in, modelo_in, stride_in, inicio_in, fin_in,
            ocr_in, anotado_in,
        ],
        outputs=[
            resumen_out, video_out, players_out, events_out,
            heatmaps_out, highlights_out,
        ],
        api_name="analizar",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--share", action="store_true", help="Generar enlace público (Colab)"
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo.launch(share=args.share, server_port=args.port)
