"""Interfaz web del análisis de video de fútbol (Gradio).

Uso:
    python app.py            # local: http://127.0.0.1:7860
    python app.py --share    # además genera un enlace público (para Colab)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import gradio as gr
import pandas as pd

from analisis_video.pipeline import PipelineConfig, run_pipeline

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"

KIND_LABELS = {"goal": "⚽ Gol", "shot": "🎯 Remate", "corner": "🚩 Córner"}
TEAM_LABELS = {"team_a": "Equipo A", "team_b": "Equipo B", "unknown": "—"}
MODEL_CHOICES = {
    "Rápido (yolov8n)": "yolov8n.pt",
    "Preciso (yolov8m, recomendado con GPU)": "yolov8m.pt",
}


def analizar(
    video,
    modelo,
    stride,
    inicio,
    fin,
    usar_ocr,
    generar_video,
    progress=gr.Progress(),
):
    if not video:
        raise gr.Error("Sube un video primero.")

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
            video_in, modelo_in, stride_in, inicio_in, fin_in, ocr_in, anotado_in
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
