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

import cv2
import gradio as gr
import pandas as pd

from analisis_video.pipeline import PipelineConfig, run_pipeline
from analisis_video.pitch import PITCH_CORNERS

OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs"
DOWNLOAD_DIR = Path(__file__).resolve().parent / "data" / "raw"

VEO_MATCH_RE = re.compile(r"app\.veo\.co/matches/([^/?#]+)")

# Algunos CDN (p.ej. Wikimedia) rechazan el User-Agent por defecto de urllib
_HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (analisis-video; jpe.aisolutions@gmail.com)"}


def _urlopen(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers=_HTTP_HEADERS))


def _resolve_video_url(url: str, prefer_panorama: bool = False) -> tuple[str, str]:
    """Devuelve (url directa, nombre de archivo). Entiende páginas de Veo.

    Veo ofrece dos renders: "standard" (cámara que sigue el juego, panea y
    hace zoom) y "panorama" (gran angular fijo, sin editar — homografía
    constante para todo el partido, pero jugadores/balón más pequeños)."""
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
    order = ("panorama", "standard") if prefer_panorama else ("standard", "panorama")
    for render_type in order:
        for video in videos:
            if video.get("render_type") == render_type and video.get("url"):
                return video["url"], f"{slug}_{render_type}.mp4"
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
CORNER_LABELS = (
    "superior-izquierda",
    "superior-derecha",
    "inferior-derecha",
    "inferior-izquierda",
)


def _resolve_local_video(video, url, panorama, progress) -> Path:
    if url and url.strip():
        progress(0, desc="Localizando el video…")
        direct_url, name = _resolve_video_url(url.strip(), prefer_panorama=panorama)
        return _download_video(direct_url, name, progress)
    if not video:
        raise gr.Error("Sube un video o pega un enlace primero.")
    return Path(video)


def _grab_frame(video, url, panorama, at_s, progress=gr.Progress()):
    """Extrae un frame del vídeo (en `at_s` segundos) para calibrar sobre él."""
    path = _resolve_local_video(video, url, panorama, progress)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise gr.Error(f"No se pudo abrir el vídeo: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(at_s or 0) * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise gr.Error("No se pudo leer ese frame — prueba con otro segundo.")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    status = f"0/4 puntos — haz clic en la esquina {CORNER_LABELS[0]}"
    return rgb, rgb, [], status


def _draw_calib_points(clean_frame, points):
    vis = clean_frame.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(vis, (int(x), int(y)), 8, (255, 60, 60), -1)
        cv2.putText(
            vis, str(i + 1), (int(x) + 10, int(y)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 60, 60), 2,
        )
    return vis


def _on_calib_click(clean_frame, points, evt: gr.SelectData):
    if clean_frame is None:
        raise gr.Error("Extrae un frame primero.")
    points = list(points)
    if len(points) >= 4:
        raise gr.Error(
            "Ya tienes los 4 puntos. Pulsa \"Reiniciar puntos\" para repetirlo."
        )
    points.append((float(evt.index[0]), float(evt.index[1])))
    vis = _draw_calib_points(clean_frame, points)
    if len(points) < 4:
        status = f"{len(points)}/4 puntos — ahora haz clic en la esquina {CORNER_LABELS[len(points)]}"
    else:
        status = (
            "4/4 puntos ✓ — se aplicará la calibración al pulsar Analizar. "
            "Pulsa \"Reiniciar puntos\" si te has equivocado."
        )
    return vis, points, status


def _reset_calib(clean_frame):
    if clean_frame is None:
        return None, [], "Extrae un frame primero."
    status = f"0/4 puntos — haz clic en la esquina {CORNER_LABELS[0]}"
    return clean_frame, [], status


def analizar(
    video,
    url,
    panorama,
    stride,
    inicio,
    fin,
    usar_ocr,
    generar_video,
    calib_points,
    progress=gr.Progress(),
):
    video_path = _resolve_local_video(video, url, panorama, progress)

    if calib_points and len(calib_points) != 4:
        raise gr.Error(
            f"La calibración tiene {len(calib_points)} puntos, necesita 4. "
            "Complétala o pulsa \"Reiniciar puntos\" para quitarla."
        )

    run_dir = OUTPUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    config = PipelineConfig(
        video_path=video_path,
        output_dir=run_dir,
        calibration_points=calib_points or None,
        start_s=float(inicio or 0),
        end_s=float(fin) if fin else None,
        stride=int(stride),
        use_scoreboard_ocr=usar_ocr,
        write_annotated_video=generar_video,
    )

    progress(0, desc="Preparando análisis…")

    def on_progress(done: int, total: int) -> None:
        progress(done / total, desc=f"Analizando… {done}/{total} frames")

    def on_status(msg: str) -> None:
        progress(1.0, desc=msg)

    try:
        result = run_pipeline(
            config, progress_callback=on_progress, status_callback=on_status
        )
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

    return (
        resumen_md,
        result["annotated_video"],
        players_df,
        events_df,
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
            panorama_in = gr.Checkbox(
                value=False,
                label="Usar plano panorámico de Veo (solo enlaces de Veo)",
                info=(
                    "Gran angular fijo, sin recorte: la calibración de cancha "
                    "vale para todo el partido y debería haber menos cortes de "
                    "tracking, a cambio de jugadores/balón más pequeños en "
                    "píxeles. Si no está disponible, se usa el plano normal."
                ),
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
            with gr.Accordion(
                "Calibración de cancha (opcional — activa eventos y da "
                "distancias/velocidades en metros reales)",
                open=False,
            ):
                gr.Markdown(
                    "Necesita un momento del vídeo donde se vean las **4 "
                    "esquinas de la cancha** a la vez (p.ej. el saque inicial "
                    "con la cámara más abierta). Extrae ese frame y haz clic "
                    "en las 4 esquinas en este orden: superior-izquierda → "
                    "superior-derecha → inferior-derecha → inferior-izquierda."
                )
                with gr.Row():
                    calib_t_in = gr.Number(
                        value=0, label="Segundo del vídeo a extraer", scale=3
                    )
                    calib_grab_btn = gr.Button("Extraer frame", scale=1)
                calib_image = gr.Image(
                    label="Haz clic en las 4 esquinas de la cancha",
                    interactive=False,
                )
                calib_status = gr.Markdown("Sin frame extraído todavía.")
                calib_reset_btn = gr.Button("Reiniciar puntos")
                calib_frame_state = gr.State(None)
                calib_points_state = gr.State([])
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
                with gr.Tab("🎬 Highlights"):
                    highlights_out = gr.Video(label="Resumen automático")

    calib_grab_btn.click(
        _grab_frame,
        inputs=[video_in, url_in, panorama_in, calib_t_in],
        outputs=[calib_image, calib_frame_state, calib_points_state, calib_status],
    )
    calib_image.select(
        _on_calib_click,
        inputs=[calib_frame_state, calib_points_state],
        outputs=[calib_image, calib_points_state, calib_status],
    )
    calib_reset_btn.click(
        _reset_calib,
        inputs=[calib_frame_state],
        outputs=[calib_image, calib_points_state, calib_status],
    )

    boton.click(
        analizar,
        inputs=[
            video_in, url_in, panorama_in, stride_in, inicio_in, fin_in,
            ocr_in, anotado_in, calib_points_state,
        ],
        outputs=[
            resumen_out, video_out, players_out, events_out, highlights_out,
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
