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
from analisis_video.player_track import MAX_GAP_S

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


def _extract_frame(video, url, panorama, at_s, progress) -> "np.ndarray":
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
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _grab_frame(video, url, panorama, at_s, progress=gr.Progress()):
    """Extrae un frame del vídeo (en `at_s` segundos) para calibrar sobre él."""
    rgb = _extract_frame(video, url, panorama, at_s, progress)
    status = f"0/4 puntos — haz clic en la esquina {CORNER_LABELS[0]}"
    return rgb, rgb, [], status


def _grab_frame_player(video, url, panorama, at_s, progress=gr.Progress()):
    """Extrae un frame del vídeo para elegir con un clic al jugador a seguir."""
    rgb = _extract_frame(video, url, panorama, at_s, progress)
    status = "Haz clic sobre el jugador que quieres seguir."
    return rgb, rgb, None, status


def _on_player_click(clean_frame, at_s, evt: gr.SelectData):
    if clean_frame is None:
        raise gr.Error("Extrae un frame primero.")
    x, y = float(evt.index[0]), float(evt.index[1])
    vis = clean_frame.copy()
    cv2.circle(vis, (int(x), int(y)), 10, (255, 60, 60), 3)
    cv2.circle(vis, (int(x), int(y)), 2, (255, 60, 60), -1)
    status = (
        f"Jugador marcado en el segundo {float(at_s or 0):.0f}. "
        "Vuelve a hacer clic si quieres corregirlo."
    )
    return vis, (float(at_s or 0), x, y), status


def _reset_player(clean_frame):
    if clean_frame is None:
        return None, None, "Extrae un frame primero."
    return clean_frame, None, "Haz clic sobre el jugador que quieres seguir."


def _pending_indices(chain: list) -> list:
    return [i for i, s in enumerate(chain) if s.get("status") == "revisar"]


def _thumb(run_dir, track_id) -> str:
    return str(Path(run_dir) / "player_thumbs" / f"{track_id}.jpg")


def _visible_candidates(seg: dict, rejected: set) -> list:
    return [c for c in seg.get("candidates", []) if c["track_id"] not in rejected]


def _render_review(run_dir, chain, idx, rejected):
    pending = _pending_indices(chain)
    if not chain:
        return [], "No se pudo localizar al jugador en el frame elegido — revisa el clic e inténtalo de nuevo."
    if not pending:
        return [], "✅ No quedan tramos por confirmar."
    idx = idx % len(pending)
    seg_idx = pending[idx]
    seg = chain[seg_idx]
    prev_end = chain[seg_idx - 1]["end_time"]
    visible = _visible_candidates(seg, rejected)
    gallery = [(_thumb(run_dir, c["track_id"]), f"#{c['track_id']}") for c in visible]
    hueco = f"hueco entre {prev_end:.0f}s y {seg['start_time']:.0f}s"
    if visible:
        status = (
            f"Tramo {idx + 1}/{len(pending)} por confirmar — {hueco}. Haz clic en "
            "el jugador, luego confirma con ✅ o descártalo con ❌."
        )
    else:
        status = (
            f"Tramo {idx + 1}/{len(pending)} por confirmar — {hueco}. Sin "
            "candidatos visibles (los rechazaste todos) — usa Siguiente, o "
            "vuelve a marcar al jugador más adelante y repite el análisis."
        )
    return gallery, status


def _review_nav(run_dir, chain, idx, delta, rejected):
    idx = idx + delta
    gallery, status = _render_review(run_dir, chain, idx, rejected)
    pending = _pending_indices(chain)
    idx = idx % len(pending) if pending else 0
    return gallery, status, idx, None


def _review_mark_selected(run_dir, chain, idx, rejected, evt: gr.SelectData):
    pending = _pending_indices(chain)
    if not pending:
        return None, "✅ No quedan tramos por confirmar."
    seg = chain[pending[idx % len(pending)]]
    visible = _visible_candidates(seg, rejected)
    if evt.index >= len(visible):
        raise gr.Error("Candidato no válido.")
    tid = visible[evt.index]["track_id"]
    return evt.index, f"Candidato #{tid} seleccionado — pulsa ✅ si es tu jugador, ❌ si no lo es."


def _review_accept(run_dir, chain, idx, rejected, selected):
    pending = _pending_indices(chain)
    if not pending:
        return chain, [], "✅ No quedan tramos por confirmar.", idx, None
    if selected is None:
        raise gr.Error("Selecciona primero un candidato haciendo clic en su miniatura.")
    seg = chain[pending[idx % len(pending)]]
    visible = _visible_candidates(seg, rejected)
    if selected >= len(visible):
        raise gr.Error("Ese candidato ya no está disponible, vuelve a seleccionar.")
    chosen = visible[selected]
    seg["track_id"] = chosen["track_id"]
    seg["status"] = "confirmado"
    seg.pop("candidates", None)
    new_idx = 0
    gallery, status = _render_review(run_dir, chain, new_idx, rejected)
    return chain, gallery, status, new_idx, None


def _review_reject(run_dir, chain, idx, rejected, selected):
    pending = _pending_indices(chain)
    if not pending:
        return rejected, [], "✅ No quedan tramos por confirmar.", None
    if selected is None:
        raise gr.Error("Selecciona primero un candidato haciendo clic en su miniatura.")
    seg = chain[pending[idx % len(pending)]]
    visible = _visible_candidates(seg, rejected)
    if selected >= len(visible):
        raise gr.Error("Ese candidato ya no está disponible, vuelve a seleccionar.")
    rejected = set(rejected)
    rejected.add(visible[selected]["track_id"])
    gallery, status = _render_review(run_dir, chain, idx, rejected)
    return rejected, gallery, status, None


def _save_review(run_dir, chain, rejected, progress=gr.Progress()):
    if not run_dir:
        raise gr.Error("No hay ningún análisis cargado todavía.")
    path = Path(run_dir) / "player_track.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data["segments"] = chain
    data["rejected_track_ids"] = sorted(rejected)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return "💾 Guardado."


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
    player_click,
    progress=gr.Progress(),
):
    video_path = _resolve_local_video(video, url, panorama, progress)

    if calib_points and len(calib_points) != 4:
        raise gr.Error(
            f"La calibración tiene {len(calib_points)} puntos, necesita 4. "
            "Complétala o pulsa \"Reiniciar puntos\" para quitarla."
        )
    if not player_click:
        raise gr.Error(
            "Marca primero al jugador que quieres seguir (extrae un frame y "
            "haz clic sobre él)."
        )

    run_dir = OUTPUT_ROOT / time.strftime("%Y%m%d_%H%M%S")
    config = PipelineConfig(
        video_path=video_path,
        output_dir=run_dir,
        calibration_points=calib_points or None,
        target_click=tuple(player_click),
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

    player_track = result.get("player_track") or {"target_track_id": None, "segments": []}
    chain = player_track["segments"]
    n_revisar = len(_pending_indices(chain))
    analyzed_end = result.get("analyzed_end_s")
    # Si el último tramo acaba mucho antes de que termine el análisis, el
    # jugador se perdió y no se encontró ningún candidato para continuar —
    # muy distinto de "se le siguió sin problema todo el partido", aunque en
    # ambos casos no haya ningún tramo marcado "revisar".
    cut_short = (
        bool(chain)
        and analyzed_end is not None
        and (analyzed_end - chain[-1]["end_time"]) > MAX_GAP_S
    )
    if not chain:
        resumen_jugador = (
            "⚠️ No se pudo localizar al jugador en el frame elegido — "
            "vuelve a marcarlo y repite el análisis."
        )
    elif cut_short:
        resumen_jugador = (
            f"⚠️ El seguimiento se corta en el segundo {chain[-1]['end_time']:.0f} "
            f"(el análisis llega hasta el {analyzed_end:.0f}) — no se encontró "
            "ningún jugador del mismo equipo cerca para continuar. Puede que "
            "haya salido del plano. Revisa el vídeo anotado en ese momento, o "
            "vuelve a marcarlo más adelante y repite el análisis desde ahí."
        )
    elif n_revisar:
        resumen_jugador = (
            f"**{len(chain)} tramos** en la trayectoria del jugador — "
            f"**{n_revisar} por confirmar** abajo."
        )
    else:
        resumen_jugador = f"**{len(chain)} tramos**, todos enlazados automáticamente y hasta el final del análisis. Nada que revisar."
    gallery, review_status_text = _render_review(str(run_dir), chain, 0, set())

    return (
        resumen_md,
        result["annotated_video"],
        events_df,
        result["highlights"],
        resumen_jugador,
        gallery,
        review_status_text,
        str(run_dir),
        chain,
        0,
    )


with gr.Blocks(title="JPE AI Solutions — Análisis de Fútbol") as demo:
    gr.Markdown(
        "# ⚽ Análisis de Video de Fútbol\n"
        "Sube un video del partido, marca al jugador que quieres seguir, y "
        "pulsa **Analizar**."
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
            gr.Markdown("### 🎯 Jugador a seguir")
            gr.Markdown(
                "Extrae un frame donde se vea bien a tu jugador y haz clic "
                "sobre él. Es obligatorio: sin esto no hay a quién analizar."
            )
            with gr.Row():
                player_t_in = gr.Number(
                    value=0, label="Segundo del vídeo a extraer", scale=3
                )
                player_grab_btn = gr.Button("Extraer frame", scale=1)
            player_image = gr.Image(
                label="Haz clic sobre el jugador a seguir", interactive=False
            )
            player_status = gr.Markdown("Sin frame extraído todavía.")
            player_reset_btn = gr.Button("Reiniciar selección")
            player_frame_state = gr.State(None)
            player_click_state = gr.State(None)

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
                with gr.Tab("⚡ Eventos"):
                    events_out = gr.Dataframe(label="Eventos detectados")
                with gr.Tab("🎯 Jugador seguido"):
                    player_track_summary = gr.Markdown()
                    review_gallery = gr.Gallery(
                        label="Candidatos — haz clic en el jugador, luego confirma o descarta",
                        columns=4,
                        height=220,
                    )
                    review_status = gr.Markdown()
                    with gr.Row():
                        review_accept_btn = gr.Button("✅ Es él/ella")
                        review_reject_btn = gr.Button("❌ No es él/ella")
                    with gr.Row():
                        review_prev_btn = gr.Button("◀ Anterior")
                        review_next_btn = gr.Button("Siguiente ▶")
                    review_save_btn = gr.Button("💾 Guardar cambios", variant="primary")
                    run_dir_state = gr.State(None)
                    player_chain_state = gr.State([])
                    review_idx_state = gr.State(0)
                    review_selected_state = gr.State(None)
                    rejected_ids_state = gr.State(set())
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

    player_grab_btn.click(
        _grab_frame_player,
        inputs=[video_in, url_in, panorama_in, player_t_in],
        outputs=[player_image, player_frame_state, player_click_state, player_status],
    )
    player_image.select(
        _on_player_click,
        inputs=[player_frame_state, player_t_in],
        outputs=[player_image, player_click_state, player_status],
    )
    player_reset_btn.click(
        _reset_player,
        inputs=[player_frame_state],
        outputs=[player_image, player_click_state, player_status],
    )

    boton.click(
        analizar,
        inputs=[
            video_in, url_in, panorama_in, stride_in, inicio_in, fin_in,
            ocr_in, anotado_in, calib_points_state, player_click_state,
        ],
        outputs=[
            resumen_out, video_out, events_out, highlights_out,
            player_track_summary, review_gallery, review_status,
            run_dir_state, player_chain_state, review_idx_state,
        ],
        api_name="analizar",
    ).then(
        lambda: (set(), None),
        outputs=[rejected_ids_state, review_selected_state],
    )
    review_prev_btn.click(
        lambda rd, c, i, rej: _review_nav(rd, c, i, -1, rej),
        inputs=[run_dir_state, player_chain_state, review_idx_state, rejected_ids_state],
        outputs=[review_gallery, review_status, review_idx_state, review_selected_state],
    )
    review_next_btn.click(
        lambda rd, c, i, rej: _review_nav(rd, c, i, 1, rej),
        inputs=[run_dir_state, player_chain_state, review_idx_state, rejected_ids_state],
        outputs=[review_gallery, review_status, review_idx_state, review_selected_state],
    )
    review_gallery.select(
        _review_mark_selected,
        inputs=[run_dir_state, player_chain_state, review_idx_state, rejected_ids_state],
        outputs=[review_selected_state, review_status],
    )
    review_accept_btn.click(
        _review_accept,
        inputs=[
            run_dir_state, player_chain_state, review_idx_state,
            rejected_ids_state, review_selected_state,
        ],
        outputs=[
            player_chain_state, review_gallery, review_status,
            review_idx_state, review_selected_state,
        ],
    )
    review_reject_btn.click(
        _review_reject,
        inputs=[
            run_dir_state, player_chain_state, review_idx_state,
            rejected_ids_state, review_selected_state,
        ],
        outputs=[rejected_ids_state, review_gallery, review_status, review_selected_state],
    )
    review_save_btn.click(
        _save_review,
        inputs=[run_dir_state, player_chain_state, rejected_ids_state],
        outputs=[review_status],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--share", action="store_true", help="Generar enlace público (Colab)"
    )
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo.launch(share=args.share, server_port=args.port)
