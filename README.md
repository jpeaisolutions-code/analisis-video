# Análisis de Video — Partidos de Fútbol

Pipeline en Python para analizar video de partidos de fútbol:

1. **Tracking** de jugadores y balón
2. **Estadísticas** del juego (posesión, distancia recorrida, velocidad, mapas de calor)
3. **Detección de eventos** clave (goles, córners, remates a puerta)
4. **Highlights** automáticos (video resumen)

## Arquitectura

- Detección: YOLOv8 (`ultralytics`)
- Tracking: ByteTrack (vía `supervision`)
- Equipos: clasificación por color de camiseta
- Cancha: detección de puntos clave + homografía (píxeles → coordenadas reales de cancha)
- Estadísticas y eventos: derivados de los tracks + homografía
- Highlights: recorte con `ffmpeg` alrededor de eventos detectados

Ver `src/analisis_video/` para cada módulo y `scripts/run_pipeline.py` como punto de entrada.

## Limitaciones conocidas

- **Goles**: detectados por cruce de línea de gol + OCR del marcador en pantalla (si el
  video lo muestra) como refuerzo. Razonablemente confiable.
- **Córners / remates a puerta**: heurísticas geométricas sobre la trayectoria del balón.
  Razonablemente confiable.
- **Tarjetas**: no hay una señal geométrica clara en el tracking para detectarlas de forma
  confiable; requeriría un clasificador de acción entrenado con datos etiquetados. Se deja
  como función de menor prioridad / mejor esfuerzo.

## Requisitos de cómputo

Este pipeline usa detección de objetos por frame + tracking, lo cual es intensivo en
cómputo. **Se recomienda GPU** para procesar partidos completos (~90 min). Sin GPU
(CPU), el procesamiento es viable solo para clips cortos de prueba (1-2 min); un
partido completo puede tardar horas.

Entornos recomendados para el procesamiento pesado: Google Colab (GPU gratuita),
o una instancia cloud con GPU (RunPod, Lambda, AWS/GCP).

## Setup

```bash
git clone https://github.com/jpeaisolutions-code/analisis-video.git
cd analisis-video
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requiere `ffmpeg` instalado a nivel de sistema (para lectura de video y generación de
highlights):

```bash
sudo apt-get install ffmpeg
```

## Uso

### Aplicación web (recomendado)

```bash
python app.py            # local: abre http://127.0.0.1:7860 en el navegador
python app.py --share    # además genera un enlace público (para Colab)
```

Subes el video desde el navegador, pulsas **Analizar** y ves los resultados en
pantalla: video anotado, posesión, estadísticas por jugador, eventos, mapas de
calor y highlights.

**En Google Colab con GPU:** abre directamente
[el notebook en Colab](https://colab.research.google.com/github/jpeaisolutions-code/analisis-video/blob/main/notebooks/colab_analisis.ipynb),
activa la GPU y ejecuta las celdas — la última abre la aplicación con un
enlace público.

### Línea de comandos (alternativa)

```bash
python scripts/run_pipeline.py --video ruta/al/video.mp4
```

Los resultados (video anotado, estadísticas, eventos, highlights) se guardan en `outputs/`.

### Recomendación de flujo de trabajo

1. Probar con un clip corto (1-2 min) o con `--end 120` primero.
2. Una vez validado, correr el partido completo en un entorno con GPU.
