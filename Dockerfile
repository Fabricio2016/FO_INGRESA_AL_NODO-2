# ── STAGE 1: Descargar y convertir modelo a ONNX ────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir ultralytics gdown

# Descarga el modelo .pt y lo convierte a .onnx + extrae nombres de clases
RUN python -c "\
import gdown, json; \
from ultralytics import YOLO; \
gdown.download('https://drive.google.com/uc?id=1tSSKvZesgWzhhX8vKBnk-7UoxkRTgKXI', 'model.pt', quiet=False); \
model = YOLO('model.pt'); \
json.dump({str(k): v for k, v in model.names.items()}, open('names.json','w')); \
model.export(format='onnx', imgsz=640, simplify=True); \
print('Conversion completa') \
"

# ── STAGE 2: Runtime liviano (sin PyTorch) ───────────────────
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Solo copia el modelo ONNX y los nombres de clases del stage anterior
COPY --from=builder /build/model.onnx .
COPY --from=builder /build/names.json .

COPY app.py .

EXPOSE 10000

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
