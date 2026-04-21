from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, Response
import onnxruntime as ort
import numpy as np
from PIL import Image
import io, json

app = FastAPI()

# ── Cargar nombres de clases ─────────────────────────────────
with open("names.json") as f:
    NAMES = json.load(f)  # {"0": "clase0", "1": "clase1", ...}

# ── Cargar modelo ONNX (sin PyTorch, ~100MB RAM) ─────────────
session = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name
IMG_SIZE   = session.get_inputs()[0].shape[2] or 640
print(f"Modelo ONNX cargado. Clases: {NAMES} | Input: {IMG_SIZE}px")


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # [1, 3, H, W]
    return arr


def postprocess(outputs, conf_threshold: float = 0.40):
    # YOLOv8 output: [1, 4+num_classes, num_anchors]
    pred = outputs[0][0]   # [4+nc, na]
    pred = pred.T          # [na, 4+nc]

    detecciones = []
    for row in pred:
        box    = row[:4]
        scores = row[4:]
        class_id   = int(np.argmax(scores))
        confidence = float(scores[class_id])

        if confidence >= conf_threshold:
            cx, cy, w, h = box
            detecciones.append({
                "clase":     NAMES.get(str(class_id), str(class_id)),
                "confianza": round(confidence, 3),
                "bbox": [
                    round(float(cx - w / 2), 1),
                    round(float(cy - h / 2), 1),
                    round(float(cx + w / 2), 1),
                    round(float(cy + h / 2), 1),
                ]
            })

    return detecciones


# ── Health check (GET y HEAD para Render) ────────────────────
@app.api_route("/", methods=["GET", "HEAD"])
def health(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    return JSONResponse({"status": "ok", "clases": NAMES})


# ── Deteccion ─────────────────────────────────────────────────
@app.post("/detectar")
async def detectar(
    file:       UploadFile = File(...),
    confianza:  float = 0.40
):
    contents = await file.read()
    image    = Image.open(io.BytesIO(contents)).convert("RGB")
    arr      = preprocess(image)
    outputs  = session.run(None, {INPUT_NAME: arr})
    detecciones = postprocess(outputs, conf_threshold=confianza)

    return JSONResponse({
        "aprobada":    len(detecciones) > 0,
        "total":       len(detecciones),
        "confianza_minima": confianza,
        "detecciones": detecciones
    })
