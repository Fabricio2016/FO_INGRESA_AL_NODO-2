from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import JSONResponse, Response
from typing import Optional
import onnxruntime as ort
import numpy as np
from PIL import Image
import io, json, os

app = FastAPI()

# ── Cargar nombres de clases ─────────────────────────────────
with open("names.json") as f:
    NAMES = json.load(f)

# ── Cargar modelo ONNX ────────────────────────────────────────
session    = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name
IMG_SIZE   = session.get_inputs()[0].shape[2] or 640

# ── Token de Telegram (env var en Render) ────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

print(f"Modelo ONNX cargado. Clases: {NAMES} | Input: {IMG_SIZE}px")
print(f"Telegram token: {'configurado' if TELEGRAM_TOKEN else 'NO configurado'}")


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    return arr


def postprocess(outputs, conf_threshold: float = 0.40):
    pred = outputs[0][0].T
    detecciones = []
    for row in pred:
        scores     = row[4:]
        class_id   = int(np.argmax(scores))
        confidence = float(scores[class_id])
        if confidence >= conf_threshold:
            cx, cy, w, h = row[:4]
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


async def get_image_bytes(
    file:    Optional[UploadFile],
    file_id: Optional[str]
) -> bytes:
    """Acepta imagen por upload binario O por file_id de Telegram."""
    if file is not None:
        return await file.read()

    if file_id and TELEGRAM_TOKEN:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            # 1. Obtener file_path desde Telegram
            r = await client.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getFile",
                params={"file_id": file_id}
            )
            r.raise_for_status()
            file_path = r.json()["result"]["file_path"]
            # 2. Descargar el archivo
            dl = await client.get(
                f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}"
            )
            dl.raise_for_status()
            return dl.content

    raise ValueError(
        "Proporciona 'file' (upload binario) o 'file_id' "
        "(requiere TELEGRAM_BOT_TOKEN como variable de entorno en Render)"
    )


# ── Health check ──────────────────────────────────────────────
@app.api_route("/", methods=["GET", "HEAD"])
def health(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    return JSONResponse({
        "status": "ok",
        "clases": NAMES,
        "telegram_token": bool(TELEGRAM_TOKEN)
    })


# ── Deteccion ─────────────────────────────────────────────────
@app.post("/detectar")
async def detectar(
    file:      Optional[UploadFile] = File(default=None),
    file_id:   Optional[str]        = Form(default=None),
    confianza: float                 = 0.40
):
    contents    = await get_image_bytes(file, file_id)
    image       = Image.open(io.BytesIO(contents)).convert("RGB")
    arr         = preprocess(image)
    outputs     = session.run(None, {INPUT_NAME: arr})
    detecciones = postprocess(outputs, conf_threshold=confianza)

    return JSONResponse({
        "aprobada":         len(detecciones) > 0,
        "total":            len(detecciones),
        "confianza_minima": confianza,
        "detecciones":      detecciones
    })
