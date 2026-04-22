from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from PIL import Image
import io, json, base64

app = FastAPI()

with open("names.json") as f:
    NAMES = json.load(f)

session    = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name
IMG_SIZE   = session.get_inputs()[0].shape[2] or 640
print(f"Model loaded. Classes: {NAMES} | Input size: {IMG_SIZE}px")


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    return arr


def postprocess(outputs, conf_threshold: float = 0.40):
    pred = outputs[0][0].T
    detecciones = []
    for row in pred:
        box        = row[:4]
        scores     = row[4:]
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


class DetectarRequest(BaseModel):
    image_base64: str
    confianza: float = 0.40


@app.api_route("/", methods=["GET", "HEAD"])
def health(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    return JSONResponse({"status": "ok", "clases": NAMES})


@app.post("/detectar")
async def detectar(req: DetectarRequest):
    contents    = base64.b64decode(req.image_base64)
    image       = Image.open(io.BytesIO(contents)).convert("RGB")
    arr         = preprocess(image)
    outputs     = session.run(None, {INPUT_NAME: arr})
    detecciones = postprocess(outputs, conf_threshold=req.confianza)

    clases = list({d["clase"] for d in detecciones})

    return JSONResponse({
        "aprobada":        len(detecciones) > 0,
        "total":           len(detecciones),
        "clases":          clases,
        "confianza_minima": req.confianza,
        "detecciones":     detecciones
    })
