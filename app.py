from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, Response
from ultralytics import YOLO
import gdown, os
from PIL import Image
import io

app = FastAPI()

# ── Configuracion del modelo ─────────────────────────────────
MODEL_PATH = "model.pt"
GDRIVE_ID  = "1tSSKvZesgWzhhX8vKBnk-7UoxkRTgKXI"

if not os.path.exists(MODEL_PATH):
    print("Descargando modelo desde Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_ID}",
        MODEL_PATH,
        quiet=False
    )
    print("Modelo descargado correctamente.")

model = YOLO(MODEL_PATH)
print("Modelo YOLO cargado y listo.")


# ── Health check — acepta GET y HEAD (Render usa HEAD para verificar) ──
@app.api_route("/", methods=["GET", "HEAD"])
def health(request: Request):
    if request.method == "HEAD":
        return Response(status_code=200)
    return JSONResponse({"status": "ok", "mensaje": "API YOLO FO_INGRESA_AL_NODO funcionando"})


# ── Endpoint de deteccion ─────────────────────────────────────
@app.post("/detectar")
async def detectar(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image)

    detecciones = []
    for r in results:
        for box in r.boxes:
            detecciones.append({
                "clase":     model.names[int(box.cls)],
                "confianza": round(float(box.conf), 3),
                "bbox":      [round(x, 1) for x in box.xyxy[0].tolist()]
            })

    return JSONResponse({
        "aprobada":    len(detecciones) > 0,
        "total":       len(detecciones),
        "detecciones": detecciones
    })
