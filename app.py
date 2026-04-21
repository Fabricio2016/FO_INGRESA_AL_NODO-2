from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import gdown, os
from PIL import Image
import io

app = FastAPI()

# ── Configuracion del modelo ─────────────────────────────────
MODEL_PATH = "model.pt"
GDRIVE_ID  = "1tSSKvZesgWzhhX8vKBnk-7UoxkRTgKXI"

# Descarga el modelo al iniciar si no existe
if not os.path.exists(MODEL_PATH):
    print("Descargando modelo desde Google Drive...")
    gdown.download(
        f"https://drive.google.com/uc?id={GDRIVE_ID}",
        MODEL_PATH,
        quiet=False
    )
    print("Modelo descargado correctamente.")

# Cargar modelo YOLO
model = YOLO(MODEL_PATH)
print("Modelo YOLO cargado y listo.")


# ── Endpoint de salud ────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "mensaje": "API YOLO FO_INGRESA_AL_NODO funcionando"}


# ── Endpoint de deteccion ────────────────────────────────────
@app.post("/detectar")
async def detectar(file: UploadFile = File(...)):
    """
    Recibe una imagen y devuelve las detecciones del modelo YOLO.
    """
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

    aprobada = len(detecciones) > 0

    return JSONResponse({
        "aprobada":    aprobada,
        "total":       len(detecciones),
        "detecciones": detecciones
    })
