from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import time

app = FastAPI()

# CORS (VERY IMPORTANT for browser camera)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")  # fire detection model

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    start = time.time()

    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    results = model(img, conf=0.25, verbose=False)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "label": "fire",
                "confidence": float(box.conf[0]),
                "bbox": [x1, y1, x2, y2]
            })

    return {
        "latency_ms": int((time.time() - start) * 1000),
        "detections": detections
    }
