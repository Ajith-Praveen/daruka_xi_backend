from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt")
model.fuse()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((640, 480))

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

    return detections
