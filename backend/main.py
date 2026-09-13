import io
import os
import base64

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "models/best.pt")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", 0.25))

app = FastAPI(title="Object Detection API")

# Allow the frontend (served from a different origin/port) to call this API.
# Lock this down to your actual frontend URL before deploying for real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(WEIGHTS_PATH):
        raise RuntimeError(
            f"Model weights not found at '{WEIGHTS_PATH}'. "
            f"Train your model (see the Lab 6 notebook), download best.pt, "
            f"and place it at backend/models/best.pt."
        )
    model = YOLO(WEIGHTS_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/classes")
def classes():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"classes": model.names}


@app.post("/predict")
async def predict(file: UploadFile = File(...), conf: float | None = None):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded image")

    conf_override = conf if conf is not None else CONF_THRESHOLD
    results = model.predict(image, conf=conf_override, verbose=False)
    result = results[0]

    detections = []
    counts = {name: 0 for name in model.names.values()}
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": round(float(box.conf[0]), 4),
            "box": {
                "x1": round(x1, 1), "y1": round(y1, 1),
                "x2": round(x2, 1), "y2": round(y2, 1),
            },
        })
        counts[cls_name] = counts.get(cls_name, 0) + 1

    # Ratios between class counts (only meaningful for this BCCD-style
    # dataset, but harmless/empty for any other class set).
    ratios = {}
    if counts.get("RBC", 0) > 0:
        if "WBC" in counts:
            ratios["WBC_to_RBC"] = round(counts["WBC"] / counts["RBC"], 4)
        if "Platelets" in counts:
            ratios["Platelets_to_RBC"] = round(counts["Platelets"] / counts["RBC"], 4)

    # Also return an annotated image (boxes drawn) as base64, so the
    # frontend can display it without re-implementing box drawing.
    annotated = result.plot()  # numpy array, BGR
    annotated_image = Image.fromarray(annotated[:, :, ::-1])  # BGR -> RGB
    buf = io.BytesIO()
    annotated_image.save(buf, format="JPEG")
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "detections": detections,
        "total_count": len(detections),
        "counts_by_class": counts,
        "ratios": ratios,
        "confidence_used": conf_override,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
