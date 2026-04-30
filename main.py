from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io, numpy as np, os

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], 
                   allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = "emining_model.pt"
model = YOLO(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

PRICES = {
    "chip":       {"vnd": 45000, "credits": 15, "co2": 120, "metal": "Au, Ag"},
    "capacitor":  {"vnd": 8000,  "credits": 3,  "co2": 22,  "metal": "Al"},
    "connector":  {"vnd": 32000, "credits": 10, "co2": 85,  "metal": "Au"},
    "resistor":   {"vnd": 2500,  "credits": 1,  "co2": 8,   "metal": "Ni"},
    "transistor": {"vnd": 12000, "credits": 4,  "co2": 35,  "metal": "Si"},
    "diode":      {"vnd": 5000,  "credits": 2,  "co2": 15,  "metal": "Si"},
}

@app.get("/")
def root():
    return {"status": "E-Mining AI API running", "model_loaded": model is not None}

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    if model is None:
        # Trả về dữ liệu mẫu nếu chưa có model — dùng cho demo
        return {
            "status": "demo_mode",
            "components": [
                {"class": "chip", "confidence": 0.91, "value_vnd": 45000,
                 "green_credits": 15, "co2_saved_g": 120, "metal": "Au, Ag"},
                {"class": "capacitor", "confidence": 0.85, "value_vnd": 8000,
                 "green_credits": 3, "co2_saved_g": 22, "metal": "Al"},
                {"class": "connector", "confidence": 0.78, "value_vnd": 32000,
                 "green_credits": 10, "co2_saved_g": 85, "metal": "Au"},
            ],
            "total_vnd": 85000,
            "total_credits": 28,
        }
    
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    results = model(np.array(img), conf=0.5)[0]
    
    detected, total_vnd, total_credits = [], 0, 0
    for box in results.boxes:
        cls = model.names[int(box.cls)].lower()
        p = PRICES.get(cls, {"vnd": 1000, "credits": 1, "co2": 5, "metal": "?"})
        detected.append({
            "class": cls,
            "confidence": round(float(box.conf), 2),
            "value_vnd": p["vnd"],
            "green_credits": p["credits"],
            "co2_saved_g": p["co2"],
            "metal": p["metal"],
        })
        total_vnd += p["vnd"]
        total_credits += p["credits"]
    
    return {"status": "ok", "components": detected,
            "total_vnd": total_vnd, "total_credits": total_credits}
