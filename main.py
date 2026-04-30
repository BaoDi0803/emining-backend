from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io, numpy as np, os, random

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thử load model thật nếu có
MODEL = None
try:
    from ultralytics import YOLO
    if os.path.exists("emining_model.pt"):
        MODEL = YOLO("emining_model.pt")
        print("✅ Model thật đã load")
    else:
        print("⚠️ Không tìm thấy model, dùng demo mode")
except Exception as e:
    print(f"⚠️ Không load được model: {e}")

PRICES = {
    "chip":       {"vnd": 45000, "credits": 15, "co2": 120, "metal": "Au, Ag"},
    "capacitor":  {"vnd": 8000,  "credits": 3,  "co2": 22,  "metal": "Al, Ta"},
    "connector":  {"vnd": 32000, "credits": 10, "co2": 85,  "metal": "Au"},
    "resistor":   {"vnd": 2500,  "credits": 1,  "co2": 8,   "metal": "Ni, C"},
    "transistor": {"vnd": 12000, "credits": 4,  "co2": 35,  "metal": "Si"},
    "diode":      {"vnd": 5000,  "credits": 2,  "co2": 15,  "metal": "Si, Ge"},
}

ALL_CLASSES = list(PRICES.keys())

def analyze_image_features(img: Image.Image) -> dict:
    """
    Phân tích đặc trưng ảnh để tạo kết quả demo đa dạng.
    Dùng màu sắc và độ sáng trung bình của ảnh làm seed.
    """
    img_small = img.resize((64, 64)).convert("RGB")
    arr = np.array(img_small)
    
    # Dùng giá trị pixel trung bình làm seed → mỗi ảnh cho kết quả khác nhau
    seed = int(arr.mean() * 1000 + arr.std() * 100 + arr[0][0][0] * 10)
    rng = random.Random(seed)
    
    # Số lượng linh kiện: 2–5, phụ thuộc vào độ phức tạp ảnh (std)
    complexity = float(arr.std())
    if complexity > 60:
        count = rng.randint(4, 5)
    elif complexity > 40:
        count = rng.randint(3, 4)
    else:
        count = rng.randint(2, 3)
    
    # Chọn các class ngẫu nhiên dựa theo seed của ảnh
    chosen = rng.sample(ALL_CLASSES, min(count, len(ALL_CLASSES)))
    
    # Confidence cũng biến động theo ảnh
    results = []
    for cls in chosen:
        conf = round(rng.uniform(0.65, 0.96), 2)
        p = PRICES[cls]
        results.append({
            "class": cls,
            "confidence": conf,
            "value_vnd": p["vnd"],
            "green_credits": p["credits"],
            "co2_saved_g": p["co2"],
            "metal": p["metal"],
        })
    
    return {
        "status": "demo_mode",
        "components": results,
        "total_vnd": sum(r["value_vnd"] for r in results),
        "total_credits": sum(r["green_credits"] for r in results),
    }

@app.get("/")
def root():
    return {
        "status": "E-Mining AI API running",
        "model_loaded": MODEL is not None,
        "mode": "real" if MODEL else "demo"
    }

@app.post("/scan")
async def scan(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Nếu có model thật → dùng model thật
    if MODEL is not None:
        arr = np.array(img)
        results_raw = MODEL(arr, conf=0.5)[0]
        detected, total_vnd, total_credits = [], 0, 0
        for box in results_raw.boxes:
            cls = MODEL.names[int(box.cls)].lower()
            p = PRICES.get(cls, {"vnd": 1000, "credits": 1, "co2": 5, "metal": "?"})
            item = {
                "class": cls,
                "confidence": round(float(box.conf), 2),
                "value_vnd": p["vnd"],
                "green_credits": p["credits"],
                "co2_saved_g": p["co2"],
                "metal": p["metal"],
            }
            detected.append(item)
            total_vnd += p["vnd"]
            total_credits += p["credits"]
        return {
            "status": "ok",
            "components": detected,
            "total_vnd": total_vnd,
            "total_credits": total_credits,
        }
    
    # Không có model → phân tích đặc trưng ảnh để demo đa dạng
    return analyze_image_features(img)
