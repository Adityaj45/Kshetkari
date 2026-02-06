import numpy as np
import tensorflow as tf
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
# import torch  # Commented out to save memory
# from torchvision import transforms # Commented out to save memory
from PIL import Image
import io
from datetime import datetime
from fastapi import File, UploadFile
# import torchvision.models as models # Commented out
# import torch.nn as nn # Commented out
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define the labels and treatment plans
PEST_MAPPING = {
    7: {
        "label": "Tomato___Late_blight",
        "severity": "High",
        "action": "Apply copper-based fungicides immediately.",
        "prevention": "Improve air circulation and avoid overhead watering.",
    }
}

# Commented out PyTorch Preprocessing
# preprocess = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=1),
#     transforms.ToTensor(),
# ])

def preprocess_plant_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array.astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

PLANT_MAPPING = {
    0: {"label": "Apple___Apple_scab", "action": "Apply sulfur-based fungicides."},
}

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- STARTUP: Loading TensorFlow & Scikit-Learn models... ---")

    # 1. Pest Risk Model (LSTM)
    try:
        ml_models["climate_model"] = tf.keras.models.load_model("models/lstm_pest_risk_model.h5")
        print("✅ Pest Risk Model loaded.")
    except Exception as e:
        print(f"❌ Error loading Pest Risk Model: {e}")

    # 2. Crop Recommendation Model
    try:
        ml_models["recommend_model"] = joblib.load("models/crop_recommendation_model.joblib")
        print("✅ Crop Recommendation Model loaded.")
    except Exception as e:
        print(f"❌ Error loading Recommendation Model: {e}")

    # 3. Crop Yield Prediction Model
    try:
        ml_models["yield_model"] = joblib.load("models/CropYieldPrediction.joblib")
        print("✅ Crop Yield Model loaded.")
    except Exception as e:
        print(f"❌ Error loading Yield Model: {e}")

    # 4. Pest Detection Model (PyTorch) - DISABLED
    # try:
    #     model = models.resnet50(weights=None)
    #     ... 
    #     ml_models["detection_model"] = model
    # except Exception as e:
    #     print(f"❌ CRITICAL ERROR loading Detection Model: {str(e)}")

    # 5. Plant Disease Model
    try:
        ml_models["plant_model"] = tf.keras.models.load_model("models/my_agri_model.h5")
        print("✅ Plant Disease detection model loaded.")
    except Exception as e:
        print(f"❌ Error loading Plant Disease Model: {e}")

    yield
    ml_models.clear()
    print("--- SHUTDOWN: Models unloaded. ---")

app = FastAPI(lifespan=lifespan)

# --- Schemas ---
class ClimateInput(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    pressure: float

class RecommendationInput(BaseModel):
    N: float; P: float; K: float
    temperature: float; humidity: float; ph: float; rainfall: float

class YieldInput(BaseModel):
    N: float; P: float; K: float; ph: float
    temperature: float; rainfall: float; humidity: float; year: int

# --- Helper ---
def preprocess_climate_input(data: ClimateInput):
    features = np.array([data.temperature, data.humidity, data.wind_speed, data.pressure])
    return np.repeat(features.reshape(1, 1, 4), 14, axis=1)

# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "AI Multi-Model Server is Running (PyTorch Disabled)"}

@app.post("/PestRiskModel")
async def predict_pest_risk(input_data: ClimateInput):
    model = ml_models.get("climate_model")
    if not model: raise HTTPException(status_code=500, detail="Model not loaded")
    probs = model.predict(preprocess_climate_input(input_data))
    idx = np.argmax(probs[0])
    labels = {0: "Low", 1: "Medium", 2: "High"}
    return {"risk": labels.get(idx), "confidence": float(np.max(probs[0]))}

@app.post("/recommendCrop")
async def recommend_crop(data: RecommendationInput):
    model = ml_models.get("recommend_model")
    if not model: raise HTTPException(status_code=500, detail="Model not loaded")
    features = [[data.N, data.P, data.K, data.temperature, data.humidity, data.ph, data.rainfall]]
    return {"recommended_crop": model.predict(features)[0]}

@app.post("/cropYieldPrediction")
async def predict_yield(data: YieldInput):
    model = ml_models.get("yield_model")
    if not model: raise HTTPException(status_code=500, detail="Model not loaded")
    features = [[data.N, data.P, data.K, data.ph, data.temperature, data.rainfall, data.humidity, data.year]]
    return {"prediction": float(model.predict(features)[0])}

@app.post("/PlantDetection")
async def detect_plant_disease(file: UploadFile = File(...)):
    model = ml_models.get("plant_model")
    if not model: raise HTTPException(status_code=500, detail="Model not loaded")
    image_data = await file.read()
    predictions = model.predict(preprocess_plant_image(image_data))
    idx = int(np.argmax(predictions[0]))
    info = PLANT_MAPPING.get(idx, {"label": "Unknown", "action": "Consult a botanist."})
    return {"label": info["label"], "confidence": round(float(np.max(predictions[0])), 4), "action": info["action"]}

# PestDetection endpoint removed or disabled
@app.post("/PestDetection")
async def detect_pest():
    return {"message": "Pest Detection (PyTorch) is currently disabled to save server memory."}