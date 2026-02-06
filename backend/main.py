import numpy as np
import tensorflow as tf
import joblib
import io
import os
from PIL import Image
from datetime import datetime  # Added missing import
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Disable GPU if you're running on a memory-constrained server
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# =========================================================
# CONFIGURATION & MAPPINGS
# =========================================================

PLANT_MAPPING = {
    0: {"label": "Apple___Apple_scab", "action": "Apply sulfur-based fungicides."},
    7: {
        "label": "Tomato___Late_blight",
        "action": "Apply copper-based fungicides immediately and improve air circulation.",
    },
    # Add your specific plant indices here...
}

RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}

# =========================================================
# INPUT SCHEMAS
# =========================================================


class ClimateInput(BaseModel):
    temperature: float = Field(..., example=18.5)
    humidity: float = Field(..., example=77.22)
    wind_speed: float = Field(..., example=2.89)
    pressure: float = Field(..., example=1018.27)


class RecommendationInput(BaseModel):
    N: float = Field(..., example=90.0)
    P: float = Field(..., example=42.0)
    K: float = Field(..., example=43.0)
    temperature: float = Field(..., example=25.4)
    humidity: float = Field(..., example=82.1)
    ph: float = Field(..., example=6.5)
    rainfall: float = Field(..., example=202.9)


class YieldInput(BaseModel):
    N: float
    P: float
    K: float
    ph: float  # <--- THIS WILL THROW A SYNTAX ERROR

    # Fix it like this:
    N: float = Field(..., example=90.0)
    P: float = Field(..., example=42.0)
    K: float = Field(..., example=43.0)
    ph: float = Field(..., example=6.5)
    temperature: float = Field(..., example=25.0)
    rainfall: float = Field(..., example=200.0)
    humidity: float = Field(..., example=80.0)
    year: int = Field(..., example=2024)


# =========================================================
# MODEL LIFESPAN MANAGEMENT
# =========================================================

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- STARTUP: Loading AI Models ---")

    def load_model(name, path, loader_func):
        if os.path.exists(path):
            try:
                ml_models[name] = loader_func(path)
                print(f"✅ {name} loaded.")
            except Exception as e:
                print(f"❌ Error loading {name}: {e}")
        else:
            print(f"⚠️ Warning: Model file not found at {path}")

    load_model(
        "climate_model", "models/lstm_pest_risk_model.h5", tf.keras.models.load_model
    )
    load_model(
        "recommend_model", "models/crop_recommendation_model.joblib", joblib.load
    )
    load_model("yield_model", "models/CropYieldPrediction.joblib", joblib.load)
    load_model("plant_model", "models/my_agri_model.h5", tf.keras.models.load_model)

    yield
    ml_models.clear()
    print("--- SHUTDOWN: Models unloaded ---")


app = FastAPI(title="Agri-AI Unified Server", lifespan=lifespan)

# =========================================================
# PREPROCESSING HELPERS
# =========================================================


def preprocess_plant_image(image_bytes):
    """Converts image to (1, 224, 224, 3) NHWC format for TensorFlow."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)


def preprocess_climate_input(data: ClimateInput):
    # LSTM models usually expect: [samples, time_steps, features]
    # Here we assume a single time step (1, 1, 4)
    features = np.array(
        [[data.temperature, data.humidity, data.wind_speed, data.pressure]]
    )
    return np.reshape(features, (1, 1, 4))


# =========================================================
# ENDPOINTS
# =========================================================


@app.get("/")
async def root():
    return {
        "status": "online",
        "models_loaded": list(ml_models.keys()),
        "pytorch_status": "disabled",
    }


@app.post("/PestRiskModel")
async def predict_pest_risk(input_data: ClimateInput):
    model = ml_models.get("climate_model")
    if not model:
        raise HTTPException(500, "Climate Model not active")

    probs = model.predict(preprocess_climate_input(input_data))
    idx = np.argmax(probs[0])
    return {
        "risk": RISK_LABELS.get(idx, "Unknown"),
        "confidence": f"{float(np.max(probs[0]) * 100):.2f}%",
        "probabilities": probs[0].tolist(),
    }


@app.post("/recommendCrop")
async def recommend_crop(data: RecommendationInput):
    model = ml_models.get("recommend_model")
    if not model:
        raise HTTPException(500, "Recommendation Model not active")

    features = [
        [
            data.N,
            data.P,
            data.K,
            data.temperature,
            data.humidity,
            data.ph,
            data.rainfall,
        ]
    ]
    prediction = model.predict(features)
    return {"recommended_crop": prediction[0]}


@app.post("/cropYieldPrediction")
async def predict_yield(data: YieldInput):
    model = ml_models.get("yield_model")
    if not model:
        raise HTTPException(500, "Yield Model not active")

    features = [
        [
            data.N,
            data.P,
            data.K,
            data.ph,
            data.temperature,
            data.rainfall,
            data.humidity,
            data.year,
        ]
    ]
    prediction = model.predict(features)
    return {"predicted_yield": float(prediction[0])}


@app.post("/PlantDetection")
async def detect_plant_disease(file: UploadFile = File(...)):
    model = ml_models.get("plant_model")
    if not model:
        raise HTTPException(status_code=500, detail="Plant Model not active")

    try:
        image_data = await file.read()
        processed_image = preprocess_plant_image(image_data)

        # TensorFlow Inference
        predictions = model.predict(processed_image, verbose=0)

        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        info = PLANT_MAPPING.get(
            predicted_idx, {"label": "Unknown Disease", "action": "Consult a botanist."}
        )

        return {
            "status": "success",
            "prediction": {
                "class_index": predicted_idx,
                "label": info["label"],
                "confidence": round(confidence, 4),
                "timestamp": datetime.utcnow().isoformat(),
            },
            "recommendation": info["action"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")


@app.post("/PestDetection")
async def detect_pest_placeholder():
    return {
        "message": "Pest Detection (PyTorch) is currently disabled to optimize memory usage."
    }


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)