import numpy as np
import tensorflow as tf
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import torch
from torchvision import transforms
from PIL import Image
import io
from datetime import datetime
from fastapi import File, UploadFile
import torchvision.models as models
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define the labels and treatment plans (Example mapping)
PEST_MAPPING = {
    7: {
        "label": "Tomato___Late_blight",
        "severity": "High",
        "action": "Apply copper-based fungicides immediately.",
        "prevention": "Improve air circulation and avoid overhead watering.",
    }
    # Add other indices here...
}

# Image Preprocessing Pipeline
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # <--- ADD THIS LINE
        transforms.ToTensor(),
        # Note: If your model expects 0-1 values, you might not need the Normalize
        # for RGB. If you do use it, ensure it's a single value for 1 channel:
        # transforms.Normalize([0.5], [0.5]),
    ]
)


def preprocess_plant_image(image_bytes):
    """
    Specific preprocessor for TensorFlow .h5 models.
    Converts image to (1, 224, 224, 3) NHWC format.
    """
    # 1. Load image and ensure RGB
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # 2. Resize to 224x224
    img = img.resize((224, 224))

    # 3. Convert to Numpy Array (Standard TF format: H, W, C)
    img_array = np.array(img)

    # 4. Normalize (Scale pixels to 0-1)
    img_array = img_array.astype("float32") / 255.0

    # 5. Add Batch Dimension -> (1, 224, 224, 3)
    return np.expand_dims(img_array, axis=0)


# 1. Add another mapping for Plants
PLANT_MAPPING = {
    0: {"label": "Apple___Apple_scab", "action": "Apply sulfur-based fungicides."},
    # Add your specific plant indices here...
}

# 2. Add an RGB Preprocessing Pipeline (since your Pest model uses Grayscale)
preprocess_rgb = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# =========================================================
# GLOBAL VARIABLES & LIFESPAN
# =========================================================

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- STARTUP: Loading models... ---")

    # 1. Pest Risk Model (LSTM)
    try:
        ml_models["climate_model"] = tf.keras.models.load_model(
            "models/lstm_pest_risk_model.h5"
        )
        print("✅ Pest Risk Model loaded.")
    except Exception as e:
        print(f"❌ Error loading Pest Risk Model: {e}")

    # 2. Crop Recommendation Model
    try:
        ml_models["recommend_model"] = joblib.load(
            "models/crop_recommendation_model.joblib"
        )
        print("✅ Crop Recommendation Model loaded.")
    except Exception as e:
        print(f"❌ Error loading Recommendation Model: {e}")

    # 3. Crop Yield Prediction Model
    try:
        ml_models["yield_model"] = joblib.load("models/CropYieldPrediction.joblib")
        print("✅ Crop Yield Model loaded.")
    except Exception as e:
        print(f"❌ Error loading Yield Model: {e}")

    # 4. Pest Detection Model (PyTorch)
    try:
        model = models.resnet50(weights=None)

        # CHANGE: Modify the very first convolutional layer to accept 1 channel instead of 3
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

        # Step B: Match the final layer (Example: 10 classes)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

        # Step C: Load weights
        state_dict = torch.load(
            "models/new_best_model.pth", map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict)
        model.eval()

        ml_models["detection_model"] = model
        print("✅ Binary Pest Detection Model loaded successfully.")

    except Exception as e:
        print(f"❌ CRITICAL ERROR loading Detection Model: {str(e)}")
        # We don't raise here so the other models can still work,
        # but this is why your endpoint says "not active"

    try:
        ml_models["plant_model"] = tf.keras.models.load_model("models/my_agri_model.h5")
        print("✅ Plant Disease detection model loaded.")
    except Exception as e:
        print(f"❌ Error loading Plant Disease Model: {e}")

    yield
    ml_models.clear()
    print("--- SHUTDOWN: Models unloaded. ---")


app = FastAPI(lifespan=lifespan)

# =========================================================
# INPUT SCHEMAS
# =========================================================


class ClimateInput(BaseModel):
    date: str
    temperature: float
    humidity: float
    wind_speed: float
    pressure: float

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2017-01-02",
                "temperature": 18.5,
                "humidity": 77.22,
                "wind_speed": 2.89,
                "pressure": 1018.27,
            }
        }


class RecommendationInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class YieldInput(BaseModel):
    crop_name: str = Field(..., example="Rice")
    season: str = Field(..., example="Kharif")
    state: str = Field(..., example="Maharashtra")
    N: float = Field(..., example=90.0)
    P: float = Field(..., example=42.0)
    K: float = Field(..., example=43.0)
    ph: float = Field(..., example=6.5)
    temperature: float = Field(..., example=25.4)
    rainfall: float = Field(..., example=202.9)
    humidity: float = Field(..., example=82.1)
    fertilizer: float = Field(..., example=150.5)
    pesticide: float = Field(..., example=10.2)
    year: int = Field(..., example=2024)


# =========================================================
# HELPER FUNCTIONS
# =========================================================


def preprocess_climate_input(data: ClimateInput):
    # Extracts features and reshapes for LSTM (1, 14, 4)
    features = [data.temperature, data.humidity, data.wind_speed, data.pressure]
    features_array = np.array(features)
    # Reshape to (1, 1, 4)
    features_reshaped = features_array.reshape(1, 1, 4)
    # Repeat across the time-step dimension to reach (1, 14, 4)
    final_features = np.repeat(features_reshaped, 14, axis=1)
    return final_features


# =========================================================
# ENDPOINTS
# =========================================================


@app.get("/")
async def root():
    return {"message": "AI Multi-Model Server is Running"}


@app.post("/PestRiskModel")
async def predict_pest_risk(input_data: ClimateInput):
    model = ml_models.get("climate_model")
    if not model:
        raise HTTPException(status_code=500, detail="Climate Model not loaded")

    try:
        # 1. Preprocess and Reshape for LSTM
        features = preprocess_climate_input(input_data)

        # 2. Run Prediction
        prediction_probs = model.predict(features)

        # 3. Interpret Results
        predicted_index = np.argmax(prediction_probs[0])
        confidence = np.max(prediction_probs[0]) * 100
        risk_labels = {0: "Low", 1: "Medium", 2: "High"}

        return {
            "summary": f"Predicted Risk = {risk_labels.get(predicted_index, 'Unknown')}, Confidence = {confidence:.2f}%",
            "details": {
                "probabilities": prediction_probs[
                    0
                ].tolist(),  # Convert to list for JSON serialization
                "predicted_label": risk_labels.get(predicted_index),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LSTM Prediction Error: {str(e)}")


@app.post("/recommendCrop")
async def recommend_crop(data: RecommendationInput):
    model = ml_models.get("recommend_model")
    if not model:
        raise HTTPException(status_code=500, detail="Recommendation Model not active")

    try:
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
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction Error: {str(e)}")


@app.post("/cropYieldPrediction")
def predict_yield(data: YieldInput):
    model = ml_models.get("yield_model")
    if not model:
        raise HTTPException(status_code=500, detail="Yield Model not active")

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

    try:
        prediction = model.predict(features)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================================================
# NEW: PEST DETECTION ENDPOINT
# =========================================================


@app.post("/PestDetection")
async def detect_pest(file: UploadFile = File(...)):
    model = ml_models.get("detection_model")
    if not model:
        raise HTTPException(status_code=500, detail="Detection Model not active")

    try:
        # 1. Read and Preprocess Image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)

        # 2. Run Inference
        with torch.no_grad():
            # CHANGE: Ensure we are calling the model correctly
            # If 'model' is a dict, it means you saved state_dict and
            # need to load it into a model class first.
            if isinstance(model, dict):
                raise HTTPException(
                    status_code=500,
                    detail="Model loaded as dict. Initialize class first.",
                )

            outputs = model(input_tensor)

            # If the model returns a tuple (common in some architectures), grab the first element
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, class_idx = torch.max(probabilities, 0)

        idx = int(class_idx.item())
        info = PEST_MAPPING.get(
            idx,
            {
                "label": "Unknown",
                "severity": "N/A",
                "action": "Consult an expert.",
                "prevention": "N/A",
            },
        )

        # 3. Format Response
        return {
            "status": "success",
            "prediction": {
                "class_index": idx,
                "label": info["label"],
                "confidence": round(float(confidence.item()), 4),
                "detected_at": datetime.utcnow().isoformat() + "Z",
            },
            "treatment_plan": {
                "severity": info["severity"],
                "action": info["action"],
                "prevention": info["prevention"],
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image Processing Error: {str(e)}")


@app.post("/PlantDetection")
async def detect_plant_disease(file: UploadFile = File(...)):
    model = ml_models.get("plant_model")
    if not model:
        raise HTTPException(status_code=500, detail="Plant Model not active")

    try:
        # 1. Read and Preprocess using the NEW helper
        image_data = await file.read()
        processed_image = preprocess_plant_image(image_data)

        # 2. Run Inference using TensorFlow's .predict()
        predictions = model.predict(processed_image)
        
        # 3. Interpret Results
        predicted_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        info = PLANT_MAPPING.get(
            predicted_idx, 
            {"label": "Unknown Disease", "action": "Consult a botanist."}
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
        raise HTTPException(status_code=500, detail=f"Plant Image Error: {str(e)}")