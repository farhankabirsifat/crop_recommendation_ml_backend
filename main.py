from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Change this later to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scalers
# model = joblib.load("ml_models/model.pkl")


MODEL_PATH = os.getenv("MODEL_PATH", "ml_models2/model.pkl")
model = joblib.load(MODEL_PATH)

minmax_scaler = joblib.load("ml_models2/minmaxscaler.pkl")
standard_scaler = joblib.load("ml_models2/standscaler.pkl")

# MODEL_PATH = os.getenv("MODEL_PATH", "ml_models/model.pkl")
# model = joblib.load(MODEL_PATH)
#
# minmax_scaler = joblib.load("ml_models/minmaxscaler.pkl")
# standard_scaler = joblib.load("ml_models/standard_scaler.pkl")

# Input schema
class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/predict")
def predict_crop(data: CropInput):
    try:
        import pandas as pd

        # Mapping crop IDs to names
        crop_mapping = {
            1: "Rice", 2: "Maize", 3: "Chickpea", 4: "Kidney Beans", 5: "Pigeon Peas",
            6: "Moth Beans", 7: "Mung Bean", 8: "Black Gram", 9: "Lentil",
            10: "Pomegranate", 11: "Banana", 12: "Mango", 13: "Grapes",
            14: "Watermelon", 15: "Muskmelon", 16: "Apple", 17: "Orange",
            18: "Papaya", 19: "Coconut", 20: "Cotton", 21: "Jute", 22: "Coffee"
        }

        input_df = pd.DataFrame([{
            "N": data.N,
            "P": data.P,
            "K": data.K,
            "temperature": data.temperature,
            "humidity": data.humidity,
            "ph": data.ph,
            "rainfall": data.rainfall
        }])

        input_scaled = minmax_scaler.transform(input_df)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_scaled)[0]
            class_labels = model.classes_

            ranked_predictions = sorted(
                zip(class_labels, probabilities),
                key=lambda x: x[1],
                reverse=True
            )

            top_5 = ranked_predictions[:5]

            return {
                "recommended_crops": [
                    {
                        "crop": crop_mapping.get(int(crop), f"Unknown ({crop})")
                        # "probability": float(round(prob, 4))
                    }
                    for crop, prob in top_5
                ]
            }
        else:
            prediction = model.predict(input_scaled)
            return {
                "recommended_crops": [
                    {"crop": crop_mapping.get(int(prediction[0]), "Unknown"), "probability": None}
                ]
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
