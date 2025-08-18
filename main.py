from http.client import HTTPException

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from starlette.middleware.cors import CORSMiddleware

from soil_fertility_adjustment_engin import crop_list, get_ml_recommendations, crop_stats

# Load artifacts
model = joblib.load('ml_models3/crop_model.pkl')
label_encoder = joblib.load('ml_models3/label_encoder.pkl')
# Load all artifacts
artifacts = joblib.load('SFAR_Model/fertility_models_reduced.pkl')
models = artifacts['models']
crop_stats = artifacts['crop_stats']
crop_list = artifacts['crop_list']
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CropRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
class FertilityRequest(BaseModel):
    crop: str
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float

@app.post("/predict")
def predict_crop(request: CropRequest):
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([[
            request.N,
            request.P,
            request.K,
            request.temperature,
            request.humidity,
            request.ph
        ]], columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph'])

        # Get probabilities for all crops
        probabilities = model.predict_proba(input_data)[0]

        # Create sorted list of (crop, probability)
        crop_probs = sorted(
            zip(label_encoder.classes_, probabilities),
            key=lambda x: x[1],
            reverse=True
        )

        # Get top 5 recommendations
        top_5 = [{"crop": crop, "probability": round(float(prob * 100), 2)} for crop, prob in crop_probs[:5]]

        return {"recommendations": top_5}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


# @app.get("/crops")
# def list_crops():
#     return {"crops": list(label_encoder.classes_)}
#
#
# @app.get("/")
# def read_root():
#     return {"message": "Crop Recommendation API - Returns top 5 crops"}



# Map lowercase crop names to original names
crop_map = {c.lower(): c for c in crop_list}

@app.post("/ml-fertility-recommendations")
def get_recommendations(request: FertilityRequest):
    crop_input = request.crop.strip().lower()
    if crop_input not in crop_map:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid crop. Available crops: {crop_list}"
        )
    crop_name = crop_map[crop_input]
    try:
        recommendations = get_ml_recommendations(
            crop_name,
            request.N,
            request.P,
            request.K,
            request.temperature,
            request.humidity,
            request.ph
        )
        input_df = pd.DataFrame([[request.temperature, request.humidity, request.ph]],
                                columns=['temperature', 'humidity', 'ph'])
        ideal_values = {
            'N': f"{crop_stats[crop_name]['median']['N']:.2f}",
            'P': f"{crop_stats[crop_name]['median']['P']:.2f}",
            'K': f"{crop_stats[crop_name]['median']['K']:.2f}",
            'temperature': f"{models['temperature'][crop_name].predict(input_df)[0]:.2f}",
            'humidity': f"{models['humidity'][crop_name].predict(input_df)[0]:.2f}",
            'ph': f"{models['ph'][crop_name].predict(input_df)[0]:.2f}"
        }
        return {
            "crop": crop_name,
            "current_parameters": request.dict(),
            "predicted_ideal": ideal_values,
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )