
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from starlette.middleware.cors import CORSMiddleware

# Load artifacts
model = joblib.load('ml_models3/crop_model.pkl')
label_encoder = joblib.load('ml_models3/label_encoder.pkl')

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this later to specific domains in production
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