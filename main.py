from http.client import HTTPException
import os
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import google.genai as genai
from google.genai import types

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

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-001")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY missing. Put it in a .env file.")

# Create a single client for the app
client = genai.Client(api_key=API_KEY)
app = FastAPI(title="Foshol_doot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---- In-memory session store (swap to Redis/DB in prod) ----
SESSIONS: Dict[str, List[types.Content]] = {}

# ---- System instruction shapes the bot’s behavior ----
AGRI_SYSTEM = """You are AgriExpert - an AI assistant specialized in agriculture. Provide professional advice on:
1. Crop cultivation (wheat, rice, corn, vegetables, fruits)
2. Soil management and fertilization
3. Pest/disease identification and organic control
4. Irrigation techniques and water conservation
5. Livestock management and animal health
6. Agricultural machinery and precision farming
7. Market trends and crop economics

Guidelines:
- Be concise and practical
- Ask clarifying questions when needed (location, crop type, farm size)
- For pesticide recommendations, include safety precautions
- For medical queries, recommend consulting a veterinarian
- For non-agricultural topics, politely decline
"""
# ---- Shared generation config (you can tune) ----
BASE_CONFIG = types.GenerateContentConfig(
    system_instruction=AGRI_SYSTEM,
    temperature=0.4,
    top_p=0.9,
    max_output_tokens=800,
    safety_settings=[  # relax or tighten as needed
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH",
        ),
    ],
)
# Safety settings reference. :contentReference[oaicite:3]{index=3}

# ---------- Schemas ----------
class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client-side chat session ID")
    message: str = Field(..., description="User message")

class ChatResponse(BaseModel):
    text: str

# Structured (JSON) advice shape using Pydantic schema → model returns valid JSON
class AgriAdvice(BaseModel):
    crop: Optional[str] = None
    diagnosis: Optional[str] = None
    key_factors: List[str] = []
    recommended_actions: List[str] = []
    monitoring: List[str] = []
    safety_notes: List[str] = []

class StructuredChatRequest(ChatRequest):
    pass

class StructuredChatResponse(BaseModel):
    advice: AgriAdvice

# ---------- Helpers ----------
def get_history(session_id: str) -> List[types.Content]:
    return SESSIONS.setdefault(session_id, [])

def add_user_msg(history: List[types.Content], text: str):
    history.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))

def add_model_msg(history: List[types.Content], text: str):
    history.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))


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

    # Agrichatbot
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
        """Standard chat (non-streaming)"""
        history = get_history(req.session_id)
        add_user_msg(history, req.message)

        # Use generate_content with conversation history
        resp = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=BASE_CONFIG,
        )
        text = resp.text or ""
        if not text:
            raise HTTPException(500, "Empty model response")

        # append model reply to history
        add_model_msg(history, text)
        return ChatResponse(text=text)

@app.get("/models")
def list_models():
        """List available base models (useful during setup)"""
        return [m.name for m in client.models.list()]

@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
        """Server-Sent Events (SSE) streaming endpoint"""

        history = get_history(req.session_id)
        add_user_msg(history, req.message)

        def event_generator():
            try:
                # Streaming API from google-genai
                # Yields chunks with .text as they arrive
                for chunk in client.models.generate_content_stream(
                        model=MODEL, contents=history, config=BASE_CONFIG
                ):
                    if chunk.text:
                        yield f"data: {chunk.text}\n\n"
            finally:
                # On stream end, finalize last text into history
                # (Optional) You could buffer text to store a single message.
                pass

        return StreamingResponse(event_generator(),
                                 media_type="text/event-stream")

@app.post("/chat/structured", response_model=StructuredChatResponse)
def chat_structured(req: StructuredChatRequest):
        """
        Returns a strictly-typed JSON object (great for UI forms or DB storage).
        Uses response_schema so Gemini returns valid JSON per our Pydantic model.
        """
        history = get_history(req.session_id)
        add_user_msg(history, req.message)

        base_dict = BASE_CONFIG.model_dump()

        resp = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=types.GenerateContentConfig(
                **{k: v for k, v in base_dict.items() if v is not None},  # safe unpack
                response_mime_type="application/json",
                response_schema=AgriAdvice,
            ),
        )
        if not resp.text:
            raise HTTPException(500, "Empty model response")

        # Save a simple textual echo in history too (optional)
        add_model_msg(history, resp.text)
        return StructuredChatResponse(advice=AgriAdvice.model_validate_json(resp.text))
