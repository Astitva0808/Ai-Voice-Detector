from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import time

from app.auth import verify_token
from app.audio import load_and_preprocess, load_audio_from_url
from app.model import predict_chunked   # ✅ IMPORTANT CHANGE
from app.config import ALLOWED_EXTENSIONS

# ================== APP INIT ==================
app = FastAPI(title="AI Voice Detection API")

# ================== CORS (REQUIRED FOR FRONTEND) ==================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Hackathon-safe
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== MAIN ENDPOINT ==================
@app.post("/v1/detect-voice")
async def detect_voice(
    request: Request,
    file: Optional[UploadFile] = File(None),
    token: str = Depends(verify_token)
):
    start_time = time.time()

    audio = None
    request_id = None
    content_type = request.headers.get("content-type", "")

    # ===== CASE 1: FILE UPLOAD =====
    if "multipart/form-data" in content_type:
        if file is None:
            raise HTTPException(status_code=400, detail="No file uploaded")

        extension = file.filename.split(".")[-1].lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=415, detail="Unsupported audio format")

        file_bytes = await file.read()
        audio = load_and_preprocess(file_bytes)

    # ===== CASE 2: AUDIO URL (TESTER) =====
    elif "application/json" in content_type:
        body = await request.json()
        audio_url = body.get("audio_url")
        request_id = body.get("request_id")

        if not audio_url:
            raise HTTPException(status_code=400, detail="audio_url is required")

        audio = load_audio_from_url(audio_url)

    else:
        raise HTTPException(status_code=415, detail="Unsupported Content-Type")

    # ===== CHUNK-BASED MODEL INFERENCE (FIX) =====
    confidence, chunk_probs = predict_chunked(audio)
    prediction = "ai_generated" if confidence >= 0.5 else "human"

    # ===== EXPLANATION LOGIC =====
    if confidence >= 0.75:
        explanation = (
            "The audio shows strong indicators of synthetic speech, including "
            "highly consistent spectral patterns and reduced natural variation."
        )
    elif confidence >= 0.5:
        explanation = (
            "The audio contains a mix of characteristics, with some segments "
            "resembling AI-generated speech and others appearing more natural."
        )
    elif confidence >= 0.25:
        explanation = (
            "The audio largely resembles natural human speech, though minor "
            "irregularities were detected."
        )
    else:
        explanation = (
            "The audio exhibits natural variations in pitch, timing, and tone "
            "that are consistent with human speech."
        )

    # ===== RESPONSE =====
    return {
        "success": True,
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "explanation": explanation,
        "model_version": "v1.2",
        "request_id": request_id,
        "processing_time_ms": int((time.time() - start_time) * 1000)
    }

# ================== HEALTH CHECK ==================
@app.get("/health")
def health():
    return {"status": "ok"}
