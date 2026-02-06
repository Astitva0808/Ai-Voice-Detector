import io
import librosa
import numpy as np
import requests
from fastapi import HTTPException
from app.config import TARGET_SAMPLE_RATE, MAX_AUDIO_DURATION


def _post_process(audio: np.ndarray):
    """
    Common post-processing for all audio inputs
    """
    if audio is None or len(audio) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty or invalid audio file"
        )

    # Normalize amplitude
    audio = librosa.util.normalize(audio)
    return audio


def load_and_preprocess(file_bytes: bytes):
    """
    Load audio from uploaded file bytes
    Optimized to avoid decoding long audio files
    """
    try:
        audio_buffer = io.BytesIO(file_bytes)
        audio, _ = librosa.load(
            audio_buffer,
            sr=TARGET_SAMPLE_RATE,
            mono=True,
            duration=MAX_AUDIO_DURATION  # 🔥 critical optimization
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid audio file"
        )

    return _post_process(audio)


def load_audio_from_url(audio_url: str):
    """
    Load audio by downloading it from a URL
    Optimized for hackathon endpoint testers
    """

    try:
        response = requests.get(
            audio_url,
            headers={
                "User-Agent": "AI-Voice-Detector",
                "Accept": "*/*"
            },
            timeout=10,
            allow_redirects=True
        )
        response.raise_for_status()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Failed to download audio from URL"
        )

    if not response.content or len(response.content) < 1000:
        raise HTTPException(
            status_code=400,
            detail="Downloaded audio file is empty or corrupted"
        )

    try:
        audio_buffer = io.BytesIO(response.content)
        audio, _ = librosa.load(
            audio_buffer,
            sr=TARGET_SAMPLE_RATE,
            mono=True,
            duration=MAX_AUDIO_DURATION  # 🔥 critical optimization
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid audio file from URL"
        )

    return _post_process(audio)
