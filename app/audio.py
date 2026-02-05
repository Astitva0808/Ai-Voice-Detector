import io
import librosa
import numpy as np
import requests
from fastapi import HTTPException
from app.config import TARGET_SAMPLE_RATE, MAX_AUDIO_DURATION


def _post_process(audio: np.ndarray, sr: int):
    """
    Common post-processing for all audio inputs
    """
    if audio is None or len(audio) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file"
        )

    # Safety trim (extra guard)
    max_samples = TARGET_SAMPLE_RATE * MAX_AUDIO_DURATION
    if len(audio) > max_samples:
        audio = audio[:max_samples]

    # Normalize audio
    audio = librosa.util.normalize(audio)
    return audio


def load_and_preprocess(file_bytes: bytes):
    """
    Load audio from uploaded file bytes
    Optimized to avoid decoding long audio files
    """
    try:
        audio_buffer = io.BytesIO(file_bytes)
        audio, sr = librosa.load(
            audio_buffer,
            sr=TARGET_SAMPLE_RATE,
            mono=True,
            duration=MAX_AUDIO_DURATION   # 🔥 CRITICAL OPTIMIZATION
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid audio file"
        )

    return _post_process(audio, sr)


def load_audio_from_url(audio_url: str):
    """
    Load audio by downloading it from a URL
    Hardened + optimized for hackathon testers and CDNs
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AI-Voice-Detector/1.0)",
        "Accept": "*/*"
    }

    try:
        response = requests.get(
            audio_url,
            headers=headers,
            timeout=15,
            allow_redirects=True,
            verify=True
        )
        response.raise_for_status()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Failed to download audio from URL"
        )

    # Sanity check
    if len(response.content) < 1000:
        raise HTTPException(
            status_code=400,
            detail="Downloaded audio file is too small or invalid"
        )

    try:
        audio_buffer = io.BytesIO(response.content)
        audio, sr = librosa.load(
            audio_buffer,
            sr=TARGET_SAMPLE_RATE,
            mono=True,
            duration=MAX_AUDIO_DURATION   # 🔥 CRITICAL OPTIMIZATION
        )
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid audio file from URL"
        )

    return _post_process(audio, sr)
