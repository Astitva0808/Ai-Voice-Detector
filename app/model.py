import os
import joblib
import librosa
import numpy as np

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "voice_detector_model.joblib")

# ================= LOAD MODEL (ONCE) =================
MODEL = joblib.load(MODEL_PATH)

# ================= CONSTANTS =================
SR = 16000
N_MFCC = 13


# ================= FEATURE EXTRACTION =================
def extract_features(audio: np.ndarray):
    """
    Extract MFCC mean + std (same as training)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SR,
        n_mfcc=N_MFCC
    )

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

    return features.reshape(1, -1)


# ================= SINGLE-CHUNK PREDICTION =================
def predict(audio: np.ndarray) -> float:
    """
    Predict AI probability for a single audio segment
    """
    features = extract_features(audio)
    probability = MODEL.predict_proba(features)[0][1]
    return float(probability)


# ================= CHUNK-BASED INFERENCE (IMPORTANT) =================
def predict_chunked(audio: np.ndarray, chunk_sec: float = 2.0):
    """
    Predict AI probability using sliding-window chunks.
    Handles mixed AI + human audio more robustly.

    Returns:
        avg_confidence (float)
        chunk_probabilities (list)
    """

    chunk_size = int(SR * chunk_sec)

    if len(audio) < chunk_size:
        # Too short → fallback to single prediction
        p = predict(audio)
        return p, [p]

    chunk_probs = []

    for start in range(0, len(audio), chunk_size):
        chunk = audio[start:start + chunk_size]

        # Ignore very small tail chunks
        if len(chunk) < chunk_size * 0.5:
            continue

        try:
            p = predict(chunk)
            chunk_probs.append(p)
        except Exception:
            continue

    if not chunk_probs:
        return 0.5, []

    avg_confidence = float(np.mean(chunk_probs))
    return avg_confidence, chunk_probs
