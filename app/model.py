import os
import joblib
import librosa
import numpy as np

# ===== STABLE PATH RESOLUTION =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "voice_detector_model.joblib")

# ===== LOAD MODEL ONCE (IMPORTANT) =====
MODEL = joblib.load(MODEL_PATH)

SR = 16000
N_MFCC = 13

def extract_features(audio: np.ndarray):
    """
    Extract the same features used during training:
    MFCC mean + MFCC std
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

    return features.reshape(1, -1)

def predict(audio: np.ndarray) -> float:
    features = extract_features(audio)
    probability = MODEL.predict_proba(features)[0][1]
    return float(probability)
