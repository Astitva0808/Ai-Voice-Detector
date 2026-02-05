import os
import librosa
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ===== PATH SETUP (STABLE) =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# ===== CONSTANTS =====
SR = 16000
N_MFCC = 13

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SR, mono=True)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

    return features

X, y = [], []

# Human = 0
for file in os.listdir(os.path.join(DATA_DIR, "human")):
    path = os.path.join(DATA_DIR, "human", file)
    X.append(extract_features(path))
    y.append(0)

# AI = 1
for file in os.listdir(os.path.join(DATA_DIR, "ai")):
    path = os.path.join(DATA_DIR, "ai", file)
    X.append(extract_features(path))
    y.append(1)

X = np.array(X)
y = np.array(y)

# ===== TRAIN / TEST SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== MODEL TRAINING =====
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===== EVALUATION =====
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ===== SAVE MODEL =====
model_path = os.path.join(MODEL_DIR, "voice_detector_model.joblib")
joblib.dump(model, model_path)

print(f"\n✅ Model saved at: {model_path}")
