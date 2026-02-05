import os

API_KEY = os.getenv("API_KEY", "hackathon_test_key_123")

MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac"}
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_DURATION = 60  # seconds
