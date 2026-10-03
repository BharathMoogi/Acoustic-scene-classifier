from pathlib import Path
import io

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from fastapi import FastAPI, File, HTTPException, UploadFile
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "model.keras"
EMBEDDINGS_PATH = BASE_DIR / "embeddings.npz"
CLASS_NAMES = [
    "Bus",
    "Metro",
    "Metro_Station",
    "Park",
    "Restaurant",
    "Shopping_Mall",
    "University",
]
YAMNET_SAMPLE_RATE = 16000

app = FastAPI(title="Acoustic Scene Classifier")

MODEL = None
YAMNET = None
SCALER = None


def load_model_and_assets():
    global MODEL, YAMNET, SCALER
    if MODEL is None:
        MODEL = tf.keras.models.load_model(str(MODEL_PATH))
    if YAMNET is None:
        YAMNET = hub.load("https://tfhub.dev/google/yamnet/1")
    if SCALER is None:
        data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
        scaler = StandardScaler()
        scaler.fit(data["embeddings"])
        SCALER = scaler


def extract_embedding_from_bytes(file_bytes: bytes):
    wav, _ = librosa.load(io.BytesIO(file_bytes), sr=YAMNET_SAMPLE_RATE, mono=True)
    if wav.size == 0:
        raise ValueError("Audio file is empty.")
    wav = wav.astype("float32")
    _, embeddings, _ = YAMNET(wav)
    emb = embeddings.numpy().mean(axis=0)
    return emb.reshape(1, -1)


def predict_audio_bytes(file_bytes: bytes):
    load_model_and_assets()
    emb = extract_embedding_from_bytes(file_bytes)
    emb_scaled = SCALER.transform(emb)
    scores = MODEL.predict(emb_scaled, verbose=0)[0]
    pred_idx = int(np.argmax(scores))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(scores[pred_idx])

    probs = pd.DataFrame({
        "class": CLASS_NAMES,
        "probability": np.round(scores, 4),
    }).sort_values("probability", ascending=False)
    return {
        "label": pred_label,
        "confidence": confidence,
        "probabilities": probs.to_dict(orient="records"),
    }


@app.get("/")
def root():
    return {
        "message": "Acoustic Scene Classifier API",
        "status": "ok",
        "classes": CLASS_NAMES,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith((".wav", ".mp3", ".flac")):
        raise HTTPException(status_code=400, detail="Please upload a .wav, .mp3, or .flac file.")

    try:
        contents = await file.read()
        result = predict_audio_bytes(contents)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(exc)}")
