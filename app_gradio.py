import os
from pathlib import Path
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import StandardScaler
import gradio as gr

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "outputs" / "model.keras"
EMBEDDINGS_PATH = BASE_DIR / "embeddings.npz"

CLASS_NAMES = [
    "Bus",
    "Metro",
    "Metro Station",
    "Park",
    "Restaurant",
    "Shopping Mall",
    "University",
]

YAMNET_SAMPLE_RATE = 16000

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


def classify_audio(audio_file_path):
    if not audio_file_path:
        return {}

    load_model_and_assets()

    # Load audio and resample to 16kHz mono
    wav, _ = librosa.load(audio_file_path, sr=YAMNET_SAMPLE_RATE, mono=True)
    if wav.size == 0:
        return {"Error": 1.0}

    wav = wav.astype("float32")

    # Extract YAMNet embeddings and pool over time
    _, embeddings, _ = YAMNET(wav)
    emb = embeddings.numpy().mean(axis=0).reshape(1, -1)

    # Scale and predict
    emb_scaled = SCALER.transform(emb)
    scores = MODEL.predict(emb_scaled, verbose=0)[0]

    # Return dictionary of class probabilities for Gradio Label component
    confidence_dict = {
        CLASS_NAMES[i]: float(scores[i]) for i in range(len(CLASS_NAMES))
    }
    return confidence_dict


# Example audio files if present
examples = []
for ex in ["Bus-5_chunk_172.wav", "Metro-3_chunk_114.wav"]:
    ex_path = BASE_DIR / ex
    if ex_path.exists():
        examples.append([str(ex_path)])

demo = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="filepath", label="Upload Environmental Audio Clip"),
    outputs=gr.Label(num_top_classes=7, label="Acoustic Scene Prediction"),
    title="🔊 AcousticSceneBD Environmental Audio Classifier",
    description="Upload a real-world environmental audio clip (or record via microphone) to classify the scene using **YAMNet + MLP Transfer Learning**.",
    examples=examples if examples else None,
    theme="soft"
)

if __name__ == "__main__":
    demo.launch()
