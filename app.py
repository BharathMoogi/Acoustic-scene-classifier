from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
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


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(str(MODEL_PATH))


@st.cache_resource
def load_yamnet():
    return hub.load("https://tfhub.dev/google/yamnet/1")


@st.cache_resource
def load_scaler():
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    scaler = StandardScaler()
    scaler.fit(data["embeddings"])
    return scaler


def extract_embedding(yamnet, audio_path: Path) -> np.ndarray:
    wav, _ = librosa.load(str(audio_path), sr=YAMNET_SAMPLE_RATE, mono=True)
    if wav.size == 0:
        raise ValueError("Audio file is empty.")

    wav = wav.astype("float32")
    _, embeddings, _ = yamnet(wav)
    emb = embeddings.numpy().mean(axis=0)
    return emb.reshape(1, -1)


def predict_audio(audio_path: Path):
    model = load_model()
    yamnet = load_yamnet()
    scaler = load_scaler()

    emb = extract_embedding(yamnet, audio_path)
    emb_scaled = scaler.transform(emb)
    scores = model.predict(emb_scaled, verbose=0)[0]
    pred_idx = int(np.argmax(scores))
    pred_label = CLASS_NAMES[pred_idx]
    confidence = float(scores[pred_idx])

    prob_df = pd.DataFrame(
        {
            "Class": CLASS_NAMES,
            "Probability": np.round(scores, 4),
        }
    ).sort_values("Probability", ascending=False)

    return pred_label, confidence, prob_df


def find_sample_audio() -> Path | None:
    candidates = sorted(BASE_DIR.parent.glob("**/*.wav"))
    # prefer a clear class sample if available, otherwise the first wav in the dataset
    for path in candidates:
        if "Park" in path.parts or "Bus" in path.parts:
            return path
    return candidates[0] if candidates else None


st.set_page_config(page_title="Acoustic Scene Classifier", page_icon="🎵", layout="wide")
st.title("Acoustic Scene Classifier")
st.caption("Upload a .wav clip or try a sample to classify the environment.")

with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])
    sample_path = find_sample_audio()
    use_sample = st.checkbox("Use a sample dataset file", value=bool(sample_path))

    if use_sample and sample_path:
        st.info(f"Sample selected: {sample_path.name}")

if uploaded_file is not None:
    path = BASE_DIR / uploaded_file.name
    path.write_bytes(uploaded_file.getvalue())
    selected_path = path
elif use_sample and sample_path:
    selected_path = sample_path
else:
    selected_path = None

if selected_path is not None:
    try:
        label, confidence, probs = predict_audio(selected_path)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Prediction")
            st.markdown(f"## {label}")
            st.metric("Confidence", f"{confidence * 100:.2f}%")

        with col2:
            st.subheader("Top probabilities")
            st.dataframe(probs.head(7), use_container_width=True)

        st.subheader("Probability distribution")
        st.bar_chart(probs.set_index("Class")["Probability"])

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
else:
    st.info("Upload an audio file or enable the sample option to start.")
