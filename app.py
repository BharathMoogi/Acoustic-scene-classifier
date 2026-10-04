from pathlib import Path
import io

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from sklearn.preprocessing import StandardScaler

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


HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Acoustic Scene Classifier</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.85);
            --border: rgba(255, 255, 255, 0.1);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
        }

        body {
            background: radial-gradient(circle at 50% 0%, #1e1b4b 0%, #0f172a 100%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 24px 16px;
        }

        .container {
            width: 100%;
            max-width: 680px;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 36px 32px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5), 0 0 40px rgba(79, 70, 229, 0.15);
            backdrop-filter: blur(16px);
        }

        .header {
            text-align: center;
            margin-bottom: 28px;
        }

        .badge {
            display: inline-block;
            background: rgba(79, 70, 229, 0.2);
            color: #818cf8;
            border: 1px solid rgba(129, 140, 248, 0.3);
            font-size: 12px;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 9999px;
            margin-bottom: 12px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        h1 {
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 8px;
            background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        p.subtitle {
            color: var(--text-muted);
            font-size: 14px;
            line-height: 1.5;
        }

        .drop-zone {
            border: 2px dashed rgba(129, 140, 248, 0.35);
            border-radius: 14px;
            padding: 30px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            background: rgba(15, 23, 42, 0.5);
            margin-bottom: 20px;
        }

        .drop-zone:hover, .drop-zone.drag-over {
            border-color: #818cf8;
            background: rgba(79, 70, 229, 0.1);
            transform: translateY(-1px);
        }

        .upload-icon {
            font-size: 38px;
            margin-bottom: 10px;
            display: block;
        }

        .file-info {
            font-size: 14px;
            color: #cbd5e1;
            margin-top: 8px;
            font-weight: 500;
        }

        input[type="file"] {
            display: none;
        }

        button.btn-predict {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
        }

        button.btn-predict:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
        }

        button.btn-predict:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        #audio-preview {
            width: 100%;
            margin-top: 15px;
            border-radius: 8px;
            display: none;
        }

        .result-card {
            margin-top: 28px;
            padding-top: 24px;
            border-top: 1px solid var(--border);
            display: none;
        }

        .top-prediction {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 20px;
        }

        .top-label-title {
            font-size: 13px;
            color: #a7f3d0;
            text-transform: uppercase;
            font-weight: 600;
        }

        .top-label-name {
            font-size: 22px;
            font-weight: 700;
            color: #ffffff;
        }

        .top-conf {
            font-size: 24px;
            font-weight: 700;
            color: #34d399;
        }

        .prob-item {
            margin-bottom: 12px;
        }

        .prob-header {
            display: flex;
            justify-content: space-between;
            font-size: 13px;
            margin-bottom: 6px;
            color: #cbd5e1;
        }

        .progress-bar-bg {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 6px;
            height: 9px;
            overflow: hidden;
        }

        .progress-bar-fill {
            background: linear-gradient(90deg, #6366f1, #818cf8);
            height: 100%;
            border-radius: 6px;
            transition: width 0.6s ease;
        }

        .loader {
            display: none;
            margin: 20px auto;
            text-align: center;
            color: var(--text-muted);
            font-size: 14px;
        }

        .spinner {
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-top: 3px solid #818cf8;
            border-radius: 50%;
            width: 28px;
            height: 28px;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 10px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <span class="badge">AcousticSceneBD • YAMNet</span>
        <h1>Acoustic Scene Classifier</h1>
        <p class="subtitle">Classify real-world environmental sounds (Bus, Metro, Park, Restaurant, Shopping Mall, University) with Deep Audio Embeddings.</p>
    </div>

    <div class="drop-zone" id="drop-zone">
        <span class="upload-icon">🎧</span>
        <div style="font-weight:600; font-size:15px; margin-bottom: 4px;">Click to select or drag & drop audio</div>
        <div style="font-size:12px; color:var(--text-muted);">Supported formats: .WAV, .MP3, .FLAC</div>
        <div class="file-info" id="file-info"></div>
    </div>
    <input type="file" id="file-input" accept=".wav,.mp3,.flac">

    <audio id="audio-preview" controls></audio>

    <button class="btn-predict" id="btn-predict" disabled style="margin-top: 15px;">Analyze Acoustic Scene</button>

    <div class="loader" id="loader">
        <div class="spinner"></div>
        <span>Extracting YAMNet features & analyzing audio...</span>
    </div>

    <div class="result-card" id="result-card">
        <div class="top-prediction">
            <div>
                <div class="top-label-title">Detected Scene</div>
                <div class="top-label-name" id="pred-class">-</div>
            </div>
            <div class="top-conf" id="pred-conf">0%</div>
        </div>

        <div style="font-size: 13px; font-weight: 600; color: var(--text-muted); margin-bottom: 12px; text-transform: uppercase;">Confidence Breakdown</div>
        <div id="prob-list"></div>
    </div>
</div>

<script>
    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');
    const fileInfo = document.getElementById('file-info');
    const audioPreview = document.getElementById('audio-preview');
    const btnPredict = document.getElementById('btn-predict');
    const loader = document.getElementById('loader');
    const resultCard = document.getElementById('result-card');
    const predClass = document.getElementById('pred-class');
    const predConf = document.getElementById('pred-conf');
    const probList = document.getElementById('prob-list');

    let selectedFile = null;

    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        selectedFile = file;
        fileInfo.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        btnPredict.disabled = false;
        resultCard.style.display = 'none';

        const url = URL.createObjectURL(file);
        audioPreview.src = url;
        audioPreview.style.display = 'block';
    }

    btnPredict.addEventListener('click', async () => {
        if (!selectedFile) return;

        btnPredict.disabled = true;
        loader.style.display = 'block';
        resultCard.style.display = 'none';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || 'Prediction failed');
            }

            const data = await response.json();

            predClass.textContent = data.label;
            predConf.textContent = (data.confidence * 100).toFixed(1) + '%';

            probList.innerHTML = '';
            data.probabilities.forEach(item => {
                const percent = (item.probability * 100).toFixed(1);
                const itemDiv = document.createElement('div');
                itemDiv.className = 'prob-item';
                itemDiv.innerHTML = `
                    <div class="prob-header">
                        <span>${item.class}</span>
                        <span>${percent}%</span>
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width: ${percent}%;"></div>
                    </div>
                `;
                probList.appendChild(itemDiv);
            });

            resultCard.style.display = 'block';
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            loader.style.display = 'none';
            btnPredict.disabled = false;
        }
    });
</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=HTML_UI, status_code=200)


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
