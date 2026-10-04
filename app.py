from contextlib import asynccontextmanager
from pathlib import Path
import io
import math
import pickle
import threading
import traceback

import numpy as np
import pandas as pd
import scipy.signal as sps
import soundfile as sf
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

BASE_DIR = Path(__file__).resolve().parent
TFLITE_PATH = BASE_DIR / "outputs" / "yamnet_quant.tflite"
YAMNET_DIR = BASE_DIR / "outputs" / "yamnet_model"
WEIGHTS_PATH = BASE_DIR / "outputs" / "model_weights.pkl"
SCALER_PATH = BASE_DIR / "outputs" / "scaler.pkl"

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

INTERPRETER = None
YAMNET = None
WEIGHTS = None
SCALER = None
MODEL_LOCK = threading.Lock()


IS_INITIALIZED = False


def load_model_and_assets():
    global INTERPRETER, YAMNET, WEIGHTS, SCALER, IS_INITIALIZED
    if IS_INITIALIZED:
        return
    try:
        if INTERPRETER is None and TFLITE_PATH.exists():
            interp = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
            interp.allocate_tensors()
            INTERPRETER = interp
            print("Loaded ultra-lightweight YAMNet TFLite engine.")
        elif YAMNET is None:
            if YAMNET_DIR.exists():
                YAMNET = tf.saved_model.load(str(YAMNET_DIR))
            else:
                import tensorflow_hub as hub
                YAMNET = hub.load("https://tfhub.dev/google/yamnet/1")

        if WEIGHTS is None and WEIGHTS_PATH.exists():
            with open(WEIGHTS_PATH, "rb") as f:
                WEIGHTS = pickle.load(f)

        if SCALER is None and SCALER_PATH.exists():
            with open(SCALER_PATH, "rb") as f:
                SCALER = pickle.load(f)

        IS_INITIALIZED = True

        # Full warm-up pass (audio decoding + TFLite + NumPy forward pass)
        buf = io.BytesIO()
        sf.write(buf, np.zeros(YAMNET_SAMPLE_RATE * 2, dtype=np.float32), YAMNET_SAMPLE_RATE, format="WAV")
        _ = predict_audio_bytes(buf.getvalue())

        print("ML runtime fully pre-warmed and ready for instant inference.")
    except Exception as e:
        print(f"Error loading assets: {e}")
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_assets()
    yield

app = FastAPI(title="Acoustic Scene Classifier", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def safe_load_audio(file_bytes: bytes, target_sr=16000, max_seconds=10) -> np.ndarray:
    try:
        data, sr = sf.read(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not decode audio file: {exc}")

    if data.ndim > 1:
        data = data.mean(axis=1)

    data = data.astype(np.float32)

    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        up = target_sr // gcd
        down = sr // gcd
        data = sps.resample_poly(data, up, down).astype(np.float32)

    max_samples = target_sr * max_seconds
    if len(data) > max_samples:
        data = data[:max_samples]

    return data


def extract_embedding_from_bytes(file_bytes: bytes):
    wav = safe_load_audio(file_bytes, target_sr=YAMNET_SAMPLE_RATE)
    if wav.size == 0:
        raise ValueError("Audio file is empty.")

    if INTERPRETER is not None:
        with MODEL_LOCK:
            input_details = INTERPRETER.get_input_details()
            output_details = INTERPRETER.get_output_details()
            INTERPRETER.resize_tensor_input(input_details[0]["index"], [len(wav)], strict=False)
            INTERPRETER.allocate_tensors()
            INTERPRETER.set_tensor(input_details[0]["index"], wav)
            INTERPRETER.invoke()
            embeddings = INTERPRETER.get_tensor(output_details[0]["index"])
            emb = embeddings.mean(axis=0)
            return emb.reshape(1, -1)
    elif YAMNET is not None:
        _, embeddings, _ = YAMNET(wav)
        emb = embeddings.numpy().mean(axis=0)
        return emb.reshape(1, -1)
    else:
        raise RuntimeError("Embedding engine not initialized.")


def predict_audio_bytes(file_bytes: bytes):
    load_model_and_assets()
    if (INTERPRETER is None and YAMNET is None) or WEIGHTS is None or SCALER is None:
        raise RuntimeError("Server is starting up. Please try again.")

    emb = extract_embedding_from_bytes(file_bytes)
    emb_scaled = SCALER.transform(emb)

    # Ultra-fast pure NumPy forward pass (< 1 ms)
    h1 = np.maximum(0, np.dot(emb_scaled, WEIGHTS[0]) + WEIGHTS[1])
    h2 = np.maximum(0, np.dot(h1, WEIGHTS[2]) + WEIGHTS[3])
    logits = np.dot(h2, WEIGHTS[4]) + WEIGHTS[5]
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    scores = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True))[0]

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

        .error-banner {
            display: none;
            margin-top: 15px;
            padding: 12px 16px;
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 10px;
            color: #fca5a5;
            font-size: 14px;
            text-align: center;
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
        <span class="badge">AcousticSceneBD • YAMNet Lite</span>
        <h1>Acoustic Scene Classifier</h1>
        <p class="subtitle">Classify real-world environmental sounds (Bus, Metro, Park, Restaurant, Shopping Mall, University) with Deep Audio Embeddings.</p>
    </div>

    <div class="drop-zone" id="drop-zone">
        <span class="upload-icon">🎧</span>
        <div style="font-weight:600; font-size:15px; margin-bottom: 4px;">Click to select or drag & drop audio</div>
        <div style="font-size:12px; color:var(--text-muted);">Supported formats: .WAV, .MP3, .FLAC, .OGG</div>
        <div class="file-info" id="file-info"></div>
    </div>
    <input type="file" id="file-input" accept=".wav,.mp3,.flac,.ogg">

    <audio id="audio-preview" controls></audio>

    <button class="btn-predict" id="btn-predict" disabled style="margin-top: 15px;">Analyze Acoustic Scene</button>

    <div class="error-banner" id="error-banner"></div>

    <div class="loader" id="loader">
        <div class="spinner"></div>
        <span>Analyzing acoustic features...</span>
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
    const errorBanner = document.getElementById('error-banner');

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
        errorBanner.style.display = 'none';

        const url = URL.createObjectURL(file);
        audioPreview.src = url;
        audioPreview.style.display = 'block';
    }

    btnPredict.addEventListener('click', async () => {
        if (!selectedFile) return;

        btnPredict.disabled = true;
        loader.style.display = 'block';
        resultCard.style.display = 'none';
        errorBanner.style.display = 'none';

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const text = await response.text();
            let data;
            try {
                data = JSON.parse(text);
            } catch (e) {
                throw new Error('Server returned unexpected format. Please try again.');
            }

            if (!response.ok) {
                throw new Error(data.detail || 'Prediction failed');
            }

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
            errorBanner.textContent = error.message;
            errorBanner.style.display = 'block';
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
def predict_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        contents = file.file.read()
        result = predict_audio_bytes(contents)
        return result
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")
