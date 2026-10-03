"""
Extract mean-pooled YAMNet embeddings for every clip in the AcousticSceneBD
dataset and save them (with labels + session IDs) to embeddings.npz.

Usage:
    python src/extract_features.py --data_dir /path/to/AcousticSceneBD
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import librosa
from tqdm import tqdm

YAMNET_SR = 16000
FILENAME_SESSION_RE = re.compile(r"^(?P<class>.+?)-(?P<session>[^_]+)_chunk_(?P<n>\d+)\.wav$", re.IGNORECASE)

CLASS_FOLDERS = [
    "Bus", "Metro", "Metro_Station", "Park",
    "Restaurant", "Shopping_Mall", "University",
]


def load_yamnet():
    import tensorflow_hub as hub
    print("Loading YAMNet from TF-Hub (downloads once, then cached)...")
    return hub.load("https://tfhub.dev/google/yamnet/1")


def embed_clip(yamnet, path: Path) -> np.ndarray:
    """Load a clip, resample to 16kHz mono, return mean-pooled 1024-d embedding."""
    wav, _ = librosa.load(str(path), sr=YAMNET_SR, mono=True)
    if wav.size == 0:
        raise ValueError(f"Empty audio: {path}")
    # YAMNet expects float32 waveform in [-1, 1]
    wav = wav.astype("float32")
    _, embeddings, _ = yamnet(wav)  # embeddings: [num_frames, 1024]
    return embeddings.numpy().mean(axis=0)


def parse_session(filename: str, class_name: str) -> str:
    m = FILENAME_SESSION_RE.match(filename)
    if m:
        return m.group("session")
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to AcousticSceneBD root folder")
    parser.add_argument("--out", default="embeddings.npz", help="Output .npz file")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"Data directory not found: {data_dir}")

    yamnet = load_yamnet()

    embeddings, labels, sessions, filenames = [], [], [], []
    unknown_session_count = 0
    total_clips = 0

    for class_name in CLASS_FOLDERS:
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"WARNING: folder not found, skipping: {class_dir}")
            continue

        wav_files = sorted(class_dir.glob("*.wav"))
        print(f"\n{class_name}: {len(wav_files)} clips")

        for wav_path in tqdm(wav_files, desc=class_name):
            try:
                emb = embed_clip(yamnet, wav_path)
            except Exception as e:
                print(f"  skipping {wav_path.name}: {e}")
                continue

            session = parse_session(wav_path.name, class_name)
            if session == "UNKNOWN":
                unknown_session_count += 1

            embeddings.append(emb)
            labels.append(class_name)
            sessions.append(f"{class_name}::{session}")
            filenames.append(wav_path.name)
            total_clips += 1

    if total_clips == 0:
        sys.exit("No clips were processed. Check --data_dir and folder names.")

    embeddings = np.stack(embeddings)
    labels = np.array(labels)
    sessions = np.array(sessions)
    filenames = np.array(filenames)

    np.savez_compressed(
        args.out,
        embeddings=embeddings,
        labels=labels,
        sessions=sessions,
        filenames=filenames,
    )

    print(f"\nSaved {total_clips} embeddings to {args.out}")
    if unknown_session_count:
        pct = 100 * unknown_session_count / total_clips
        print(
            f"NOTE: {unknown_session_count} clips ({pct:.1f}%) didn't match the "
            f"expected '<Class>-<Session>_chunk_<N>.wav' filename pattern. "
            f"train.py will fall back to a random stratified split for those."
        )


if __name__ == "__main__":
    main()
