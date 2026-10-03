# AcousticSceneBD Classifier (YAMNet transfer learning)

This reproduces the best-performing approach from the AcousticSceneBD paper
(YAMNet embeddings + MLP head, 82.99% test accuracy) so you can train it on
your own copy of the dataset.

## 1. Unzip your dataset

Unzip it so it looks like this (matches the paper's structure):

```
AcousticSceneBD/
  Bus/
    Bus-1_chunk_1.wav
    Bus-1_chunk_2.wav
    ...
  Metro/
  Metro Station/
  Park/
  Restaurant/
  Shopping Mall/
  University/
  metadata.csv        <- optional, not required
```

If your folder names or structure differ slightly, just edit `DATA_DIR` in
`src/extract_features.py`.

## 2. Install dependencies (one-time)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

This pulls in TensorFlow, TensorFlow Hub (for YAMNet), librosa, scikit-learn,
pandas, matplotlib.

## 3. Extract YAMNet embeddings (run once)

```bash
python src/extract_features.py --data_dir /path/to/AcousticSceneBD
```

This walks every class folder, resamples each clip to 16kHz mono, runs it
through YAMNet (downloaded automatically from TF-Hub the first time — needs
internet once), mean-pools the per-frame embeddings into a single
1024-dim vector per clip, and saves everything to `embeddings.npz`.

This step takes the longest (minutes, depending on your CPU) but only needs
to run once. Re-run only if you add new audio.

## 4. Train the classifier

```bash
python src/train.py
```

This:
- Splits data by **recording session** (not by individual clip) into
  70/10/20 train/val/test, same as the paper — this stops clips from the
  same recording session leaking between splits.
- Trains a 3-layer MLP on top of the frozen YAMNet embeddings
  (Adam, lr=1e-3, batch 32, up to 50 epochs, early stopping patience 10).
- Prints accuracy, macro/weighted F1, and a full per-class report.
- Saves a confusion matrix plot to `outputs/confusion_matrix.png` and the
  trained model to `outputs/model.keras`.

## Notes on the session split

The paper's filenames follow `<Class>-<Session>_chunk_<N>.wav`
(e.g. `Bus-1_chunk_3.wav` → session `1`). `extract_features.py` parses the
session ID from the filename automatically. If your filenames don't match
this pattern, it falls back to a random stratified split — you'll see a
warning printed if that happens.

## Why YAMNet over the other three baselines

- **Random Forest on MFCCs** (58.81% in the paper) — the weakest; hand-crafted
  features can't capture the temporal/textural nuance needed to tell
  Restaurant from Shopping Mall apart.
- **Wav2Vec2** (47.11%) — pretrained on speech, not environmental sound, so
  it's a domain mismatch.
- **PANNs CNN14** (78.15%) — solid, but slightly behind YAMNet here.
- **YAMNet** (82.99%, best) — pretrained on general audio (AudioSet), which
  covers ambient/environmental sounds much better than a speech-only model,
  and its embeddings are lighter (1024-d, faster) than PANNs (2048-d).

If you want to also run PANNs CNN14 for comparison (like the paper's Table 8),
say so and I'll add a second extraction script — the training script is
written to work with either embedding type.
