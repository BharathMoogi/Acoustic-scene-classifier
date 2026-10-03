"""
Train a 3-layer MLP classifier on top of frozen YAMNet embeddings, with a
session-level train/val/test split to avoid leakage between splits.

Usage:
    python src/train.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow import keras

EMBEDDINGS_PATH = "embeddings.npz"
OUT_DIR = Path("outputs")
RANDOM_STATE = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.10, 0.20


def session_level_split(sessions: np.ndarray, labels: np.ndarray):
    """
    Assign whole sessions (not individual clips) to train/val/test, per class,
    so acoustically-overlapping clips from the same session never span splits.

    Done manually (rather than via sklearn's StratifiedShuffleSplit) because
    some classes may only have a handful of recording sessions -- too few for
    sklearn's stratification minimums when nesting two splits. This assigns
    sessions class-by-class instead, which works no matter how few sessions
    a class has (falling back to "at least 1 in train" for very small
    classes).
    """
    rng = np.random.RandomState(RANDOM_STATE)

    # group sessions by class
    session_to_class = {}
    for s in np.unique(sessions):
        session_to_class[s] = labels[sessions == s][0]

    sessions_by_class = {}
    for s, cls in session_to_class.items():
        sessions_by_class.setdefault(cls, []).append(s)

    train_sessions, val_sessions, test_sessions = [], [], []

    for cls, sess_list in sessions_by_class.items():
        sess_list = list(sess_list)
        rng.shuffle(sess_list)
        n = len(sess_list)

        if n == 1:
            # only one session for this class -- has to go to train,
            # this class just won't be evaluated in val/test
            train_sessions.extend(sess_list)
            continue

        n_test = max(1, round(n * TEST_FRAC))
        n_test = min(n_test, n - 1)  # leave at least 1 for train

        remaining_after_test = n - n_test
        if remaining_after_test > 1:
            n_val = max(1, round(n * VAL_FRAC))
            n_val = min(n_val, remaining_after_test - 1)
        else:
            n_val = 0

        test_sessions.extend(sess_list[:n_test])
        val_sessions.extend(sess_list[n_test:n_test + n_val])
        train_sessions.extend(sess_list[n_test + n_val:])

    train_mask = np.isin(sessions, train_sessions)
    val_mask = np.isin(sessions, val_sessions)
    test_mask = np.isin(sessions, test_sessions)

    return train_mask, val_mask, test_mask


def build_model(input_dim: int, num_classes: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    OUT_DIR.mkdir(exist_ok=True)

    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    X = data["embeddings"]
    labels_raw = data["labels"]
    sessions = data["sessions"]

    print(f"Loaded {X.shape[0]} clips, embedding dim = {X.shape[1]}")

    if (labels_raw != "UNKNOWN").sum() < len(labels_raw):
        pass  # informational only, handled in session split via random session ids

    le = LabelEncoder()
    y = le.fit_transform(labels_raw)
    class_names = le.classes_
    print("Classes:", list(class_names))

    train_mask, val_mask, test_mask = session_level_split(sessions, labels_raw)
    print(f"Train: {train_mask.sum()}  Val: {val_mask.sum()}  Test: {test_mask.sum()}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    X_val = scaler.transform(X[val_mask])
    X_test = scaler.transform(X[test_mask])

    y_train, y_val, y_test = y[train_mask], y[val_mask], y[test_mask]

    model = build_model(input_dim=X.shape[1], num_classes=len(class_names))
    model.summary()

    callbacks = [
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )

    # --- Evaluation ---
    y_pred = model.predict(X_test).argmax(axis=1)

    print("\n=== Classification report (test set) ===")
    print(classification_report(y_test, y_pred, target_names=class_names, digits=3))

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    test_acc = (y_pred == y_test).mean()
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    # --- Confusion matrix ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=class_names, ax=ax, xticks_rotation=45, colorbar=False
    )
    plt.title("YAMNet + MLP — Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=150)
    print(f"Saved confusion matrix to {OUT_DIR / 'confusion_matrix.png'}")

    # --- Training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_curves.png", dpi=150)
    print(f"Saved training curves to {OUT_DIR / 'training_curves.png'}")

    # --- Save model ---
    model.save(OUT_DIR / "model.keras")
    print(f"Saved trained model to {OUT_DIR / 'model.keras'}")


if __name__ == "__main__":
    main()
