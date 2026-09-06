"""
train.py  —  Complete Corrected Training Script
================================================
Fixes applied:
  1. Aggressive audio augmentation (×8 per sample) to combat tiny dataset.
  2. MobileNetV2 transfer learning (pretrained ImageNet weights).
  3. Two-phase training: frozen base → fine-tune top layers.
  4. GlobalAveragePooling instead of Flatten to reduce overfitting.
  5. Class-weight balancing for imbalanced datasets.
  6. Proper mel_stats saved as (mean, std) per channel for app.py.
  7. Early stopping with generous patience + ReduceLROnPlateau.
  8. Stratified train/val split.
  9. Full logging to training_log.csv.

Usage (PowerShell):
    python train.py --dataset_dir "dataset/archive (1)/clean/clean" --csv_path "dataset/archive (1)/instruments.csv" --output_dir ./models

Usage (single line, any shell):
    python train.py --dataset_dir "dataset/archive (1)/clean/clean" --csv_path "dataset/archive (1)/instruments.csv" --output_dir ./models
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Audio / spectrogram constants ─────────────────────────────────────────────
SAMPLE_RATE = 22050
DURATION    = 3                    # seconds
N_MELS      = 128
N_FFT       = 2048
HOP_LENGTH  = 512
IMG_SIZE    = (128, 128)           # (H, W) fed to MobileNetV2
CHANNELS    = 3                    # mel_dB, Δ, Δ²

# ── Training hyper-parameters ─────────────────────────────────────────────────
BATCH_SIZE      = 16
EPOCHS_FROZEN   = 30               # phase 1 — base frozen
EPOCHS_FINETUNE = 50               # phase 2 — top layers unfrozen
AUGMENT_FACTOR  = 8                # synthetic copies per real sample
VAL_RATIO       = 0.20
MIN_SAMPLES_PER_CLASS = 5          # warn if a class has fewer than this


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  AUDIO UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio(path: str) -> np.ndarray:
    """Load, trim/pad to DURATION seconds, return mono waveform."""
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
    target = SAMPLE_RATE * DURATION
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)), mode="constant")
    else:
        audio = audio[:target]
    return audio


def augment_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Randomly apply one or more of: time-stretch, pitch-shift, noise injection,
    time-shift, and gain variation.  Returns augmented waveform same length.
    """
    aug = audio.copy()

    # Time stretch  (keeps length via centre-crop / pad)
    if np.random.rand() < 0.6:
        rate = np.random.uniform(0.75, 1.25)
        stretched = librosa.effects.time_stretch(aug, rate=rate)
        target = SAMPLE_RATE * DURATION
        if len(stretched) < target:
            stretched = np.pad(stretched, (0, target - len(stretched)))
        aug = stretched[:target]

    # Pitch shift  (±3 semitones)
    if np.random.rand() < 0.6:
        steps = np.random.uniform(-3, 3)
        aug = librosa.effects.pitch_shift(aug, sr=sr, n_steps=steps)

    # Background noise
    if np.random.rand() < 0.5:
        noise_amp = np.random.uniform(0.001, 0.010)
        aug = aug + noise_amp * np.random.randn(len(aug))

    # Time shift  (roll waveform)
    if np.random.rand() < 0.4:
        shift = np.random.randint(0, SAMPLE_RATE // 2)
        aug = np.roll(aug, shift)

    # Gain variation
    if np.random.rand() < 0.4:
        gain = np.random.uniform(0.7, 1.3)
        aug = aug * gain

    # Clip to [-1, 1]
    aug = np.clip(aug, -1.0, 1.0)
    return aug.astype(np.float32)


def audio_to_mel(audio: np.ndarray) -> np.ndarray:
    """Waveform → (128, 128, 3) float32  [mel_dB, Δ, Δ²]"""
    mel    = librosa.feature.melspectrogram(
                 y=audio, sr=SAMPLE_RATE,
                 n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    delta  = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)

    stacked = np.stack([mel_db, delta, delta2], axis=-1)        # (128, T, 3)
    resized = tf.image.resize(stacked, IMG_SIZE).numpy()        # (128, 128, 3)
    return resized.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def discover_files(dataset_dir: str, csv_path: str | None):
    """
    Returns list of (filepath, label) tuples.
    Strategy:
      1. If csv_path exists and has 'path'/'label' columns → use it.
      2. Otherwise walk dataset_dir expecting  <dataset_dir>/<label>/<*.wav>
    """
    pairs = []

    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Normalise column names
        df.columns = [c.strip().lower() for c in df.columns]

        path_col = next((c for c in df.columns if "path" in c or "file" in c or "fname" in c or "name" in c), None)

        label_col = next((c for c in df.columns if "label" in c or "class" in c
                          or "instrument" in c or "category" in c), None)
        if path_col and label_col:
            for _, row in df.iterrows():
                fp = str(row[path_col])
                if not os.path.isabs(fp):
                    fp = os.path.join(dataset_dir, fp)
                if os.path.exists(fp):
                    pairs.append((fp, str(row[label_col])))
            print(f"[CSV] Loaded {len(pairs)} samples from {csv_path}")

    if not pairs:
        print("[DIR] Scanning directory structure …")
        for label in sorted(os.listdir(dataset_dir)):
            label_dir = os.path.join(dataset_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith((".wav", ".mp3", ".ogg", ".flac", ".m4a")):
                    pairs.append((os.path.join(label_dir, fname), label))
        print(f"[DIR] Found {len(pairs)} samples across "
              f"{len(set(p[1] for p in pairs))} classes")

    return pairs


def build_dataset(pairs, augment: bool = True):
    """
    Load audio → mel for every (path, label) pair.
    If augment=True, generate AUGMENT_FACTOR extra copies per sample.
    Returns X (N, 128, 128, 3), y_str (N,).
    """
    X, y = [], []
    total = len(pairs)

    for i, (path, label) in enumerate(pairs):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing {i+1}/{total} …", end="\r")
        try:
            audio = load_audio(path)
            mel   = audio_to_mel(audio)
            X.append(mel)
            y.append(label)

            if augment:
                for _ in range(AUGMENT_FACTOR):
                    aug_audio = augment_audio(audio)
                    aug_mel   = audio_to_mel(aug_audio)
                    X.append(aug_mel)
                    y.append(label)
        except Exception as e:
            print(f"\n  ⚠️  Skipping {path}: {e}")

    print(f"\n  ✅ Built dataset: {len(X)} samples total")
    return np.array(X, dtype=np.float32), np.array(y)


def compute_and_save_stats(X_train: np.ndarray, output_dir: str):
    """Compute per-channel mean/std over training set, save to mel_stats.pkl."""
    # X_train shape: (N, 128, 128, 3)
    mean = X_train.mean(axis=(0, 1, 2))   # (3,)
    std  = X_train.std (axis=(0, 1, 2))   # (3,)
    std  = np.where(std < 1e-8, 1e-8, std)
    joblib.dump((mean, std), os.path.join(output_dir, "mel_stats.pkl"))
    print(f"  Mel stats — mean: {mean.round(2)}, std: {std.round(2)}")
    return mean, std


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Broadcast-normalize (N,128,128,3) with per-channel stats."""
    return (X - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int) -> tf.keras.Model:
    """
    MobileNetV2 backbone (ImageNet weights) + custom classification head.
    Input: (128, 128, 3)  — same shape as standard MobileNetV2 minimum input.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, CHANNELS),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False   # Phase 1: freeze all base layers

    inputs = tf.keras.Input(shape=(*IMG_SIZE, CHANNELS))

    # MobileNetV2 expects [0,255] OR [-1,1]; we pass normalised mel — add a
    # learned rescaling layer so the network can adapt.
    x = tf.keras.layers.Rescaling(scale=1.0)(inputs)   # identity, trainable=False
    x = base(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation="relu",
                               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


def unfreeze_top_layers(base: tf.keras.Model, num_layers: int = 30):
    """Unfreeze the last `num_layers` layers of the base for fine-tuning."""
    base.trainable = True
    for layer in base.layers[:-num_layers]:
        layer.trainable = False
    print(f"  Unfrozen top {num_layers} layers of MobileNetV2 for fine-tuning")


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def get_callbacks(output_dir: str, phase: str):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15 if phase == "frozen" else 20,
            min_delta=0.005,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(output_dir, f"training_log_{phase}.csv"),
            append=False,
        ),
    ]


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Discover files ───────────────────────────────────────────────────
    print("\n[1/6] Discovering audio files …")
    pairs = discover_files(args.dataset_dir, args.csv_path)
    if not pairs:
        raise RuntimeError("No audio files found. Check --dataset_dir and --csv_path.")

    # Class distribution report
    from collections import Counter
    dist = Counter(p[1] for p in pairs)
    print("\n  Class distribution (raw files):")
    for cls, cnt in sorted(dist.items()):
        flag = "⚠️ " if cnt < MIN_SAMPLES_PER_CLASS else "  "
        print(f"    {flag}{cls}: {cnt}")

    # ── 2. Encode labels ────────────────────────────────────────────────────
    print("\n[2/6] Encoding labels …")
    le = LabelEncoder()
    all_labels = [p[1] for p in pairs]
    le.fit(all_labels)
    num_classes = len(le.classes_)
    print(f"  {num_classes} classes: {list(le.classes_)}")
    joblib.dump(le, os.path.join(args.output_dir, "label_encoder.pkl"))

    # ── 3. Build full dataset with augmentation ─────────────────────────────
    print(f"\n[3/6] Loading audio + augmenting ×{AUGMENT_FACTOR} …")
    X, y_str = build_dataset(pairs, augment=True)
    y = le.transform(y_str)

    # ── 4. Train / val split ────────────────────────────────────────────────
    print("\n[4/6] Splitting train / val …")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_RATIO, random_state=SEED)
    train_idx, val_idx = next(sss.split(X, y))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    print(f"  Train: {len(X_train)}  Val: {len(X_val)}")

    # ── 5. Normalize ────────────────────────────────────────────────────────
    print("\n[5/6] Normalizing …")
    mean, std = compute_and_save_stats(X_train, args.output_dir)
    X_train = normalize(X_train, mean, std)
    X_val   = normalize(X_val,   mean, std)

    # Class weights (handle imbalance)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    cw_dict = {i: w for i, w in enumerate(class_weights)}

    # ── 6. Build + train model ──────────────────────────────────────────────
    print("\n[6/6] Building model …")
    model, base = build_model(num_classes)
    model.summary(line_length=80)

    # ── Phase 1: Train head only ────────────────────────────────────────────
    print("\n── Phase 1: Training classification head (base frozen) ──")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_FROZEN,
        batch_size=BATCH_SIZE,
        class_weight=cw_dict,
        callbacks=get_callbacks(args.output_dir, "frozen"),
        verbose=1,
    )

    # ── Phase 2: Fine-tune top MobileNetV2 layers ───────────────────────────
    print("\n── Phase 2: Fine-tuning top MobileNetV2 layers ──")
    unfreeze_top_layers(base, num_layers=30)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),   # lower LR for fine-tuning
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS_FINETUNE,
        batch_size=BATCH_SIZE,
        class_weight=cw_dict,
        callbacks=get_callbacks(args.output_dir, "finetune"),
        verbose=1,
    )

    # ── Final evaluation ────────────────────────────────────────────────────
    # AFTER (fixed):
    best_model = tf.keras.models.load_model(
    os.path.join(args.output_dir, "best_model.keras"), compile=False)
    best_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)
    print(f"\n{'='*50}")
    print(f"[SUCCESS] Training complete!")
    print(f"  Best val accuracy : {val_acc:.4f}  ({val_acc*100:.1f}%)")
    print(f"  Best val loss     : {val_loss:.4f}")
    print(f"  Artefacts saved to: {args.output_dir}")
    print(f"{'='*50}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train instrument classifier")
    parser.add_argument("--dataset_dir", required=True,
                        help="Root folder containing class subfolders of audio files")
    parser.add_argument("--csv_path", default=None,
                        help="Optional CSV with 'path' and 'label' columns")
    parser.add_argument("--output_dir", default="./models",
                        help="Where to save model + artefacts (default: ./models)")
    args = parser.parse_args()
    train(args)
