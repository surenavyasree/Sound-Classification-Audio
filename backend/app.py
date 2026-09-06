"""
app.py  —  Complete Corrected Flask Inference Server
=====================================================
Fixes applied vs original app.py:
  1. Loads best_model.keras (saved by train.py).
  2. Normalization uses per-channel mel_stats (mean, std) shape (3,)
     broadcast correctly over (N, 128, 128, 3) tensors.
  3. Audio preprocessing EXACTLY matches train.py pipeline.
  4. /classes endpoint for frontend label discovery.
  5. /health endpoint for diagnostics.
  6. Supports single file (/classify) and multi-file mixing (/predict).
  7. Graceful errors with clear messages.
  8. Optional RF baseline support (backward-compatible).

Run:
    python backend/app.py
Server starts on http://0.0.0.0:5000
"""

import os
import io
import traceback

import numpy as np
import librosa
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
LE_PATH    = os.path.join(MODEL_DIR, "label_encoder.pkl")
STATS_PATH = os.path.join(MODEL_DIR, "mel_stats.pkl")

# Optional RF baseline
RF_PATH  = os.path.join(MODEL_DIR, "rf_baseline.pkl")
PCA_PATH = os.path.join(MODEL_DIR, "pca_256.pkl")

# ── Audio constants  (MUST match train.py exactly) ────────────────────────────
SAMPLE_RATE = 22050
DURATION    = 3
N_MELS      = 128
N_FFT       = 2048
HOP_LENGTH  = 512
IMG_SIZE    = (128, 128)

ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "flac", "m4a"}

# ── Globals loaded once at startup ────────────────────────────────────────────
deep_model    = None
label_encoder = None
mel_mean      = None    # shape (3,)  — per-channel mean
mel_std       = None    # shape (3,)  — per-channel std
rf_model      = None
pca_model     = None


# ═══════════════════════════════════════════════════════════════════════════════
# STARTUP — Load all artefacts
# ═══════════════════════════════════════════════════════════════════════════════

def load_artefacts():
    global deep_model, label_encoder, mel_mean, mel_std, rf_model, pca_model

    print("\n📦 Loading artefacts …")

    # ── Label encoder ────────────────────────────────────────────────────────
    if os.path.exists(LE_PATH):
        label_encoder = joblib.load(LE_PATH)
        print(f"   ✅ Label encoder  — {len(label_encoder.classes_)} classes: "
              f"{list(label_encoder.classes_)}")
    else:
        print(f"   ❌ label_encoder.pkl NOT FOUND at {LE_PATH}")

    # ── Mel normalisation stats ───────────────────────────────────────────────
    if os.path.exists(STATS_PATH):
        loaded = joblib.load(STATS_PATH)
        mel_mean = np.array(loaded[0], dtype=np.float32).reshape(3)
        mel_std  = np.array(loaded[1], dtype=np.float32).reshape(3)
        mel_std  = np.where(mel_std < 1e-8, 1e-8, mel_std)
        print(f"   ✅ Mel stats — mean: {mel_mean.round(2)}, std: {mel_std.round(2)}")
    else:
        print("   ⚠️  mel_stats.pkl not found — using per-sample normalisation (less accurate)")

    # ── Deep model ───────────────────────────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        deep_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print(f"   ✅ CNN model loaded  (input shape: {deep_model.input_shape})")
    else:
        print(f"   ❌ Model NOT FOUND at {MODEL_PATH}")
        print("      → Run train.py first to generate best_model.keras")

    # ── RF baseline (optional) ────────────────────────────────────────────────
    if os.path.exists(RF_PATH) and os.path.exists(PCA_PATH):
        rf_model  = joblib.load(RF_PATH)
        pca_model = joblib.load(PCA_PATH)
        print("   ✅ RF baseline + PCA loaded")
    else:
        print("   ℹ️  RF / PCA not found — baseline predictions skipped")

    print("🚀 Server ready.\n")


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO HELPERS  (identical pipeline to train.py)
# ═══════════════════════════════════════════════════════════════════════════════

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """Decode bytes → mono waveform, trimmed/padded to DURATION seconds."""
    with io.BytesIO(audio_bytes) as buf:
        audio, _ = librosa.load(buf, sr=SAMPLE_RATE, duration=DURATION, mono=True)
    target = SAMPLE_RATE * DURATION
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)), mode="constant")
    else:
        audio = audio[:target]
    return audio.astype(np.float32)


def audio_to_mel(audio: np.ndarray) -> np.ndarray:
    """Waveform → (128, 128, 3) float32  [mel_dB, Δ, Δ²]"""
    mel    = librosa.feature.melspectrogram(
                 y=audio, sr=SAMPLE_RATE,
                 n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    delta  = librosa.feature.delta(mel_db)
    delta2 = librosa.feature.delta(mel_db, order=2)

    stacked = np.stack([mel_db, delta, delta2], axis=-1)     # (128, T, 3)
    resized = tf.image.resize(stacked, IMG_SIZE).numpy()     # (128, 128, 3)
    return resized.astype(np.float32)


def normalize_mel(mel: np.ndarray) -> np.ndarray:
    """
    Normalize using saved per-channel stats from training.
    mel shape: (128, 128, 3)
    """
    if mel_mean is not None and mel_std is not None:
        # Broadcast (3,) stats over (128, 128, 3)
        return (mel - mel_mean.reshape(1, 1, 3)) / mel_std.reshape(1, 1, 3)

    # Fallback: per-sample z-score per channel
    out = np.empty_like(mel)
    for c in range(mel.shape[-1]):
        ch = mel[..., c]
        out[..., c] = (ch - ch.mean()) / (ch.std() + 1e-8)
    return out


def mix_audio_clips(clips: list) -> np.ndarray:
    """Average-mix multiple waveforms and normalise peak to 0.95."""
    mixed = np.stack(clips, axis=0).mean(axis=0)
    peak  = np.abs(mixed).max()
    if peak > 1e-6:
        mixed = mixed / peak * 0.95
    return mixed


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def run_inference(mel_norm: np.ndarray) -> dict:
    """
    Run available models on a single normalised mel spectrogram.
    mel_norm shape: (128, 128, 3)
    Returns dict with 'deep_model' and optionally 'rf_baseline' results.
    """
    result = {}

    # ── CNN / MobileNetV2 ────────────────────────────────────────────────────
    if deep_model is not None and label_encoder is not None:
        batch = mel_norm[np.newaxis, ...]                  # (1, 128, 128, 3)
        probs = deep_model.predict(batch, verbose=0)[0]    # (num_classes,)

        top5_idx  = np.argsort(probs)[::-1][:5]
        top5_lbls = label_encoder.inverse_transform(top5_idx)
        top5_conf = probs[top5_idx].tolist()

        result["deep_model"] = {
            "top_label":      top5_lbls[0],
            "top_confidence": float(top5_conf[0]),
            "top5": [
                {"label": lbl, "confidence": round(float(c), 4)}
                for lbl, c in zip(top5_lbls, top5_conf)
            ],
        }

    # ── RF baseline (optional) ────────────────────────────────────────────────
    if rf_model is not None and pca_model is not None and label_encoder is not None:
        flat     = mel_norm.reshape(1, -1)
        pca_feat = pca_model.transform(flat)
        rf_proba = rf_model.predict_proba(pca_feat)[0]

        top_idx  = np.argsort(rf_proba)[::-1][:5]
        top_lbls = label_encoder.inverse_transform(top_idx)
        top_conf = rf_proba[top_idx].tolist()

        result["rf_baseline"] = {
            "top_label":      top_lbls[0],
            "top_confidence": float(top_conf[0]),
            "top5": [
                {"label": lbl, "confidence": round(float(c), 4)}
                for lbl, c in zip(top_lbls, top_conf)
            ],
        }

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """Quick health / status check."""
    return jsonify({
        "status":        "ok",
        "deep_model":    deep_model is not None,
        "rf_baseline":   rf_model is not None,
        "label_encoder": label_encoder is not None,
        "classes":       label_encoder.classes_.tolist() if label_encoder else [],
    })


@app.route("/classes", methods=["GET"])
def get_classes():
    """Return list of recognisable instrument labels."""
    if label_encoder is None:
        return jsonify({"error": "label_encoder not loaded"}), 500
    return jsonify({"classes": label_encoder.classes_.tolist()})


@app.route("/predict", methods=["POST"])
@app.route("/classify", methods=["POST"])
def predict():
    """
    Accepts one or more audio files.
    Single file  → classify directly.
    Multiple files → mix then classify.
    """
    try:
        # Accept 'files[]' (multi) or 'file' (single)
        files = request.files.getlist("files[]")
        if not files:
            single = request.files.get("file")
            if single:
                files = [single]

        if not files or all(f.filename == "" for f in files):
            return jsonify({"error": "No audio files uploaded."}), 400

        clips = []
        for f in files:
            if not allowed_file(f.filename):
                return jsonify({
                    "error": f"Invalid file type '{f.filename}'. "
                             f"Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                }), 400
            clips.append(load_audio_bytes(f.read()))

        audio_final = mix_audio_clips(clips) if len(clips) > 1 else clips[0]

        mel      = audio_to_mel(audio_final)
        mel_norm = normalize_mel(mel)

        predictions = run_inference(mel_norm)
        if not predictions:
            return jsonify({"error": "No models are loaded. Check server logs."}), 500

        return jsonify({
            "success":     True,
            "mixed":       len(clips) > 1,
            "num_files":   len(clips),
            "predictions": predictions,
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/generate", methods=["POST"])
def generate():
    """Placeholder for AI audio generation (future feature)."""
    try:
        files = request.files.getlist("files[]")
        if not files or len(files) < 2:
            return jsonify({"error": "Please upload at least two reference audio files."}), 400
        return jsonify({
            "success": True,
            "message": "AI audio generation placeholder — integrate a GAN/diffusion model here.",
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    load_artefacts()
    app.run(host="0.0.0.0", port=5000, debug=False)
