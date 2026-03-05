import os
os.environ["NUMBA_DISABLE_JIT"] = "1"  # precaution for numba issues

import io
import time
import sqlite3
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import numpy as np
import soundfile as sf
import warnings
warnings.filterwarnings("ignore", message="resume_download")
warnings.filterwarnings("ignore", message="gradient_checkpointing")

# --- Import our utils (audio preprocessing, ffmpeg conversion etc.) ---
from app.utils.audio_utils import preprocess_audio, ffmpeg_convert_to_wav, load_wav_with_soundfile

# --- NEW IMPORTS for Cloudinary + HTTP requests ---
import cloudinary
import cloudinary.uploader
import requests

# Optional heavy imports (lazy loaded)
USE_W2V = False
W2V_MODEL_NAME = None
W2V_PROCESSOR = None
W2V_MODEL = None
CLASSIFIER = None
CLASSIFIER_N_FEATURES = None

# Load env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL_ID = os.getenv("HF_MODEL_ID")
USE_W2V = os.getenv("USE_W2V", "false").lower() in ("1", "true", "yes")
W2V_MODEL_NAME = os.getenv("W2V_MODEL_NAME", "facebook/wav2vec2-base")
CLASSIFIER_PATH = os.getenv("CLASSIFIER_PATH", "models/gender_classifier.joblib")

# === Cloudinary & Auth backend config (from your auth logs) ===
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")

VOICE_SYNC_SECRET = os.getenv("VOICE_SYNC_SECRET")

AUTH_BACKEND_URL = os.getenv("AUTH_BACKEND_URL")

# Configure cloudinary (will use provided values)
try:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True
    )
    logger_cloud = logging.getLogger("cloudinary")
    logger_cloud.debug("Cloudinary configured.")
except Exception:
    # If cloudinary import/config fails, we'll continue but log later where used.
    pass

# Storage paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(BASE_DIR, "enrollments.db")

os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("talkify-backend")

# --- DB utilities ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS enrollments (
            user_id TEXT PRIMARY KEY,
            created_at TEXT,
            template_embedding BLOB,
            template_pitch REAL
        )
        """
    )
    conn.commit()
    conn.close()

def np_to_blob(x: Optional[np.ndarray]) -> Optional[bytes]:
    if x is None:
        return None
    bio = io.BytesIO()
    np.save(bio, x, allow_pickle=False)
    bio.seek(0)
    return bio.read()

def blob_to_np(b: Optional[bytes]) -> Optional[np.ndarray]:
    if b is None:
        return None
    bio = io.BytesIO(b)
    bio.seek(0)
    return np.load(bio, allow_pickle=False)

# --- Audio helpers ---
def save_audio_file(upload: UploadFile, dest_path: str) -> None:
    data = upload.file.read()
    with open(dest_path, "wb") as f:
        f.write(data)
    upload.file.close()

def load_audio_mono(path: str, sr: int = 16000):
    # Use preprocessing pipeline: FFmpeg conversion + read + noise reduce + normalize
    y, sr = preprocess_audio(path, target_sr=sr)
    return y, sr

# --- pitch extractor (autocorrelation, robust - avoids librosa/numba issues) ---
def frame_signal(y, frame_length, hop_length):
    n_frames = 1 + (len(y) - frame_length) // hop_length
    frames = np.stack([y[i*hop_length:i*hop_length+frame_length] for i in range(n_frames)])
    return frames

def pitch_from_autocorr(frame, sr, fmin=50, fmax=600):
    # center clip (simple), windowing
    frame = frame * np.hamming(len(frame))
    # compute autocorrelation via FFT (faster)
    corr = np.fft.ifft(np.abs(np.fft.fft(frame))**2).real
    corr = corr[:len(corr)//2]
    # find lag range
    min_lag = int(sr / fmax) if fmax > 0 else 1
    max_lag = int(sr / fmin) if fmin > 0 else len(corr)-1
    if max_lag <= min_lag:
        return None
    segment = corr[min_lag:max_lag+1]
    peak = np.argmax(segment) + min_lag
    if corr[peak] <= 0:
        return None
    # convert lag to frequency
    f0 = float(sr) / float(peak) if peak > 0 else None
    return f0

def extract_pitch_median(y: np.ndarray, sr: int = 16000) -> Optional[float]:
    try:
        # simple framing
        frame_length = int(0.03 * sr)  # 30ms
        hop_length = int(0.01 * sr)    # 10ms
        if len(y) < frame_length:
            # pad
            y = np.pad(y, (0, frame_length - len(y)))
        frames = frame_signal(y, frame_length, hop_length)
        pitches = []
        for frame in frames:
            f0 = pitch_from_autocorr(frame, sr, fmin=50, fmax=600)
            if f0 is not None and 50 <= f0 <= 600:
                pitches.append(f0)
        if len(pitches) == 0:
            return None
        return float(np.median(pitches))
    except Exception as e:
        logger.exception("pitch extraction failed: %s", e)
        return None
# --- optional wav2vec embedding extractor (lazy) ---
def lazy_load_w2v():
    global W2V_MODEL, W2V_PROCESSOR, USE_W2V
    if not USE_W2V:
        return
    if W2V_MODEL is None or W2V_PROCESSOR is None:
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            import torch
        except Exception as e:
            logger.error("transformers/torch not installed, disabling W2V. %s", e)
            USE_W2V = False
            return
        logger.info("Loading Wav2Vec2 model: %s", W2V_MODEL_NAME)
        W2V_PROCESSOR = Wav2Vec2Processor.from_pretrained(W2V_MODEL_NAME)
        W2V_MODEL = Wav2Vec2Model.from_pretrained(W2V_MODEL_NAME)
        W2V_MODEL.eval()
        logger.info("Wav2Vec2 loaded.")

def extract_w2v_embedding(y: np.ndarray, sr: int = 16000):
    if not USE_W2V:
        return None
    lazy_load_w2v()
    if W2V_MODEL is None:
        return None
    import torch
    inputs = W2V_PROCESSOR(y, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = W2V_MODEL(inputs.input_values).last_hidden_state
    emb = outputs.mean(dim=1).squeeze(0).cpu().numpy()
    return emb

# --- classifier loader ---
def lazy_load_classifier():
    global CLASSIFIER, CLASSIFIER_N_FEATURES
    if CLASSIFIER is not None:
        return
    if not os.path.exists(CLASSIFIER_PATH):
        logger.info("No classifier file found at %s -- will use heuristic", CLASSIFIER_PATH)
        return
    try:
        import joblib
        CLASSIFIER = joblib.load(CLASSIFIER_PATH)
        # try to detect expected feature count
        CLASSIFIER_N_FEATURES = getattr(CLASSIFIER, "n_features_in_", None)
        logger.info("Loaded classifier from %s (n_features=%s)", CLASSIFIER_PATH, CLASSIFIER_N_FEATURES)
    except Exception as e:
        logger.exception("Failed to load classifier: %s", e)
        CLASSIFIER = None
        CLASSIFIER_N_FEATURES = None

# --- gender prediction ---
def predict_gender_from_pitch(pitch_hz: Optional[float]):
    if pitch_hz is None:
        return {"gender": "unknown", "confidence": 0.0}
    x = pitch_hz
    prob_female = 1 / (1 + np.exp(-(x - 160) / 10))
    gender = "female" if prob_female >= 0.5 else "male"
    return {"gender": gender, "confidence": float(prob_female)}

def predict_gender(audio_path: str):
    y, sr = load_audio_mono(audio_path)
    pitch = extract_pitch_median(y, sr)
    lazy_load_classifier()

    # If classifier is present and we are using W2V and classifier expects W2V-sized input,
    # then try to use it; otherwise fallback to pitch heuristic.
    if CLASSIFIER is not None and USE_W2V:
        emb = extract_w2v_embedding(y, sr)
        if emb is not None:
            try:
                # safety: check shape compatibility
                if CLASSIFIER_N_FEATURES is None or CLASSIFIER_N_FEATURES == emb.size:
                    proba = CLASSIFIER.predict_proba(emb.reshape(1, -1))[0]
                    classes = getattr(CLASSIFIER, "classes_", None)
                    if classes is not None and len(classes) == 2:
                        idx_female = int(np.where(classes == "female")[0]) if "female" in classes else 1
                        conf_female = float(proba[idx_female])
                        gender = "female" if conf_female >= 0.5 else "male"
                        return {"gender": gender, "confidence": conf_female, "method": "classifier"}
                else:
                    logger.warning("Classifier feature mismatch: classifier expects %s but emb has %s. Skipping classifier.",
                                   CLASSIFIER_N_FEATURES, emb.size)
            except Exception as e:
                logger.exception("classifier predict failed: %s", e)

    res = predict_gender_from_pitch(pitch)
    res["method"] = "pitch"
    res["pitch_hz"] = pitch
    return res

# --- enrollment and verification ---
def save_enrollment(user_id: str, template_embedding: Optional[np.ndarray], template_pitch: Optional[float]):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    created_at = datetime.utcnow().isoformat()
    emb_blob = np_to_blob(template_embedding) if template_embedding is not None else None
    c.execute(
        """
        INSERT OR REPLACE INTO enrollments(user_id, created_at, template_embedding, template_pitch)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, created_at, emb_blob, template_pitch),
    )
    conn.commit()
    conn.close()

def get_enrollment(user_id: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT user_id, created_at, template_embedding, template_pitch FROM enrollments WHERE user_id = ?", (user_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    emb = blob_to_np(row[2]) if row[2] is not None else None
    return {"user_id": row[0], "created_at": row[1], "template_embedding": emb, "template_pitch": row[3]}

def cosine_similarity(a: np.ndarray, b: np.ndarray):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    num = (a * b).sum(axis=1)
    den = (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    den = np.where(den == 0, 1e-8, den)
    return float(num / den)

# FastAPI app
app = FastAPI(title="Talkify Voice Verification Backend")

@app.on_event("startup")
def startup_event():
    init_db()
    lazy_load_classifier()
    if USE_W2V:
        lazy_load_w2v()
    logger.info("Startup complete. USE_W2V=%s, CLASSIFIER_LOADED=%s", USE_W2V, CLASSIFIER is not None)

class EnrollResponse(BaseModel):
    user_id: str
    message: str

@app.post("/enroll", response_model=EnrollResponse)
async def enroll(user_id: str = Form(...), file: UploadFile = File(...)):
    timestamp = int(time.time())
    user_dir = os.path.join(STORAGE_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    filename = f"enroll_{timestamp}.wav"
    dest = os.path.join(user_dir, filename)

    save_audio_file(file, dest)
    try:
        y, sr = load_audio_mono(dest)
        pitch = extract_pitch_median(y, sr)
        emb = extract_w2v_embedding(y, sr) if USE_W2V else None
        save_enrollment(user_id, emb, pitch)
        return EnrollResponse(user_id=user_id, message="Enrolled successfully")
    except Exception as e:
        logger.exception("Enroll failed: %s", e)
        raise HTTPException(status_code=500, detail="Enroll failed")
@app.post("/verify-voice")
async def verify_voice(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    user_token: Optional[str] = Form(None)  # <-- optional token from Android client
):
    tmp_name = f"tmp_verify_{int(time.time())}.wav"
    tmp_path = os.path.join(STORAGE_DIR, tmp_name)
    save_audio_file(file, tmp_path)

    enrolled = get_enrollment(user_id)
    if enrolled is None:
        raise HTTPException(status_code=404, detail="No enrollment found for user")

    try:
        incoming_info = predict_gender(tmp_path)
    except Exception as e:
        logger.exception("Predict failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed")

    sim_score = None
    match = False
    if enrolled["template_embedding"] is not None and USE_W2V:
        try:
            y, sr = load_audio_mono(tmp_path)
            emb_in = extract_w2v_embedding(y, sr)
            if emb_in is not None and enrolled["template_embedding"] is not None:
                # verify dims match
                if enrolled["template_embedding"].shape == emb_in.shape:
                    sim_score = cosine_similarity(enrolled["template_embedding"], emb_in)
                    match = sim_score >= 0.6
                else:
                    logger.warning("Enrollment embedding shape %s != incoming emb shape %s", enrolled["template_embedding"].shape, emb_in.shape)
        except Exception as e:
            logger.exception("Similarity check failed: %s", e)

    gender = incoming_info.get("gender", "unknown")
    confidence = incoming_info.get("confidence", 0.0)
    result = "failed"
    reason = []
    if gender == "female":
        if enrolled["template_embedding"] is not None and USE_W2V:
            if match:
                result = "success"
            else:
                reason.append("speaker_mismatch")
        else:
            result = "success"
    else:
        reason.append("gender_not_female")

    # --- NEW: If success & female, upload audio to Cloudinary and notify Auth backend ---
        # --- NEW: If success & female, upload audio to Cloudinary and notify Auth backend ---
    voice_url = None
    auth_update_status = None

    if result == "success" and gender == "female":
        try:
            if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET:
                upload_result = cloudinary.uploader.upload(
                    tmp_path,
                    resource_type="auto",
                    folder="talkify/voice_samples",
                    public_id=f"voice_{user_id}_{int(time.time())}"
                )

                # ✅ Cloudinary URL
                voice_url = upload_result.get("secure_url") or upload_result.get("url")
                logger.info("Cloudinary upload success: %s", voice_url)

                # 🔔 Notify Auth Backend (INTERNAL API)
                try:
                    requests.post(
                        f"{AUTH_BACKEND_URL}/api/user/internal/update-voice-sample",
                        json={
                            "user_id": user_id,
                            "voice_sample": voice_url
                        },
                        headers={
                            "x-internal-secret": VOICE_SYNC_SECRET,
                            "Content-Type": "application/json"
                        },
                        timeout=5
                    )
                    logger.info("Auth backend voice_sample updated successfully")
                except Exception as e:
                    logger.exception("Failed to notify auth backend: %s", e)

            else:
                logger.warning("Cloudinary credentials not configured. Skipping upload.")

        except Exception as e:
            logger.exception("Cloudinary upload failed: %s", e)

        # Notify Auth backend with voice_sample URL (if available)
        # if voice_url and AUTH_BACKEND_URL:
        #     try:
        #         payload = {
        #             "voice_sample": voice_url,
        #             "gender": "female"  # keep consistent
        #         }
        #         headers = {"Content-Type": "application/json"}
        #         if user_token:
        #             headers["Authorization"] = f"Bearer {user_token}"
        #         # We use PUT to /user/update-profile. Your backend originally accepts multipart PUT
        #         # for full profile updates. Here we send a minimal JSON update with voice_sample URL.
        #         # If your backend requires multipart, swap to multipart/form-data (requests-toolbelt can help).
        #        # resp = requests.put(
        #           #  f"{AUTH_BACKEND_URL.rstrip('/')}/user/update-profile",
        #            # json=payload,
        #            # headers=headers,
        #            # timeout=10
        #        # )
        #         auth_update_status = {
        #             "status_code": resp.status_code,
        #             "body": resp.text
        #         }
        #         logger.info("Auth backend update response: %s", auth_update_status)
        #     except Exception as e:
        #         logger.exception("Auth backend update failed: %s", e)
        #         auth_update_status = {"error": str(e)} 
        else:
            if not AUTH_BACKEND_URL:
                logger.warning("AUTH_BACKEND_URL not configured. Skipping auth backend update.")
            elif not voice_url:
                logger.warning("Voice URL not available; skipping auth backend update.")

    response = {
        "user_id": user_id,
        "result": result,
        "gender": gender,
        "confidence": confidence,
        "similarity": sim_score,
        "method": incoming_info.get("method", None),
        "reason": reason,
        "voice_url": voice_url,
        "auth_update": auth_update_status
    }
    return JSONResponse(content=response)

@app.get("/")
def root():
    return {"message": "Talkify Voice Verification Backend is running!"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "Talkify Backend"}
