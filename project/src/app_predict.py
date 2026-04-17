import json
import subprocess
import tempfile
from pathlib import Path
 
import numpy as np
import librosa
import joblib
from scipy.sparse import hstack, csr_matrix
import cv2
import mediapipe as mp
 
import whisper
 
from src.llm_explainer import extract_frames, generate_explanation, generate_visual_explanation, generate_image_explanation
 
MODEL_DIR = Path(__file__).parent.parent / "models" / "best_text_audio_mfcc"
IMAGE_MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model_image_combined.pth"
 
_WHISPER_MODEL = None
 
 
# ─────────────────────────────────────────
# AUTO-DOWNLOAD IMAGE MODEL IF MISSING
# ─────────────────────────────────────────
def _ensure_image_model_downloaded():
    pass  # Model already in repo
 
 
# ─────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────
def load_bundle():
    vec    = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    scaler = joblib.load(MODEL_DIR / "audio_scaler.joblib")
    clf    = joblib.load(MODEL_DIR / "logreg_model.joblib")
    meta   = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    return vec, scaler, clf, meta
 
def run_ffmpeg_extract_audio(video_path: str, wav_out: str):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", wav_out]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
 
def run_whisper_transcribe(wav_path: str) -> str:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model("base")
    try:
        res = _WHISPER_MODEL.transcribe(wav_path, fp16=False, language="en", task="transcribe")
        return (res.get("text") or "").strip()
    except Exception:
        return ""
 
def mfcc_stats(wav_path: str, sr: int, n_mfcc: int) -> np.ndarray:
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    if y is None or y.size == 0:
        return np.zeros(n_mfcc * 2, dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)], axis=0)
    return feat.astype(np.float32)
 
 
# ─────────────────────────────────────────
# AUDIO FORENSICS
# ─────────────────────────────────────────
def audio_forensics_score(wav_path: str) -> float:
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        if y is None or y.size == 0:
            return 0.5
 
        suspicious = 0
        total      = 0
 
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        mean_flat = float(np.mean(flatness))
        total += 1
        if mean_flat > 0.15:
            suspicious += 1
 
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_std = float(np.std(zcr))
        total += 1
        if zcr_std < 0.02:
            suspicious += 1
 
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        rolloff_std = float(np.std(rolloff))
        total += 1
        if rolloff_std < 500:
            suspicious += 1
 
        intervals = librosa.effects.split(y, top_db=30)
        if len(intervals) > 0:
            speech_samples = sum(e - s for s, e in intervals)
            silence_ratio  = 1.0 - (speech_samples / max(len(y), 1))
            total += 1
            if silence_ratio > 0.6 or silence_ratio < 0.05:
                suspicious += 1
 
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_var = float(np.mean(np.var(chroma, axis=1)))
        total += 1
        if chroma_var < 0.01:
            suspicious += 1
 
        return suspicious / total if total > 0 else 0.5
 
    except Exception:
        return 0.5
 
 
# ─────────────────────────────────────────
# FACE GEOMETRY
# ─────────────────────────────────────────
def face_geometry_score(video_path: str) -> float:
    try:
        mp_face_mesh = mp.solutions.face_mesh
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return 0.5
 
        sample_indices = np.linspace(0, total_frames - 1, 6, dtype=int)
        asymmetry_scores = []
 
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if not ret:
                    continue
 
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if not results.multi_face_landmarks:
                    continue
 
                lm = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]
                pts = np.array([[l.x * w, l.y * h] for l in lm])
 
                nose     = pts[1]
                left_eye = pts[263]
                right_eye= pts[33]
 
                dist_left  = np.linalg.norm(left_eye  - nose)
                dist_right = np.linalg.norm(right_eye - nose)
 
                if dist_left + dist_right > 0:
                    asymmetry = abs(dist_left - dist_right) / ((dist_left + dist_right) / 2)
                    asymmetry_scores.append(asymmetry)
 
        cap.release()
 
        if not asymmetry_scores:
            return 0.5
 
        mean_asym = float(np.mean(asymmetry_scores))
        fake_score = min(mean_asym / 0.30, 1.0)
        return round(fake_score, 4)
 
    except Exception:
        return 0.5
 
 
# ─────────────────────────────────────────
# LIP SYNC CHECK
# ─────────────────────────────────────────
def lip_sync_score(video_path: str, wav_path: str) -> float:
    try:
        mp_face_mesh = mp.solutions.face_mesh
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return 0.5
 
        mouth_openness = []
        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if results.multi_face_landmarks:
                    lm  = results.multi_face_landmarks[0].landmark
                    h, w = frame.shape[:2]
                    upper = np.array([lm[13].x * w, lm[13].y * h])
                    lower = np.array([lm[14].x * w, lm[14].y * h])
                    mouth_openness.append(float(np.linalg.norm(upper - lower)))
                else:
                    mouth_openness.append(0.0)
        cap.release()
 
        if not mouth_openness:
            return 0.5
 
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
        hop   = int(sr / fps)
        rms   = librosa.feature.rms(y=y, hop_length=max(hop, 1))[0]
 
        n = min(len(mouth_openness), len(rms))
        if n < 5:
            return 0.5
 
        mouth_arr = np.array(mouth_openness[:n])
        audio_arr = np.array(rms[:n])
 
        def norm(x):
            rng = x.max() - x.min()
            return (x - x.min()) / rng if rng > 0 else x * 0
 
        mouth_n = norm(mouth_arr)
        audio_n = norm(audio_arr)
 
        correlation = float(np.corrcoef(mouth_n, audio_n)[0, 1])
        if np.isnan(correlation):
            return 0.5
 
        fake_score = (1.0 - max(correlation, 0.0)) / 2.0
        return round(fake_score, 4)
 
    except Exception:
        return 0.5
 
 
# ─────────────────────────────────────────
# COMBINE SCORES
# ─────────────────────────────────────────
def combine_scores(
    model_fake: float,
    lip_fake:   float,
    geo_fake:   float,
    audio_fake: float
) -> float:
    combined = (
        0.40 * model_fake +
        0.20 * lip_fake   +
        0.20 * geo_fake   +
        0.20 * audio_fake
    )
    return round(float(combined), 4)
 
 
# ─────────────────────────────────────────
# IMAGE MODEL (MobileNetV3 Small)
# ─────────────────────────────────────────
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
 
_IMAGE_MODEL   = None
_IMAGE_TFM     = None
_IMAGE_CLASSES = None
_IMAGE_DEVICE  = None
 
def _get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
 
def _load_image_model():
    _ensure_image_model_downloaded()
 
    global _IMAGE_MODEL, _IMAGE_TFM, _IMAGE_CLASSES, _IMAGE_DEVICE
    if _IMAGE_MODEL is not None:
        return
 
    _IMAGE_DEVICE  = _get_device()
    _IMAGE_CLASSES = ["REAL", "FAKE"]
 
    # Build MobileNetV3 Small architecture (matches training)
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
 
    ckpt = torch.load(IMAGE_MODEL_PATH, map_location=_IMAGE_DEVICE)
    state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.to(_IMAGE_DEVICE)
    model.eval()
    _IMAGE_MODEL = model
 
    _IMAGE_TFM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
 
 
# ─────────────────────────────────────────
# FACE GEOMETRY FOR IMAGE
# ─────────────────────────────────────────
def face_geometry_score_image(pil_img: Image.Image) -> float:
    try:
        mp_face_mesh = mp.solutions.face_mesh
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w  = frame.shape[:2]
 
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return 0.5
 
            lm        = results.multi_face_landmarks[0].landmark
            pts       = np.array([[l.x * w, l.y * h] for l in lm])
            nose      = pts[1]
            left_eye  = pts[263]
            right_eye = pts[33]
 
            dist_left  = np.linalg.norm(left_eye  - nose)
            dist_right = np.linalg.norm(right_eye - nose)
 
            if dist_left + dist_right == 0:
                return 0.5
 
            asymmetry  = abs(dist_left - dist_right) / ((dist_left + dist_right) / 2)
            fake_score = min(asymmetry / 0.30, 1.0)
            return round(float(fake_score), 4)
 
    except Exception:
        return 0.5
 
 
# =========================
# VIDEO DEEPFAKE
# =========================
def predict_video(video_file_path: str) -> dict:
    vec, scaler, clf, meta = load_bundle()
    sr     = int(meta["sr"])
    n_mfcc = int(meta["n_mfcc"])
    inv    = {int(k): v for k, v in meta["inverse_label_map"].items()}
 
    with tempfile.TemporaryDirectory() as td:
        wav_path = str(Path(td) / "audio.wav")
        run_ffmpeg_extract_audio(video_file_path, wav_path)
 
        transcript  = run_whisper_transcribe(wav_path)
        X_text      = vec.transform([transcript])
        x_audio     = mfcc_stats(wav_path, sr=sr, n_mfcc=n_mfcc).reshape(1, -1)
        x_audio_s   = scaler.transform(x_audio)
        X           = hstack([X_text, csr_matrix(x_audio_s)])
        proba       = clf.predict_proba(X)[0]
        model_fake  = float(proba[1])
 
        lip_fake   = lip_sync_score(video_file_path, wav_path)
        geo_fake   = face_geometry_score(video_file_path)
        audio_fake = audio_forensics_score(wav_path)
 
        final_fake = combine_scores(model_fake, lip_fake, geo_fake, audio_fake)
        final_real = round(1.0 - final_fake, 4)
        pred_label = "FAKE" if final_fake >= 0.5 else "REAL"
        confidence = final_fake if final_fake >= 0.5 else final_real
 
        text_explanation   = generate_explanation(transcript, final_real * 100, final_fake * 100)
        frames             = extract_frames(video_file_path, num_frames=3)
        visual_explanation = generate_visual_explanation(frames, final_real * 100, final_fake * 100, transcript)
 
        return {
            "modality":           "video",
            "prediction":         pred_label,
            "confidence":         round(confidence, 4),
            "prob_real":          final_real,
            "prob_fake":          final_fake,
            "transcript":         transcript,
            "explanation":        text_explanation,
            "visual_explanation": visual_explanation,
            "score_lip_sync":     round(1.0 - lip_fake,   4),
            "score_face_geo":     round(1.0 - geo_fake,   4),
            "score_audio_forensics": round(1.0 - audio_fake, 4),
        }
 
 
# =========================
# AUDIO DEEPFAKE
# =========================
def predict_audio(audio_file_path: str) -> dict:
    vec, scaler, clf, meta = load_bundle()
    sr     = int(meta["sr"])
    n_mfcc = int(meta["n_mfcc"])
    inv    = {int(k): v for k, v in meta["inverse_label_map"].items()}
 
    with tempfile.TemporaryDirectory() as td:
        wav_path = str(Path(td) / "audio.wav")
 
        if audio_file_path.lower().endswith(".wav"):
            Path(wav_path).write_bytes(Path(audio_file_path).read_bytes())
        else:
            cmd = ["ffmpeg", "-y", "-i", audio_file_path, "-ac", "1", "-ar", "16000", wav_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
 
        transcript  = run_whisper_transcribe(wav_path)
        X_text      = vec.transform([transcript])
        x_audio     = mfcc_stats(wav_path, sr=sr, n_mfcc=n_mfcc).reshape(1, -1)
        x_audio_s   = scaler.transform(x_audio)
        X           = hstack([X_text, csr_matrix(x_audio_s)])
        proba       = clf.predict_proba(X)[0]
        model_fake  = float(proba[1])
 
        audio_fake = audio_forensics_score(wav_path)
 
        final_fake = round(0.60 * model_fake + 0.40 * audio_fake, 4)
        final_real = round(1.0 - final_fake, 4)
        pred_label = "FAKE" if final_fake >= 0.5 else "REAL"
        confidence = final_fake if final_fake >= 0.5 else final_real
 
        text_explanation = generate_explanation(transcript, final_real * 100, final_fake * 100)
 
        return {
            "modality":    "audio",
            "prediction":  pred_label,
            "confidence":  round(confidence, 4),
            "prob_real":   final_real,
            "prob_fake":   final_fake,
            "transcript":  transcript,
            "explanation": text_explanation,
            "score_audio_forensics": round(1.0 - audio_fake, 4),
        }
 
 
# =========================
# IMAGE DEEPFAKE
# =========================
def predict_image(pil_img: Image.Image) -> dict:
    _load_image_model()
 
    # ── MobileNetV3 prediction ───────────────────
    tensor = _IMAGE_TFM(pil_img.convert("RGB")).unsqueeze(0).to(_IMAGE_DEVICE)
    with torch.no_grad():
        logits = _IMAGE_MODEL(tensor)
        proba  = torch.softmax(logits, dim=1)[0].cpu().numpy()
 
    model_fake = float(proba[1])  # index 1 = FAKE
 
    geo_fake = face_geometry_score_image(pil_img)
 
    final_fake = round(0.70 * model_fake + 0.30 * geo_fake, 4)
    final_real = round(1.0 - final_fake, 4)
    pred_label = "FAKE" if final_fake >= 0.5 else "REAL"
    confidence = final_fake if final_fake >= 0.5 else final_real
 
    # ── GPT-4o Vision explanation for image ──────
    visual_explanation = generate_image_explanation(
        pil_img,
        final_real * 100,
        final_fake * 100,
    )
 
    return {
        "modality":          "image",
        "prediction":        pred_label,
        "confidence":        round(confidence, 4),
        "prob_real":         final_real,
        "prob_fake":         final_fake,
        "transcript":        "",
        "explanation":       visual_explanation,
        "score_face_geo":    round(1.0 - geo_fake, 4),
    }
 
