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
 
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
 
from src.llm_explainer import extract_frames, generate_explanation, generate_visual_explanation, generate_image_explanation
 
# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
MODEL_DIR         = Path(__file__).parent.parent / "models" / "best_text_audio_mfcc"
IMAGE_MODEL_PATH  = Path(__file__).parent.parent / "models" / "best_model_image_combined.pth"
VIDEO_MODEL_PATH  = Path(__file__).parent.parent / "models" / "best_model_video_combined.pth"
 
_WHISPER_MODEL = None
 
 
# ─────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────
def load_bundle():
    vec    = joblib.load(MODEL_DIR / "tfidf_vectorizer.joblib")
    scaler = joblib.load(MODEL_DIR / "audio_scaler.joblib")
    clf    = joblib.load(MODEL_DIR / "logreg_model.joblib")
    meta   = json.loads((MODEL_DIR / "meta.json").read_text(encoding="utf-8"))
    return vec, scaler, clf, meta
 
def _get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
 
 
# ─────────────────────────────────────────
# AUDIO / WHISPER HELPERS
# ─────────────────────────────────────────
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
        total += 1
        if float(np.mean(flatness)) > 0.15:
            suspicious += 1
 
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        total += 1
        if float(np.std(zcr)) < 0.02:
            suspicious += 1
 
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        total += 1
        if float(np.std(rolloff)) < 500:
            suspicious += 1
 
        intervals = librosa.effects.split(y, top_db=30)
        if len(intervals) > 0:
            speech_samples = sum(e - s for s, e in intervals)
            silence_ratio  = 1.0 - (speech_samples / max(len(y), 1))
            total += 1
            if silence_ratio > 0.6 or silence_ratio < 0.05:
                suspicious += 1
 
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        total += 1
        if float(np.mean(np.var(chroma, axis=1))) < 0.01:
            suspicious += 1
 
        return suspicious / total if total > 0 else 0.5
 
    except Exception:
        return 0.5
 
 
# ─────────────────────────────────────────
# FACE GEOMETRY (VIDEO)
# ─────────────────────────────────────────
def face_geometry_score(video_path: str) -> float:
    try:
        mp_face_mesh = mp.solutions.face_mesh
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            return 0.5
 
        sample_indices   = np.linspace(0, total_frames - 1, 6, dtype=int)
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
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)
                if not results.multi_face_landmarks:
                    continue
                lm  = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]
                pts  = np.array([[l.x * w, l.y * h] for l in lm])
                nose      = pts[1]
                left_eye  = pts[263]
                right_eye = pts[33]
                dist_left  = np.linalg.norm(left_eye  - nose)
                dist_right = np.linalg.norm(right_eye - nose)
                if dist_left + dist_right > 0:
                    asymmetry = abs(dist_left - dist_right) / ((dist_left + dist_right) / 2)
                    asymmetry_scores.append(asymmetry)
 
        cap.release()
 
        if not asymmetry_scores:
            return 0.5
        return round(min(float(np.mean(asymmetry_scores)) / 0.30, 1.0), 4)
 
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
                    lm   = results.multi_face_landmarks[0].landmark
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
 
        correlation = float(np.corrcoef(norm(mouth_arr), norm(audio_arr))[0, 1])
        if np.isnan(correlation):
            return 0.5
 
        return round((1.0 - max(correlation, 0.0)) / 2.0, 4)
 
    except Exception:
        return 0.5
 
 
# ─────────────────────────────────────────
# FACE GEOMETRY (IMAGE)
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
            asymmetry = abs(dist_left - dist_right) / ((dist_left + dist_right) / 2)
            return round(min(float(asymmetry) / 0.30, 1.0), 4)
 
    except Exception:
        return 0.5
 
 
# ─────────────────────────────────────────
# FRIEND'S VIDEO FRAME MODEL (better accuracy)
# ─────────────────────────────────────────
_VIDEO_MODEL  = None
_VIDEO_DEVICE = None
_VIDEO_TFM    = None
 
def _load_video_model():
    global _VIDEO_MODEL, _VIDEO_DEVICE, _VIDEO_TFM
    if _VIDEO_MODEL is not None:
        return
 
    _VIDEO_DEVICE = _get_device()
 
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, 2)
 
    state = torch.load(str(VIDEO_MODEL_PATH), map_location="cpu")
    new_state = {
        k.replace("backbone.", ""): v
        for k, v in state.items()
        if k.startswith("backbone.")
    }
    model.load_state_dict(new_state, strict=False)
    model.eval().to(_VIDEO_DEVICE)
 
    _VIDEO_TFM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    _VIDEO_MODEL = model
 
@torch.no_grad()
def _predict_video_frame(video_file_path: str) -> list:
    """Returns [fake_prob, real_prob] from friend's frame-level model."""
    _load_video_model()
 
    with tempfile.TemporaryDirectory() as td:
        frame_path = str(Path(td) / "frame.jpg")
        cmd = [
            "ffmpeg", "-y", "-i", video_file_path,
            "-vf", "select=eq(n\\,0)", "-vframes", "1", frame_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
 
        img    = Image.open(frame_path).convert("RGB")
        x      = _VIDEO_TFM(img).unsqueeze(0).to(_VIDEO_DEVICE)
        logits = _VIDEO_MODEL(x)
        probs  = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
        return probs  # [fake, real]
 
 
# ─────────────────────────────────────────
# IMAGE MODEL (friend's weights, your pipeline)
# ─────────────────────────────────────────
_IMAGE_MODEL   = None
_IMAGE_TFM     = None
_IMAGE_DEVICE  = None
 
def _load_image_model():
    global _IMAGE_MODEL, _IMAGE_TFM, _IMAGE_DEVICE
    if _IMAGE_MODEL is not None:
        return
 
    _IMAGE_DEVICE = _get_device()
 
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, 2)
 
    # Load friend's state dict (no backbone prefix remapping needed for image model)
    state = torch.load(str(IMAGE_MODEL_PATH), map_location=_IMAGE_DEVICE)
    state_dict = state.get("model_state_dict", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(_IMAGE_DEVICE)
 
    _IMAGE_TFM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    _IMAGE_MODEL = model
 
 
# =========================
# VIDEO DEEPFAKE
# =========================
def predict_video(video_file_path: str) -> dict:
    vec, scaler, clf, meta = load_bundle()
    sr     = int(meta["sr"])
    n_mfcc = int(meta["n_mfcc"])
 
    with tempfile.TemporaryDirectory() as td:
        wav_path = str(Path(td) / "audio.wav")
        run_ffmpeg_extract_audio(video_file_path, wav_path)
 
        # ── Text + MFCC logistic regression ──────────
        transcript  = run_whisper_transcribe(wav_path)
        X_text      = vec.transform([transcript])
        x_audio_s   = scaler.transform(mfcc_stats(wav_path, sr=sr, n_mfcc=n_mfcc).reshape(1, -1))
        X           = hstack([X_text, csr_matrix(x_audio_s)])
        logreg_proba = clf.predict_proba(X)[0]  # [real, fake]
 
        # ── Friend's frame-level visual model ────────
        frame_probs = _predict_video_frame(video_file_path)  # [fake, real]
 
        # ── Friend's combined score (their proven formula) ───
        friend_fake = (0.5 * logreg_proba[1]) + (0.5 * frame_probs[0])
 
        # ── Your analysis signals ─────────────────────
        lip_fake   = lip_sync_score(video_file_path, wav_path)
        geo_fake   = face_geometry_score(video_file_path)
        audio_fake = audio_forensics_score(wav_path)
 
        # ── Final: friend's score as base + your analysis ──
        # friend_fake replaces the old model_fake (40% weight)
        final_fake = round(
            0.40 * friend_fake +
            0.20 * lip_fake    +
            0.20 * geo_fake    +
            0.20 * audio_fake,
            4
        )
        final_real = round(1.0 - final_fake, 4)
        pred_label = "FAKE" if final_fake >= 0.5 else "REAL"
        confidence = final_fake if final_fake >= 0.5 else final_real
 
        # ── Explanations ──────────────────────────────
        text_explanation   = generate_explanation(transcript, final_real * 100, final_fake * 100)
        frames             = extract_frames(video_file_path, num_frames=3)
        visual_explanation = generate_visual_explanation(frames, final_real * 100, final_fake * 100, transcript)
 
        return {
            "modality":              "video",
            "prediction":            pred_label,
            "confidence":            round(confidence, 4),
            "prob_real":             final_real,
            "prob_fake":             final_fake,
            "transcript":            transcript,
            "explanation":           text_explanation,
            "visual_explanation":    visual_explanation,
            "score_lip_sync":        round(1.0 - lip_fake,   4),
            "score_face_geo":        round(1.0 - geo_fake,   4),
            "score_audio_forensics": round(1.0 - audio_fake, 4),
        }
 
 
# =========================
# AUDIO DEEPFAKE
# =========================
def predict_audio(audio_file_path: str) -> dict:
    vec, scaler, clf, meta = load_bundle()
    sr     = int(meta["sr"])
    n_mfcc = int(meta["n_mfcc"])
 
    with tempfile.TemporaryDirectory() as td:
        wav_path = str(Path(td) / "audio.wav")
 
        if audio_file_path.lower().endswith(".wav"):
            Path(wav_path).write_bytes(Path(audio_file_path).read_bytes())
        else:
            cmd = ["ffmpeg", "-y", "-i", audio_file_path, "-ac", "1", "-ar", "16000", wav_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
 
        transcript   = run_whisper_transcribe(wav_path)
        X_text       = vec.transform([transcript])
        x_audio_s    = scaler.transform(mfcc_stats(wav_path, sr=sr, n_mfcc=n_mfcc).reshape(1, -1))
        X            = hstack([X_text, csr_matrix(x_audio_s)])
        logreg_proba = clf.predict_proba(X)[0]
 
        audio_fake   = audio_forensics_score(wav_path)
 
        # Same weighted formula as your original audio path
        final_fake = round(0.60 * float(logreg_proba[1]) + 0.40 * audio_fake, 4)
        final_real = round(1.0 - final_fake, 4)
        pred_label = "FAKE" if final_fake >= 0.5 else "REAL"
        confidence = final_fake if final_fake >= 0.5 else final_real
 
        text_explanation = generate_explanation(transcript, final_real * 100, final_fake * 100)
 
        return {
            "modality":              "audio",
            "prediction":            pred_label,
            "confidence":            round(confidence, 4),
            "prob_real":             final_real,
            "prob_fake":             final_fake,
            "transcript":            transcript,
            "explanation":           text_explanation,
            "score_audio_forensics": round(1.0 - audio_fake, 4),
        }
 
 
# =========================
# IMAGE DEEPFAKE
# =========================
def predict_image(pil_img: Image.Image) -> dict:
    _load_image_model()
 
    # ── Friend's model inference ──────────────────
    tensor = _IMAGE_TFM(pil_img.convert("RGB")).unsqueeze(0).to(_IMAGE_DEVICE)
    with torch.no_grad():
        logits = _IMAGE_MODEL(tensor)
        proba  = torch.softmax(logits, dim=1)[0].cpu().numpy()
 
    # friend's image model: index 0 = fake, index 1 = real
    model_fake = float(proba[0])
 
    # ── Your face geometry signal ─────────────────
    geo_fake = face_geometry_score_image(pil_img)
 
    # ── Combine (your weights + lowered threshold) ─
    final_fake = round(0.80 * model_fake + 0.20 * geo_fake, 4)
    final_real = round(1.0 - final_fake, 4)
    pred_label = "FAKE" if final_fake >= 0.35 else "REAL"
    confidence = final_fake if final_fake >= 0.35 else final_real
 
    # ── LLM visual explanation ────────────────────
    visual_explanation = generate_image_explanation(pil_img, final_real * 100, final_fake * 100)
 
    return {
        "modality":       "image",
        "prediction":     pred_label,
        "confidence":     round(confidence, 4),
        "prob_real":      final_real,
        "prob_fake":      final_fake,
        "transcript":     "",
        "explanation":    visual_explanation,
        "score_face_geo": round(1.0 - geo_fake, 4),
    }
