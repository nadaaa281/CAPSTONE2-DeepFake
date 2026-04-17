import os
import base64
import cv2
from PIL import Image
import io
 
# ── Lazy client — initialized only when first needed, not at import time ──────
_client = None
 
def _get_client():
    global _client
    if _client is None:
        import streamlit as st
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        _client = OpenAI(api_key=api_key)
    return _client
 
 
# ─────────────────────────────────────────
# FRAME EXTRACTION
# ─────────────────────────────────────────
def extract_frames(video_path: str, num_frames: int = 3) -> list:
    """Extract evenly spaced frames from a video as base64 strings."""
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(total * i / num_frames) for i in range(num_frames)]
 
    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        frames_b64.append(b64)
 
    cap.release()
    return frames_b64
 
 
# ─────────────────────────────────────────
# IMAGE TO BASE64
# ─────────────────────────────────────────
def pil_image_to_b64(pil_img: Image.Image) -> str:
    """Convert a PIL image to a base64 JPEG string."""
    buffer = io.BytesIO()
    pil_img.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
 
 
# ─────────────────────────────────────────
# EXPLANATION — TEXT ONLY (audio/transcript)
# ─────────────────────────────────────────
def generate_explanation(transcript: str, real_prob: float, fake_prob: float) -> str:
    client = _get_client()
    prompt = f"""
You are a deepfake detection expert.
 
A model analyzed a media file and produced:
- REAL probability: {real_prob:.2f}%
- FAKE probability: {fake_prob:.2f}%
 
Transcript (if available):
\"\"\"{transcript}\"\"\"
 
Explain clearly and simply WHY this is likely real or fake.
Base your reasoning on:
- Speech naturalness
- Repetition or inconsistencies
- Context coherence
- Suspicious linguistic patterns
 
Keep the explanation concise (3–5 sentences).
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content
    except Exception:
        return (
            "Explanation could not be generated due to an error with the AI service. "
            "Please review the confidence scores above for the prediction result."
        )
 
 
# ─────────────────────────────────────────
# EXPLANATION — VISUAL IMAGE
# ─────────────────────────────────────────
def generate_image_explanation(
    pil_img:   Image.Image,
    real_prob: float,
    fake_prob: float,
) -> str:
    """Send a still image to GPT-4o Vision for visual deepfake explanation."""
    client = _get_client()
    b64 = pil_image_to_b64(pil_img)
 
    content = [
        {
            "type": "text",
            "text": f"""You are a deepfake detection expert analyzing a still image.
 
The detection model produced:
- REAL probability: {real_prob:.2f}%
- FAKE probability: {fake_prob:.2f}%
 
Carefully examine the image and explain WHY this image is likely real or fake.
Focus on:
- Unnatural facial geometry or asymmetry
- Blurring or artifacts around face edges, hair, or ears
- Inconsistent lighting or shadows on the face
- Skin texture anomalies (too smooth, waxy, or plastic-looking)
- Unnatural eyes (glassy, asymmetric, or missing reflections)
- Facial expressions that look stiff or unnatural
- Background inconsistencies near the face
 
Keep the explanation concise (3–5 sentences).
""",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
    ]
 
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            temperature=0.4,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception:
        return (
            "Visual explanation could not be generated due to an error with the AI service. "
            "Please review the confidence scores above for the prediction result."
        )
 
 
# ─────────────────────────────────────────
# EXPLANATION — VISUAL (image/video frames)
# ─────────────────────────────────────────
def generate_visual_explanation(
    frames_b64: list,
    real_prob:  float,
    fake_prob:  float,
    transcript: str = "",
) -> str:
    """Send extracted frames to GPT-4o Vision for visual deepfake explanation."""
    client = _get_client()
 
    content = [
        {
            "type": "text",
            "text": f"""You are a deepfake detection expert analyzing video frames.
 
The detection model produced:
- REAL probability: {real_prob:.2f}%
- FAKE probability: {fake_prob:.2f}%
 
Transcript (if available): {transcript}
 
Analyze the provided frames and explain WHY this media is likely real or fake.
Focus on:
- Unnatural facial geometry or texture
- Blurring or artifacts around face edges
- Inconsistent lighting or shadows
- Unnatural eye blinking or lip movement
- Skin texture anomalies
 
Keep the explanation concise (3–5 sentences).
""",
        }
    ]
 
    for b64 in frames_b64:import os
import base64
import cv2
from PIL import Image
import io
 
# ── Lazy client — initialized only when first needed, not at import time ──────
_client = None
 
def _get_client():
    global _client
    if _client is None:
        import streamlit as st
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
        _client = OpenAI(api_key=api_key)
    return _client
 
 
# ─────────────────────────────────────────
# FRAME EXTRACTION
# ─────────────────────────────────────────
def extract_frames(video_path: str, num_frames: int = 3) -> list:
    """Extract evenly spaced frames from a video as base64 strings."""
    cap    = cv2.VideoCapture(video_path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(total * i / num_frames) for i in range(num_frames)]
 
    frames_b64 = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        img    = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        frames_b64.append(b64)
 
    cap.release()
    return frames_b64
 
 
# ─────────────────────────────────────────
# IMAGE TO BASE64
# ─────────────────────────────────────────
def pil_image_to_b64(pil_img: Image.Image) -> str:
    """Convert a PIL image to a base64 JPEG string."""
    buffer = io.BytesIO()
    pil_img.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
 
 
# ─────────────────────────────────────────
# EXPLANATION — TEXT ONLY (audio/transcript)
# ─────────────────────────────────────────
def generate_explanation(transcript: str, real_prob: float, fake_prob: float) -> str:
    client = _get_client()
    prompt = f"""
You are a deepfake detection expert.
 
A model analyzed a media file and produced:
- REAL probability: {real_prob:.2f}%
- FAKE probability: {fake_prob:.2f}%
 
Transcript (if available):
\"\"\"{transcript}\"\"\"
 
Explain clearly and simply WHY this is likely real or fake.
Base your reasoning on:
- Speech naturalness
- Repetition or inconsistencies
- Context coherence
- Suspicious linguistic patterns
 
Keep the explanation concise (3–5 sentences).
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        return response.choices[0].message.content
    except Exception:
        return (
            "Explanation could not be generated due to an error with the AI service. "
            "Please review the confidence scores above for the prediction result."
        )
 
 
# ─────────────────────────────────────────
# EXPLANATION — VISUAL IMAGE
# ─────────────────────────────────────────
def generate_image_explanation(
    pil_img:   Image.Image,
    real_prob: float,
    fake_prob: float,
) -> str:
    """Send a still image to GPT-4o Vision for visual deepfake explanation."""
    client = _get_client()
    b64 = pil_image_to_b64(pil_img)
 
    content = [
        {
            "type": "text",
            "text": f"""You are a deepfake detection expert analyzing a still image.
 
The detection model produced:
- REAL probability: {real_prob:.2f}%
- FAKE probability: {fake_prob:.2f}%
 
Carefully examine the image and explain WHY this image is likely real or fake.
Focus on:
- Unnatural facial geometry or asymmetry
- Blurring or artifacts around face edges, hair, or ears
- Inconsistent lighting or shadows on the face
- Skin texture anomalies (too smooth, waxy, or plastic-looking)
- Unnatural eyes (glassy, asymmetric, or missing reflections)
- Facial expressions that look stiff or unnatural
- Background inconsistencies near the face
 
Keep the explanation concise (3–5 sentences).
""",
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
    ]
 
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            temperature=0.4,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception:
        return (
            "Visual explanation could not be generated due to an error with the AI service. "
            "Please review the confidence scores above for the prediction result."
        )
 
 
# ─────────────────────────────────────────
# EXPLANATION — VISUAL (image/video frames)
# ─────────────────────────────────────────
def generate_visual_explanation(
    frames_b64: list,
    real_prob:  float,
    fake_prob:  float,
    transcript: str = "",
) -> str:
    """Send extracted frames to GPT-4o Vision for visual deepfake explanation."""
    client = _get_client()
 
    content = [
        {
            "type": "text",
            "text": f"""You are a deepfake detection expert analyzing video frames.
 
The detection model produced:
- REAL probability: {real_prob:.2f}%
- FAKE probability: {fake_prob:.2f}%
 
Transcript (if available): {transcript}
 
Analyze the provided frames and explain WHY this media is likely real or fake.
Focus on:
- Unnatural facial geometry or texture
- Blurring or artifacts around face edges
- Inconsistent lighting or shadows
- Unnatural eye blinking or lip movement
- Skin texture anomalies
 
Keep the explanation concise (3–5 sentences).
""",
        }
    ]
 
    for b64 in frames_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
 
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            temperature=0.4,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception:
        return (
            "Visual explanation could not be generated due to an error with the AI service. "
            "Please review the confidence scores above for the prediction result."
        )
 
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
 
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            temperature=0.4,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception:
        return (
            "Visual explanation could not be generated due to an error with the AI service. "
            "Please review the confidence scores above for the prediction result."
        )
 
