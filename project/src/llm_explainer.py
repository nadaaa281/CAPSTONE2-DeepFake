import streamlit as st
api_key = st.secrets["OPENAI_API_KEY"]

# Use environment variable instead of hardcoding
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_explanation(transcript, real_prob, fake_prob):
    prompt = f"""
You are an AI expert in deepfake detection.

A model analyzed a media file and produced:
- REAL probability: {real_prob}%
- FAKE probability: {fake_prob}%

Transcript (if available):
\"\"\"{transcript}\"\"\"

Explain clearly and simply WHY this is likely real or fake.

Base your reasoning on:
- Speech naturalness
- Repetition or inconsistencies
- Context coherence
- Suspicious linguistic patterns

Keep the explanation concise (3–5 lines).
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )

    return response.choices[0].message.content