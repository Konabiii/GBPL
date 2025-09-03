#!/usr/bin/env python3
import os
import io
import json
import time
from typing import Optional

import streamlit as st
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, db

# ---------- Load environment ----------
load_dotenv()
st.set_page_config(page_title="Plant Diagnosis", layout="centered")

# ---------- Firebase Init ----------
FIREBASE_DB_URL = ""
if not firebase_admin._apps:
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path or not os.path.exists(cred_path):
        st.error("GOOGLE_APPLICATION_CREDENTIALS is not set or file not found.")
        st.stop()
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DB_URL})

# ---------- Gemini Init ----------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY is not set.")
    st.stop()
genai.configure(api_key=api_key)
MODEL_NAME = "models/gemini-1.5-flash-latest"
REALTIME_PATH = "/sensors2"

# ---------- Utils ----------
def load_image_bytes(file) -> bytes:
    with Image.open(file) as im:
        im = im.convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

def load_candidates(file) -> Optional[dict]:
    if not file:
        return None
    return json.load(file)

def build_system_instruction() -> str:
    return (
        "You are an agronomy expert assisting field diagnosis. "
        "Given a plant image, ambient temperature/humidity, and optional candidate disease names, "
        "produce an English-only, plain-text diagnosis for a farmer. "
        "Structure the output with clear section headers:\n"
        "1) Likely diagnosis (disease name)\n"
        "2) Why (key symptoms and environmental consistency)\n"
        "3) Immediate actions (today)\n"
        "4) Treatment — Non-chemical\n"
        "5) Treatment — Chemical (if applicable; avoid specific brand recommendations)\n"
        "6) Monitoring (next 3–7 days)\n"
        "7) Prevention (longer-term)\n"
        "8) Safety notes and disclaimer\n"
        "Be cautious: note uncertainty if applicable and avoid unsafe instructions."
    )

def build_user_prompt(sensor: dict, candidates: Optional[dict],
                     crop: Optional[str], location: Optional[str],
                     growth_stage: Optional[str]) -> str:
    lines = ["Generate an English diagnosis text using the following information."]
    if crop: lines.append(f"- Crop: {crop}")
    if growth_stage: lines.append(f"- Growth stage: {growth_stage}")
    if location: lines.append(f"- Location: {location}")
    lines.append(f"- Temperature: {sensor.get('temp_c', 'NaN')} °C")
    lines.append(f"- Humidity: {sensor.get('humidity', 'NaN')} %")
    if candidates and 'candidates' in candidates:
        names = [c.get('name') for c in candidates['candidates'] if c.get('name')]
        if names: lines.append("- Candidate diseases: " + ", ".join(names))
    lines.append("Return plain English text only (no JSON).")
    return "\n".join(lines)

def read_sensor_from_realtime(path: str) -> dict:
    ref = db.reference(path)
    data = ref.get()
    if not data:
        st.error(f"No data found at path: {path}")
        st.stop()
    return {
        "temp_c": float(data.get("temperature", "nan")),
        "humidity": float(data.get("humidity", "nan"))
    }

def save_feedback_to_firebase(rating: int, feedback_text: str, diagnosis: str,
                             crop: Optional[str] = None,
                             location: Optional[str] = None,
                             growth_stage: Optional[str] = None):
    ref = db.reference("/feedback")
    entry = {
        "rating": rating,
        "feedback": feedback_text,
        "diagnosis": diagnosis,
        "timestamp": int(time.time())
    }
    if crop: entry["crop"] = crop
    if location: entry["location"] = location
    if growth_stage: entry["growth_stage"] = growth_stage
    ref.push(entry)

# ---------- Session state ----------
if "feedback_rating" not in st.session_state:
    st.session_state.feedback_rating = 3
if "diagnosis_text" not in st.session_state:
    st.session_state.diagnosis_text = ""
if "submitted_feedback" not in st.session_state:
    st.session_state.submitted_feedback = False
if "generated_diagnosis" not in st.session_state:
    st.session_state.generated_diagnosis = False
if "capture_started" not in st.session_state:
    st.session_state.capture_started = False
if "last_image_file" not in st.session_state:
    st.session_state.last_image_file = None

st.title("🌿 Plant Diagnosis App")

# ---------- Step 1: choose method ----------
st.markdown("### 📷 Step 1: Choose Input Method")
input_method = st.radio("Select input method:", ["Upload Image", "Camera Capture"])

if st.button("Start Capture/Upload"):
    st.session_state.capture_started = True

# ---------- Step 2: actually provide image ----------
image_file = None
if st.session_state.capture_started:
    if input_method == "Upload Image":
        image_file = st.file_uploader("Upload plant image (jpg/png)", type=["jpg","jpeg","png"])
    else:
        camera_file = st.camera_input("Take a picture of the plant")
        if camera_file:
            image_file = camera_file

# ---------- Preview selected image ----------
if image_file:
    st.markdown("### 👀 Preview Image")
    st.image(image_file, caption="Selected Plant Image", use_container_width=True)

candidates_file = st.file_uploader("Candidate diseases JSON (optional)", type=["json"])
crop = st.text_input("Crop (optional, e.g., tomato)")
location = st.text_input("Location (optional, e.g., Hanoi, Vietnam)")
growth_stage = st.text_input("Growth stage (optional, e.g., seedling/vegetative/fruiting)")

# ---------- Generate Diagnosis ----------
if st.button("Generate Diagnosis"):
    if not image_file:
        st.warning("Please capture or upload a plant image first.")
    else:
        st.session_state.generated_diagnosis = False
        st.session_state.diagnosis_text = ""
        st.session_state.submitted_feedback = False
        st.session_state.last_image_file = image_file

        with st.spinner("Fetching sensor data and generating diagnosis..."):
            try:
                sensor = read_sensor_from_realtime(REALTIME_PATH)
                img_bytes = load_image_bytes(image_file)
                candidates = load_candidates(candidates_file)

                model = genai.GenerativeModel(
                    model_name=MODEL_NAME,
                    system_instruction=build_system_instruction(),
                    generation_config={"response_mime_type": "text/plain"}
                )

                user_prompt = build_user_prompt(sensor, candidates, crop, location, growth_stage)
                parts = [user_prompt, {"mime_type": "image/jpeg", "data": img_bytes}]
                if candidates:
                    parts.append(json.dumps({"candidates": candidates}, ensure_ascii=False))

                resp = model.generate_content(parts)
                st.session_state.diagnosis_text = resp.text or ""
                st.session_state.generated_diagnosis = True
            except Exception as e:
                st.error(f"Error: {e}")

# ---------- Show Diagnosis Result ----------
if st.session_state.generated_diagnosis:
    st.subheader("📝 Diagnosis Result")
    st.text_area("Diagnosis", st.session_state.diagnosis_text, height=300)

    if not st.session_state.submitted_feedback:
        st.subheader("⭐ Rate & Feedback")

        st.session_state.feedback_rating = st.slider(
            "Rating (1–5 stars)",
            min_value=1,
            max_value=5,
            value=st.session_state.feedback_rating
        )
        feedback_text = st.text_area("Your feedback (optional)", height=100)

        if st.button("Submit Feedback"):
            save_feedback_to_firebase(
                st.session_state.feedback_rating,
                feedback_text,
                st.session_state.diagnosis_text,
                crop,
                location,
                growth_stage
            )
            st.session_state.submitted_feedback = True
            st.success("Thank you for your feedback!")
            time.sleep(1)

            # 🔹 Reset toàn bộ session_state rồi reload app
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
