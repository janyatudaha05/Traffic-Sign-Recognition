"""
app.py — Python 3.14 + PyTorch 2.10+ compatible
Streamlit web UI for traffic sign recognition.

Run: streamlit run app/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

from preprocess import apply_clahe, IMG_SIZE, VAL_TRANSFORM
from model import build_model

NUM_CLASSES = 43
CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing (3.5t+)", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles",
    "No vehicles (3.5t+)", "No entry", "General caution",
    "Dangerous curve left", "Dangerous curve right", "Double curve",
    "Bumpy road", "Slippery road", "Road narrows right",
    "Road work", "Traffic signals", "Pedestrians",
    "Children crossing", "Bicycles crossing", "Beware of ice/snow",
    "Wild animals crossing", "End speed + passing limits",
    "Turn right ahead", "Turn left ahead", "Ahead only",
    "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End no passing (3.5t+)",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    model = build_model(NUM_CLASSES).to(DEVICE)
    ckpt  = torch.load("models/best_model.pt", map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict(model, pil_img: Image.Image):
    img = np.array(pil_img.convert("RGB"))
    img = apply_clahe(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    tensor = VAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)[0]
        probs  = torch.softmax(logits, dim=0).cpu().numpy()

    top5_idx  = probs.argsort()[::-1][:5]
    top5_conf = probs[top5_idx]
    top5_name = [CLASS_NAMES[i] for i in top5_idx]
    return list(zip(top5_name, top5_conf.tolist(), top5_idx.tolist()))


# ── UI ─────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦", layout="centered")
st.title("Traffic Sign Recognition")
st.caption("CNN trained on GTSRB — 43 German traffic sign classes · Python 3.14 + PyTorch")

model = load_model()

uploaded = st.file_uploader(
    "Upload a traffic sign image (JPG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    pil_img = Image.open(uploaded)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Predicting..."):
            results = predict(model, pil_img)

        top_name, top_conf, top_id = results[0]
        color = "green" if top_conf > 0.80 else "orange" if top_conf > 0.50 else "red"
        st.markdown(f"### :{color}[{top_name}]")
        st.metric("Confidence", f"{top_conf*100:.1f}%", delta=f"Class ID: {top_id}")

        st.markdown("**Top 5 predictions**")
        for name, conf, cid in results:
            st.progress(float(conf), text=f"[{cid:02d}] {name} — {conf*100:.1f}%")

st.markdown("---")
st.caption("Run with: `streamlit run app/app.py`")
