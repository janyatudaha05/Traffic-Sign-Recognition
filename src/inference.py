"""
inference.py — Python 3.14 + PyTorch 2.10+ compatible
Real-time webcam inference OR single-image prediction.

Usage:
    python inference.py                    # webcam mode
    python inference.py --image path.jpg   # single image mode
"""

import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path

from preprocess import apply_clahe, IMG_SIZE, VAL_TRANSFORM
from model import build_model

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("../models/best_model.pt")
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


def load_model() -> torch.nn.Module:
    model = build_model(NUM_CLASSES).to(DEVICE)
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_frame(model, frame_bgr: np.ndarray):
    """Predict from a raw OpenCV BGR frame. Returns (class_id, confidence, label)."""
    rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb   = apply_clahe(rgb)
    rgb   = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    tensor = VAL_TRANSFORM(rgb).unsqueeze(0).to(DEVICE)  # [1,3,32,32]

    with torch.no_grad():
        logits = model(tensor)[0]
        probs  = torch.softmax(logits, dim=0)
        conf, cls_id = probs.max(dim=0)

    cls_id = cls_id.item()
    conf   = conf.item()
    label  = CLASS_NAMES[cls_id]
    return cls_id, conf, label


def webcam_mode(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print("Webcam running. Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cls_id, conf, label = predict_frame(model, frame)

        # Overlay bounding suggestion + label
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        box_half = min(w, h) // 4
        cv2.rectangle(frame,
                      (cx - box_half, cy - box_half),
                      (cx + box_half, cy + box_half),
                      (0, 255, 0), 2)

        text     = f"{label[:30]} ({conf:.1%})"
        bg_color = (0, 180, 0) if conf > 0.80 else (0, 165, 255)
        cv2.rectangle(frame, (0, 0), (w, 36), bg_color, -1)
        cv2.putText(frame, text, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        cv2.imshow("Traffic Sign Recognition — press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def single_image_mode(model, img_path: str):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Cannot read image: {img_path}")
        return
    cls_id, conf, label = predict_frame(model, frame)
    print(f"\nImage     : {img_path}")
    print(f"Prediction: [{cls_id}] {label}")
    print(f"Confidence: {conf:.4f} ({conf*100:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single image (omit for webcam mode)")
    args = parser.parse_args()

    model = load_model()
    print(f"Model loaded on {DEVICE}")

    if args.image:
        single_image_mode(model, args.image)
    else:
        webcam_mode(model)
