"""
evaluate.py — Python 3.14 + PyTorch 2.10+ compatible
Loads best checkpoint, runs test-set evaluation, prints report,
and saves confusion matrix heatmap.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

from preprocess import get_dataloaders
from model import build_model

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("../models/best_model.pt")
NUM_CLASSES = 43

# 43 GTSRB class names
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
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val acc {ckpt['val_acc']:.4f})")
    return model


def evaluate():
    _, _, test_loader = get_dataloaders()
    model = load_model()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = (all_preds == all_labels).mean()
    print(f"\nTest accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    print(classification_report(
        all_labels, all_preds,
        target_names=[f"{i}: {n[:20]}" for i, n in enumerate(CLASS_NAMES)]
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=range(NUM_CLASSES),
                yticklabels=range(NUM_CLASSES))
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion matrix — GTSRB (43 classes)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Saved confusion_matrix.png")

    # Export TorchScript (for deployment without Python source)
    scripted = torch.jit.script(load_model())
    scripted.save("../models/model_scripted.pt")
    print("Saved models/model_scripted.pt (TorchScript)")


if __name__ == "__main__":
    evaluate()
