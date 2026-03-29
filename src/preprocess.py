"""
preprocess.py — Python 3.14 + PyTorch compatible
Handles image loading, CLAHE, resize, normalization, and augmentation.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image


IMG_SIZE = 32  # resize target: 32x32 px
NUM_CLASSES = 43
DATA_ROOT = Path("data")


# ── CLAHE preprocessing (CPU/OpenCV, Python 3.14 compatible) ─────────────────

def apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def load_and_preprocess(img_path: str) -> np.ndarray:
    """Load image, apply CLAHE, resize. Returns uint8 HxWxC array."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = apply_clahe(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return img  # uint8, [0, 255]


# ── PyTorch transforms ────────────────────────────────────────────────────────

# Training: normalize + augmentation
TRAIN_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.RandomRotation(degrees=15),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    # NOTE: Do NOT use horizontal flip — traffic signs are directional!
    T.ToTensor(),                         # [0,255] → [0.0,1.0], CHW
    T.Normalize(mean=[0.5, 0.5, 0.5],    # center around 0
                std=[0.5, 0.5, 0.5]),
])

# Validation / Test: only normalize, no augmentation
VAL_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
])


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class TrafficSignDataset(Dataset):
    """
    Reads Train.csv / Test.csv produced by GTSRB.
    CSV format: Width, Height, Roi.X1, Roi.Y1, Roi.X2, Roi.Y2, ClassId, Path
    """

    def __init__(self, csv_path: str, root: str = "data", transform=None):
        self.df = pd.read_csv(csv_path)
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.root / row["Path"]
        img = load_and_preprocess(img_path)  # numpy uint8 HxWxC
        label = int(row["ClassId"])
        if self.transform:
            img = self.transform(img)
        return img, label


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(batch_size: int = 64, num_workers: int = 2):
    train_ds = TrafficSignDataset("data/Train.csv", transform=TRAIN_TRANSFORM)
    test_ds  = TrafficSignDataset("data/Test.csv",  transform=VAL_TRANSFORM)

    # 80/20 split of train set → train / validation
    n_val   = int(0.2 * len(train_ds))
    n_train = len(train_ds) - n_val
    train_subset, val_subset = (
        __import__("torch").utils.data.random_split(train_ds, [n_train, n_val])
    )
    # Val subset should use VAL_TRANSFORM — wrap it
    val_subset.dataset = TrafficSignDataset(
        "data/Train.csv", transform=VAL_TRANSFORM
    )

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    tl, vl, tel = get_dataloaders()
    imgs, labels = next(iter(tl))
    print(f"Batch shape : {imgs.shape}")   # [64, 3, 32, 32]
    print(f"Labels      : {labels[:8]}")
    print(f"Train batches: {len(tl)} | Val: {len(vl)} | Test: {len(tel)}")
