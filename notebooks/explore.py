"""
explore.py — Python 3.14 compatible EDA script
Run this first to understand your dataset before training.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path

DATA_ROOT = Path("data")


def main():
    print("=" * 60)
    print("GTSRB Dataset Exploration")
    print("=" * 60)

    # Load CSVs
    train_df = pd.read_csv(DATA_ROOT / "Train.csv")
    test_df  = pd.read_csv(DATA_ROOT / "Test.csv")

    print(f"\nTraining samples : {len(train_df):,}")
    print(f"Test samples     : {len(test_df):,}")
    print(f"Classes          : {train_df['ClassId'].nunique()}")
    print(f"\nColumn names: {list(train_df.columns)}")
    print(f"\nFirst 3 rows:\n{train_df.head(3)}")

    # Class distribution
    class_counts = train_df["ClassId"].value_counts().sort_index()
    print(f"\nMin samples per class : {class_counts.min()} (class {class_counts.idxmin()})")
    print(f"Max samples per class : {class_counts.max()} (class {class_counts.idxmax()})")
    print(f"Imbalance ratio       : {class_counts.max() / class_counts.min():.1f}x")

    # Plot class distribution
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.bar(class_counts.index, class_counts.values, color="#378ADD", edgecolor="none")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Number of images")
    ax.set_title("Training set class distribution (note imbalance)")
    plt.tight_layout()
    plt.savefig("class_distribution.png", dpi=120)
    print("\nSaved class_distribution.png")

    # Image size distribution
    widths, heights = [], []
    for _, row in train_df.sample(min(500, len(train_df)), random_state=42).iterrows():
        img = cv2.imread(str(DATA_ROOT / row["Path"]))
        if img is not None:
            heights.append(img.shape[0])
            widths.append(img.shape[1])

    print(f"\nImage width  — min:{min(widths)} max:{max(widths)} mean:{np.mean(widths):.0f}")
    print(f"Image height — min:{min(heights)} max:{max(heights)} mean:{np.mean(heights):.0f}")
    print("→ All images will be resized to 32×32 during preprocessing")

    # Sample images per class (first 9 classes × 3 samples)
    fig, axes = plt.subplots(9, 3, figsize=(8, 24))
    for cls in range(9):
        samples = train_df[train_df["ClassId"] == cls].head(3)
        for col, (_, row) in enumerate(samples.iterrows()):
            img = cv2.imread(str(DATA_ROOT / row["Path"]))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[cls][col].imshow(img)
            axes[cls][col].axis("off")
            if col == 0:
                axes[cls][col].set_ylabel(f"Class {cls}", fontsize=9)
    plt.suptitle("Sample images — classes 0–8 (3 samples each)", y=1.01)
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=120, bbox_inches="tight")
    print("Saved sample_images.png")

    print("\nEDA complete. Review the saved plots before training.")


if __name__ == "__main__":
    main()
