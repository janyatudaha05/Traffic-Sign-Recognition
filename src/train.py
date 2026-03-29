"""
train.py — Python 3.14 + PyTorch 2.10+ compatible
Full training loop with class-weight handling, LR scheduling, and checkpointing.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from preprocess import get_dataloaders
from model import build_model

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE  = 64
EPOCHS      = 50
LR          = 1e-3
PATIENCE    = 10          # early stopping patience
NUM_CLASSES = 43
SAVE_DIR    = Path("../models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Python 3.14 + PyTorch 2.10 CPU only (no CUDA wheels yet for cp314)
# If CUDA wheels become available, this will automatically use GPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ── Class weights (handle imbalance) ─────────────────────────────────────────

def get_class_weights(train_loader) -> torch.Tensor:
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(NUM_CLASSES),
        y=all_labels,
    )
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


# ── One epoch pass ────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, training: bool):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False,
                                  desc="train" if training else "val "):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if training:
                optimizer.zero_grad()

            logits = model(imgs)
            loss   = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += labels.size(0)

    return total_loss / total, correct / total


# ── Main training loop ────────────────────────────────────────────────────────

def train():
    train_loader, val_loader, _ = get_dataloaders(batch_size=BATCH_SIZE)

    model     = build_model(NUM_CLASSES).to(DEVICE)
    weights   = get_class_weights(train_loader)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=True
    )

    best_val_acc  = 0.0
    no_improve    = 0
    history       = {"train_loss": [], "train_acc": [],
                     "val_loss":   [], "val_acc":   []}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, training=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, training=False)
        elapsed = time.time() - t0

        scheduler.step(vl_acc)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"Val loss {vl_loss:.4f} acc {vl_acc:.4f} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Save best checkpoint
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve   = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc":    vl_acc,
            }, SAVE_DIR / "best_model.pt")
            print(f"  ✓ Saved best model (val acc {vl_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
    return history


if __name__ == "__main__":
    train()
