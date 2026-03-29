"""
model.py — Python 3.14 + PyTorch 2.10+ compatible
CNN architecture for 43-class traffic sign recognition.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → Conv → BN → ReLU → MaxPool → Dropout"""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.25):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x):
        return self.block(x)


class TrafficSignCNN(nn.Module):
    """
    Input : (B, 3, 32, 32)  — RGB normalized tensor
    Output: (B, 43)          — raw logits (use CrossEntropyLoss)

    Architecture:
        ConvBlock 1 : 3  → 32  channels | 32×32 → 16×16
        ConvBlock 2 : 32 → 64  channels | 16×16 →  8×8
        ConvBlock 3 : 64 → 128 channels |  8×8  →  4×4
        Flatten     : 128 × 4 × 4 = 2048
        FC 1        : 2048 → 512, ReLU, Dropout 0.5
        FC 2        : 512  → 256, ReLU, Dropout 0.4
        FC out      : 256  → 43  (logits)
    """

    def __init__(self, num_classes: int = 43):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3,  32,  dropout=0.25),   # → (B, 32, 16, 16)
            ConvBlock(32, 64,  dropout=0.25),   # → (B, 64,  8,  8)
            ConvBlock(64, 128, dropout=0.30),   # → (B,128,  4,  4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def build_model(num_classes: int = 43) -> TrafficSignCNN:
    return TrafficSignCNN(num_classes=num_classes)


if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(4, 3, 32, 32)
    out = model(dummy)
    print(f"Output shape : {out.shape}")   # [4, 43]

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params : {total_params:,}")
