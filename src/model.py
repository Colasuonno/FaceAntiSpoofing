"""
Modello deep per anti-spoofing: backbone pre-addestrato (timm) con testa a 2 classi
(live / spoof).
"""
from __future__ import annotations
import timm
import torch
import torch.nn as nn


def build_model(backbone: str = "resnet18", pretrained: bool = True, num_classes: int = 2):
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
    return model


@torch.no_grad()
def predict_scores(model, loader, device):
    """Ritorna (spoof_scores, labels): spoof_score = softmax della classe spoof (1)."""
    model.eval()
    scores, labels = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(xb)
        p = torch.softmax(logits.float(), dim=1)[:, 1]
        scores.append(p.cpu())
        labels.append(yb)
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()
