"""
Transforms per il modello deep.

Le immagini di CelebA-Spoof (cropped) sono gia' ritagliate sul volto: il preprocessing
si riduce a resize/normalize + augmentation.

L'augmentation e' importante per la ROBUSTEZZA e il cross-dataset (Fase 7): print/replay
attack cambiano texture, illuminazione, compressione. Simuliamo queste variazioni con
color jitter, blur, JPEG compression, random erasing.

Normalizzazione ImageNet perche' usiamo backbone pre-addestrati su ImageNet (Fase 5).
"""
from __future__ import annotations
import io
import random

from PIL import Image
from torchvision import transforms

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomJPEG:
    """Ricompressione JPEG casuale: simula gli artefatti di acquisizione/replay."""

    def __init__(self, quality_range=(30, 90), p=0.5):
        self.qmin, self.qmax = quality_range
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        q = random.randint(self.qmin, self.qmax)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def build_train_transform(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((int(img_size * 1.15), int(img_size * 1.15))),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        RandomJPEG(quality_range=(30, 90), p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def build_eval_transform(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(tensor):
    """Riporta un tensor normalizzato in [0,1] per la visualizzazione."""
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)
