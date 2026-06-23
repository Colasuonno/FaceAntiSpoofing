"""
Modulo di inferenza condiviso per la demo anti-spoofing.

- Carica il checkpoint CNN (outputs/cnn_best.pt o un checkpoint calibrato).
- Face detection con Haar cascade di OpenCV (bundled, nessun download).
- predict(): dato un volto -> spoof score in [0,1] e verdetto LIVE/SPOOF.

Convenzione: score = probabilita' della classe spoof (1). Predetto SPOOF se score >= threshold.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
import torch
from PIL import Image

import sys
sys.path.insert(0, "src")
from model import build_model
from transforms import build_eval_transform

DEFAULT_CKPT = "outputs/cnn_best.pt"


class SpoofDetector:
    def __init__(self, ckpt_path: str = DEFAULT_CKPT, threshold: float = 0.5, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model = build_model(ckpt["backbone"], pretrained=False).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.tf = build_eval_transform()
        self.threshold = threshold
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade)

    def detect_faces(self, bgr: np.ndarray):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                                   minSize=(60, 60))
        return list(faces)  # (x,y,w,h)

    def _crop(self, bgr, box, margin=0.2):
        x, y, w, h = box
        mx, my = int(w * margin), int(h * margin)
        x0, y0 = max(0, x - mx), max(0, y - my)
        x1, y1 = min(bgr.shape[1], x + w + mx), min(bgr.shape[0], y + h + my)
        return bgr[y0:y1, x0:x1]

    @torch.no_grad()
    def predict_crop(self, bgr_crop: np.ndarray) -> float:
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        x = self.tf(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
            logits = self.model(x)
        return float(torch.softmax(logits.float(), dim=1)[0, 1].cpu())

    def predict_image(self, bgr: np.ndarray):
        """Ritorna lista di (box, score, is_spoof). Se nessun volto, usa l'intera immagine."""
        faces = self.detect_faces(bgr)
        results = []
        if not faces:
            score = self.predict_crop(bgr)
            results.append((None, score, score >= self.threshold))
        else:
            for box in faces:
                score = self.predict_crop(self._crop(bgr, box))
                results.append((box, score, score >= self.threshold))
        return results

    def annotate(self, bgr: np.ndarray) -> np.ndarray:
        """Disegna box + verdetto sul frame (per la webcam)."""
        for box, score, is_spoof in self.predict_image(bgr):
            label = "SPOOF" if is_spoof else "REAL"
            color = (0, 0, 255) if is_spoof else (0, 200, 0)
            text = f"{label} {score:.2f}"
            if box is None:
                cv2.putText(bgr, text + " (no face)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                x, y, w, h = box
                cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
                cv2.putText(bgr, text, (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return bgr
