"""
Fase 11.5 — Sistema biometrico COMPLETO: anti-spoofing + riconoscimento in cascata.

Pipeline per ogni immagine:
  1. ANTI-SPOOF (CNN, Fase 5): il volto e' live o spoof?
       - se SPOOF  -> BLOCCA (attacco), non si procede al riconoscimento.
  2. RECOGNITION (facenet, Fase 11): solo se LIVE, calcola l'embedding e lo confronta
     con la gallery (open-set):
       - similarita' col miglior soggetto >= soglia -> IDENTIFICATO: <nome>
       - altrimenti                                  -> SCONOSCIUTO (non iscritto)

Open-set = il sistema puo' anche dire "non e' nessuno degli iscritti", a differenza del
closed-set dove e' costretto a scegliere un nome.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageOps

import sys
sys.path.insert(0, "src")
from infer import SpoofDetector
from recognition import FaceEmbedder, cosine_similarity


class BiometricSystem:
    def __init__(self, cnn_ckpt="outputs/cnn_best.pt", recog_threshold=None,
                 spoof_threshold=0.5):
        self.spoof = SpoofDetector(cnn_ckpt, threshold=spoof_threshold)
        self.embedder = FaceEmbedder()
        if recog_threshold is None:
            p = Path("outputs/recog_threshold.npy")
            recog_threshold = float(np.load(p)[0]) if p.exists() else 0.40
        self.recog_threshold = recog_threshold
        self.gallery: dict[str, np.ndarray] = {}   # nome -> embedding medio (L2-norm)

    # ---------------- enrollment ----------------
    def enroll(self, name: str, image_paths: list[str]) -> int:
        """Iscrive un soggetto: media degli embedding delle sue foto. Ritorna #foto usate."""
        embs = []
        for p in image_paths:
            e = self.embedder.embed_path(p)
            if e is not None:
                embs.append(e)
        if not embs:
            return 0
        v = np.mean(embs, axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        self.gallery[name] = v
        return len(embs)

    # ---------------- identificazione ----------------
    @staticmethod
    def _pil_from_path(path: str) -> Image.Image:
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im).convert("RGB")

    def identify_pil(self, pil: Image.Image) -> dict:
        pil = ImageOps.exif_transpose(pil).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # 1) anti-spoof sul volto principale
        faces = self.spoof.predict_image(bgr)            # [(box, spoof_score, is_spoof)]
        box, spoof_score, is_spoof = max(faces, key=lambda r: r[1])  # volto piu' "sospetto"
        if is_spoof:
            return {"live": False, "spoof_score": float(spoof_score),
                    "identity": None, "best_sim": None, "decision": "SPOOF (attacco bloccato)"}

        # 2) riconoscimento (solo se live)
        emb = self.embedder.embed_pil(pil)
        if emb is None:
            return {"live": True, "spoof_score": float(spoof_score),
                    "identity": None, "best_sim": None, "decision": "nessun volto per il match"}
        if not self.gallery:
            return {"live": True, "spoof_score": float(spoof_score),
                    "identity": None, "best_sim": None, "decision": "gallery vuota"}

        sims = {n: cosine_similarity(emb, v) for n, v in self.gallery.items()}
        best_name = max(sims, key=sims.get)
        best_sim = sims[best_name]
        if best_sim >= self.recog_threshold:
            decision = f"IDENTIFICATO: {best_name}"
            identity = best_name
        else:
            decision = "SCONOSCIUTO (non iscritto)"
            identity = None
        return {"live": True, "spoof_score": float(spoof_score),
                "identity": identity, "best_match": best_name, "best_sim": float(best_sim),
                "all_sims": {n: float(s) for n, s in sims.items()}, "decision": decision}

    def identify_path(self, path: str) -> dict:
        return self.identify_pil(self._pil_from_path(path))
