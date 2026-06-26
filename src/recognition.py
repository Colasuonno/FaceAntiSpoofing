"""
Fase 11 — Modulo di RICONOSCIMENTO del volto (face recognition).

A differenza dell'anti-spoofing (live vs spoof), qui rispondiamo a "CHI sei?".
Pipeline:
  - MTCNN: rileva e ALLINEA il volto (ritaglio 160x160 centrato sugli occhi/naso/bocca);
  - InceptionResnetV1 (pre-addestrata su VGGFace2): trasforma il volto in un EMBEDDING,
    un vettore di 512 numeri che rappresenta l'identita' (volti della stessa persona ->
    vettori vicini; persone diverse -> vettori lontani).
  - Confronto tra due volti = SIMILARITA' COSENO tra i loro embedding.

L'embedding e' L2-normalizzato, quindi la similarita' coseno = prodotto scalare in [-1, 1]
(1 = identici, ~0 = diversi).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN, InceptionResnetV1


class FaceEmbedder:
    def __init__(self, device=None, image_size: int = 160, margin: int = 14):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # MTCNN: rileva + allinea il volto. select_largest=True -> tiene il volto piu' grande.
        self.mtcnn = MTCNN(image_size=image_size, margin=margin, select_largest=True,
                           post_process=True, device=self.device)
        # rete di embedding (512-d), pesi VGGFace2
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    @torch.no_grad()
    def embed_pil(self, img: Image.Image) -> np.ndarray | None:
        """PIL RGB -> embedding 512-d L2-normalizzato, oppure None se non trova un volto."""
        img = ImageOps.exif_transpose(img).convert("RGB")  # raddrizza foto da telefono (EXIF)
        face = self.mtcnn(img)                   # tensor (3,160,160) allineato, o None
        if face is None:
            return None
        emb = self.model(face.unsqueeze(0).to(self.device))   # (1,512)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)  # L2-normalize
        return emb[0].cpu().numpy()

    def embed_path(self, path: str) -> np.ndarray | None:
        with Image.open(path) as im:
            return self.embed_pil(im)

    @torch.no_grad()
    def embed_face_tensor(self, face: torch.Tensor) -> np.ndarray:
        """Embedding da un volto gia' allineato (tensor 3x160x160), usato in integrazione."""
        emb = self.model(face.unsqueeze(0).to(self.device))
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb[0].cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similarita' coseno tra due embedding L2-normalizzati = prodotto scalare in [-1,1]."""
    return float(np.dot(a, b))
