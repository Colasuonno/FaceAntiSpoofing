"""
Feature classiche per anti-spoofing: COLOR-TEXTURE LBP.

Metodo di riferimento (Boulkenafet et al., 2015, "Face anti-spoofing based on color
texture analysis"): le immagini spoof (foto/replay) hanno una micro-texture diversa,
piu' evidente nei canali di crominanza. Calcoliamo istogrammi LBP multi-scala su due
spazi colore (YCbCr e HSV), per canale, e li concateniamo in un unico feature vector.

Dimensione vettore: 2 colorspace x 3 canali x 2 scale LBP uniform = 12 istogrammi
  scala (P=8,R=1) -> 10 bin ; scala (P=16,R=2) -> 18 bin
  totale = 6*10 + 6*18 = 168 feature.
"""
from __future__ import annotations
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from joblib import Parallel, delayed

LBP_SCALES = [(8, 1), (16, 2)]
COLOR_SPACES = ["YCbCr", "HSV"]
IMG_SIZE = 128


def _lbp_hist(channel: np.ndarray, P: int, R: int) -> np.ndarray:
    lbp = local_binary_pattern(channel, P, R, method="uniform")
    n_bins = P + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_features(img: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    img = img.convert("RGB").resize((size, size))
    feats = []
    for cs in COLOR_SPACES:
        arr = np.asarray(img.convert(cs))
        for c in range(3):
            ch = arr[:, :, c]
            for (P, R) in LBP_SCALES:
                feats.append(_lbp_hist(ch, P, R))
    return np.concatenate(feats)


def _extract_path(img_root, relpath, size):
    with Image.open(f"{img_root}/{relpath}") as im:
        return extract_features(im, size)


def extract_dataframe(df, img_root: str, size: int = IMG_SIZE, n_jobs: int = 4) -> np.ndarray:
    """Estrae le feature per tutte le righe di un DataFrame (colonna 'filepath'), in parallelo."""
    paths = df["filepath"].tolist()
    feats = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(_extract_path)(img_root, p, size) for p in paths
    )
    return np.vstack(feats)
