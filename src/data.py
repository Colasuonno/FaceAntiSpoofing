"""
Data loading per CelebA-Spoof (versione cropped) — basato su file PNG su disco.

Usare prima `python src/extract.py` per generare:
  data/images/<split>/<live|spoof>/*.png
  data/train.csv , data/test.csv   (colonne: filepath,label ; label 0=live, 1=spoof)

NOTA ANTI-LEAKAGE:
La versione cropped NON contiene il subject_id, quindi non possiamo costruire uno split
per-soggetto da soli. Per questo:
  - TRAIN/VAL -> dal repo di train (split ufficiale CelebA-Spoof)
  - TEST      -> dal repo di test  (disgiunto per soggetto)
Il VAL viene ritagliato dal train in modo casuale: puo' esserci sovrapposizione di soggetti
tra train e val (usato solo per early stopping), MA il TEST resta pulito e disgiunto.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

LIVE, SPOOF = 0, 1
DATA_ROOT = Path("data")
IMG_ROOT = DATA_ROOT / "images"


def load_csv(split: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / f"{split}.csv")
    return df


def train_val_split(val_fraction: float = 0.15, seed: int = 42):
    """Split casuale train/val a livello di riga (val solo per early stopping)."""
    df = load_csv("train")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    n_val = int(round(len(df) * val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


class CelebASpoofDataset(Dataset):
    """Dataset PyTorch leggero: legge i PNG da disco on-the-fly."""

    def __init__(self, df: pd.DataFrame, transform=None, img_root: Path = IMG_ROOT):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.img_root = Path(img_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_root / row["filepath"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])

    def class_counts(self) -> dict:
        vc = self.df["label"].value_counts().to_dict()
        return {"live": int(vc.get(LIVE, 0)), "spoof": int(vc.get(SPOOF, 0))}


def describe(df: pd.DataFrame, name: str):
    n = len(df)
    live = int((df["label"] == LIVE).sum())
    spoof = int((df["label"] == SPOOF).sum())
    print(f"{name}: {n} img | live={live} ({live/n:.1%}) | spoof={spoof} ({spoof/n:.1%})")


if __name__ == "__main__":
    tr = load_csv("train")
    te = load_csv("test")
    print("== STATISTICHE DATASET (gate Fase 2) ==")
    describe(tr, "TRAIN ")
    describe(te, "TEST  ")
    trs, vas = train_val_split()
    describe(trs, "  -> train")
    describe(vas, "  -> val  ")
