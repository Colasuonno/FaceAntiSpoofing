"""
Estrae i parquet CelebA-Spoof (cropped) in file PNG su disco + un CSV di indice.

Perche': tenere 80k+ immagini PNG dentro pandas satura la RAM (OOM). Scrivendo i file
su disco una volta sola, il training puo' usare un Dataset leggero con DataLoader workers.
I byte sono gia' PNG -> li scriviamo cosi' come sono (nessuna ricompressione, lossless).

Output:
  data/images/<split>/<live|spoof>/<shard>_<row>.png
  data/<split>.csv   con colonne: filepath,label   (label: 0=live, 1=spoof)
"""
from __future__ import annotations
import csv
import gc
import glob
from pathlib import Path

import pyarrow.parquet as pq

LABEL_NAME = {0: "live", 1: "spoof"}


def _is_valid(cell) -> bool:
    return isinstance(cell, dict) and cell.get("bytes") is not None


def extract_split(parquet_dir: str, out_img_root: str, csv_path: str, split: str):
    files = sorted(glob.glob(str(Path(parquet_dir) / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"Nessun parquet in {parquet_dir}")
    for lab in ("live", "spoof"):
        Path(out_img_root, split, lab).mkdir(parents=True, exist_ok=True)

    n_ok = n_bad = 0
    counts = {0: 0, 1: 0}
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["filepath", "label"])
        for fi, f in enumerate(files):
            shard = Path(f).stem.split("-")[1]  # es. '00000'
            df = pq.read_table(f, columns=["cropped_image", "labels"]).to_pandas()
            for i, row in enumerate(df.itertuples(index=False)):
                cell = row.cropped_image
                if not _is_valid(cell):
                    n_bad += 1
                    continue
                label = int(row.labels)
                rel = f"{split}/{LABEL_NAME[label]}/{shard}_{i:05d}.png"
                with open(Path(out_img_root, rel), "wb") as img:
                    img.write(cell["bytes"])
                w.writerow([rel, label])
                counts[label] += 1
                n_ok += 1
            del df
            gc.collect()
            print(f"  [{split}] shard {fi+1}/{len(files)} ({shard}) -> ok={n_ok} bad={n_bad}", flush=True)
    print(f"[{split}] TOT validi={n_ok} | scartati={n_bad} | live={counts[0]} spoof={counts[1]}", flush=True)
    return n_ok, n_bad, counts


if __name__ == "__main__":
    root = "data/images"
    extract_split("data/celeba_spoof_hf", root, "data/train.csv", "train")
    extract_split("data/celeba_spoof_test_hf", root, "data/test.csv", "test")
    print("EXTRACT DONE", flush=True)
