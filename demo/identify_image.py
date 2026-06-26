"""
Demo sistema completo su una singola immagine: anti-spoof + identificazione.

Uso:
  python demo/identify_image.py <immagine> [--gallery data/gallery] [--out outputs/identify_demo.png]

Iscrive i soggetti trovati in --gallery (tutte le foto per soggetto), poi sull'immagine data
stampa: live/spoof e, se live, IDENTIFICATO:<nome> / SCONOSCIUTO.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import cv2
sys.path.insert(0, "src")
from biometric_system import BiometricSystem
from openset_eval import list_imgs, annotate_save


def enroll_gallery(sysbio, gallery_dir):
    for d in sorted(Path(gallery_dir).glob("*")):
        imgs = list_imgs(d)
        if imgs:
            n = sysbio.enroll(d.name, [str(p) for p in imgs])
            print(f"iscritto '{d.name}' ({n} foto)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--gallery", default="data/gallery")
    ap.add_argument("--out", default="outputs/identify_demo.png")
    args = ap.parse_args()

    sysbio = BiometricSystem()
    enroll_gallery(sysbio, args.gallery)
    print("gallery:", list(sysbio.gallery.keys()), "\n")

    r = sysbio.identify_path(args.image)
    print("LIVE" if r["live"] else "SPOOF", "| spoof_score=%.3f" % r["spoof_score"])
    print("DECISIONE:", r["decision"], "" if r.get("best_sim") is None else "(sim=%.3f)" % r["best_sim"])
    annotate_save(args.image, r, args.out)
    print("immagine annotata ->", args.out)


if __name__ == "__main__":
    main()
