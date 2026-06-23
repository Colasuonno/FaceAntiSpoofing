"""
Demo su singola immagine: stampa il verdetto LIVE/SPOOF + score e salva l'immagine annotata.

Uso:
  python demo/single_image.py <path_immagine> [--ckpt outputs/cnn_best.pt] [--threshold 0.5]
"""
import argparse, sys
import cv2
sys.path.insert(0, "src")
from infer import SpoofDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--ckpt", default="outputs/cnn_best.pt")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="outputs/demo_single.png")
    args = ap.parse_args()

    det = SpoofDetector(args.ckpt, threshold=args.threshold)
    bgr = cv2.imread(args.image)
    if bgr is None:
        print("Impossibile leggere", args.image); return
    results = det.predict_image(bgr)
    for i, (box, score, is_spoof) in enumerate(results):
        print(f"volto {i+1}: {'SPOOF' if is_spoof else 'LIVE'}  (spoof score={score:.3f}, "
              f"soglia={args.threshold})")
    cv2.imwrite(args.out, det.annotate(bgr))
    print("immagine annotata ->", args.out)


if __name__ == "__main__":
    main()
