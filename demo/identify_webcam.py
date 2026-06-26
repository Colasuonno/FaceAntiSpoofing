"""
Demo sistema completo in tempo reale (webcam): anti-spoof + identificazione.

Uso:
  python demo/identify_webcam.py [--gallery data/gallery] [--cam 0]
                                 [--spoof-th 0.5] [--recog-th <override>]

A schermo, sul volto:
  - rosso  "SPOOF"                 -> attacco bloccato
  - verde  "Benvenuto, <nome>"     -> iscritto riconosciuto
  - giallo "SCONOSCIUTO"           -> persona live ma non in gallery

Tasti: q = esci.  Per stabilita' sul tuo setup conviene prima calibrare l'anti-spoof
(demo/calibrate.py) e passare --recog-th se serve.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, "src")
from biometric_system import BiometricSystem
from openset_eval import list_imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default="data/gallery")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--spoof-th", type=float, default=0.5)
    ap.add_argument("--recog-th", type=float, default=None)
    args = ap.parse_args()

    sysbio = BiometricSystem(spoof_threshold=args.spoof_th, recog_threshold=args.recog_th)
    for d in sorted(Path(args.gallery).glob("*")):
        imgs = list_imgs(d)
        if imgs:
            sysbio.enroll(d.name, [str(p) for p in imgs])
    print("gallery:", list(sysbio.gallery.keys()), "| soglia sim=%.3f" % sysbio.recog_threshold)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("Webcam non disponibile (cam", args.cam, ")"); return
    prev = time.time()
    print("Premi 'q' per uscire.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        r = sysbio.identify_pil(Image.fromarray(rgb))
        if not r["live"]:
            color, txt = (0, 0, 255), "SPOOF"
        elif r["identity"]:
            color, txt = (0, 200, 0), f"Benvenuto, {r['identity']} ({r['best_sim']:.2f})"
        else:
            color, txt = (0, 215, 255), "SCONOSCIUTO"
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), color, 6)
        cv2.putText(frame, txt, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        now = time.time(); fps = 1.0/max(1e-6, now-prev); prev = now
        cv2.putText(frame, f"{fps:.0f} FPS", (15, frame.shape[0]-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Sistema biometrico - anti-spoof + ID (q per uscire)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
