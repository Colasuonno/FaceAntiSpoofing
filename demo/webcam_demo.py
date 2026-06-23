"""
Demo webcam in tempo reale: REAL (verde) / SPOOF (rosso) sul volto.

Uso:
  python demo/webcam_demo.py [--ckpt outputs/cnn_best.pt] [--threshold 0.5] [--cam 0]

Tasti:  q = esci

Prova: mostra la tua faccia (REAL) e poi mostra alla webcam la tua foto sul telefono o
una stampa (SPOOF). Per risultati stabili nel tuo setup, esegui prima demo/calibrate.py.
"""
import argparse, sys, time
import cv2
sys.path.insert(0, "src")
from infer import SpoofDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/cnn_best.pt")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cam", type=int, default=0)
    args = ap.parse_args()

    det = SpoofDetector(args.ckpt, threshold=args.threshold)
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("Webcam non disponibile (cam index", args.cam, ")"); return

    prev = time.time()
    print("Premi 'q' per uscire.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = det.annotate(frame)
        now = time.time(); fps = 1.0 / max(1e-6, now - prev); prev = now
        cv2.putText(frame, f"{fps:.0f} FPS", (10, frame.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Face Anti-Spoofing - demo (q per uscire)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
