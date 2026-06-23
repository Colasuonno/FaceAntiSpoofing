"""
Calibrazione sui propri dati: cattura frame LIVE (la tua faccia) e SPOOF (tua foto su
telefono / stampa) dalla webcam, poi fa un fine-tuning veloce del modello per adattarlo
al tuo setup d'esame (luce, webcam). Riduce il domain gap nel caso reale della demo.

Uso:
  1) Cattura:   python demo/calibrate.py capture --label live   --n 60
                python demo/calibrate.py capture --label spoof  --n 60
     (durante la cattura: premi SPAZIO per iniziare/fermare, q per uscire)
  2) Fine-tune: python demo/calibrate.py finetune --epochs 5
     -> salva outputs/cnn_calibrated.pt  (usalo poi: webcam_demo.py --ckpt outputs/cnn_calibrated.pt)
"""
import argparse, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, "src")
from infer import SpoofDetector
from model import build_model, predict_scores
from transforms import build_train_transform, build_eval_transform
from data import CelebASpoofDataset
import pandas as pd

CAL_ROOT = Path("data/calibration")


def capture(label: str, n: int, cam: int):
    out = CAL_ROOT / label
    out.mkdir(parents=True, exist_ok=True)
    det = SpoofDetector()  # usa solo il face detector
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        print("Webcam non disponibile"); return
    saved, capturing = 0, False
    print("SPAZIO = start/stop cattura, q = esci. Inquadra il volto", flush=True)
    while saved < n:
        ok, frame = cap.read()
        if not ok:
            break
        faces = det.detect_faces(frame)
        disp = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(disp, f"{label}: {saved}/{n} {'REC' if capturing else 'pausa'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("calibrazione", disp)
        if capturing and faces:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            crop = det._crop(frame, (x, y, w, h))
            cv2.imwrite(str(out / f"{int(time.time()*1000)}_{saved}.png"), crop)
            saved += 1
        k = cv2.waitKey(1) & 0xFF
        if k == ord(" "):
            capturing = not capturing
        elif k == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows()
    print(f"salvati {saved} frame in {out}")


def _build_df():
    rows = []
    for lab, name in [(0, "live"), (1, "spoof")]:
        for p in (CAL_ROOT / name).glob("*.png"):
            rows.append({"filepath": str(p), "label": lab})
    return pd.DataFrame(rows)


def finetune(epochs: int, lr: float):
    df = _build_df()
    if df.empty or df["label"].nunique() < 2:
        print("Servono frame sia 'live' sia 'spoof'. Esegui prima la cattura."); return
    print("frame calibrazione:", dict(df["label"].value_counts()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("outputs/cnn_best.pt", map_location=device, weights_only=False)
    model = build_model(ckpt["backbone"], pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])

    # i filepath sono gia' assoluti -> img_root vuoto
    ds = CelebASpoofDataset(df, build_train_transform(), img_root="")
    dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item()
        print(f"epoch {ep+1}/{epochs} loss={tot/len(dl):.4f}", flush=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "backbone": ckpt["backbone"], "threshold": 0.5},
               "outputs/cnn_calibrated.pt")
    print("salvato outputs/cnn_calibrated.pt -> usalo con webcam_demo.py --ckpt outputs/cnn_calibrated.pt")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture"); c.add_argument("--label", required=True, choices=["live", "spoof"])
    c.add_argument("--n", type=int, default=60); c.add_argument("--cam", type=int, default=0)
    f = sub.add_parser("finetune"); f.add_argument("--epochs", type=int, default=5)
    f.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    if args.cmd == "capture":
        capture(args.label, args.n, args.cam)
    else:
        finetune(args.epochs, args.lr)


if __name__ == "__main__":
    main()
