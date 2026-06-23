"""
Fase 5 — Training CNN fine-tuned per anti-spoofing.

- Backbone pre-addestrato (timm), testa 2 classi.
- Augmentation (src/transforms.py), class weights per lo sbilanciamento ~2:1.
- AMP (mixed precision) per stare negli 8 GB di VRAM ed essere veloci.
- Early stopping su EER di validation; salva il best checkpoint.
- Valutazione finale sul TEST: EER, AUC, ACER, ROC/DET + score salvati per la fusione.

Target gate: battere la baseline (EER 7,69% / AUC 0,976).
"""
from __future__ import annotations
import sys, time, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
sys.path.insert(0, "src")

from data import train_val_split, load_csv, CelebASpoofDataset
from transforms import build_train_transform, build_eval_transform
from model import build_model, predict_scores
from metrics import compute_eer, far_frr_at, iso_metrics, plot_roc, plot_det

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)


def make_loader(df, transform, batch, shuffle, workers=4):
    ds = CelebASpoofDataset(df, transform)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, drop_last=shuffle, persistent_workers=workers > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--max_train", type=int, default=0, help="0 = tutto il train")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| backbone:", args.backbone, flush=True)

    tr_df, va_df = train_val_split()
    if args.max_train > 0:
        tr_df = tr_df.sample(args.max_train, random_state=42).reset_index(drop=True)
    test_df = load_csv("test")
    print(f"train={len(tr_df)} val={len(va_df)} test={len(test_df)}", flush=True)

    train_loader = make_loader(tr_df, build_train_transform(), args.batch, True)
    val_loader = make_loader(va_df, build_eval_transform(), 128, False)
    test_loader = make_loader(test_df, build_eval_transform(), 128, False)

    model = build_model(args.backbone, pretrained=True).to(device)

    # class weights (inverse frequency) per lo sbilanciamento
    n_live = int((tr_df["label"] == 0).sum()); n_spoof = int((tr_df["label"] == 1).sum())
    w = torch.tensor([(n_live + n_spoof) / (2 * n_live),
                      (n_live + n_spoof) / (2 * n_spoof)], dtype=torch.float32, device=device)
    print(f"class weights: live={w[0]:.3f} spoof={w[1]:.3f}", flush=True)
    criterion = nn.CrossEntropyLoss(weight=w)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    hist = {"train_loss": [], "val_eer": []}
    best_eer, best_state, since_best = 1.0, None, 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        running, nb = 0.0, 0
        te = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optim); scaler.update()
            running += loss.item(); nb += 1
        sched.step()
        train_loss = running / max(nb, 1)

        vs, vl = predict_scores(model, val_loader, device)
        val_eer, _ = compute_eer(vs, vl)
        hist["train_loss"].append(train_loss); hist["val_eer"].append(val_eer)
        print(f"epoch {ep}/{args.epochs} | loss={train_loss:.4f} | val_EER={val_eer:.4f} "
              f"| {time.time()-te:.0f}s (tot {time.time()-t0:.0f}s)", flush=True)

        if val_eer < best_eer:
            best_eer = val_eer
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= args.patience:
                print(f"early stopping (no improvement per {args.patience} epoche)", flush=True)
                break

    # ripristina best e valuta sul test
    model.load_state_dict(best_state)
    ts, tl = predict_scores(model, test_loader, device)
    eer, thr = compute_eer(ts, tl)
    far, frr = far_frr_at(ts, tl, thr)
    iso = iso_metrics(ts, tl, thr)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(tl, ts)

    print("\n==== RISULTATI CNN (test) ====")
    print(f"best val EER = {best_eer:.4f}")
    print(f"TEST: EER={eer:.4f} AUC={auc:.4f} | FAR={far:.4f} FRR={frr:.4f} "
          f"| APCER={iso['APCER']:.4f} BPCER={iso['BPCER']:.4f} ACER={iso['ACER']:.4f}")

    plot_roc({f"CNN {args.backbone}": (ts, tl)}, str(OUT / "roc_cnn.png"), title="ROC — CNN")
    plot_det({f"CNN {args.backbone}": (ts, tl)}, str(OUT / "det_cnn.png"), title="DET — CNN")
    torch.save({"state_dict": best_state, "backbone": args.backbone,
                "threshold": thr, "val_eer": best_eer}, OUT / "cnn_best.pt")
    np.savez(OUT / "scores_cnn_test.npz", scores=ts, labels=tl)

    # curva di training
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(hist["train_loss"], "b-o", label="train loss"); ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="b")
    ax2 = ax1.twinx(); ax2.plot(hist["val_eer"], "r-s", label="val EER")
    ax2.set_ylabel("val EER", color="r")
    plt.title("Training curve"); fig.tight_layout(); plt.savefig(OUT / "train_curve.png", dpi=130)

    with open(OUT / "metrics_cnn.txt", "w") as f:
        f.write(f"EER={eer:.4f} AUC={auc:.4f} thr={thr:.3f} FAR={far:.4f} FRR={frr:.4f} "
                f"APCER={iso['APCER']:.4f} BPCER={iso['BPCER']:.4f} ACER={iso['ACER']:.4f} "
                f"best_val_eer={best_eer:.4f}\n")
    print(f"\nsalvati: cnn_best.pt, scores_cnn_test.npz, roc_cnn.png, det_cnn.png, train_curve.png")
    print(f"TEMPO TOTALE: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
