"""
Fase 7 — Valutazione CROSS-DATASET (il pezzo "da voto alto").

Protocollo realistico:
  - SORGENTE  = CelebA-Spoof. La soglia di decisione viene fissata sul VAL sorgente (EER).
  - Si valuta poi:
      * INTRA  = CelebA-Spoof test            (stesso dominio)
      * CROSS  = CASIA-FASD                    (dominio mai visto)
    applicando la STESSA soglia sorgente -> HTER cross = misura del gap di generalizzazione.
  - Riportiamo anche l'EER "proprio" su CASIA (best-case con soglia ri-tarata) per separare
    lo "shift di soglia" dal vero "gap di feature".

Vale per entrambi i modelli: Baseline SVM (color-LBP) e CNN (resnet18).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, "src")

from sklearn.metrics import roc_auc_score
from data import train_val_split, load_csv, CelebASpoofDataset
from transforms import build_eval_transform
from model import build_model, predict_scores
from features_classic import extract_dataframe
from metrics import compute_eer, hter_at, iso_metrics, plot_roc

OUT = Path("outputs")
CELEBA_ROOT = "data/images"
CASIA_ROOT = "data/casia_fasd"


def cnn_scores_on(df, img_root, device, model, workers=4):
    ds = CelebASpoofDataset(df, build_eval_transform(), img_root=img_root)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=workers, pin_memory=True)
    return predict_scores(model, loader, device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- carica dati ----------
    _, val_df = train_val_split()
    casia_df = pd.read_csv("data/casia.csv")
    print(f"CelebA val={len(val_df)} | CASIA={len(casia_df)} "
          f"(live={int((casia_df.label==0).sum())} spoof={int((casia_df.label==1).sum())})")

    results = []  # righe tabella

    # ================= CNN =================
    ckpt = torch.load(OUT / "cnn_best.pt", map_location=device, weights_only=False)
    cnn = build_model(ckpt["backbone"], pretrained=False).to(device)
    cnn.load_state_dict(ckpt["state_dict"])

    # soglia sorgente da CelebA val
    vs, vl = cnn_scores_on(val_df, CELEBA_ROOT, device, cnn)
    _, src_thr_cnn = compute_eer(vs, vl)

    # intra (CelebA test) — riuso score salvati
    d = np.load(OUT / "scores_cnn_test.npz"); cs, cl = d["scores"], d["labels"]
    eer_in, _ = compute_eer(cs, cl)
    results.append(["CNN", "INTRA (CelebA test)", eer_in, roc_auc_score(cl, cs),
                    hter_at(cs, cl, src_thr_cnn)])
    # cross (CASIA)
    xs, xl = cnn_scores_on(casia_df, CASIA_ROOT, device, cnn)
    eer_x, _ = compute_eer(xs, xl)
    results.append(["CNN", "CROSS (CASIA-FASD)", eer_x, roc_auc_score(xl, xs),
                    hter_at(xs, xl, src_thr_cnn)])
    cnn_casia = (xs, xl)

    # ================= SVM =================
    pack = joblib.load(OUT / "svm_baseline.joblib")
    scaler, svm = pack["scaler"], pack["svm"]

    # soglia sorgente: feature su un subsample del val
    val_sub = val_df.sample(min(4000, len(val_df)), random_state=0)
    Xval = scaler.transform(extract_dataframe(val_sub, CELEBA_ROOT, n_jobs=4))
    sval = svm.decision_function(Xval)
    _, src_thr_svm = compute_eer(sval, val_sub["label"].to_numpy())

    # intra (CelebA test) — score salvati
    d = np.load(OUT / "scores_svm_test.npz"); ss, sl = d["scores"], d["labels"]
    eer_in_s, _ = compute_eer(ss, sl)
    results.append(["SVM", "INTRA (CelebA test)", eer_in_s, roc_auc_score(sl, ss),
                    hter_at(ss, sl, src_thr_svm)])
    # cross (CASIA)
    Xc = scaler.transform(extract_dataframe(casia_df, CASIA_ROOT, n_jobs=4))
    scs = svm.decision_function(Xc)
    scl = casia_df["label"].to_numpy()
    eer_x_s, _ = compute_eer(scs, scl)
    results.append(["SVM", "CROSS (CASIA-FASD)", eer_x_s, roc_auc_score(scl, scs),
                    hter_at(scs, scl, src_thr_svm)])
    svm_casia = (scs, scl)

    # ---------- tabella ----------
    df = pd.DataFrame(results, columns=["Model", "Scenario", "EER", "AUC", "HTER@src_thr"])
    df["EER%"] = (df["EER"] * 100).round(2)
    df["HTER%"] = (df["HTER@src_thr"] * 100).round(2)
    df["AUC"] = df["AUC"].round(4)
    show = df[["Model", "Scenario", "EER%", "AUC", "HTER%"]]
    print("\n==== CROSS-DATASET (soglia fissata su CelebA val) ====")
    print(show.to_string(index=False))
    show.to_csv(OUT / "cross_dataset.csv", index=False)

    # gap
    for mdl in ["CNN", "SVM"]:
        a = df[(df.Model == mdl) & (df.Scenario.str.startswith("INTRA"))]["HTER@src_thr"].values[0]
        b = df[(df.Model == mdl) & (df.Scenario.str.startswith("CROSS"))]["HTER@src_thr"].values[0]
        print(f"  {mdl}: HTER intra={a*100:.2f}%  ->  cross={b*100:.2f}%  (gap +{(b-a)*100:.2f} punti)")

    plot_roc({"CNN @CASIA": cnn_casia, "SVM @CASIA": svm_casia},
             str(OUT / "roc_cross_casia.png"), title="ROC cross-dataset (CASIA-FASD)")
    print("\nsalvati: cross_dataset.csv, roc_cross_casia.png")


if __name__ == "__main__":
    main()
