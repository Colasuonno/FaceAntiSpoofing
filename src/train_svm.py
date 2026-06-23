"""
Fase 4 — Baseline classica: color-texture LBP + SVM (RBF).

- Fit su un subsample BILANCIATO del train (l'SVM RBF non scala a decine di migliaia
  di campioni); class_weight='balanced'.
- Valutazione sul TEST ufficiale (disgiunto): EER, ROC, ACER.
- Salva modello (scaler + svm) e gli score del test (per la fusione, Fase 8).
"""
from __future__ import annotations
import sys, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
sys.path.insert(0, "src")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from data import load_csv
from features_classic import extract_dataframe
from metrics import compute_eer, far_frr_at, iso_metrics, plot_roc, plot_det

IMG_ROOT = "data/images"
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)


def balanced_subsample(df, n_per_class, seed=42):
    rng = np.random.default_rng(seed)
    parts = []
    for lab in (0, 1):
        sub = df[df["label"] == lab]
        take = min(n_per_class, len(sub))
        parts.append(sub.sample(take, random_state=seed))
    out = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_per_class", type=int, default=6000)
    ap.add_argument("--test_max", type=int, default=0, help="0 = test completo")
    ap.add_argument("--n_jobs", type=int, default=4)
    args = ap.parse_args()

    t0 = time.time()
    train_df = load_csv("train")
    test_df = load_csv("test")

    fit_df = balanced_subsample(train_df, args.train_per_class)
    if args.test_max > 0:
        test_df = balanced_subsample(test_df, args.test_max // 2)
    print(f"fit su {len(fit_df)} img | test su {len(test_df)} img")

    print("estrazione feature train...", flush=True)
    Xtr = extract_dataframe(fit_df, IMG_ROOT, n_jobs=args.n_jobs)
    ytr = fit_df["label"].to_numpy()
    print(f"  Xtr {Xtr.shape}  ({time.time()-t0:.0f}s)", flush=True)

    print("estrazione feature test...", flush=True)
    Xte = extract_dataframe(test_df, IMG_ROOT, n_jobs=args.n_jobs)
    yte = test_df["label"].to_numpy()
    print(f"  Xte {Xte.shape}  ({time.time()-t0:.0f}s)", flush=True)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    print("fit SVM RBF...", flush=True)
    svm = SVC(C=10.0, gamma="scale", kernel="rbf", class_weight="balanced", cache_size=1000)
    svm.fit(Xtr_s, ytr)
    print(f"  fit done ({time.time()-t0:.0f}s), #SV={svm.n_support_}", flush=True)

    # score = "spoofness" (decision_function: positivo -> classe 1 = spoof)
    scores = svm.decision_function(Xte_s)

    eer, thr = compute_eer(scores, yte)
    far, frr = far_frr_at(scores, yte, thr)
    iso = iso_metrics(scores, yte, thr)

    print("\n==== RISULTATI BASELINE SVM (test) ====")
    print(f"EER  = {eer:.4f}  (soglia={thr:.3f})")
    print(f"FAR  = {far:.4f}  FRR = {frr:.4f}  @soglia EER")
    print(f"APCER={iso['APCER']:.4f}  BPCER={iso['BPCER']:.4f}  ACER={iso['ACER']:.4f}")

    plot_roc({"SVM color-LBP": (scores, yte)}, str(OUT / "roc_svm.png"), title="ROC — baseline SVM")
    plot_det({"SVM color-LBP": (scores, yte)}, str(OUT / "det_svm.png"), title="DET — baseline SVM")

    joblib.dump({"scaler": scaler, "svm": svm}, OUT / "svm_baseline.joblib")
    np.savez(OUT / "scores_svm_test.npz", scores=scores, labels=yte)
    with open(OUT / "metrics_svm.txt", "w") as f:
        f.write(f"EER={eer:.4f} thr={thr:.3f} FAR={far:.4f} FRR={frr:.4f} "
                f"APCER={iso['APCER']:.4f} BPCER={iso['BPCER']:.4f} ACER={iso['ACER']:.4f}\n")
    print(f"\nsalvati: svm_baseline.joblib, scores_svm_test.npz, roc_svm.png, det_svm.png")
    print(f"TEMPO TOTALE: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
