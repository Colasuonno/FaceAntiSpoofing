"""
Fase 8 — Fusione SCORE-LEVEL (Baseline SVM + CNN). Collega al Cap. 11 del manuale.

Problema (da slide): gli score dei due matcher NON sono omogenei
  - SVM: decision_function (illimitato)
  - CNN: softmax della classe spoof in [0,1]
=> serve NORMALIZZAZIONE prima di fondere.

Protocollo rigoroso:
  - parametri di normalizzazione (min-max) e peso della weighted-sum stimati sul VAL sorgente;
  - valutazione finale sul TEST (score gia' salvati e allineati).

Regole di fusione provate: sum, mean, product, max, min, weighted-sum (peso tunato sul val).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, "src")

from sklearn.metrics import roc_auc_score
import joblib
from data import train_val_split, CelebASpoofDataset
from transforms import build_eval_transform
from model import build_model, predict_scores
from features_classic import extract_dataframe
from metrics import compute_eer, iso_metrics, plot_roc

OUT = Path("outputs")
CELEBA_ROOT = "data/images"


def minmax_fit(x):
    return float(np.min(x)), float(np.max(x))


def minmax_apply(x, lo, hi):
    return np.clip((np.asarray(x) - lo) / (hi - lo + 1e-12), 0, 1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- VAL: stessi campioni per i due modelli (per tunare norm + peso) ----
    _, val_df = train_val_split()
    val_sub = val_df.sample(min(5000, len(val_df)), random_state=0).reset_index(drop=True)
    y_val = val_sub["label"].to_numpy()

    # CNN val
    ckpt = torch.load(OUT / "cnn_best.pt", map_location=device, weights_only=False)
    cnn = build_model(ckpt["backbone"], pretrained=False).to(device)
    cnn.load_state_dict(ckpt["state_dict"])
    loader = DataLoader(CelebASpoofDataset(val_sub, build_eval_transform(), img_root=CELEBA_ROOT),
                        batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    cnn_val, _ = predict_scores(cnn, loader, device)

    # SVM val
    pack = joblib.load(OUT / "svm_baseline.joblib")
    scaler, svm = pack["scaler"], pack["svm"]
    svm_val = svm.decision_function(scaler.transform(extract_dataframe(val_sub, CELEBA_ROOT, n_jobs=4)))

    # fit normalizzatori sul val
    c_lo, c_hi = minmax_fit(cnn_val)
    s_lo, s_hi = minmax_fit(svm_val)
    cnn_val_n = minmax_apply(cnn_val, c_lo, c_hi)
    svm_val_n = minmax_apply(svm_val, s_lo, s_hi)

    # tuning peso weighted-sum sul val (min EER)
    best_w, best_e = 0.5, 1.0
    for w in np.linspace(0, 1, 21):
        e, _ = compute_eer(w * cnn_val_n + (1 - w) * svm_val_n, y_val)
        if e < best_e:
            best_e, best_w = e, w
    print(f"peso ottimo (val): w_cnn={best_w:.2f} (val EER fusa={best_e:.4f})")

    # ---- TEST: score salvati (allineati) ----
    dc = np.load(OUT / "scores_cnn_test.npz"); cnn_te, y_te = dc["scores"], dc["labels"]
    ds = np.load(OUT / "scores_svm_test.npz"); svm_te, y_te2 = ds["scores"], ds["labels"]
    assert np.array_equal(y_te, y_te2), "le label di test dei due modelli non coincidono!"

    cnn_te_n = minmax_apply(cnn_te, c_lo, c_hi)
    svm_te_n = minmax_apply(svm_te, s_lo, s_hi)

    rules = {
        "SVM (solo)": svm_te_n,
        "CNN (solo)": cnn_te_n,
        "Fusion sum": cnn_te_n + svm_te_n,
        "Fusion mean": (cnn_te_n + svm_te_n) / 2,
        "Fusion product": cnn_te_n * svm_te_n,
        "Fusion max": np.maximum(cnn_te_n, svm_te_n),
        "Fusion min": np.minimum(cnn_te_n, svm_te_n),
        f"Fusion weighted (w={best_w:.2f})": best_w * cnn_te_n + (1 - best_w) * svm_te_n,
    }

    import pandas as pd
    rows = []
    for name, sc in rules.items():
        eer, thr = compute_eer(sc, y_te)
        iso = iso_metrics(sc, y_te, thr)
        rows.append({"Method": name, "EER%": round(eer * 100, 2),
                     "AUC": round(roc_auc_score(y_te, sc), 4),
                     "ACER%": round(iso["ACER"] * 100, 2)})
    df = pd.DataFrame(rows)
    print("\n==== FUSIONE SCORE-LEVEL (test) ====")
    print(df.to_string(index=False))
    df.to_csv(OUT / "fusion.csv", index=False)

    best_fusion = min([r for r in rows if r["Method"].startswith("Fusion")], key=lambda r: r["EER%"])
    print(f"\nmigliore fusione: {best_fusion['Method']} -> EER {best_fusion['EER%']}% (CNN sola: "
          f"{[r for r in rows if r['Method']=='CNN (solo)'][0]['EER%']}%)")

    plot_roc({"CNN (solo)": (cnn_te_n, y_te),
              "SVM (solo)": (svm_te_n, y_te),
              best_fusion["Method"]: (rules[best_fusion["Method"]], y_te)},
             str(OUT / "roc_fusion.png"), title="ROC — fusione score-level")
    print("salvati: fusion.csv, roc_fusion.png")


if __name__ == "__main__":
    main()
