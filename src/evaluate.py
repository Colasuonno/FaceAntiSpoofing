"""
Fase 6 — Framework di valutazione e confronto.

Carica gli score salvati su test (baseline SVM e CNN), calcola tutte le metriche,
produce ROC/DET sovrapposte e una tabella riassuntiva (CSV + stampa).

Esegue anche i test di sanita' delle metriche (gate Fase 6):
  - predizioni perfette  -> EER = 0
  - predizioni casuali   -> EER ~ 0.5
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "src")

from sklearn.metrics import roc_auc_score
from metrics import compute_eer, far_frr_at, iso_metrics, hter_at, plot_roc, plot_det

OUT = Path("outputs")

SCORE_FILES = {
    "Baseline SVM (color-LBP)": OUT / "scores_svm_test.npz",
    "CNN (resnet18)": OUT / "scores_cnn_test.npz",
}


def sanity_checks():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 500 + [1] * 500)
    perfect = labels.astype(float)                    # score=label -> separazione perfetta
    randomp = rng.random(1000)                        # casuale
    eer_p, _ = compute_eer(perfect, labels)
    eer_r, _ = compute_eer(randomp, labels)
    print(f"[sanity] EER predizioni perfette = {eer_p:.3f} (atteso ~0)")
    print(f"[sanity] EER predizioni casuali  = {eer_r:.3f} (atteso ~0.5)")
    assert eer_p < 1e-6, "EER perfetto dovrebbe essere 0"
    assert 0.4 < eer_r < 0.6, "EER casuale dovrebbe essere ~0.5"
    print("[sanity] OK\n")


def evaluate_all():
    curves = {}
    rows = []
    for name, f in SCORE_FILES.items():
        if not f.exists():
            print(f"!! manca {f}, salto {name}")
            continue
        d = np.load(f)
        scores, labels = d["scores"], d["labels"]
        eer, thr = compute_eer(scores, labels)
        far, frr = far_frr_at(scores, labels, thr)
        iso = iso_metrics(scores, labels, thr)
        auc = roc_auc_score(labels, scores)
        rows.append({
            "Model": name, "N": len(labels),
            "EER%": round(eer * 100, 2), "AUC": round(auc, 4),
            "ACER%": round(iso["ACER"] * 100, 2),
            "APCER%": round(iso["APCER"] * 100, 2), "BPCER%": round(iso["BPCER"] * 100, 2),
            "thr@EER": round(thr, 3),
        })
        curves[name] = (scores, labels)

    df = pd.DataFrame(rows)
    print("==== TABELLA RIASSUNTIVA (test) ====")
    print(df.to_string(index=False))
    df.to_csv(OUT / "summary.csv", index=False)

    plot_roc(curves, str(OUT / "roc_compare.png"), title="ROC — confronto modelli")
    plot_det(curves, str(OUT / "det_compare.png"), title="DET — confronto modelli")
    print(f"\nsalvati: summary.csv, roc_compare.png, det_compare.png")


if __name__ == "__main__":
    sanity_checks()
    evaluate_all()
