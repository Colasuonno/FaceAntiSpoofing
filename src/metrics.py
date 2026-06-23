"""
Metriche di valutazione per anti-spoofing / biometria.

Convenzione: lo SCORE e' "spoofness" (alto = piu' probabile spoof).
La classe positiva e' SPOOF (label 1). Live = 0.

Implementate qui (usate da tutte le fasi):
  - compute_eer : Equal Error Rate + soglia
  - far_frr_at  : FAR/FRR a una soglia
  - iso metrics : APCER / BPCER / ACER (Fase 6/7)
  - plot ROC e DET
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- definizioni in chiave biometrica ---------------------------------------
# Trattiamo lo SPOOF come "impostore": un attacco accettato come live = False Acceptance.
# FAR = frazione di spoof classificati come live (sotto soglia di spoofness)
# FRR = frazione di live classificati come spoof (sopra soglia)


def compute_eer(scores: np.ndarray, labels: np.ndarray):
    """EER usando la curva ROC (positivo = spoof). Ritorna (eer, soglia)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    fpr, tpr, thr = roc_curve(labels, scores)  # positivo=1=spoof
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer, float(thr[idx])


def far_frr_at(scores: np.ndarray, labels: np.ndarray, threshold: float):
    """A soglia data su 'spoofness': predetto spoof se score>=threshold.
    FAR = spoof predetti live / tot spoof ; FRR = live predetti spoof / tot live."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred_spoof = scores >= threshold
    spoof = labels == 1
    live = labels == 0
    far = float(np.mean(~pred_spoof[spoof])) if spoof.any() else float("nan")  # spoof accettati come live
    frr = float(np.mean(pred_spoof[live])) if live.any() else float("nan")     # live rifiutati come spoof
    return far, frr


def iso_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float,
                attack_types: np.ndarray | None = None):
    """Metriche ISO/IEC 30107-3.
    APCER = max sui tipi di attacco della frazione di spoof classificati come live (bonafide).
    BPCER = frazione di bonafide (live) classificati come attacco.
    ACER  = (APCER + BPCER)/2.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred_attack = scores >= threshold
    live = labels == 0
    spoof = labels == 1
    bpcer = float(np.mean(pred_attack[live])) if live.any() else float("nan")
    if attack_types is None:
        apcer = float(np.mean(~pred_attack[spoof])) if spoof.any() else float("nan")
    else:
        attack_types = np.asarray(attack_types)
        rates = []
        for t in np.unique(attack_types[spoof]):
            m = spoof & (attack_types == t)
            rates.append(np.mean(~pred_attack[m]))
        apcer = float(np.max(rates)) if rates else float("nan")
    acer = (apcer + bpcer) / 2
    return {"APCER": apcer, "BPCER": bpcer, "ACER": acer}


def hter_at(scores, labels, threshold):
    far, frr = far_frr_at(scores, labels, threshold)
    return (far + frr) / 2


def plot_roc(curves: dict, path: str, title: str = "ROC"):
    """curves: {nome: (scores, labels)}. Salva una ROC con tutte le curve."""
    plt.figure(figsize=(6, 5))
    for name, (scores, labels) in curves.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.7)
    plt.xlabel("FPR (spoof accettati come live)")
    plt.ylabel("TPR (spoof rilevati)")
    plt.title(title); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_det(curves: dict, path: str, title: str = "DET"):
    """DET: FAR vs FRR in scala normale (semplificata)."""
    plt.figure(figsize=(6, 5))
    for name, (scores, labels) in curves.items():
        fpr, tpr, _ = roc_curve(labels, scores)
        fnr = 1 - tpr
        plt.plot(fpr, fnr, label=name)
    plt.xlabel("FAR"); plt.ylabel("FRR")
    plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
