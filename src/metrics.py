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
    plt.xlabel("FPR (genuine rejected as spoof)")
    plt.ylabel("TPR (spoof detected)")
    plt.title(title); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def plot_far_frr(scores, labels, path: str, title: str = "FAR/FRR vs soglia",
                 n: int = 400, logx: bool = False):
    """Curve FAR e FRR al variare della soglia, con il punto di incrocio (EER).
    logx=True: asse x logaritmico (utile quando gli score si ammassano vicino a 0/1,
    es. softmax di una CNN molto sicura -> l'incrocio EER sarebbe schiacciato a sinistra)."""
    scores = np.asarray(scores, dtype=float); labels = np.asarray(labels, dtype=int)
    if logx:
        ts = np.logspace(-4, 0, n)  # da 1e-4 a 1 (score softmax in [0,1])
    else:
        ts = np.linspace(scores.min(), scores.max(), n)
    fars, frrs = [], []
    for t in ts:
        far, frr = far_frr_at(scores, labels, t)
        fars.append(far); frrs.append(frr)
    fars, frrs = np.array(fars), np.array(frrs)
    idx = np.nanargmin(np.abs(fars - frrs))
    eer = float((fars[idx] + frrs[idx]) / 2); thr = float(ts[idx])
    plt.figure(figsize=(6, 5))
    plt.plot(ts, fars * 100, label="FAR (attacks accepted)", color="#c0392b")
    plt.plot(ts, frrs * 100, label="FRR (genuine rejected)", color="#2c6fbb")
    plt.axvline(thr, color="gray", ls="--", lw=0.8)
    plt.plot(thr, eer * 100, "ko", ms=6)
    plt.annotate(f"EER = {eer*100:.2f}%\n(thr = {thr:.3f})", (thr, eer * 100),
                 textcoords="offset points", xytext=(12, 8), fontsize=9)
    if logx:
        plt.xscale("log")
    plt.xlabel("threshold (spoofness score)"); plt.ylabel("error rate (%)")
    plt.title(title); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    return eer, thr


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
