"""
Fase 11.2-11.4 — Valutazione del modulo di riconoscimento su LFW (Labeled Faces in the Wild),
il benchmark pubblico standard per il face recognition.

Perche' LFW: il nostro dataset anti-spoofing (CelebA-Spoof cropped) NON ha le identita' dei
soggetti, quindi non possiamo valutarci il riconoscimento. LFW ha identita' etichettate e un
protocollo ufficiale -> lo usiamo come benchmark indipendente per il modulo di recognition.

  - VERIFICATION 1:1  (11.3): coppie genuine/impostore -> EER + soglia (riusata poi come
    soglia di accettazione nell'identification open-set e nella demo).
  - IDENTIFICATION 1:N (11.4): gallery (1 foto/persona) + probe -> rank-1 e curva CMC.

Output: outputs/recognition_eval.csv, outputs/roc_verification.png, outputs/cmc_identification.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "src")

from sklearn.datasets import fetch_lfw_pairs, fetch_lfw_people
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recognition import FaceEmbedder, cosine_similarity
from metrics import compute_eer

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)


def _to_pil(arr: np.ndarray) -> Image.Image:
    """Le immagini sklearn-LFW sono float [0,1] HxWx3 -> PIL uint8."""
    return Image.fromarray((arr * 255).astype(np.uint8))


@torch.no_grad()
def _embed_arr(embedder: FaceEmbedder, arr: np.ndarray):
    return embedder.embed_pil(_to_pil(arr))


# ---------------------------------------------------------------- VERIFICATION
def verification(embedder: FaceEmbedder):
    print("\n== 11.3 VERIFICATION 1:1 (LFW pairs) ==", flush=True)
    pairs = fetch_lfw_pairs(subset="test", color=True, resize=1.0,
                            slice_=(slice(0, 250), slice(0, 250)))
    X, y = pairs.pairs, pairs.target  # X:(N,2,H,W,3) y:1=stessa persona,0=diversa
    sims, labels, skipped = [], [], 0
    for i in range(len(X)):
        a = _embed_arr(embedder, X[i, 0]); b = _embed_arr(embedder, X[i, 1])
        if a is None or b is None:
            skipped += 1; continue
        sims.append(cosine_similarity(a, b)); labels.append(int(y[i]))
    sims, labels = np.array(sims), np.array(labels)
    # positivo = stessa persona (genuino), score = similarita'
    eer, thr = compute_eer(sims, labels)
    auc_v = roc_auc_score(labels, sims)
    acc = float(np.mean((sims >= thr).astype(int) == labels))
    print(f"coppie usate={len(sims)} (saltate {skipped}) | EER={eer*100:.2f}% "
          f"AUC={auc_v:.4f} acc@thr={acc*100:.2f}% | soglia sim={thr:.3f}", flush=True)

    fpr, tpr, _ = roc_curve(labels, sims)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"verification (AUC={auc(fpr,tpr):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.7)
    plt.xlabel("FAR (impostori accettati)"); plt.ylabel("TPR (genuini accettati)")
    plt.title("ROC — verification 1:1 (LFW)"); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "roc_verification.png", dpi=130); plt.close()
    return {"task": "verification 1:1 (LFW)", "EER%": round(eer*100, 2),
            "AUC": round(auc_v, 4), "thr_sim": round(thr, 3), "N": len(sims)}, thr


# ---------------------------------------------------------------- IDENTIFICATION
def identification(embedder: FaceEmbedder, min_faces=5, max_people=60):
    print("\n== 11.4 IDENTIFICATION 1:N (LFW) ==", flush=True)
    people = fetch_lfw_people(min_faces_per_person=min_faces, color=True, resize=1.0,
                              slice_=(slice(0, 250), slice(0, 250)))
    images, target, names = people.images, people.target, people.target_names
    # costruisci embedding per persona; per ogni identita': 1 foto in gallery, le altre probe
    gallery, gallery_id, probes, probe_id = [], [], [], []
    used_ids = sorted(set(target))[:max_people]
    for pid in used_ids:
        idx = np.where(target == pid)[0]
        embs = []
        for j in idx:
            e = _embed_arr(embedder, images[j])
            if e is not None: embs.append(e)
        if len(embs) < 2:
            continue
        gallery.append(embs[0]); gallery_id.append(pid)
        for e in embs[1:]:
            probes.append(e); probe_id.append(pid)
    G = np.vstack(gallery); gallery_id = np.array(gallery_id)
    P = np.vstack(probes); probe_id = np.array(probe_id)
    # matrice di similarita' probe x gallery
    S = P @ G.T  # embedding L2-normalizzati -> coseno
    order = np.argsort(-S, axis=1)             # ranking gallery per ogni probe
    ranked_ids = gallery_id[order]
    correct_rank = (ranked_ids == probe_id[:, None])
    # CMC: rank-k = frazione di probe il cui match corretto e' entro i primi k
    n_gal = G.shape[0]
    cmc = np.array([correct_rank[:, :k+1].any(axis=1).mean() for k in range(n_gal)])
    rank1 = float(cmc[0])
    print(f"identita'={len(gallery_id)} gallery, {len(probe_id)} probe | "
          f"rank-1={rank1*100:.2f}% rank-5={cmc[min(4,n_gal-1)]*100:.2f}%", flush=True)

    plt.figure(figsize=(6, 5))
    plt.plot(range(1, min(20, n_gal)+1), cmc[:min(20, n_gal)]*100, "b-o", ms=3)
    plt.xlabel("rank k"); plt.ylabel("identificazione corretta entro rank-k (%)")
    plt.title("CMC — identification 1:N (LFW)"); plt.grid(alpha=0.3); plt.ylim(0, 101)
    plt.tight_layout(); plt.savefig(OUT / "cmc_identification.png", dpi=130); plt.close()
    return {"task": "identification 1:N (LFW)", "rank1%": round(rank1*100, 2),
            "rank5%": round(float(cmc[min(4,n_gal-1)])*100, 2),
            "gallery": len(gallery_id), "probes": len(probe_id)}


if __name__ == "__main__":
    import pandas as pd
    emb = FaceEmbedder()
    row_v, thr = verification(emb)
    row_i = identification(emb)
    df = pd.DataFrame([row_v, row_i])
    print("\n==== RIASSUNTO RECOGNITION (LFW) ====")
    print(df.to_string(index=False))
    df.to_csv(OUT / "recognition_eval.csv", index=False)
    # salva la soglia di verifica per identification open-set / demo
    np.save(OUT / "recog_threshold.npy", np.array([thr]))
    print(f"\nsalvati: recognition_eval.csv, roc_verification.png, cmc_identification.png, "
          f"recog_threshold.npy (soglia sim={thr:.3f})")
