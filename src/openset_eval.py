"""
Fase 11.6 — Test OPEN-SET sul tuo scenario reale:
  - ISCRIVI i soggetti di data/gallery/ (1 foto a testa); le altre foto restano come
    PROBE GENUINI (per verificare che il sistema li riconosca).
  - IMPOSTORI: le tue foto in data/impostors/ + un campione di LFW (persone garantite
    diverse). Devono essere RIFIUTATI ("SCONOSCIUTO").
Per ogni probe stampa: live/spoof + identita' decisa + similarita'.

Metriche:
  - tasso di identificazione corretta sui genuini
  - tasso di corretto rifiuto sugli impostori (1 - falsi accessi)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, "src")

from biometric_system import BiometricSystem
from sklearn.datasets import fetch_lfw_people

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_imgs(d: Path):
    return sorted([p for p in d.iterdir() if p.suffix.lower() in EXT]) if d.is_dir() else []


def annotate_save(path_in: str, res: dict, path_out: str):
    pil = BiometricSystem._pil_from_path(path_in)
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    h = bgr.shape[0]
    green, red, yellow = (0, 200, 0), (0, 0, 255), (0, 215, 255)
    if not res["live"]:
        color, txt = red, res["decision"]
    elif res["identity"]:
        color, txt = green, f"{res['decision']}  sim={res['best_sim']:.2f}"
    else:
        color, txt = yellow, f"{res['decision']}  sim={res.get('best_sim') or 0:.2f}"
    scale = max(0.7, bgr.shape[1] / 900)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1] - 1, bgr.shape[0] - 1), color, 6)
    cv2.putText(bgr, txt, (15, int(40 * scale)), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
    cv2.imwrite(path_out, bgr)


def main():
    sys_bio = BiometricSystem()
    print(f"soglia riconoscimento (sim) = {sys_bio.recog_threshold:.3f}\n")

    # ---------- enrollment + probe genuini ----------
    genuine_probes = []  # (path, vero_nome)
    for d in sorted(Path("data/gallery").glob("*")):
        imgs = list_imgs(d)
        if len(imgs) < 1:
            continue
        n = sys_bio.enroll(d.name, [str(imgs[0])])   # iscrivi con la 1a foto
        print(f"iscritto '{d.name}' ({n} foto). probe genuini: {len(imgs)-1}")
        for p in imgs[1:]:
            genuine_probes.append((str(p), d.name))
    print(f"gallery: {list(sys_bio.gallery.keys())}\n")

    # ---------- impostori: tue foto + campione LFW ----------
    impostor_imgs = [str(p) for p in list_imgs(Path("data/impostors"))]
    impostor_imgs += [str(p) for sub in Path("data/impostors").glob("*") if sub.is_dir()
                      for p in list_imgs(sub)]

    rows = []

    print("== PROBE GENUINI (atteso: identificato col nome giusto) ==")
    g_ok = 0
    for path, true_name in genuine_probes:
        r = sys_bio.identify_path(path)
        ok = (r["identity"] == true_name)
        g_ok += ok
        print(f"  {Path(path).name:28s} -> {r['decision']:35s} "
              f"[{'LIVE' if r['live'] else 'SPOOF'}] {'OK' if ok else 'X'}")
        rows.append({"tipo": "genuino", "file": Path(path).name, "atteso": true_name,
                     "live": r["live"], "decisione": r["decision"],
                     "sim": r.get("best_sim"), "corretto": ok})

    print("\n== IMPOSTORI tuoi (atteso: SCONOSCIUTO) ==")
    i_ok = 0; i_tot = 0
    for path in impostor_imgs:
        r = sys_bio.identify_path(path)
        correct = (r["identity"] is None)   # corretto se NON viene identificato come iscritto
        i_ok += correct; i_tot += 1
        print(f"  {Path(path).name:28s} -> {r['decision']:35s} "
              f"[{'LIVE' if r['live'] else 'SPOOF'}] {'OK' if correct else 'FALSO ACCESSO!'}")
        rows.append({"tipo": "impostore", "file": Path(path).name, "atteso": "SCONOSCIUTO",
                     "live": r["live"], "decisione": r["decision"],
                     "sim": r.get("best_sim"), "corretto": correct})

    # ---------- impostori LFW (campione) ----------
    print("\n== IMPOSTORI LFW (campione, atteso: SCONOSCIUTO) ==")
    people = fetch_lfw_people(min_faces_per_person=20, color=True, resize=1.0,
                              slice_=(slice(0, 250), slice(0, 250)))
    lfw_ok = 0; lfw_tot = 0; lfw_blocked = 0
    seen = set()
    for img, pid in zip(people.images, people.target):
        if pid in seen:
            continue
        seen.add(pid)
        if lfw_tot >= 50:
            break
        pil = Image.fromarray((img * 255).astype(np.uint8))
        r = sys_bio.identify_pil(pil)
        correct = (r["identity"] is None)
        lfw_ok += correct; lfw_tot += 1
        lfw_blocked += (not r["live"])
    print(f"  LFW impostori: {lfw_ok}/{lfw_tot} correttamente NON identificati "
          f"({lfw_blocked} bloccati da anti-spoof, gli altri rifiutati dal riconoscimento)")

    # ---------- riepilogo ----------
    print("\n==== RIEPILOGO OPEN-SET ====")
    if genuine_probes:
        print(f"Genuini identificati correttamente : {g_ok}/{len(genuine_probes)}")
    print(f"Impostori tuoi rifiutati           : {i_ok}/{i_tot}")
    print(f"Impostori LFW rifiutati            : {lfw_ok}/{lfw_tot}")
    falsi_accessi = (i_tot - i_ok) + (lfw_tot - lfw_ok)
    print(f"FALSI ACCESSI totali               : {falsi_accessi}")

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT / "openset_results.csv", index=False)

    # immagini annotate per il report: 1 genuino + 1 impostore
    if genuine_probes:
        r = sys_bio.identify_path(genuine_probes[0][0])
        annotate_save(genuine_probes[0][0], r, str(OUT / "identify_genuine.png"))
    if impostor_imgs:
        r = sys_bio.identify_path(impostor_imgs[0])
        annotate_save(impostor_imgs[0], r, str(OUT / "identify_impostor.png"))
    print("\nsalvati: openset_results.csv, identify_genuine.png, identify_impostor.png")


if __name__ == "__main__":
    main()
