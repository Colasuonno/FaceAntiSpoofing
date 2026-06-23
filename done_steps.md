# Done steps — Face Anti-Spoofing

Riepilogo di quello che è stato fatto finora. Per il piano completo vedi `PIANO_PROGETTO.md`.

---

## ✅ Fase 1 — Setup ambiente
- Creata virtualenv del progetto in `.venv/` (Python 3.14.5).
- Installato: PyTorch 2.12.1 **+cu130**, torchvision, timm, OpenCV, scikit-learn, scikit-image, pandas, matplotlib, ecc.
- **GPU verificata**: NVIDIA RTX 2060 SUPER, 8 GB VRAM, `torch.cuda.is_available() == True`, test di calcolo su GPU ok.

## ✅ Fase 2 — Dataset
- Scelto **CelebA-Spoof** (versione *cropped*, volti già ritagliati) da Hugging Face:
  - train: `nguyenkhoa/antispoofing`
  - test:  `nguyenkhoa/celeba-spoof-for-face-antispoofing-test`
- Scaricato un **subset**: 15 shard di train + test completo (10 shard) ≈ 12 GB di parquet.
- **Estratti i PNG su disco** (i byte erano già PNG, nessuna ricompressione) + indice CSV.
- Numeri finali:
  | Split | Immagini | Live | Spoof |
  |---|---|---|---|
  | TRAIN | 79.183 | 33,2% | 66,8% |
  | → train (fit) | 67.306 | 33,0% | 67,0% |
  | → val (early stopping) | 11.877 | 34,1% | 65,9% |
  | TEST (disgiunto) | 66.787 | 29,7% | 70,3% |
- Gestite automaticamente ~955 righe corrotte (immagine None) scartandole.
- **Note rilevanti**:
  - Classi sbilanciate ~2:1 (spoof:live) → in training useremo class weights / sampling bilanciato.
  - La versione cropped non ha `subject_id` → train/val possono condividere soggetti (ok, il val serve solo per early stopping); il **TEST resta disgiunto** (split ufficiale CelebA-Spoof) → niente leakage sulla valutazione finale.

## ✅ Fase 3 — Preprocessing / transforms
- Immagini già croppate → preprocessing ridotto a resize/normalize + **augmentation**.
- Definite le transforms (normalizzazione ImageNet per i backbone pre-addestrati):
  - **train**: RandomResizedCrop, horizontal flip, color jitter, gaussian blur, **JPEG random**, random erasing (simulano le variazioni di print/replay → utili al cross-dataset).
  - **eval**: resize 224 + normalize.
- **Verificato**: tensori `(3,224,224)`, range normalizzato corretto, batch DataLoader `(64,3,224,224)`, throughput ~607 img/s con 4 workers, anteprima augmentation coerente.

---

## ✅ Fase 4 — Baseline classica (color-texture LBP + SVM)
- Feature **color-texture LBP** (metodo Boulkenafet 2015): istogrammi LBP multi-scala su
  YCbCr + HSV, per canale → **168 feature** per immagine.
- Fit **SVM RBF** (`class_weight='balanced'`) su subsample bilanciato (12.000 img).
- Valutazione sul **TEST completo** (66.787 img):
  | Metrica | Valore |
  |---|---|
  | **EER** | **7,69%** |
  | AUC (ROC) | 0,976 |
  | FAR / FRR @EER | 7,69% / 7,69% |
  | APCER / BPCER / **ACER** | 7,69% / 7,69% / **7,69%** |
- ✅ Gate superato (atteso 10–25% per baseline texture, ottenuto 7,7% → meglio).
- **Questo è il riferimento da battere**: la CNN della Fase 5 deve fare meglio di EER 7,7% / AUC 0,976.
- Creato anche `src/metrics.py` (EER, FAR/FRR, APCER/BPCER/ACER ISO, HTER, plot ROC/DET) —
  servirà anche nelle Fasi 6/7.
- Salvati gli score del test (`scores_svm_test.npz`) per la fusione in Fase 8.

## ✅ Fase 5 — Modello deep (CNN fine-tuned)
- Backbone **resnet18** pre-addestrato (timm), testa a 2 classi (live/spoof).
- Training su GPU con **AMP** (mixed precision), augmentation, **class weights** per lo sbilanciamento,
  early stopping su EER di validation, scheduler cosine. 5 epoche, ~24 min totali (~5 min/epoca).
- Training sano: train loss 0,19 → 0,06; **val EER fino a 0,54%** (nessun overfitting).
- Valutazione sul **TEST completo** e confronto con la baseline:
  | Metrica (test) | Baseline SVM | **CNN resnet18** |
  |---|---|---|
  | **EER** | 7,69% | **3,69%** |
  | AUC | 0,976 | **0,994** |
  | ACER | 7,69% | **3,69%** |
- ✅ Gate superato: la CNN **dimezza l'EER** della baseline.
- Salvati: checkpoint `cnn_best.pt` (con soglia), score test `scores_cnn_test.npz` (per fusione Fase 8).

## ✅ Fase 6 — Framework di valutazione e confronto
- `src/evaluate.py`: carica gli score salvati su test, calcola tutte le metriche e produce
  ROC/DET **sovrapposte** + tabella riassuntiva (`summary.csv`).
- **Test di sanita' metriche** (gate): predizioni perfette → EER 0,000 ✅ ; casuali → EER 0,529 ✅.
- Tabella riassuntiva (test 66.787 img):
  | Model | EER% | AUC | ACER% |
  |---|---|---|---|
  | Baseline SVM | 7,69 | 0,976 | 7,69 |
  | CNN resnet18 | 3,69 | 0,994 | 3,69 |
- Prodotti: `summary.csv`, `roc_compare.png`, `det_compare.png`.

## ✅ Fase 7 — Valutazione CROSS-DATASET (il pezzo "da voto alto")
- Secondo dataset (dominio mai visto): **CASIA-FASD** croppato (`Bahareh0281/CASIA-FASD-Cropped`),
  1655 volti (404 live, 1251 spoof) → `data/casia_fasd/` + `data/casia.csv`.
- Protocollo corretto: **soglia fissata sul VAL sorgente (CelebA)** e applicata sia INTRA sia CROSS
  (niente tuning sul target). Valutati entrambi i modelli.
- Risultati:
  | Model | Scenario | EER% | AUC | HTER% |
  |---|---|---|---|---|
  | CNN | INTRA (CelebA test) | 3,69 | 0,994 | 7,91 |
  | **CNN** | **CROSS (CASIA-FASD)** | **26,51** | **0,832** | **35,36** |
  | SVM | INTRA (CelebA test) | 7,69 | 0,976 | 8,50 |
  | **SVM** | **CROSS (CASIA-FASD)** | **34,43** | **0,730** | **44,37** |
- **Gap di generalizzazione documentato**: CNN HTER 7,9% → 35,4% (+27,5 punti); SVM +35,9 punti.
- **Tesi**: la CNN generalizza meglio della baseline classica anche cross-dataset (AUC 0,83 vs 0,73).
- Prodotti: `src/cross_dataset.py`, `cross_dataset.csv`, `roc_cross_casia.png`.

## ✅ Fase 8 — Fusione score-level (SVM + CNN) — collega al Cap. 11
- Problema "score non omogenei" (SVM decision_function illimitato vs CNN softmax [0,1])
  → normalizzazione min-max con parametri stimati sul VAL (no tuning sul test).
- Regole provate: sum, mean, product, max, min, weighted (peso tunato sul val).
- Risultati (test):
  | Metodo | EER% | AUC | ACER% |
  |---|---|---|---|
  | CNN (solo) | 3,69 | 0,994 | 3,69 |
  | **Fusion sum** | **3,35** | 0,991 | **3,35** |
  | Fusion product | 3,50 | **0,9948** | 3,50 |
  | Fusion weighted (w=0,10) | 5,14 | 0,987 | 5,14 |
- ✅ Gate superato: la fusione migliora il miglior singolo (EER 3,69 → 3,35; AUC → 0,9948 col product).
- **Insight**: il peso ottimizzato sul val **overfitta** (test 5,14%); la regola semplice a peso
  uguale generalizza meglio. Miglioramento contenuto perche' la CNN domina la baseline.
- Prodotti: `src/fusion.py`, `fusion.csv`, `roc_fusion.png`.

## ✅ Fase 9 — Demo (webcam + calibrazione)
- `src/infer.py`: `SpoofDetector` (carica CNN + face detection Haar bundled + predict/annotate).
- `demo/single_image.py`: immagine → verdetto LIVE/SPOOF + score + immagine annotata.
- `demo/webcam_demo.py`: webcam real-time, box verde REAL / rosso SPOOF + FPS.
- `demo/calibrate.py`: cattura frame propri (live/spoof) + fine-tuning veloce → `cnn_calibrated.pt`
  (riduce il domain gap sul setup d'esame).
- Verifiche headless (la webcam va lanciata sul PC dell'utente):
  - inferenza singola: LIVE→0.000, SPOOF→1.000 ✅
  - demo-path su 200 campioni: 92% accuracy @0.5 ✅ ; annotazione visiva ok.
- Nota: su crop strettissimi Haar usa l'intera immagine (fallback corretto); su frame webcam
  reali rileva il volto normalmente.
- **Come lanciarla**: `python demo/webcam_demo.py` (prima, consigliata, la calibrazione coi propri dati).

## File prodotti finora
```
Project - Face Anti-Spoofing/
├── PIANO_PROGETTO.md        # piano a fasi con verifiche
├── done_steps.md            # questo file
├── .venv/                   # ambiente Python
├── data/
│   ├── celeba_spoof_hf/         # parquet train scaricati
│   ├── celeba_spoof_test_hf/    # parquet test scaricati
│   ├── images/{train,test}/{live,spoof}/*.png   # 145.970 immagini estratte
│   ├── train.csv, test.csv      # indice filepath,label
├── src/
│   ├── data.py              # Dataset PyTorch (legge da disco) + split train/val
│   ├── extract.py           # estrazione parquet -> PNG + CSV
│   ├── transforms.py        # augmentation (train) + transform (eval)
│   ├── features_classic.py  # color-texture LBP (168 feature)
│   ├── metrics.py           # EER, FAR/FRR, APCER/BPCER/ACER, HTER, ROC/DET
│   ├── train_svm.py         # baseline SVM
│   ├── model.py             # backbone timm + predict_scores
│   ├── train_cnn.py         # training CNN (AMP, early stopping)
│   ├── evaluate.py          # framework valutazione/confronto (Fase 6)
│   ├── cross_dataset.py     # valutazione cross-dataset (Fase 7)
│   ├── fusion.py            # fusione score-level (Fase 8)
│   └── infer.py             # SpoofDetector per la demo (Fase 9)
├── demo/
│   ├── single_image.py      # demo su singola immagine
│   ├── webcam_demo.py       # demo webcam real-time
│   └── calibrate.py         # cattura propri dati + fine-tuning
└── outputs/
    ├── crops_preview.png    # anteprima live vs spoof
    ├── aug_preview.png      # anteprima augmentation
    ├── svm_baseline.joblib  # modello baseline (scaler+svm)
    ├── scores_svm_test.npz  # score test baseline (per fusione Fase 8)
    ├── roc_svm.png, det_svm.png, metrics_svm.txt
    ├── cnn_best.pt          # checkpoint CNN (state_dict + soglia)
    ├── scores_cnn_test.npz  # score test CNN (per fusione Fase 8)
    ├── roc_cnn.png, det_cnn.png, train_curve.png, metrics_cnn.txt
    ├── summary.csv, roc_compare.png, det_compare.png   # Fase 6
    ├── cross_dataset.csv, roc_cross_casia.png          # Fase 7
    ├── fusion.csv, roc_fusion.png                      # Fase 8
    ├── demo_live.png, demo_spoof.png                   # Fase 9 (immagini annotate)
```
(in `src/` anche: `evaluate.py` (Fase 6), `cross_dataset.py` (Fase 7), `fusion.py` (Fase 8).
 in `data/` anche: `casia_fasd/` + `casia.csv` (dataset cross).)

## Dettagli tecnici da ricordare
- **Python 3.14 + DataLoader**: usa `forkserver`, quindi gli script con `num_workers>0` devono essere **file .py reali** (non heredoc/stdin).
- **OOM**: caricare molti shard parquet (con i PNG dentro) in RAM satura i 15 GB → per questo si lavora con i PNG su disco.

---

## Stato: Fasi 1→9 completate ✅

## ⏭️ Prossimi step
- **Fase 10 — Report + slide** (ultima fase del piano).
- **(Bonus, opzionale) — Ridurre il gap cross-dataset**: piu' augmentation / training su mix
  di dati e rimisurare l'HTER cross.
