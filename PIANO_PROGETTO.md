# Progetto — Face Anti-Spoofing (Presentation Attack Detection)

**Corso:** Biometric Systems — Sapienza
**Tipo:** modulo biometrico in Python (OpenCV + PyTorch), gruppo consigliato (2–3)
**Obiettivo:** costruire un modulo che, data un'immagine/un frame di volto, decide **LIVE vs SPOOF**,
con valutazione biometrica rigorosa e una **demo webcam in tempo reale**.

---

## 0. Obiettivi e deliverable

**Cosa consegniamo:**
- [ ] Codice Python riproducibile (repo ordinato + `requirements.txt` + README).
- [ ] Due classificatori: **baseline classica** (texture + SVM) e **deep** (CNN fine-tuned).
- [ ] Valutazione completa: **EER, APCER/BPCER/ACER, ROC, DET** + esperimento **cross-dataset**.
- [ ] **Demo live** (webcam) + script di **calibrazione** sui nostri dati.
- [ ] Report + slide per l'orale.

**Il "30" non è il modello, è l'analisi:** generalizzazione cross-dataset, confronto classico vs deep,
comportamento per tipo di attacco (print vs replay), eventuale fusione score-level.

---

## 1. Setup ambiente  ⏱️ ~mezza giornata

**Task**
- [ ] Creare venv: `python3 -m venv .venv && source .venv/bin/activate`
- [ ] Installare: `pip install torch torchvision opencv-python scikit-learn scikit-image numpy pandas matplotlib seaborn tqdm timm`
- [ ] Verificare la GPU.

**✅ Verifica (gate)**
```python
import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
# Deve stampare: True  <nome GPU>
import cv2, sklearn, skimage, timm  # nessun errore di import
```
Se `cuda.is_available()` è `False` → sistemare driver/CUDA prima di proseguire.

---

## 2. Dataset  ⏱️ ~1 giorno (gran parte è download)

**Dataset principale (training/test in-domain):** **CelebA-Spoof**
- Repo ufficiale: https://github.com/ZhangYuanhan-AI/CelebA-Spoof (download da Google Drive; è grande, ~75 GB).
- ⚠️ Strategia: scaricare e lavorare prima su un **subset** (es. 30–50k immagini bilanciate) per iterare veloce, poi scalare.

**Dataset secondario (solo test, per il cross-dataset):** scegliere UNO tra i più accessibili
- **CASIA-FASD**, **Replay-Attack (Idiap)**, **MSU-MFSD**, **NUAA**, **OULU-NPU**.
- (alcuni richiedono una richiesta di licenza accademica → inviarla SUBITO, possono volerci giorni).

**Task**
- [ ] Scaricare CelebA-Spoof (o subset) in `data/celeba_spoof/`.
- [ ] Inviare richiesta licenza per il dataset cross (parte in parallelo).
- [ ] Scrivere `src/data.py` che legge immagini + label (0=live, 1=spoof) e crea split **train/val/test per-soggetto** (mai lo stesso soggetto in train e test!).

**✅ Verifica (gate)**
- [ ] Script che stampa: n° immagini per split e **distribuzione live/spoof** (deve essere ragionevolmente bilanciata).
- [ ] Nessuna sovrapposizione di soggetti tra train e test (stampare l'intersezione → deve essere vuota).
- [ ] Visualizzare a campione 10 live + 10 spoof per controllare le label a occhio.

---

## 3. Preprocessing — face crop  ⏱️ ~1 giorno

**Task**
- [ ] `src/preprocess.py`: face detection (OpenCV DNN face detector o `cv2.CascadeClassifier`), crop quadrato del volto, resize a 224×224, salvataggio cache.
- [ ] Gestire i frame senza volto rilevato (scartare o log).

**✅ Verifica (gate)**
- [ ] Su 200 immagini a campione, il volto è croppato correttamente (montaggio visivo salvato in `outputs/crops_preview.png`).
- [ ] Percentuale di "volto non trovato" < ~5% (altrimenti rivedere il detector/threshold).

---

## 4. Baseline classica (texture + SVM)  ⏱️ ~1–2 giorni

> Questo è anche il pezzo "da slide": LBP / color texture, esattamente la teoria del corso.

**Task**
- [ ] `src/features_classic.py`: estrarre **LBP** (`skimage.feature.local_binary_pattern`) sui canali, opz. in spazio colore **HSV/YCbCr** (color texture). Concatenare in un feature vector.
- [ ] `src/train_svm.py`: standardizzare (`StandardScaler`) + **SVM** (`sklearn.svm.SVC(probability=True)`); grid search leggera su `C`/`gamma`.
- [ ] Salvare il modello (`joblib`).

**✅ Verifica (gate)**
- [ ] Training completa senza errori e produce uno **score continuo** per immagine (non solo 0/1).
- [ ] EER sul test in-domain calcolato e **sensato** (baseline texture: tipicamente EER ~10–25% a seconda del subset).
- [ ] Curva ROC salvata in `outputs/roc_svm.png`.

---

## 5. Modello deep (CNN fine-tuned)  ⏱️ ~2–3 giorni

**Task**
- [ ] `src/model.py`: backbone pre-addestrato (es. `timm.create_model('resnet18'/'mobilenetv3', pretrained=True, num_classes=2)`).
- [ ] `src/train_cnn.py`: DataLoader con **augmentation** (flip, color jitter, blur, JPEG compression, random resized crop) — fondamentale per la robustezza/cross-dataset.
- [ ] Loss `CrossEntropy`, ottimizzatore `AdamW`, early stopping su val, salvataggio best checkpoint.
- [ ] Lo score di spoof = probabilità softmax della classe spoof.

**✅ Verifica (gate)**
- [ ] La **loss di training scende** e la val accuracy sale (salvare la curva in `outputs/train_curve.png`).
- [ ] Il modello deep **batte la baseline SVM** in EER/ACER in-domain (se non lo fa → debug: label, normalizzazione, leakage).
- [ ] Inference su singola immagine funziona e restituisce uno score in [0,1].

---

## 6. Framework di valutazione  ⏱️ ~1–2 giorni (può partire in parallelo dalla Fase 4)

> Cuore del voto. Le metriche del corso + quelle ISO dell'anti-spoofing.

**Metriche da implementare in `src/metrics.py`:**
- [ ] **FAR/FRR** e **EER** (soglia dove FAR≈FRR).
- [ ] **APCER** = % di attacchi spoof classificati come live (per tipo di attacco, si prende il max).
- [ ] **BPCER** = % di volti live classificati come spoof.
- [ ] **ACER** = (APCER + BPCER) / 2.
- [ ] **HTER** = (FAR + FRR)/2 a soglia fissata (utile nel cross-dataset).
- [ ] Curve **ROC** e **DET** (plot con matplotlib).

**✅ Verifica (gate)**
- [ ] Test di sanità: su predizioni perfette EER=0; su predizioni random EER≈50%.
- [ ] Tabella riassuntiva (CSV) con EER/ACER/HTER per ciascun modello.
- [ ] ROC e DET salvate per SVM e CNN nello stesso grafico (confronto).

---

## 7. Esperimento clou — Cross-dataset  ⏱️ ~1–2 giorni

> Il pezzo che fa la differenza: il modello generalizza su un dominio mai visto?

**Task**
- [ ] Fissare la soglia sul **val di CelebA-Spoof**, poi valutare sul **dataset secondario** (CASIA-FASD/Replay-Attack/…).
- [ ] Riportare **HTER intra-dataset vs HTER cross-dataset** → il "gap".
- [ ] (Bonus) provare a ridurre il gap con più augmentation / training su mix di dati e mostrare il miglioramento.
- [ ] Analisi **per tipo di attacco** (print vs replay vs maschera): dove sbaglia di più?

**✅ Verifica (gate)**
- [ ] Tabella `outputs/cross_dataset.csv` con HTER intra vs cross (ci si aspetta un **peggioramento netto** cross — è il risultato atteso e va commentato).
- [ ] Almeno una contromisura testata con numeri a confronto.

---

## 8. (Estensione opzionale) Fusione score-level  ⏱️ ~1 giorno

> Collega al Cap. 11 del manuale (sistemi multibiometrici / fusione).

**Task**
- [ ] Normalizzare gli score (min-max o z-score) di SVM e CNN.
- [ ] Fondere (sum / weighted sum) e rivalutare EER/ACER.

**✅ Verifica (gate)**
- [ ] La fusione **migliora** (o qumeno pareggia) il miglior singolo modello → tabella di confronto. Se peggiora, analizzare perché (utile comunque a fini di report).

---

## 9. Demo live (webcam) + calibrazione  ⏱️ ~1–2 giorni

**Task**
- [ ] `demo/webcam_demo.py`: `cv2.VideoCapture(0)` → face detection → modello → overlay **REAL (verde) / SPOOF (rosso)** + score, in tempo reale.
- [ ] `demo/single_image.py`: dai un file immagine → stampa verdetto + score.
- [ ] `demo/calibrate.py`: cattura N frame **nostri** (live: la nostra faccia; spoof: nostra foto su telefono + stampa) e fa un **fine-tuning veloce** per adattare il modello al nostro setup d'esame.

**✅ Verifica (gate)**
- [ ] La demo gira **real-time** sulla GPU (≥ ~15–30 fps).
- [ ] Dopo la calibrazione coi nostri dati: faccia vera → REAL; foto/telefono/stampa → SPOOF, in modo **stabile** nelle condizioni di luce dell'esame.
- [ ] Provarla nell'ambiente reale dell'esame almeno una volta.

---

## 10. Report + slide  ⏱️ ~2 giorni

**Checklist contenuti report**
- [ ] Introduzione al problema (cos'è il PAD, attacchi 2D/3D, liveness — dalle slide del corso).
- [ ] Dataset e protocollo (split per-soggetto, niente leakage).
- [ ] Metodo: baseline classica + deep + (fusione).
- [ ] Metriche (definizioni APCER/BPCER/ACER/EER/HTER) e perché.
- [ ] Risultati: tabelle + ROC/DET, in-domain.
- [ ] **Cross-dataset gap** + contromisura (la parte forte).
- [ ] Analisi per tipo di attacco.
- [ ] Demo (screenshot) + limiti onesti.
- [ ] Conclusioni e lavori futuri.

**✅ Verifica finale (pronti per l'orale)**
- [ ] Un collega esterno clona il repo e riproduce un risultato seguendo il README.
- [ ] Sappiamo spiegare a voce ogni metrica e perché il cross-dataset peggiora.
- [ ] La demo è stata provata e funziona.

---

## Struttura repo consigliata
```
Project - Face Anti-Spoofing/
├── PIANO_PROGETTO.md        # questo file
├── README.md
├── requirements.txt
├── data/                    # dataset (NON committare: aggiungere a .gitignore)
├── src/
│   ├── data.py              # caricamento + split per-soggetto
│   ├── preprocess.py        # face crop
│   ├── features_classic.py  # LBP / color texture
│   ├── train_svm.py         # baseline
│   ├── model.py             # backbone CNN
│   ├── train_cnn.py         # training deep
│   ├── metrics.py           # EER/APCER/BPCER/ACER/HTER/ROC/DET
│   └── evaluate.py          # esegue tutte le valutazioni
├── demo/
│   ├── webcam_demo.py
│   ├── single_image.py
│   └── calibrate.py
└── outputs/                 # grafici, tabelle, checkpoint
```

---

## Divisione del lavoro (gruppo da 3)
- **Persona A** — Dati & preprocessing (Fasi 2–3) + baseline classica (Fase 4).
- **Persona B** — Modello deep (Fase 5) + demo (Fase 9).
- **Persona C** — Valutazione & metriche (Fase 6) + cross-dataset (Fase 7) + fusione (Fase 8).
- Report (Fase 10): tutti.
> Le Fasi 4/5/6 possono andare **in parallelo** una volta pronta la Fase 3.

---

## Timeline indicativa (~4–5 settimane)
| Settimana | Obiettivo |
|---|---|
| 1 | Fasi 0–3: setup, dataset, preprocessing → **gate: crop ok + split senza leakage** |
| 2 | Fasi 4 + 6: baseline SVM + framework metriche → **gate: EER/ROC baseline** |
| 3 | Fase 5: CNN fine-tuned → **gate: deep batte baseline** |
| 4 | Fasi 7–8: cross-dataset + fusione → **gate: tabella gap** |
| 5 | Fasi 9–10: demo + report → **gate: demo live + repo riproducibile** |

---

## Rischi & mitigazioni
- **Download CelebA-Spoof enorme** → iniziare con un subset bilanciato; scalare dopo.
- **Licenza dataset cross lenta** → richiederla il **giorno 1**; ripiego su NUAA/CASIA-FASD (più accessibili).
- **Data leakage (stesso soggetto in train/test)** → split rigorosamente **per-soggetto** (verificato in Fase 2).
- **Demo instabile sul nostro setup** → `calibrate.py` con i nostri dati + augmentation + prova nelle condizioni d'esame.
- **Modello "troppo bravo" in-domain ma crolla cross** → è il risultato atteso: trasformarlo nel punto forte del report, non nasconderlo.
