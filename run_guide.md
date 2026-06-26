# Run guide — Demo del sistema biometrico

Guida ai comandi per lanciare **tutti** gli script della demo (anti-spoofing e
sistema completo anti-spoof + riconoscimento).

---

## 0. Regole valide per tutto

1. **Esegui sempre dalla cartella del progetto** (quella che contiene `src/`,
   `demo/`, `data/`, `outputs/`):
   ```bash
   cd "Project - Face Anti-Spoofing"
   ```
   Gli script usano percorsi relativi (`src/`, `data/`, `outputs/`): lanciati da
   un'altra cartella non trovano i moduli.

2. **Attiva l'ambiente virtuale** (Python 3.14, dipendenze già installate):
   ```bash
   source .venv/bin/activate
   ```
   In alternativa, prefissa ogni comando con `.venv/bin/python` invece di
   `python`.

3. **Webcam**: le demo `*_webcam.py` e `calibrate.py` aprono una finestra
   OpenCV → servono uno **schermo** e una **webcam** (non funzionano in modalità
   headless / SSH senza display). Si esce con il tasto **`q`**.

4. **Prima esecuzione del riconoscimento**: `identify_*.py` scaricano una volta
   i pesi di MTCNN + FaceNet (VGGFace2) → serve **connessione internet** al
   primo avvio; poi restano in cache.

5. **File modello necessari** (già presenti in `outputs/`):
   - `cnn_best.pt` — CNN anti-spoofing
   - `recog_threshold.npy` — soglia di riconoscimento calibrata su LFW

---

## 1. Anti-spoofing su una singola immagine

Verdetto LIVE/SPOOF + score, salva l'immagine annotata.

```bash
python demo/single_image.py <percorso_immagine>
```
Opzioni:
```bash
python demo/single_image.py foto.jpg --threshold 0.5 --ckpt outputs/cnn_best.pt --out outputs/demo_single.png
```
- `--threshold` soglia di spoofness (default `0.5`; più alta = più permissivo).
- `--out` dove salvare l'immagine annotata.

---

## 2. Anti-spoofing in tempo reale (webcam)

Box **verde = REAL**, **rosso = SPOOF**, con FPS.

```bash
python demo/webcam_demo.py
```
Opzioni:
```bash
python demo/webcam_demo.py --cam 0 --threshold 0.5 --ckpt outputs/cnn_best.pt
```
- `--cam` indice webcam (prova `1`, `2` se la `0` non è quella giusta).
- **Prova**: mostra la tua faccia (REAL), poi inquadra la tua foto sul telefono
  o una stampa (SPOOF).

> Suggerimento: per stabilità sul tuo setup d'esame, esegui prima la
> **calibrazione** (sezione 5) e poi usa `--ckpt outputs/cnn_calibrated.pt`.

---

## 3. Sistema completo su una singola immagine (anti-spoof + identità)

Iscrive i soggetti della gallery, poi sull'immagine stampa LIVE/SPOOF e, se
live, `IDENTIFICATO:<nome>` oppure `SCONOSCIUTO`.

```bash
python demo/identify_image.py <percorso_immagine>
```
Opzioni:
```bash
python demo/identify_image.py foto.jpg --gallery data/gallery --out outputs/identify_demo.png
```

---

## 4. Sistema completo in tempo reale (webcam)

Sul volto:
- **rosso "SPOOF"** → attacco bloccato
- **verde "Benvenuto, \<nome\>"** → iscritto riconosciuto
- **giallo "SCONOSCIUTO"** → persona live ma non in gallery

```bash
python demo/identify_webcam.py
```
Opzioni:
```bash
python demo/identify_webcam.py --cam 0 --gallery data/gallery --spoof-th 0.5 --recog-th 0.4
```
- `--spoof-th` soglia anti-spoof.
- `--recog-th` override della soglia di similarità coseno per il riconoscimento
  (default: quella calibrata in `recog_threshold.npy`).

### Come aggiungere/cambiare i soggetti iscritti
La gallery è una cartella con **una sottocartella per soggetto**, ognuna con una
o più foto del volto:
```
data/gallery/
├── andreea/   (2 foto)
├── giovanni/  (2 foto)
└── <nuovo_nome>/   <-- crea questa cartella e mettici le sue foto
```
I soggetti vengono iscritti **automaticamente** all'avvio leggendo `--gallery`.

---

## 5. (Opzionale) Calibrazione anti-spoof sul tuo setup

Riduce il *domain gap* (luce/webcam d'esame) facendo un fine-tuning veloce sui
tuoi frame. Consigliata prima delle demo webcam.

**Passo 1 — cattura** (premi **SPAZIO** per avviare/fermare, **`q`** per uscire):
```bash
python demo/calibrate.py capture --label live  --n 60
python demo/calibrate.py capture --label spoof --n 60
```
- `live`: la tua faccia reale davanti alla webcam.
- `spoof`: una tua foto mostrata su telefono / stampa.

**Passo 2 — fine-tuning** (produce `outputs/cnn_calibrated.pt`):
```bash
python demo/calibrate.py finetune --epochs 5
```

**Passo 3 — usa il modello calibrato** nelle demo:
```bash
python demo/webcam_demo.py --ckpt outputs/cnn_calibrated.pt
```

---

## Riepilogo rapido

| Obiettivo | Comando |
|---|---|
| Anti-spoof su immagine | `python demo/single_image.py foto.jpg` |
| Anti-spoof webcam | `python demo/webcam_demo.py` |
| Sistema completo su immagine | `python demo/identify_image.py foto.jpg` |
| Sistema completo webcam | `python demo/identify_webcam.py` |
| Calibrazione (cattura) | `python demo/calibrate.py capture --label live --n 60` |
| Calibrazione (fine-tune) | `python demo/calibrate.py finetune --epochs 5` |

> Ricorda: **`cd "Project - Face Anti-Spoofing"`** + **`source .venv/bin/activate`** prima di tutto.
