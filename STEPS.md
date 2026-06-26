# Fase 1

setup python tutto ok


# Fase 2

from data/celeba_spoof parquet to png to train
from data/celeba_spoof_test to png to test

Output dir is data/images/<split(train,test)>/<live|spoof>/<....>.png



even the metadata is put under

data/<split>.csv (filepath & label) for using non RAM Load to avoid OOM


---

data.py splits the data:

- 78k lines in total for training
- 67k is for train
- 12k (15%) is for validation (early stopping for threshold)


# Fase 3

Data augmentation

- We cut,flip orizzontal, color change, random jpeg recompression, remove some part of image


# Fase 4

Transform each image in numbers (feature vector) describing the texture.

For each pixel, LBP (Local Binary Pattern) see the closest 8 bit and compare it with center.

We do histogram.

We use LBP 2 times with 2 radius (8 bit and 16 bit)


We use YCbCr HSV. Spooffed image are seen better rather than RGB - YCbCr = Luminanza (Y) + crominanza blu (Cb) + crominanza rossa (Cr);
  - HSV = Tinta (H) + Saturazione (S) + Valore (V).


(La Prima fase della fase 4 crea feature vector per ogni immagine definendo una texture con istogrammi)


### Fase 4.5

In questa fase usiamo queste feature per allenare una SVM (Support Vector Machine, classificatore classico), a separare live da spoof.


Pesca 6000 live + 6000 spoof (Sottocampione bilanciato).

SVM is trained with RBF Kernel (curve tra le classi non solo rette).

EER is 7,69% usando test diversi (diversi soggetti), baseline che la CNN deve battere


# Fase 5


Definiamo il modello di CNN partendo da timm (Pythorch Image Models), usando resnet18 come backbone, (CNN a 18 livelli)


pretrained = true

num_classes=2. timm prende ResNet già addestrata e cambia l'ultimo layer con un layer a 2 uscite (live,spoof)


Immagine 224x224 -> ResNet-18 (Pre addestato su ImageNet) estrae feature -> ultimo layer a 2 uscite -> estra lo score di spoofness



### Fase 5.5 train


Usiamo solo 6 epoche per raggiungere la soglia precista. Ogni epoca misura EER su validation ->  se l'EER migliora salva i pesi -> se l'EER Non migliora per due epoche di fila si ferma. Continuare ad allenare fa fare overfitting e "imparare a memoria"

Arrivando a EER 3.69% AUC 0.994


# Fase 6
evaluate EER,AUC,APCER...


# Fase 7

L'idea è: provare il modello su un dataset mai visto con condizioni diverse (CASIA-FASD) (Cross dataset)

Stiamo usando la stessa threshold di CelebA su CASIA (Scenario realistico).

Mostrare le performance crollare


# Fase 8

Mix multiscorelevel SVM and CNN



  Method                     EER%    AUC      ACER%
  CNN (solo)                 3.69    0.994    3.69
  Fusion sum/mean            3.35    0.991    3.35     ← migliore EER
  Fusion product             3.50    0.9948   3.50     ← migliore AUC
  Fusion weighted (w=0.10)   5.14    0.987    5.14     ← PEGGIORA!
  Due conclusioni, entrambe da raccontare:


# IDEA FINO AD ADESSO


Abbiamo allenato due modelli diversi, SVM e CNN a performare e poi li abbiamo uniti

CNN NON È ADDESTRATA SU SVM

  
  





  
  
  




