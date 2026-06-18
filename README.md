# 🎗️ PCam Metastatic Tissue Classification Suite

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://github.com/tensorflow/tensorflow)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Academic-Project](https://img.shields.io/badge/UNICAM-Deep_Learning_Project-purple.svg)](https://www.unicam.it/)

Questo repository contiene il progetto finale sviluppato per il corso di **Deep Learning** presso l'**Università degli Studi di Camerino (UNICAM)**. Il sistema affronta la classificazione binaria di campioni istopatologici metastatici utilizzando il benchmark **PatchCamelyon (PCam)**.

L'intero lavoro segue una logica di **complessità incrementale**, valutando l'impatto di diverse architetture, regolarizzazioni e *inductive bias* geometrici, con una priorità clinica orientata alla massimizzazione della **Recall (Sensibilità)** per minimizzare i Falsi Negativi in fase di screening[cite: 2].

📌 **Per un'analisi teorica e metodologica completa, consultare il report ufficiale in formato PDF incluso nel repository:** `DL_Project.pdf`[cite: 2].

---

## 🔬 Il Benchmark Clinico: PCam Dataset

Il dataset è derivato dalla celebre competizione **Camelyon16**[cite: 2]. Ciascuna patch ha una dimensione di $96\times96$ pixel in coordinate H&E (Ematossilina ed Eosina)[cite: 2]. Tuttavia, il target binario è determinato esclusivamente dalla presenza di tessuto tumorale nella regione centrale di $32\times32$ pixel[cite: 2]. 

* **Dimensione del Subset Sperimentale:** 80.000 patch per il Training, 8.000 per la Validazione (estratte garantendo un perfetto bilanciamento delle classi 50/50 e splitting a livello di paziente)[cite: 2].
* **Inference Configuration:** Per simulare uno screening clinico asettico, tutti i modelli custom sono stati addestrati con loss simmetrica (1:1) e valutati sul Test Set (32.768 patches) a una soglia unificata di **0.35** per testarne la sensibilità nativa[cite: 2].

---

## 🚀 Architetture a Confronto & Pipeline Logica

Il progetto analizza ed implementa quattro diversi approcci neurali:

1. **From-Scratch CNN (Baseline):** Una rete custom leggera con blocchi progressivi, fortemente regolarizzata tramite strati di `SpatialDropout2D` per contrastare l'overfitting causato dagli artefatti di colorazione dei vetrini[cite: 2].
2. **Deep ResNet-50:** Introduzione delle skip connection convoluzionali per scalare la profondità dell'estrazione delle feature mantenendo stabile il flusso del gradiente[cite: 2].
3. **Hybrid G-ResNet (G-CNN):** Una rete geometrica equivariante che mappa le feature sul gruppo di simmetria discreta $p4m$ (rotazioni di 90° e riflessioni), iniettando l'invarianza all'orientamento tipica dei tessuti biologici direttamente nei pesi dei kernel[cite: 2].
4. **Transfer Learning (DenseNet121):** Integrazione di feature out-of-domain pre-addestrate su ImageNet, ottimizzate tramite una pipeline di incremento del contrasto e un protocollo di *Fine-Tuning Globale Parziale* a due stadi con cost-weighting asimmetrico[cite: 2].

---

## 📊 Risultati Sperimentali (Test Set)

Tutti i modelli sono stati testati sul medesimo set indipendente di **32.768 campioni** sotto la soglia di inferenza clinica standardizzata a $0.35$:

| Modello | Accuracy | Macro F1-Score | Recall (Tumor) | Precision (Tumor) | AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **From-Scratch CNN** | 0.75 | 0.75 | 0.71 | 0.78 | 0.91 |
| **ResNet-50** | 0.75 | 0.75 | 0.65 | 0.82 | **0.93** |
| **Hybrid G-ResNet** | 0.83 | 0.83 | 0.77 | **0.88** | 0.92 |
| **DenseNet121 (Optimized)** | **0.84** | **0.84** | **0.85** | 0.83 | 0.91 |

### 📈 Key Insights

* **Il Paradosso di ResNet:** Nonostante registri la capacità discriminatoria globale più alta ($\text{AUC} = 0.93$), se addestrata da zero soffre di un forte bias conservativo che penalizza la Recall sul tumore ($0.65$), tendendo a rifugiarsi in minimi locali difensivi[cite: 2].
* **L'Intelligenza Geometrica della G-CNN:** Vincolare i filtri convoluzionali alle simmetrie $p4m$ agisce da potentissimo regolarizzatore, permettendo alla G-ResNet di superare la ResNet classica in Accuracy ($0.83$) e Precision ($0.88$) con una frazione dei parametri[cite: 2].
* **La Soluzione Clinica (DenseNet121):** Grazie al fine-tuning mirato e alla loss asimmetrica, il modello pre-addestrato abbatte drasticamente le diagnostic omissions spingendo la Recall all'**85%** e l'Accuracy all'**84%**, confermandosi lo strumento ideale per assistere il patologo[cite: 2].

---

## 🛠️ Tecnologie Utilizzate & Ottimizzazioni I/O

Per gestire la mole del dataset (circa 8GB) senza mandare in crash gli ambienti cloud volatili, sono state implementate soluzioni ingegneristiche mirate:
* **Out-of-Core Streaming (HDF5):** I dati vengono letti direttamente su disco tramite flussi HDF5 binari sequenziali, azzerando l'overflow della RAM host[cite: 2].
* **Pipeline Asincrone (`tf.data`):** La data augmentation viene eseguita in multithreading parallelo su CPU (`AUTOTUNE`) mentre la GPU calcola i gradienti del batch corrente, eliminando i colli di bottiglia dell'hardware[cite: 2].
* **Mixed Precision (`mixed_float16`):** Addestramento ottimizzato in virgola mobile a precisione mista per massimizzare il throughput dei Tensor Core delle GPU NVIDIA T4[cite: 2].
* **XLA Compilation:** Compilazione JIT dei grafi computazionali abilitata in fase di fine-tuning per fondere le operazioni sui kernel e accelerare i tempi di esecuzione[cite: 2].

---

## 👥 Gruppo di Lavoro
* **Tommaso Cosci** - Master's Degree Student in Computer Science (Artificial Intelligence), UNICAM[cite: 2]
* **Alessio Tartufoli** - Master's Degree Student in Computer Science (Artificial Intelligence), UNICAM[cite: 2]

---
*Progetto accademico sviluppato per il corso di Deep Learning - A.A. 2025/2026.*[cite: 2]
