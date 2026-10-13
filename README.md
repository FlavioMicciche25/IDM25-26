# Introduzione al Data Mining A.A. 2025-2026

La cartella contiene i materiali sviluppati durante il corso, organizzati in:

- First Classwork
- Second Classwork
- Progetto

Tutti i lavori seguono un approccio **object-oriented**, con `main.py` come orchestratore e classi modulari dedicate alle singole fasi della pipeline.

---

## First Classwork – Market Basket Analysis

**Dataset:** AnonymizedFidelity (transazioni di un supermercato)

**Attività principali:**

- preprocessing e creazione di variabili temporali;
- analisi delle frequenze sui livelli gerarchici dei prodotti;
- analisi stratificata per fascia oraria e mensile;
- estrazione di regole di associazione tramite:
  - Apriori;
  - FP-Growth;
- clustering dei clienti dopo riduzione dimensionale con **PCA**.

---

## Second Classwork – Classificazione clinica

**Dataset:** DAES (fogli ASD, GDD, Controlli)

**Pipeline:**

- caricamento dati multi-sheet;
- preprocessing e codifica delle feature;
- visualizzazione tramite **PCA**;
- addestramento di modelli di classificazione:
  - Decision Tree;
  - Random Forest;
  - SVM;
  - K-NN;
- tecniche di **Ensemble Learning**:
  - Bagging;
  - Boosting.

---

## Progetto – Analisi dei dati Fitbit e classificazione dei gruppi clinici

**Dataset:** Fitbit Daily + metadati dei pazienti.

**Workflow:**

- caricamento e integrazione dei dataset tramite identificativo del paziente;
- preprocessing dei dati e gestione dei valori mancanti;
- visualizzazione dei dati tramite **PCA a due componenti**;
- addestramento e confronto di modelli di classificazione:
  - Decision Tree;
  - Random Forest;
  - SVM;
  - K-NN;
- ottimizzazione degli iperparametri tramite **Grid Search**;
- valutazione delle prestazioni mediante:
  - Accuracy;
  - Classification Report;
  - Confusion Matrix;
  - ROC Curve e AUC;
- confronto tra modelli base e tecniche di **Ensemble Learning**.
---
### **Requisiti**

Per eseuire il progetto è necessario installare, preferibilmente in ambiente virtuale, le librerie presenti in 'requirements.txt':

```bash
pip install -r requirements.txt
```