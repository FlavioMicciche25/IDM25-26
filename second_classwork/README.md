# IDM25-26
---
## Second classwork - Data Mining Introduction
---

## Descrizione del progetto

Il dataset impiegato per il **Second Classwork** è il **Dataset DAES**, organizzato in tre fogli distinti (**ASD**, **GDD**, **Controlli**), ciascuno rappresentativo di una diversa classe clinica.

Le fasi sviluppate durante il classwork sono:

- **Caricamento e gestione dei dati multi-sheet**
- **Preprocessing e pulizia del dataset**
- **Analisi esplorativa tramite PCA**
- **Addestramento e valutazione di modelli di classificazione**
- **Utilizzo di tecniche di ensemble (bagging e boosting)**

Durante la stesura del codice, su indicazione del professore, è stato seguito un approccio di **programmazione orientata agli oggetti**.

In particolare, il file `main.py` si occupa di istanziare gli oggetti e richiamarne i relativi metodi, mentre ciascun file contiene una specifica classe responsabile di una determinata fase della pipeline.

Seguendo l'ordine di utilizzo delle varie classi, vengono descritti di seguito i file e il loro contenuto.

---

### **1. dataloader.py**

- **Import utilizzate:** `pandas`, `numpy`

- **Classe:** `DataLoader`

- **Metodi:**
  - **__init__()**: costruttore della classe utilizzato per inizializzare i percorsi dei dataset e gli attributi interni.
  - **load()**: carica i dataset `patients` e `fitbit_daily`, verificando la correttezza dei file forniti.
  - **get_patients()**: restituisce il DataFrame dei pazienti.
  - **get_fitbit()**: restituisce il DataFrame Fitbit.
  - **printhead()**: visualizza le prime righe del dataset selezionato.
  - **printinfo()**: mostra le informazioni strutturali del DataFrame (tipi di dato e valori non nulli).

---

### **2. preprocessing.py**

- **Import utilizzate:** `pandas`, `train_test_split` (*sklearn.model_selection*)

- **Classe:** `Preprocessor`

- **Metodi:**
  - **__init__()**: inizializza i DataFrame contenenti i dati anagrafici e Fitbit.
  - **check_duplicates()**: verifica la presenza di record duplicati.
  - **drop_cols()**: rimuove le colonne specificate.
  - **encode()**: codifica variabili categoriche mediante mapping.
  - **datetoage()**: converte una data di nascita nell'età corrispondente.
  - **impute()**: gestisce i valori mancanti tramite mediana o valore nullo.
  - **drop_unmatched_patients()**: elimina dal dataset Fitbit i pazienti non presenti nel dataset anagrafico.
  - **merge()**: esegue il merge dei dataset utilizzando `PatientID`.

- **Funzioni esterne:**
  - **splitdata()**: suddivide il dataset in training set e test set utilizzando una divisione stratificata rispetto alla classe.

---

### **3. pcavisualizer.py**

- **Import utilizzate:** `pandas`, `seaborn`, `matplotlib`, `PCA`, `StandardScaler`

- **Classe:** `PCAVisualizer`

- **Metodi:**
  - **__init__()**: inizializza il numero di componenti principali, lo scaler e l'oggetto PCA.
  - **reduce()**: applica la standardizzazione e la riduzione dimensionale tramite PCA.
  - **plot()**: rappresenta i dati ridotti in uno spazio bidimensionale, distinguendo le classi tramite colori differenti.
  - **explainvariance()**: restituisce la quota di varianza spiegata dalle componenti principali.

---

### **4. classification_models.py**

- **Import utilizzate:** `scikit-learn`

- **Classe:** `Model`

- **Metodi:**
  - **_buildmodel()**: costruisce pipeline e griglia di ricerca degli iperparametri in base al classificatore scelto.
  - **train()**: addestra il modello utilizzando Grid Search con validazione incrociata.
  - **evaluate()**: valuta le prestazioni del modello sul test set.
  - **bestparams()**: restituisce i migliori iperparametri trovati.
  - **get_model()**: restituisce il miglior modello ottenuto dalla Grid Search.

- **Funzioni di ensemble:**
  - **bagging_decisiontree()**: applica Bagging a un Decision Tree.
  - **boosting_decisiontree()**: applica AdaBoost a un Decision Tree.
  - **bagging_knn()**: applica Bagging a un classificatore K-NN.
  - **bagging_svc()**: applica Bagging a un classificatore SVC calibrato.

Queste funzioni permettono di confrontare le prestazioni dei modelli base con tecniche di **Ensemble Learning**.

---

### **5. evaluation.py**

- **Import utilizzate:** `numpy`, `matplotlib`, `seaborn`, `sklearn.metrics`, `label_binarize`

- **Classe:** `Evaluator`

- **Metodi:**
  - **__init__()**: inizializza il modello, i dati di test e le predizioni.
  - **report()**: genera il Classification Report contenente le principali metriche di valutazione.
  - **confusion()**: genera e visualizza la matrice di confusione sotto forma di heatmap.

- **Funzioni esterne:**
  - **plot_roc_two_models()**: confronta due modelli tramite curva ROC micro-average e valore AUC associato.

---

## Requisiti

Per eseguire il progetto è consigliato utilizzare un ambiente virtuale e installare le dipendenze presenti nel file `requirements.txt`.

```bash
pip install -r requirements.txt
```