# **IDM25-26**
---
## Fitbit Daily - Data Mining Introduction Project
---

## Obiettivo e Dataset

Il progetto ha come scopo la **classificazione dei pazienti** in diversi gruppi clinici, tra cui controllo sani e patologie neuro-degenerative. Nel dettaglio:

- **Alzehimer (AD);**
- **Parkinson (PD);**
- **Sclerosi Multipla (SM);**
- **Controllo Sano (CONT).**

I dataset impiegati per il data mining sono:

- **patients.txt;**
- **fitbit_daily.txt.**

### **1 patients.txt**

Contiene i metadati dei pazienti, uno per riga.
Le colonne principali includono:

- **PatientID:** identificativo univoco del paziente
- **GruppoLabel:** label usata per la classificazione.
- **Altri metadati:** Centro, Gruppo, PatologiaPrincipale, Sesso, DataNascita;

Questo dataset fornisce la **classe (target)** per ogni paziente.

### **2 fitbit_daily.txt**

Contiene le misure giornaliere registrate dal Fitbit per ogni paziente. Ogni paziente utilizza il fitbit per circa 5 giorni. Le feature includono:

- **Attività fisica:** Calories, ActivityCalories, Distance, Steps, RestingHeartRate, Sedentary / LightlyActive /Moderately Active / VeryActiveMinutes.

- **Zone cardiache:** con calorie e minutaggio per ogni zona (BelowFatBurn, FatBurn, Cardio, Peak);

- **Sonno**:
    - **Parametri generali:** VO2Max, NightlySkinTemperature,MinutesAfterWakeup, SleepEfficiency, MinutesToFallAsleep, TimeInBed;
    - **Parametri per ogni fase:** conteggio e minutaggio in LightSleep, DeepSleep, REMSleep o Awake;
    - **Parametri di respirazione:** BreathingRate (complessivo e specifico per ogni fase del sonno).

Questo dataset fornisce le **feature** utilizzate per **addestrare i modelli di classificazione**.

---

## Classi implementate

### **1. dataloader.py**

- **Classe:** `DataLoader`

- **Metodi:**
  - **load()**: Carica i dataset `patients` e `fitbit_daily` dai rispettivi percorsi e verifica che i nomi dei file siano corretti.
  - **get_patients()**: Restituisce il dataset dei pazienti precedentemente caricato.
  - **get_fitbit()**: Restituisce il dataset Fitbit precedentemente caricato.
  - **printhead()**: Visualizza le prime righe del dataset selezionato.
  - **printinfo()**: Mostra le informazioni strutturali del dataset.

### **2. preprocessing.py**

- **Classe:** `Preprocessor`

- **Metodi:**
  - **check_duplicates()**: Individua eventuali record duplicati e interrompe l'esecuzione se presenti.
  - **drop_cols()**: Rimuove dal dataset le colonne specificate.
  - **encode()**: Applica una codifica numerica a una colonna categoriale tramite mapping.
  - **datetoage()**: Converte una data di nascita nell'età corrispondente e rimuove la colonna originale.
  - **impute()**: Gestisce i valori mancanti utilizzando la mediana oppure il valore zero.
  - **drop_unmatched_patients()**: Elimina dal dataset Fitbit i pazienti non presenti nel dataset anagrafico.
  - **merge()**: Esegue la fusione dei dataset `patients` e `fitbit_daily` tramite `PatientID`.

- **Funzioni:**
  - **splitdata()**: Suddivide il dataset in set di training e test mantenendo la distribuzione delle classi.

### **3. pcavisualizer.py**

- **Classe:** `PCAVisualizer`

- **Metodi:**
  - **reduce()**: Standardizza le feature ed esegue la riduzione dimensionale tramite PCA.
  - **plot()**: Visualizza i dati proiettati sulle prime due componenti principali mediante scatter plot.
  - **explainvariance()**: Restituisce la percentuale di varianza spiegata da ciascuna componente principale.

### **4. classificationmodels.py**

- **Classe:** `Model`

- **Metodi:**
  - **_buildmodel()**: Configura pipeline e griglia di ricerca in base al classificatore selezionato.
  - **train()**: Addestra il modello utilizzando Grid Search con validazione incrociata.
  - **evaluate()**: Calcola l'accuratezza del modello sul set di test.
  - **bestparams()**: Restituisce la migliore combinazione di iperparametri individuata.
  - **get_model()**: Restituisce il modello ottimizzato ottenuto dalla Grid Search.

- **Funzioni:**
  - **bagging_decisiontree()**: Addestra e valuta un ensemble Bagging basato su Decision Tree.
  - **boosting_decisiontree()**: Addestra e valuta un ensemble AdaBoost basato su Decision Tree.
  - **bagging_knn()**: Addestra e valuta un ensemble Bagging basato su K-Nearest Neighbors.
  - **bagging_svc()**: Addestra e valuta un ensemble Bagging basato su Support Vector Classifier calibrato.

### **5. evaluation.py**

- **Classe:** `Evaluator`

- **Metodi:**
  - **report()**: Genera e stampa il classification report con le principali metriche di valutazione.
  - **confusion()**: Costruisce e visualizza la confusion matrix tramite heatmap.

- **Funzioni:**
  - **plot_roc_two_models()**: Confronta due modelli mediante curve ROC micro-average e relativo valore AUC.

---

### **Requisiti**

Per eseuire il progetto è necessario installare, preferibilmente in ambiente virtuale, le librerie presenti in 'requirements.txt' posto nella repository principale:

```bash
cd ./IDM25-26
pip install -r requirements.txt
