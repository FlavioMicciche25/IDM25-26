# **IDM25-26**
---
## Fitbit Daily - Data Mining Introduction Project
---

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

