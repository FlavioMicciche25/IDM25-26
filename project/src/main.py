import pandas as pd
from dataloader import DataLoader
from preprocessing import Preprocessor
from preprocessing import splitdata

# LOAD DATASETS
loader = DataLoader(
    patients_path="../data/patients.txt",
    fitbit_path="../data/fitbit_daily.txt"
)

loader.load()

loader.printhead(loader.patients_path).printhead(loader.fitbit_path)
loader.printinfo(loader.patients_path).printinfo(loader.fitbit_path)

# CLEANING, MERGING AND DATASETS
df_pat = loader.get_patients()
df_fit = loader.get_fitbit()

col_to_drop_pat = [
    "ObjectId", "Centro", "Sede",
    "PatologiaPrincipale", "Gruppo"
]
col_to_drop_fit = [
    "TimeInBed", "AsleepMinutes",
    "LightSleepCounts", "DeepSleepCounts", "REMSleepCounts",
    "AwakeCounts", "AsleepCounts", "LightSleepBreathingRate",
    "DeepSleepBreathingRate", "REMSleepBreathingRate"
]
zero_cols_fit = [
    "MinutesAfterWakeup", "MinutesToFallAsleep", "SleepEfficiency",
    "LightSleepMinutes", "DeepSleepMinutes", "REMSleepMinutes",
    "AwakeMinutes"
]
median_cols_fit = [
     "Calories", "ActivityCalories", "RestingHeartRate",
    "VO2Max", "NightlySkinTemperature", "BreathingRate",
    "FullSleepBreathingRate",
    "BelowFatBurnCalories", "FatBurnCalories", "CardioCalories", "PeakCalories",
    "BelowFatBurnMinutes", "FatBurnMinutes", "CardioMinutes", "PeakMinutes"
]

preproc = Preprocessor(df_pat, df_fit)

preproc.check_duplicates(df_pat, "patients") \
       .drop_cols(df_pat, col_to_drop_pat) \
       .encode(df_pat, "Sesso", {'female': 0, 'male': 1}) \
       .datetoage(df_pat, "DataNascita")

preproc.check_duplicates(df_fit, "fitbit") \
       .drop_cols(df_fit, col_to_drop_fit) \
       .impute(df_fit, zero_cols_fit, "zero") \
       .impute(df_fit, median_cols_fit, "median")

preproc.drop_unmatched_patients()
df_final = preproc.merge()

print(df_final.head())
print(df_final.info())

X = df_final.drop(columns=["GruppoLabel"])
y = df_final["GruppoLabel"]
X_train, X_test, y_train, y_test = splitdata(X, y)
