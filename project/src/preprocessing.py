import pandas as pd
from sklearn.model_selection import train_test_split

"""
# OLD VERSION
class Preprocessor:

    def __init__(self):
        pass

    def preprocess_patients(self, df):
        
        # Rimozione colonne ridondanti
        cols_to_drop = [
            "ObjectId", "Centro", "Sede", 
            "PatologiaPrincipale", "Gruppo"
        ]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        # Encoding sesso
        if "Sesso" in df.columns:
            df["Sesso"] = df["Sesso"].map({'female': 0, 'male': 1})

        # DataNascita → Età
        if "DataNascita" in df.columns:
            df["Età"] = pd.to_datetime("today").year - pd.to_datetime(df["DataNascita"]).dt.year
            df = df.drop(columns=["DataNascita"])

        return df

    def preprocess_fitbit(self, df):

        # Colonne da rimuovere (ridondanti o >50% NaN)
        cols_to_drop = [
            "TimeInBed", "AsleepMinutes",
            "LightSleepCounts", "DeepSleepCounts", "REMSleepCounts",
            "AwakeCounts", "AsleepCounts",
            "LightSleepBreathingRate", "DeepSleepBreathingRate", "REMSleepBreathingRate"
        ]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

        # Imputazione NaN fisiologici → 0
        zero_impute_cols = [
            "MinutesAfterWakeup", "MinutesToFallAsleep", "SleepEfficiency",
            "LightSleepMinutes", "DeepSleepMinutes", "REMSleepMinutes",
            "AwakeMinutes"
        ]
        for col in zero_impute_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        # Imputazione NaN non fisiologici → mediana
        median_impute_cols = [
            "Calories", "ActivityCalories", "RestingHeartRate",
            "VO2Max", "NightlySkinTemperature", "BreathingRate",
            "FullSleepBreathingRate",
            "BelowFatBurnCalories", "FatBurnCalories", "CardioCalories", "PeakCalories",
            "BelowFatBurnMinutes", "FatBurnMinutes", "CardioMinutes", "PeakMinutes"
        ]
        for col in median_impute_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        return df

"""

class Preprocessor:
    def __init__(self, df_pat=None, df_fit=None):
        self.df_pat = df_pat
        self.df_fit = df_fit

    def check_duplicates(self, df, df_name="DataFrame"):
        dup = df[df.duplicated(keep=False)]
        if not dup.empty:
            print(f"\nDuplicati trovati in {df_name}:")
            print(dup.to_string(index=False))
            raise ValueError(f"Duplicates found in {df_name}.")
        print(f"No duplicates found in {df_name}.")
        return self

    def drop_cols(self, df, cols):
        df.drop(columns=[c for c in cols if c in df.columns], inplace=True)
        return self

    def encode(self, df, col, mapping):
        if col in df.columns:
            df[col] = df[col].map(mapping)
        return self

    def datetoage(self, df, col):
        if col in df.columns:
            df["Età"] = pd.to_datetime("today").year - pd.to_datetime(df[col]).dt.year
            df.drop(columns=[col], inplace=True)
        return self

    def impute(self, df, cols, strategy="median"):
        for col in cols:
            if col in df.columns:
                if strategy == "zero":
                    df[col] = df[col].fillna(0)
                elif strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
        return self

    def drop_unmatched_patients(self):
        missing_ids = set(self.df_fit['PatientID']) - set(self.df_pat['PatientID'])
        if missing_ids:
            print(f"Pazients found in fitbit but not found in patients: {missing_ids}")
            print("They will be removed from dataset fitbit_daily to avoid merge conflicts.")
            
            self.df_fit = self.df_fit[~self.df_fit["PatientID"].isin(missing_ids)]
        else:
            print("No missing patient: fitbit_daily is aligned with patients.")

    def merge(self):
        return self.df_fit.merge(self.df_pat, on="PatientID", how="left")


def splitdata(X, y, testsize=0.2, random_state=42):
    return train_test_split(X, y, test_size=testsize, random_state=random_state, stratify=y)
