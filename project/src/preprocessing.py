import pandas as pd
from sklearn.model_selection import train_test_split


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
