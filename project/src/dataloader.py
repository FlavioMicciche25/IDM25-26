import pandas as pd
import numpy as np

class DataLoader:

    def __init__(self, patients_path=None, fitbit_path=None):
        
        self.patients_path = patients_path
        self.fitbit_path = fitbit_path

        self.patients = None
        self.fitbit_daily = None

    def load(self):

        if self.patients_path is not None:
            if "patient" in self.patients_path.lower():
                self.patients = pd.read_csv(self.patients_path, sep="\t")
            else:
                raise ValueError("patients path not contains 'patient'.")

        if self.fitbit_path is not None:
            if "fitbit_daily" in self.fitbit_path.lower():
                self.fitbit_daily = pd.read_csv(self.fitbit_path, sep="\t")
            else:
                raise ValueError("fitbit_daily path not contains 'fitbit_daily'.")

        print("Dataset correctly loaded.")
        return

    def get_patients(self):
        if self.patients is None:
            raise RuntimeError("Dataset 'patients' not loaded.")
        return self.patients

    def get_fitbit(self):
        if self.fitbit_daily is None:
            raise RuntimeError("Dataset 'fitbit_daily' not loaded.")
        return self.fitbit_daily

    def printhead(self, dataset=None, occur=5):
        print(dataset)

        if dataset is None:
            dataset = self.patients_path

        
        if "patient" in dataset.lower():
            print("\n===== PATIENTS HEADER =====")
            df = self.get_patients()
        elif "fitbit" in dataset.lower():
            print("\n===== FITBIT DAILY HEADER =====")
            df = self.get_fitbit()
        else:
            raise ValueError("Dataset must be 'patients' or 'fitbit_daily'.")

        print(df.head(occur))
        return self

    def printinfo(self, dataset=None):

        if dataset is None:
            dataset = self.patients_path
        
        if "patients" in dataset.lower():
            print("\n===== PATIENTS INFO =====")
            df = self.get_patients()
        elif "fitbit" in dataset.lower():
            print("\n===== FITBIT DAILY INFO =====")
            df = self.get_fitbit()
        else:
            raise ValueError("Dataset must be 'patients' or 'fitbit_daily'.")

        print(" INFO",df.info())
        return self

    def merge(self, key="PatientID"):
        
        if self.patients is None or self.fitbit_daily is None:
            raise RuntimeError("before you merge them, first upload both datasets with load() method.")

        if key not in self.patients.columns:
            raise ValueError(f"Key '{key}' not found in patients.")

        if key not in self.fitbit_daily.columns:
            raise ValueError(f"Key '{key}' not found in fitbit_daily.")

        merged = self.fitbit_daily.merge(self.patients, on=key, how="left")
        print("Merge completed.\n===== UNIFIED FITBIT DATASET HEADER =====")
        return merged
