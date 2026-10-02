import pandas as pd
from dataloader import DataLoader

loader = DataLoader(
    patients_path="../data/patients.txt",
    fitbit_path="../data/fitbit_daily.txt"
)

loader.load()

loader.printhead(loader.patients_path).printhead(loader.fitbit_path)

loader.printinfo(loader.patients_path).printinfo(loader.fitbit_path)