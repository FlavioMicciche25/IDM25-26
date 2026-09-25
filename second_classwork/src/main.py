import pandas as pd
from dataloader import DataLoader

loader = DataLoader("./Dataset DAES.xlsx")
loader.load()

datasets = {}
dfnames = ["ASD","GDD","Controlli"]

#LOAD DATASET
for name in dfnames:
    print(f"======{name}======")
    datasets[name] = loader.get_sheet(name)
    loader.printfirstrow(name,10)
    loader.printinfo(name)