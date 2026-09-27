import pandas as pd
from dataloader import DataLoader
from preprocessing import Preprocessor

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


#CLEANING DATASETS
coltorm = ["Pazienti","Età cronologica (mesi)","Scala B","Scala D","TOT.","Score di rischio"]
for name in dfnames:
    preproc = Preprocessor(datasets[name])
    datasets[name] = preproc.rmrows()
    datasets[name] = preproc.rmcolumns(cols=coltorm)