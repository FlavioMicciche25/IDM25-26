import pandas as pd
from dataloader import DataLoader
from preprocessing import Preprocessor

link = "./AnonymizedFidelity.csv"
loader = DataLoader(link)

data : pd.DataFrame= loader.load()


print("FIRST FIVE OCCURENCIES")
loader.printfirst(5)


print("INFO")
print(loader.info())

data : pd.DataFrame = preprocessor.dropcol()

