import pandas as pd
from dataloader import DataLoader
from preprocessing import Preprocessor
from frequencyAnalisys import FrequencyAnalyzer

link = "./AnonymizedFidelity.csv"
loader = DataLoader(link)

data : pd.DataFrame= loader.load()


print("FIRST FIVE OCCURENCIES")
loader.printfirst(5)


print("INFO")
print(loader.info())

preprocessor = Preprocessor(data)
data : pd.DataFrame = preprocessor.removeshoppers().dropcol().convertdatetime().createslices().getdataset()


frequencyanalyzer = FrequencyAnalyzer(data)
frequencyanalyzer.analyzelevels()
frequencyanalyzer.stratifiedlevels()

def getdataset(self):
        return self.dataset