import pandas as pd
from dataloader import DataLoader
from preprocessing import Preprocessor
from frequencyAnalisys import FrequencyAnalyzer
from association import Association

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


associator = Association(data)
print ("APRIORI treshold = 0.05 confidence = 0.2 maxnumrules = 20")
apriori = associator.create_transaction().apriorirules(0.05,0.2).showrules(20)

print ("FP-GROWTH treshold = 0.05 confidence = 0.2 maxnumrules = 20")
apriori = associator.create_transaction().fpgrowthrules(0.05,0.15).showrules(30)
