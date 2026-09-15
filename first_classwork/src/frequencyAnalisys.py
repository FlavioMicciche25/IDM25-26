import pandas as pd
from matplotlib import pyplot as plt


class FrequencyAnalyzer:
    def __init__(self,data):
        self.database : pd.DataFrame = data


    @staticmethod
    def printplot(most,least,title1, title2):
        

        plt.figure(figsize=(10,6))
        plt.subplot(1,2,1)
        most.plot(kind='bar', title = title1)
        plt.xlabel("Element")
        plt.grid()
        plt.ylabel("Frequency")
        plt.xticks(rotation=45)

        plt.subplot(1,2,2)
        least.plot(kind='bar',title=title2)
        plt.xlabel("Element")
        plt.ylabel("Frequency")
        plt.grid()
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()


    def analyze(self,title1,title2):
        coltoanalyze =['descr_liv1','descr_liv2','descr_liv3','descr_liv4']
        i = 1
        for c in coltoanalyze:
            mostfive = self.database[c].value_counts().sort_values(ascending=False).head(5)
            leastfive = self.database[c].value_counts().sort_values(ascending=True).head(5)
            strlvl = f"- Level {i}"
            t1 = title1 + strlvl
            t2 = title2 + strlvl
            FrequencyAnalyzer.printplot(mostfive,leastfive,t1,t2)
            i = i+1


    def analyzelevels(self):
        title1 = f"Top 5 most frequent elements "
        title2 = f"Top 5 least frequent elements "
        self.analyze(title1,title2)



