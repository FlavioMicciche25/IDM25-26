import pandas as pd

class Preprocessor:

    def __init__(self, dataset):
        self.dataset = dataset

    def rmcolumns(self, cols):
        self.dataset = self.dataset.drop(columns=cols)
        return self.dataset

    def rmrows(self):
        self.dataset = self.dataset[self.dataset['Età equivalente'] >= 12]
        return self.dataset