import pandas as pd
from datetime import time


class Preprocessor:
    def __init__(self,data):
        self.dataset = data.copy()

    def dropcol(self):
        toDrop = ["db_id", "puntovendita_id","cassa","cassiere","nummero_scontrino",
            "num_riga","r_peso", "r_iva", "r_sconto",
            "tipologia", "descr_tipologia", "cod_rep", "descr_rep", "cat_mer"]

        self.dataset = self.dataset.drop(columns=toDrop)
        return self
    