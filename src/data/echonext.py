from pathlib import Path
import pandas as pd
import numpy as np


class EchonextDataset:
    def __init__(self, root:str):
        self.root = Path(root)
        self.metadata = pd.read_csv(self.root / 'metadata.csv')

    def get_ecg(self,index:int):
        row = self.metadata.iloc[index]
        ecg = np.load(self.root / row['ecg_file'])
        return ecg

    def get_lvef(self,index:int):
        return float(self.metadata.iloc[index]['lvef_value'])




