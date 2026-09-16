from pathlib import Path
import pandas as pd
import numpy as np


class EchonextDataset:
    def __init__(self, root:str):
        self.root = Path(root)
        self.metadata = pd.read_csv(self.root / 'echonext_metadata_100k.csv')

    def get_train_ecg(self,index:int):
        row = self.metadata.iloc[index]
        ecg = np.load(self.root / row['ecg_file'])
        return ecg

    def get_lvef(self,index:int):
        return float(self.metadata.iloc[index]['lvef_value'])

    def train_spliting_patients(self,df,seed):
        patients = df['patient_id'].unique()
        rng = np.random.default_rng(seed)
        rng.shuffle(patients)

        num = len(patients)

        train_p = df[df['patient_id'].isin(patients[:int(0.7*num)])]
        val_p = df[df['patient_id'].isin(patients[int(0.7*num):int(0.85*num)])]
        test_p = df[df['patient_id'].isin(patients[int(0.85*num):])]
        return train_p,val_p,test_p







