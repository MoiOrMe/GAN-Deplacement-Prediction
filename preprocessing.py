import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class ETHDataset(Dataset):

    def __init__(self, data_path, obs_len=8, pred_len=12):

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len

        # Charger le fichier texte
        data = pd.read_csv(
            data_path,
            sep=r'\s+',          # gère espaces ou tabulations
            header=None,
            names=['frame', 'ped_id', 'x', 'y'],
            engine='python'
        )

        # convertir les ID en entier
        data["ped_id"] = data["ped_id"].astype(int)

        print("Nombre total de lignes :", len(data))
        print("Nombre de piétons :", data["ped_id"].nunique())

        self.sequences = []

        # grouper par piéton
        grouped = data.groupby('ped_id')

        for ped_id, ped_data in grouped:

            ped_data = ped_data.sort_values('frame')

            coords = ped_data[['x', 'y']].values

            # ignorer les trajectoires trop courtes
            if len(coords) < self.seq_len:
                continue

            # sliding window
            for i in range(len(coords) - self.seq_len + 1):

                seq = coords[i:i+self.seq_len]

                self.sequences.append(seq)

        print("Nombre total de séquences :", len(self.sequences))


    def __len__(self):
        return len(self.sequences)


    def __getitem__(self, idx):

        seq = self.sequences[idx]

        obs = seq[:self.obs_len]
        pred = seq[self.obs_len:]

        # dernier point observé
        last_obs = obs[-1].copy()

        # coordonnées relatives
        obs_rel = obs - last_obs
        pred_rel = pred - last_obs

        sample = {
            "obs_abs": torch.tensor(obs, dtype=torch.float32),      # coordonnées absolues observées
            "pred_abs": torch.tensor(pred, dtype=torch.float32),    # coordonnées absolues prédites
            "obs_rel": torch.tensor(obs_rel, dtype=torch.float32),  # coordonnées relatives observées
            "pred_rel": torch.tensor(pred_rel, dtype=torch.float32) # coordonnées relatives prédites
        }

        return sample