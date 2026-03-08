import matplotlib.pyplot as plt
import numpy as np
import cv2
from preprocessing import ETHDataset
from utils import world_to_image
import torch
from models.gan import Generator

# path vers le dataset
data_path = "data/raw/all_data/biwi_eth.txt"
dataset = ETHDataset(data_path)
print("Nombre de séquences :", len(dataset))
sample = dataset[0]

obs = sample["obs_abs"].numpy()
pred = sample["pred_abs"].numpy()

# Instanciation du générateur (non entraîné pour l'instant)
gen = Generator()

# PyTorch s'attend à des "batchs" (lot de données). On ajoute une dimension avec unsqueeze(0)
obs_rel_tensor = sample["obs_rel"].unsqueeze(0)
last_obs_abs = sample["obs_abs"][-1].numpy()

# Génération multi-path (Best-of-K)
K = 5
predictions_abs = []

# Mode évaluation pour ne pas calculer les gradients
with torch.no_grad():
    for _ in range(K):
        # Le générateur produit des mouvements relatifs
        pred_rel_fake = gen(obs_rel_tensor).squeeze(0).numpy()
        
        # Reconversion en coordonnées absolues (somme cumulative depuis le dernier point observé)
        pred_abs_fake = last_obs_abs + np.cumsum(pred_rel_fake, axis=0)
        predictions_abs.append(pred_abs_fake)

# Visualisation
plt.figure(figsize=(8, 6))
plt.plot(obs[:,0], obs[:,1], 'bo-', label="Passé (Observé)")
plt.plot(pred[:,0], pred[:,1], 'ro-', label="Vrai Futur (Ground Truth)")

for i, pred_fake in enumerate(predictions_abs):
    label = "Futurs Générés (GAN)" if i == 0 else ""
    plt.plot(pred_fake[:,0], pred_fake[:,1], 'go--', alpha=0.5, label=label)

plt.legend()
plt.title("Test du Générateur GAN - Prédictions Multiples")
plt.axis('equal') # Pour ne pas déformer les trajectoires
plt.show()

