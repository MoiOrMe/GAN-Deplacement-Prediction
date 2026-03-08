import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import os
from preprocessing import ETHDataset
from utils import world_to_image
from models.gan import Generator 

# Initialisation du dataset
data_path = "data/raw/all_data/biwi_eth.txt"
dataset = ETHDataset(data_path)
print("Nombre de séquences :", len(dataset))
sample = dataset[0]

obs = sample["obs_abs"].numpy()
pred = sample["pred_abs"].numpy()

# Instanciation du générateur
gen = Generator()

# ==========================================
# Chargement des poids pré-entraînés GAN
# ==========================================
weights_path = "weights/gan_checkpoint.pth"

if os.path.exists(weights_path):
    print(f"Chargement des poids depuis {weights_path}...")
    checkpoint = torch.load(weights_path)
    # On charge uniquement le dictionnaire d'état du générateur
    gen.load_state_dict(checkpoint['gen_state_dict'])
    print("Poids chargés avec succès !")
else:
    print(f"Attention : Le fichier de poids '{weights_path}' est introuvable. Le modèle va générer des trajectoires aléatoires (non entraîné).")

# On passe le modèle en mode évaluation (désactive les comportements spécifiques à l'entraînement)
gen.eval()

obs_rel_tensor = sample["obs_rel"].unsqueeze(0)
last_obs_abs = sample["obs_abs"][-1].numpy()

# Génération multi-path (Best-of-K)
K = 5
predictions_abs = []

# torch.no_grad() permet d'économiser de la mémoire car on ne calcule pas les gradients lors de l'inférence
with torch.no_grad():
    for _ in range(K):
        # Le générateur produit des mouvements relatifs
        pred_rel_fake = gen(obs_rel_tensor).squeeze(0).numpy()
        
        # Reconversion en coordonnées absolues
        pred_abs_fake = last_obs_abs + np.cumsum(pred_rel_fake, axis=0)
        predictions_abs.append(pred_abs_fake)

# TODO: Implémenter et appeler la fonction pour calculer ADE@K.
# TODO: Implémenter et appeler la fonction pour calculer FDE@K.
# TODO: Implémenter et appeler la fonction pour calculer la mesure de diversité (variance à travers les échantillons).

# Visualisation des multiples futurs pour la même entrée
plt.figure(figsize=(8, 6))
plt.plot(obs[:,0], obs[:,1], 'bo-', label="Passé (Observé)")
plt.plot(pred[:,0], pred[:,1], 'ro-', label="Vrai Futur (Ground Truth)")

for i, pred_fake in enumerate(predictions_abs):
    label = "Futurs Générés (GAN)" if i == 0 else ""
    plt.plot(pred_fake[:,0], pred_fake[:,1], 'go--', alpha=0.5, label=label)

plt.legend()
plt.title("Test du Générateur GAN - Prédictions Multiples")
plt.axis('equal') 

# TODO: Superposer les trajectoires sur l'image de la scène (Overlay trajectories on scene) en utilisant la matrice d'homographie.
plt.show()