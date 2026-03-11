import matplotlib.pyplot as plt
import numpy as np
import cv2
from preprocessing import ETHDataset
from utils import world_to_image

def choose_scene():
    """Menu pour choisir une scène"""
    print("\n=== Sélectionner une scène ===")
    print("1. Eth")
    print("2. Zara01")
    print("3. Zara02")
    print("4. Hotel")
    
    choice = input("Entrez votre choix (1-4) ou le nom de la scène : ").strip()
    
    scene_map = {
        "1": "Eth",
        "2": "Zara01",
        "3": "Zara02",
        "4": "Hotel"
    }
    
    scene = scene_map.get(choice, choice)
    return scene

def load_scene_data(scene):
    """Charger les données d'une scène"""
    data_path = f"Data/{scene}_Data.txt"
    homo_path = f"Data/{scene}_Homo.txt"
    bg_path = f"Data/{scene}_Bg.png"
    
    print(f"\nChargement de la scène: {scene}")
    print(f"  - Données: {data_path}")
    print(f"  - Homographie: {homo_path}")
    print(f"  - Image de fond: {bg_path}")
    
    # Charger le dataset
    dataset = ETHDataset(data_path)
    print(f"Nombre de séquences : {len(dataset)}")
    
    # Charger la matrice d'homographie
    H = np.loadtxt(homo_path)
    H = np.linalg.inv(H)
    print(f"Matrice d'homographie inversée (world-to-image):\n{H}")
    
    # Charger l'image de fond
    scene_image = cv2.imread(bg_path)
    scene_image = cv2.cvtColor(scene_image, cv2.COLOR_BGR2RGB)
    
    return dataset, H, scene_image

# === Main ===
scene = choose_scene()
dataset, H, scene_image = load_scene_data(scene)

# Extraire un échantillon
sample = dataset[0]

print(f"Échantillon sélectionné: {sample}")
obs = sample["obs_abs"].numpy()
pred = sample["pred_abs"].numpy()

# Créer une figure avec 2 sous-graphiques côte à côte
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Visualiser en coordonnées monde (gauche)
ax1.plot(obs[:,0], obs[:,1], 'bo-', label="Observed")
ax1.plot(pred[:,0], pred[:,1], 'ro-', label="Future")
ax1.legend()
ax1.set_title("Trajectory (World coordinates)")
ax1.set_xlabel("X (meters)")
ax1.set_ylabel("Y (meters)")
ax1.grid(True)

# Projeter sur l'image
obs_img = world_to_image(obs, H)
pred_img = world_to_image(pred, H)

# Debug: Afficher les coordonnées projetées
# print(f"Coordonnées projetées - Obs: min={obs_img.min():.2f}, max={obs_img.max():.2f}")
# print(f"Coordonnées projetées - Pred: min={pred_img.min():.2f}, max={pred_img.max():.2f}")
# print(f"Échantillon obs_img: {obs_img[:3]}")
# print(f"Échantillon pred_img: {pred_img[:3]}")
# print(f"Taille image: {scene_image.shape}")

# Corriger l'axe Y si nécessaire (images ont Y=0 en haut)
if obs_img[:,1].mean() < scene_image.shape[0] / 2:  # Si en moyenne en haut
    print("Correction de l'axe Y (inversion)")
    obs_img[:,1] = scene_image.shape[0] - obs_img[:,1]
    pred_img[:,1] = scene_image.shape[0] - pred_img[:,1]

# Visualiser sur l'image de fond (droite)
ax2.imshow(scene_image)
ax2.plot(obs_img[:,0], obs_img[:,1], 'bo-', label="Observed")
ax2.plot(pred_img[:,0], pred_img[:,1], 'ro-', label="Future")
ax2.legend()
ax2.set_title("Projection on Scene Image")
ax2.set_xlabel("X (pixels)")
ax2.set_ylabel("Y (pixels)")

plt.tight_layout()
plt.show()

