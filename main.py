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
    print("4. Univ")
    
    choice = input("Entrez votre choix (1-4) ou le nom de la scène : ").strip()
    
    scene_map = {
        "1": "Eth",
        "2": "Zara01",
        "3": "Zara02",
        "4": "Univ"
    }
    
    scene = scene_map.get(choice, choice)
    return scene

def load_scene_data(scene):
    """Charger les données d'une scène"""
    data_path = f"{scene}_Data.txt"
    homo_path = f"{scene}_Homo.txt"
    bg_path = f"{scene}_Bg.png"
    
    print(f"\nChargement de la scène: {scene}")
    print(f"  - Données: {data_path}")
    print(f"  - Homographie: {homo_path}")
    print(f"  - Image de fond: {bg_path}")
    
    # Charger le dataset
    dataset = ETHDataset(data_path)
    print(f"Nombre de séquences : {len(dataset)}")
    
    # Charger la matrice d'homographie
    H = np.loadtxt(homo_path)
    
    # Charger l'image de fond
    scene_image = cv2.imread(bg_path)
    scene_image = cv2.cvtColor(scene_image, cv2.COLOR_BGR2RGB)
    
    return dataset, H, scene_image

# === Main ===
scene = choose_scene()
dataset, H, scene_image = load_scene_data(scene)

# Extraire un échantillon
sample = dataset[0]

obs = sample["obs_abs"].numpy()
pred = sample["pred_abs"].numpy()

# Visualiser en coordonnées monde
plt.figure()
plt.plot(obs[:,0], obs[:,1], 'bo-', label="Observed")
plt.plot(pred[:,0], pred[:,1], 'ro-', label="Future")
plt.legend()
plt.title("Trajectory (World coordinates)")
plt.show()

# Projeter sur l'image
obs_img = world_to_image(obs, H)
pred_img = world_to_image(pred, H)

# Visualiser sur l'image de fond
plt.figure()
plt.imshow(scene_image)
plt.plot(obs_img[:,0], obs_img[:,1], 'bo-', label="Observed")
plt.plot(pred_img[:,0], pred_img[:,1], 'ro-', label="Future")
plt.legend()
plt.title("Projection test")
plt.show()

