import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import os
from preprocessing import ETHDataset
from utils import world_to_image, load_homography
from models.gan import Generator 


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

def choose_menu():
    """Menu pour choisir une action"""
    print("\n=== Sélectionner une action ===")
    print("1. afficher les valeurs d'un échantillon")
    print("2. tester le générateur LSTM")
    print("3. tester le générateur GAN")
    print("4. Quitter")

    choice = input("Entrez votre choix (1-4) : ").strip()

    return int(choice)

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

    dataset = ETHDataset(data_path, scale_factor=SCALE_FACTOR)
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

# Le SCALE_FACTOR doit être identique à celui utilisé dans preprocessing.py
SCALE_FACTOR = 10.0 

scene = choose_scene()
dataset, H, scene_image = load_scene_data(scene)

# Extraire un échantillon
sample = dataset[0]

obs = sample["obs_abs"].numpy()
pred = sample["pred_abs"].numpy()

action = choose_menu()

if action == 1:

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

if action == 3:

    # Instanciation du générateur
    gen = Generator()

    # ==========================================
    # Chargement des poids pré-entraînés GAN
    # ==========================================
    weights_path = "weights/gan_checkpoint.pth"

    if os.path.exists(weights_path):
        print(f"Chargement des poids depuis {weights_path}...")
        checkpoint = torch.load(weights_path)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        print("Poids chargés avec succès !")
    else:
        print(f"Attention : Le fichier de poids '{weights_path}' est introuvable. Le modèle va générer des trajectoires aléatoires (non entraîné).")

    gen.eval()

    obs_rel_tensor = sample["obs_rel"].unsqueeze(0)
    last_obs_abs = sample["obs_abs"][-1].numpy()

    # Génération multi-path (Best-of-K)
    K = 5
    predictions_abs = []

    with torch.no_grad():
        for _ in range(K):
            # Le générateur produit des mouvements relatifs normalisés
            pred_rel_fake = gen(obs_rel_tensor).squeeze(0).numpy()
            
            # 1. Dé-normalisation
            pred_rel_fake_denorm = pred_rel_fake * SCALE_FACTOR
            
            # 2. Reconversion en coordonnées absolues
            # CORRECTION : Pas de cumsum ! pred_rel est l'écart direct depuis last_obs
            pred_abs_fake = last_obs_abs + pred_rel_fake_denorm
            predictions_abs.append(pred_abs_fake)

    # TODO: Implémenter et appeler la fonction pour calculer ADE@K.
    # TODO: Implémenter et appeler la fonction pour calculer FDE@K.
    # TODO: Implémenter et appeler la fonction pour calculer la mesure de diversité (variance à travers les échantillons).

    # ==========================================
    # Visualisation Scene-Aware (Image + Matrice)
    # ==========================================
    image_path = "data/eth/visual_data/frame000000.jpg" 
    homography_path = "data/H/H_eth.txt"    

    plt.figure(figsize=(10, 8))

    if os.path.exists(image_path) and os.path.exists(homography_path):
        # Chargement et affichage de l'image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        
        # 1. Chargement de la matrice H (Image -> Monde)
        H = load_homography(homography_path)
        
        # 2. LA CORRECTION : Inversion de la matrice (Monde -> Image)
        H_inv = np.linalg.inv(H)
        
        # 3. Projection en utilisant la matrice inversée
        obs_img = world_to_image(obs, H_inv)
        pred_img = world_to_image(pred, H_inv)
        
        plt.plot(obs_img[:,0], obs_img[:,1], 'bo-', label="Passé (Observé)")
        plt.plot(pred_img[:,0], pred_img[:,1], 'ro-', label="Vrai Futur (Ground Truth)")
        
        for i, pred_fake in enumerate(predictions_abs):
            # On utilise bien H_inv ici aussi !
            pred_fake_img = world_to_image(pred_fake, H_inv)
            label = "Futurs Générés (GAN)" if i == 0 else ""
            plt.plot(pred_fake_img[:,0], pred_fake_img[:,1], 'go--', alpha=0.5, label=label)

    else:
        print("Image ou matrice introuvable. Affichage standard sur fond blanc.")
        plt.plot(obs[:,0], obs[:,1], 'bo-', label="Passé (Observé)")
        plt.plot(pred[:,0], pred[:,1], 'ro-', label="Vrai Futur (Ground Truth)")
        
        for i, pred_fake in enumerate(predictions_abs):
            label = "Futurs Générés (GAN)" if i == 0 else ""
            plt.plot(pred_fake[:,0], pred_fake[:,1], 'go--', alpha=0.5, label=label)
        plt.axis('equal') 

    plt.legend()
    plt.title("Test du Générateur GAN - Prédictions Multiples sur Image")
    plt.show()

