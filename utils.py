import numpy as np

def load_homography(filepath):
    """
    Charge la matrice d'homographie H depuis un fichier texte.
    np.loadtxt gère nativement le format scientifique (e-02, e+00).
    """
    H = np.loadtxt(filepath)
    return H

def world_to_image(world_coords, H):
    """
    Projette les coordonnées du monde vers l'image en utilisant la matrice H.
    """
    # Ajout de la 3ème dimension (Z=1) pour la multiplication matricielle
    coords_homo = np.hstack([world_coords, np.ones((world_coords.shape[0], 1))])
    
    # Multiplication avec la matrice
    img_coords = (H @ coords_homo.T).T
    
    # Normalisation par la 3ème coordonnée
    img_coords = img_coords[:, :2] / img_coords[:, 2:]
    return img_coords