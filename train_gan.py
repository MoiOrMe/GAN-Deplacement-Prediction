import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import ETHDataset
from models.gan import Generator, Discriminator 

def train_gan(dataloader, epochs=50, use_label_smoothing=True, use_gradient_clipping=True, clip_value=1.0, save_path="weights/gan_checkpoint.pth", load_path=None):
    """
    Boucle d'entraînement pour le Vanilla Trajectory GAN.
    Intègre la sauvegarde et le chargement des poids pour réutiliser le modèle.
    """
    # 1. Initialisation des modèles
    gen = Generator()
    disc = Discriminator()

    # 2. Chargement des poids existants (si on veut reprendre un entraînement)
    if load_path and os.path.exists(load_path):
        print(f"Chargement des poids depuis {load_path}...")
        checkpoint = torch.load(load_path)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        print("Poids chargés avec succès !")

    # Fonction de perte binaire standard pour les GANs
    criterion = nn.BCELoss() 
    
    # Optimiseurs
    opt_g = optim.Adam(gen.parameters(), lr=0.001)
    opt_d = optim.Adam(disc.parameters(), lr=0.001)

    gen.train()
    disc.train()

    print("Début de l'entraînement...")
    for epoch in range(epochs):
        for batch in dataloader:
            # Extraction des données du batch fournies par ETHDataset
            obs_rel = batch["obs_rel"] 
            pred_rel_real = batch["pred_rel"] 

            # ==========================================
            # 1. Entraînement du Discriminateur
            # ==========================================
            opt_d.zero_grad()

            # A. Évaluation des trajectoires réelles (Ground Truth)
            full_traj_real = torch.cat((obs_rel, pred_rel_real), dim=1) 
            real_pred = disc(full_traj_real)

            real_labels = torch.ones_like(real_pred)
            
            # Stabilisation via Label smoothing
            if use_label_smoothing:
                real_labels = real_labels * 0.9

            d_loss_real = criterion(real_pred, real_labels)

            # B. Évaluation des trajectoires générées (Fake)
            pred_rel_fake = gen(obs_rel)
            
            full_traj_fake = torch.cat((obs_rel, pred_rel_fake.detach()), dim=1)
            fake_pred = disc(full_traj_fake)
            
            fake_labels = torch.zeros_like(fake_pred)
            d_loss_fake = criterion(fake_pred, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()

            # Stabilisation via Gradient clipping
            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(disc.parameters(), clip_value)
                
            opt_d.step()

            # ==========================================
            # 2. Entraînement du Générateur
            # ==========================================
            opt_g.zero_grad()
            
            full_traj_fake_for_g = torch.cat((obs_rel, pred_rel_fake), dim=1)
            pred_fake_for_g = disc(full_traj_fake_for_g)
            
            g_loss = criterion(pred_fake_for_g, torch.ones_like(pred_fake_for_g))
            g_loss.backward()

            if use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(gen.parameters(), clip_value)
                
            opt_g.step()

        print(f"Epoch [{epoch+1}/{epochs}] | Loss D: {d_loss.item():.4f} | Loss G: {g_loss.item():.4f}")

    # ==========================================
    # 3. Sauvegarde du modèle à la fin
    # ==========================================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'gen_state_dict': gen.state_dict(),
        'disc_state_dict': disc.state_dict(),
    }, save_path)
    
    print(f"Entraînement terminé ! Modèles sauvegardés dans {save_path}")
    return gen, disc

# ==========================================
# Point d'entrée principal pour lancer l'entraînement
# ==========================================
if __name__ == "__main__":
    # TODO: Vérifier que le chemin vers biwi_eth.txt est correct dans ton arborescence
    data_path = "data/raw/all_data/biwi_eth.txt"
    
    print("Initialisation du dataset...")
    dataset = ETHDataset(data_path)
    
    # Le DataLoader regroupe les séquences par lots (batch_size) et les mélange (shuffle)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Lancement de l'entraînement
    # TODO: Tu peux changer load_path="weights/gan_checkpoint.pth" si tu veux reprendre un entraînement
    train_gan(dataloader, epochs=500, save_path="weights/gan_checkpoint.pth", load_path=None)