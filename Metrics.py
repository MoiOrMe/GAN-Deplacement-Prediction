import torch

# Target = les 12 pas réels sans les 8 pas d'observation
# Pred = les 12 pas qu'on a prédit
# Pas fait pour le GAN


#--- ADE : Distance moyenne entre les points prédits et les points réels sur les 12 pas de temps ---#
def calculate_ade(pred, target):

    # distance euclidienne entre les points prédits et les points réels
    errors = torch.norm(pred - target, p=2, dim=-1) 
    ade = torch.mean(errors) 
    return ade

#--- FDE : Distance entre le point final prédit et le point final réel---#
def calculate_fde(pred, target):

    # -1 pour dernier pas
    final_pred = pred[:, -1, :]
    final_target = target[:, -1, :]
    
    # Distance entre les deux points finaux
    fde = torch.norm(final_pred - final_target, p=2, dim=-1)
    return torch.mean(fde)

#--- GAN avec k échantillons ---#

# generator = GAN
# observed_traj = les 8 pas d'observation
# target_traj = les 12 pas réels
# scene_features = les caractéristiques de la scène (ex: carte, obstacles)
# k = nb traj

def evaluate_gan_best_of_k(generator, observerd_traj, target_traj, scene_features, k):
    batch_size = observed_traj.size(0)
    best_ade = torch.full((batch_size,), float('inf'), device=observed_traj.device)
    best_fde = torch.full((batch_size,), float('inf'), device=observed_traj.device)

    for _ in range(k):
        # utilisation de l'ADE et FDE pour évaluer les k trajectoires générées
        pred_traj = generator(observed_traj, scene_features)
        ade = calculate_ade(pred_traj, target_traj)
        fde = calculate_fde(pred_traj, target_traj)

        best_ade = torch.min(best_ade, ade)
        best_fde = torch.min(best_fde, fde)

    return best_ade.mean().item(), best_fde.mean().item()