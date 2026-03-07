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