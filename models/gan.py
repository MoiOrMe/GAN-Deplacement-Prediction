import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, obs_len=8, pred_len=12, noise_dim=16, hidden_size=64):
        super(Generator, self).__init__()
        # Le dataset fournit 8 pas de temps observés pour en prédire 12 [cite: 185]
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.noise_dim = noise_dim
        self.hidden_size = hidden_size

        # Encodeur LSTM : Lit la trajectoire passée (x, y en relatif) pour en extraire le contexte
        self.encoder = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        
        # Décodeur LSTM : Va générer le futur. Son entrée est la combinaison 
        # du contexte passé (hidden_size) et du vecteur de bruit (noise_dim)
        self.decoder = nn.LSTM(input_size=hidden_size + noise_dim, hidden_size=hidden_size, batch_first=True)
        
        # Couche finale pour transformer la dimension cachée (64) en coordonnées (x, y)
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, obs_rel, z=None):
        batch_size = obs_rel.size(0)

        # 1. Encodage du passé
        # On passe la trajectoire observée dans l'encodeur
        _, (h_n, _) = self.encoder(obs_rel)
        # On ne garde que le tout dernier état caché (il résume tout le passé)
        encoded_obs = h_n[-1] 

        # 2. Ajout du bruit pour la multimodalité
        # C'est ce vecteur 'z' qui permet au GAN de proposer des chemins différents pour un même passé
        if z is None:
            # Si aucun bruit n'est fourni, on en génère un aléatoire
            z = torch.randn(batch_size, self.noise_dim, device=obs_rel.device)

        # On colle (concatène) le résumé du passé avec le bruit z
        decoder_input = torch.cat((encoded_obs, z), dim=1)
        
        # On répète cette combinaison pour chaque pas de temps futur que l'on veut prédire (ex: 12 fois)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.pred_len, 1)

        # 3. Décodage et Prédiction
        decoder_out, _ = self.decoder(decoder_input)
        
        # On transforme la sortie du décodeur en mouvements relatifs (x, y)
        pred_rel = self.output_layer(decoder_out)

        return pred_rel


class Discriminator(nn.Module):
    def __init__(self, hidden_size=64):
        super(Discriminator, self).__init__()
        
        # L'encodeur du discriminateur lit la trajectoire complète (passé + futur généré/réel)
        self.encoder = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        
        # Un classifieur simple pour dire si la trajectoire est réaliste (1) ou fausse (0)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # La Sigmoïde écrase la valeur entre 0 et 1 (probabilité)
            nn.Sigmoid()
        )

    def forward(self, full_traj_rel):
        # On passe toute la trajectoire dans le LSTM
        _, (h_n, _) = self.encoder(full_traj_rel)
        
        # On récupère le dernier état caché qui représente l'ensemble du mouvement
        encoded_traj = h_n[-1]
        
        # Le classifieur donne son verdict final
        prob = self.classifier(encoded_traj)
        return prob