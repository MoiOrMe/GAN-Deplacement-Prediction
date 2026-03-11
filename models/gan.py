import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Générateur du GAN conditionnel pour la prédiction de trajectoire.
    Prend en entrée la trajectoire observée et génère les positions futures relatives.
    """
    def __init__(self, obs_len=8, pred_len=12, noise_dim=16, hidden_size=64):
        super(Generator, self).__init__()
        # Le dataset fournit 8 pas de temps observés pour en prédire 12 
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.noise_dim = noise_dim
        self.hidden_size = hidden_size

        # Encodeur LSTM : Lit la trajectoire passée (x, y en relatif) pour extraire le contexte spatial et temporel
        self.encoder = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        
        # Décodeur LSTM : Génère le futur. 
        # Son entrée est la combinaison du contexte passé (hidden_size) et du vecteur de bruit (noise_dim)
        # TODO: Intégrer les "Optional scene feature" (caractéristiques de l'image) en entrée du décodeur si nécessaire.
        self.decoder = nn.LSTM(input_size=hidden_size + noise_dim, hidden_size=hidden_size, batch_first=True)
        
        # Couche finale pour transformer la dimension cachée en coordonnées (x, y) relatives
        self.output_layer = nn.Linear(hidden_size, 2)

    def forward(self, obs_rel, z=None):
        batch_size = obs_rel.size(0)

        # 1. Encodage du passé
        _, (h_n, _) = self.encoder(obs_rel)
        encoded_obs = h_n[-1] # On conserve le dernier état caché qui résume le passé

        # 2. Ajout du bruit pour la multimodalité (permet de générer des chemins différents)
        if z is None:
            z = torch.randn(batch_size, self.noise_dim, device=obs_rel.device)

        # Concaténation du résumé du passé avec le bruit z
        decoder_input = torch.cat((encoded_obs, z), dim=1)
        
        # On répète cette combinaison pour chaque pas de temps futur (ex: 12 fois)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.pred_len, 1)

        # 3. Décodage et Prédiction
        decoder_out, _ = self.decoder(decoder_input)
        pred_rel = self.output_layer(decoder_out)

        return pred_rel


class Discriminator(nn.Module):
    """
    Discriminateur du GAN.
    Évalue une trajectoire complète (passé + futur) et détermine si elle est réelle ou générée[cite: 114, 119].
    """
    def __init__(self, hidden_size=64):
        super(Discriminator, self).__init__()
        
        # L'encodeur lit la trajectoire complète
        self.encoder = nn.LSTM(input_size=2, hidden_size=hidden_size, batch_first=True)
        
        # Classifieur binaire : 1 (réaliste/vrai) ou 0 (faux/généré)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Écrase la valeur entre 0 et 1 pour obtenir une probabilité
        )

    def forward(self, full_traj_rel):
        _, (h_n, _) = self.encoder(full_traj_rel)
        encoded_traj = h_n[-1] # État caché final représentant l'ensemble du mouvement
        
        prob = self.classifier(encoded_traj)
        return prob