"""CNN-LSTM : bloc CNN du CD-CNN, suivi d'un LSTM qui lit la séquence dans le temps.

Idée :
  - Le CNN fait le travail de traitement du signal sur chaque petit moment :
    filtres en fréquence (convolution temporelle) et combinaisons d'électrodes
    (convolution depthwise). Il transforme 64 électrodes x 480 instants en
    16 caractéristiques x 120 pas de temps.
  - Le LSTM lit ces 120 pas dans l'ordre et garde une mémoire de ce qu'il a vu.
    Il peut ainsi repérer comment l'activité évolue pendant les 3 s.
  - Son dernier état caché résume la fenêtre. Une couche dense le transforme en logits.

Différences avec le CD-CNN :
  - On garde : convolution temporelle (8 filtres, noyau 128), BN, depthwise (D=2, MaxNorm 1),
    BN, ELU, pooling, dropout.
  - Le pooling vaut (1, 4) et non (1, 8) : 480 / 4 = 120 pas de temps pour le LSTM.
    C'est le premier pooling décrit dans l'article (section 2.2.2).
  - Le LSTM remplace la convolution séparable et le 2e pooling.
  - Pas de MaxNorm sur la couche dense : les états cachés sont entre -1 et 1, et une
    norme <= 0.25 limiterait trop la taille des logits.

Entrée : (B, 1, 64, 480). Sortie : (B, n_classes), des logits sans softmax.
"""

import math

import torch
import torch.nn as nn

from .cd_cnn import _max_norm_, _padding_same_keras


class CNNLSTM(nn.Module):
    def __init__(self, n_classes=4, hidden=32, n_channels=64, n_times=480,
                 F1=8, D=2, kernel_length=128, pool=4, dropout_rate=0.5):
        super().__init__()
        self.F1, self.D, self.n_channels, self.hidden = F1, D, n_channels, hidden
        n_feat = F1 * D               # 16 caractéristiques par pas de temps
        self.n_pas = n_times // pool  # 120 pas de temps

        # --- Bloc CNN (repris du CD-CNN) ---
        self.pad_temporel = _padding_same_keras(kernel_length)        # 63 zéros à gauche, 64 à droite
        self.conv_temporelle = nn.Conv2d(1, F1, (1, kernel_length), bias=False)
        self.bn1 = nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)
        self.depthwise = nn.Conv2d(F1, n_feat, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_feat, momentum=0.01, eps=1e-3)
        self.elu = nn.ELU()
        self.pool = nn.AvgPool2d((1, pool))
        self.drop_cnn = nn.Dropout(dropout_rate)

        # --- LSTM ---
        # batch_first=True : l'entrée est rangée (batch, pas_de_temps, caractéristiques).
        # (Sans cette option, PyTorch attend (pas_de_temps, batch, caractéristiques).)
        self.lstm = nn.LSTM(input_size=n_feat, hidden_size=hidden, num_layers=1, batch_first=True)
        self.drop_lstm = nn.Dropout(dropout_rate)

        # --- Classifieur ---
        self.classifieur = nn.Linear(hidden, n_classes)

        self._init_comme_keras()

    # ------------------------------------------------------------------
    def forward(self, x):
        # x : (B, 1, 64, 480)
        x = self.pad_temporel(x)      # (B, 1, 64, 607)
        x = self.conv_temporelle(x)   # (B, 8, 64, 480)   8 filtres temporels (0,8 s)
        x = self.bn1(x)               # (B, 8, 64, 480)
        x = self.depthwise(x)         # (B, 16, 1, 480)   2 filtres spatiaux par filtre temporel
        x = self.bn2(x)               # (B, 16, 1, 480)
        x = self.elu(x)               # (B, 16, 1, 480)
        x = self.pool(x)              # (B, 16, 1, 120)   moyenne sur 4 instants (25 ms)
        x = self.drop_cnn(x)          # (B, 16, 1, 120)

        # Remise en forme pour le LSTM : (batch, pas_de_temps, caractéristiques).
        x = x.squeeze(2)              # (B, 16, 120)      on retire la hauteur, qui vaut 1
        x = x.permute(0, 2, 1)        # (B, 120, 16)      le temps passe en 2e position

        sorties, (h_n, c_n) = self.lstm(x)
        # sorties : (B, 120, H)  état caché à chaque pas de temps
        # h_n     : (1, B, H)    dernier état caché (1 = nombre de couches)
        # c_n     : (1, B, H)    dernière mémoire de cellule (non utilisée ici)
        h = h_n[-1]                   # (B, H)  identique à sorties[:, -1, :] ici
        h = self.drop_lstm(h)         # (B, H)
        return self.classifieur(h)    # (B, n_classes)   logits

    # ------------------------------------------------------------------
    def _init_comme_keras(self):
        """Initialisation par défaut de Keras (couches Conv2D, Dense et LSTM)."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Depthwise : mêmes « fans » que Keras (voir cd_cnn.py).
        limite = math.sqrt(6.0 / (self.F1 * self.n_channels + self.D * self.n_channels))
        with torch.no_grad():
            self.depthwise.weight.uniform_(-limite, limite)

        H = self.hidden
        for nom, p in self.lstm.named_parameters():
            if nom.startswith("weight_ih"):
                # Poids entrée -> portes, forme (4H, 16). Xavier (glorot_uniform) :
                # la variance des signaux reste stable entre l'entrée et les portes.
                nn.init.xavier_uniform_(p)
            elif nom.startswith("weight_hh"):
                # Poids état caché -> portes, forme (4H, H). Orthogonale : multiplier
                # par une matrice orthogonale ne change pas la norme d'un vecteur.
                # Le signal ne grossit pas et ne s'éteint pas d'un pas à l'autre,
                # ce qui limite l'explosion ou la disparition du gradient sur 120 pas.
                nn.init.orthogonal_(p)
            elif nom.startswith("bias"):
                # PyTorch a deux biais (bias_ih et bias_hh). Ils sont additionnés, donc on
                # met tout à 0 sauf le biais de la porte d'oubli de bias_ih, mis à 1.
                # Ordre des portes dans PyTorch : entrée (i), oubli (f), candidat (g), sortie (o).
                # Biais d'oubli à 1 : au début, sigmoid(1) ≈ 0,73. La porte laisse passer
                # la mémoire, donc le LSTM n'oublie pas tout dès le départ et les gradients
                # remontent plus loin dans le temps (Jozefowicz et al., 2015).
                nn.init.zeros_(p)
                if nom.startswith("bias_ih"):
                    with torch.no_grad():
                        p[H:2 * H].fill_(1.0)

    @torch.no_grad()
    def appliquer_contraintes(self):
        """MaxNorm(1) sur la depthwise seulement (pas sur la couche dense)."""
        _max_norm_(self.depthwise.weight, 1.0)
