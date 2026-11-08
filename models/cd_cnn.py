"""CD-CNN en PyTorch : portage de `build_model_optimized` (notebooks/cd-cnn.ipynb, cellule 9).

C'est une architecture de type EEGNet :
  bloc 1 : convolution temporelle  -> apprend des filtres en fréquence (comme des passe-bande)
  bloc 2 : convolution depthwise   -> apprend des filtres spatiaux (combinaisons d'électrodes)
  bloc 3 : convolution séparable   -> combine les motifs temporels de façon compacte
  classifieur : couche dense -> logits

Convention des formes : (batch, canaux, hauteur, largeur) = (B, C, électrodes, temps).
Entrée : (B, 1, 64, 480), soit 64 électrodes et 3 s à 160 Hz.

Équivalences Keras -> PyTorch appliquées ici :
  - BatchNormalization : momentum Keras 0.99 = momentum PyTorch 0.01, et eps 1e-3 au lieu de 1e-5.
  - Initialisation : glorot_uniform (Keras) = xavier_uniform_ (PyTorch), biais à zéro.
    Exception : pour la depthwise, Keras ne calcule pas les « fans » comme PyTorch.
    On recopie donc la formule de Keras à la main (voir _init_comme_keras).
  - MaxNorm : PyTorch n'a pas de contrainte intégrée. On l'applique après chaque
    optimizer.step(), en appelant appliquer_contraintes().
  - Pas de softmax à la fin : nn.CrossEntropyLoss le calcule elle-même (log-softmax).
"""

import math

import torch
import torch.nn as nn


class CDCNN(nn.Module):
    def __init__(self, n_classes=4, n_channels=64, n_times=480,
                 F1=8, D=2, F2=16, kernel_length=128, sep_kernel=16,
                 pool1=8, pool2=8, dropout_rate=0.5):
        super().__init__()
        self.F1, self.D, self.n_channels = F1, D, n_channels

        # --- BLOC 1 : convolution temporelle ---
        # Keras : Conv2D(F1, (1, kernel_length), padding='same', use_bias=False)
        # Le noyau (1, 128) glisse sur le temps seulement (128 échantillons = 0,8 s).
        # padding='same' de Keras : la sortie garde 480 instants. Le noyau fait 128 (pair),
        # donc il faut 127 zéros : Keras en met 63 à gauche et 64 à droite.
        # On écrit ce padding explicitement (ZeroPad2d) pour éviter toute ambiguïté.
        self.pad_temporel = _padding_same_keras(kernel_length)
        self.conv_temporelle = nn.Conv2d(1, F1, (1, kernel_length), bias=False)
        self.bn1 = nn.BatchNorm2d(F1, momentum=0.01, eps=1e-3)

        # --- BLOC 2 : convolution spatiale depthwise ---
        # Keras : DepthwiseConv2D((64, 1), depth_multiplier=D, MaxNorm(1.), use_bias=False)
        # groups=F1 : chaque carte temporelle a ses propres D filtres spatiaux (64 poids chacun).
        # Le noyau (64, 1) couvre toutes les électrodes d'un coup : la hauteur passe de 64 à 1.
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=1e-3)
        self.elu = nn.ELU()  # alpha = 1.0, comme Activation('elu') en Keras
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.drop1 = nn.Dropout(dropout_rate)  # comme Keras Dropout : éteint des valeurs isolées

        # --- BLOC 3 : convolution séparable ---
        # Keras : SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)
        # Elle se décompose en deux convolutions :
        #   a) depthwise (1, 16) : un filtre temporel par carte (groups = nombre de cartes)
        #   b) pointwise (1, 1)  : mélange les cartes entre elles
        self.pad_sep = _padding_same_keras(sep_kernel)       # 7 zéros à gauche, 8 à droite
        self.sep_depthwise = nn.Conv2d(F1 * D, F1 * D, (1, sep_kernel), groups=F1 * D, bias=False)
        self.sep_pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2, momentum=0.01, eps=1e-3)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop2 = nn.Dropout(dropout_rate)

        # --- CLASSIFICATION ---
        # Après les deux poolings : 480 // 8 = 60, puis 60 // 8 = 7 (division entière,
        # comme le padding 'valid' de Keras). Donc 16 cartes x 7 instants = 112 valeurs.
        self.n_features = F2 * ((n_times // pool1) // pool2)
        self.flatten = nn.Flatten()
        # Keras : Dense(n_classes, activation='softmax', kernel_constraint=MaxNorm(0.25))
        self.classifieur = nn.Linear(self.n_features, n_classes)

        self._init_comme_keras()

    # ------------------------------------------------------------------
    def forward(self, x):
        # x : (B, 1, 64, 480)
        x = self.pad_temporel(x)     # (B, 1, 64, 607)  480 + 63 + 64 zéros sur le temps
        x = self.conv_temporelle(x)  # (B, 8, 64, 480)  8 filtres temporels
        x = self.bn1(x)              # (B, 8, 64, 480)

        x = self.depthwise(x)        # (B, 16, 1, 480)  8 x D=2 filtres spatiaux
        x = self.bn2(x)              # (B, 16, 1, 480)
        x = self.elu(x)              # (B, 16, 1, 480)
        x = self.pool1(x)            # (B, 16, 1, 60)   moyenne sur 8 instants
        x = self.drop1(x)            # (B, 16, 1, 60)

        x = self.pad_sep(x)          # (B, 16, 1, 75)   60 + 7 + 8
        x = self.sep_depthwise(x)    # (B, 16, 1, 60)
        x = self.sep_pointwise(x)    # (B, 16, 1, 60)
        x = self.bn3(x)              # (B, 16, 1, 60)
        x = self.elu(x)              # (B, 16, 1, 60)
        x = self.pool2(x)            # (B, 16, 1, 7)
        x = self.drop2(x)            # (B, 16, 1, 7)

        x = self.flatten(x)          # (B, 112)
        return self.classifieur(x)   # (B, n_classes)  logits, sans softmax

    # ------------------------------------------------------------------
    def _init_comme_keras(self):
        """Reproduit l'initialisation par défaut de Keras."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)   # = glorot_uniform
                if m.bias is not None:
                    nn.init.zeros_(m.bias)          # Keras : biais à zéro
        # Cas particulier de la depthwise. Keras range le noyau en (64, 1, F1, D) et calcule
        # fan_in = F1*64 = 512 et fan_out = D*64 = 128. PyTorch range le poids en
        # (F1*D, 1, 64, 1) et calcule fan_in = 64 et fan_out = 1024 : la borne serait différente.
        fan_in = self.F1 * self.n_channels
        fan_out = self.D * self.n_channels
        limite = math.sqrt(6.0 / (fan_in + fan_out))
        with torch.no_grad():
            self.depthwise.weight.uniform_(-limite, limite)
        # (Pour la séparable, Keras et PyTorch tombent sur la même somme fan_in + fan_out = 272.)

    @torch.no_grad()
    def appliquer_contraintes(self):
        """MaxNorm de Keras, à appeler après chaque optimizer.step()."""
        _max_norm_(self.depthwise.weight, 1.0)     # chaque filtre spatial : norme <= 1
        _max_norm_(self.classifieur.weight, 0.25)  # poids de chaque classe : norme <= 0.25


def _padding_same_keras(k):
    """Padding 'same' de Keras sur l'axe du temps : (k-1)//2 zéros à gauche, le reste à droite."""
    total = k - 1
    return nn.ZeroPad2d((total // 2, total - total // 2, 0, 0))  # (gauche, droite, haut, bas)


def _max_norm_(w, max_value, eps=1e-7):
    """Même formule que keras.constraints.MaxNorm : w * min(n, max) / (eps + n).

    La norme n est calculée pour chaque filtre de sortie (première dimension du poids PyTorch).
    Ça correspond à axis=0 du noyau Keras, qui est rangé dans l'autre sens.
    """
    normes = w.flatten(1).norm(dim=1).view(-1, *([1] * (w.dim() - 1)))
    w.mul_(normes.clamp(max=max_value) / (eps + normes))
