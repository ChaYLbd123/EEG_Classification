"""Registre des modèles et petits outils (nombre de paramètres, résumé couche par couche)."""

import torch

from .cd_cnn import CDCNN
from .cnn_lstm import CNNLSTM

MODELES = {
    "cd_cnn": CDCNN,
    "cnn_lstm": CNNLSTM,   # taille de l'état caché : option hidden (32 par défaut, 64 en option)
}


def construire_modele(nom, n_classes, hidden=32, pool="dernier"):
    """pool (CNN-LSTM seulement) : "dernier" = dernier état caché (modèle validé),
    "moyenne" = moyenne des 120 états cachés (variante cnn_lstm_moy)."""
    if nom == "cnn_lstm":
        return CNNLSTM(n_classes=n_classes, hidden=hidden, lecture=pool)
    return MODELES[nom](n_classes=n_classes)


def nom_modele(nom, hidden=32, pool="dernier"):
    """Nom affiché et nom de dossier : cd_cnn, cnn_lstm32, cnn_lstm64, cnn_lstm_moy32..."""
    if nom != "cnn_lstm":
        return nom
    return f"cnn_lstm_moy{hidden}" if pool == "moyenne" else f"cnn_lstm{hidden}"


def compter_parametres(model):
    """Compte les paramètres, en séparant ce que Keras et PyTorch comptent différemment.

    Keras compte dans « Total params » les moyennes et variances glissantes de la BatchNorm
    (« non-trainable »). PyTorch les range dans des *buffers*, qui ne sont pas des paramètres.
    """
    entrainables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers_bn = sum(b.numel() for n, b in model.named_buffers()
                     if n.endswith("running_mean") or n.endswith("running_var"))
    return {"entrainables": entrainables, "buffers_bn": buffers_bn,
            "total_equivalent_keras": entrainables + buffers_bn}


def resume_modele(model, forme_entree=(1, 1, 64, 480)):
    """Affiche la forme de sortie et le nombre de paramètres de chaque couche,
    un peu comme model.summary() en Keras."""
    lignes = []

    def hook(module, _inp, out):
        n = sum(p.numel() for p in module.parameters(recurse=False))
        if isinstance(out, tuple):      # le LSTM renvoie (sorties, (h_n, c_n))
            out = out[0]
        lignes.append((module._nom, type(module).__name__, tuple(out.shape), n))

    handles = []
    for nom, m in model.named_children():
        m._nom = nom
        handles.append(m.register_forward_hook(hook))
    etat = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(forme_entree))
    model.train(etat)
    for h in handles:
        h.remove()

    print(f"{'Couche':<18}{'Type':<14}{'Forme de sortie':<22}{'Param.':>8}")
    print("-" * 62)
    for nom, typ, forme, n in lignes:
        print(f"{nom:<18}{typ:<14}{str(forme):<22}{n:>8}")
    c = compter_parametres(model)
    print("-" * 62)
    print(f"Paramètres entraînables (PyTorch)     : {c['entrainables']:,}")
    print(f"Buffers BatchNorm (moyenne/variance)  : {c['buffers_bn']:,}")
    print(f"Total équivalent Keras model.summary(): {c['total_equivalent_keras']:,}")
    return c
