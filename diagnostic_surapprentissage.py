"""Test de sur-apprentissage d'un petit lot (« overfit a small batch »).

Idée : on donne au modèle très peu de données (64 fenêtres) et beaucoup de pas
d'optimisation. Un modèle correct, sans bug, doit pouvoir les apprendre par cœur :
la perte sur ces 64 fenêtres doit tendre vers 0 et la précision vers 100 %.
Si ça n'arrive pas, le problème est dans le modèle ou la boucle (formes, gradient,
initialisation...), pas dans la quantité de données.

Ce test ne mesure PAS la généralisation : on évalue sur les mêmes 64 fenêtres.

Réglages : 4 classes, 16 fenêtres par classe (10 premiers sujets), Adam lr 0.001,
batch 16, dropout et MaxNorm laissés actifs (le vrai modèle), 600 pas.
La perte est mesurée en mode eval (sans dropout) tous les 10 pas.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from data import charger_donnees
from models import construire_modele, nom_modele
from rapport import courbe_surapprentissage
from training import creer_optimiseur, faire_loader, fixer_graine


def tester(modele, X, y, n_pas, seed=42, hidden=32):
    fixer_graine(seed)
    model = construire_modele(modele, 4, hidden)
    opt = creer_optimiseur(model, 1e-3)
    critere = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    tous = np.arange(len(y))
    pas, pertes, precs = [], [], []
    k = 0
    while k < n_pas:
        for xb, yb in faire_loader(Xt, yt, tous, 16, shuffle=True, seed=seed + k):
            model.train()
            opt.zero_grad()
            critere(model(xb), yb).backward()
            opt.step()
            model.appliquer_contraintes()
            k += 1
            if k % 10 == 0 or k == 1:
                model.eval()
                with torch.no_grad():
                    logits = model(Xt)
                pas.append(k)
                pertes.append(critere(logits, yt).item())
                precs.append((logits.argmax(1) == yt).float().mean().item())
            if k >= n_pas:
                break
    return {"pas": pas, "perte_eval": pertes, "precision_eval": precs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="C:/Users/DELL/Downloads/eegmmidb-1.0.0/files")
    p.add_argument("--out-dir", default="resultats")
    p.add_argument("--pas", type=int, default=600)
    a = p.parse_args()

    X, y, _ = charger_donnees(a.data_dir, 4, 10, True, os.path.join(a.out_dir, "cache"))
    rng = np.random.default_rng(42)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], 16, replace=False) for c in range(4)])
    X, y = X[idx], y[idx]
    print(f"Petit lot : {X.shape}, classes {np.bincount(y).tolist()}")

    res = {}
    for modele in ("cnn_lstm", "cd_cnn"):        # CD-CNN = référence
        cle = nom_modele(modele, 32)
        res[cle] = tester(modele, X, y, a.pas)
        r = res[cle]
        print(f"{cle:11s} perte eval : pas 1 = {r['perte_eval'][0]:.3f}, "
              + ", ".join(f"pas {s} = {l:.3f}" for s, l in zip(r["pas"], r["perte_eval"]) if s in (100, 200, 300, 400, 600))
              + f" | précision finale {r['precision_eval'][-1]:.0%}")

    dossier = os.path.join(a.out_dir, "diagnostics")
    os.makedirs(dossier, exist_ok=True)
    with open(os.path.join(dossier, "surapprentissage.json"), "w", encoding="utf-8") as f:
        json.dump(res, f)
    courbe_surapprentissage(res, os.path.join(dossier, "surapprentissage.png"))

    r = res["cnn_lstm32"]
    ok = r["perte_eval"][-1] < 0.05 and r["precision_eval"][-1] == 1.0
    print("VERDICT CNN-LSTM-32 :", "OK, le modèle peut apprendre (perte -> 0, 100 %)" if ok
          else "ÉCHEC : le modèle n'arrive pas à apprendre 64 fenêtres par cœur")


if __name__ == "__main__":
    main()
