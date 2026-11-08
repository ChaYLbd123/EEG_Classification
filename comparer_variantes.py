"""Compare les variantes de réglage (LECTURE SEULE : aucun résultat n'est modifié).

Critère de choix (fixé dans NOTES.md §6 AVANT les essais) :
  perte de validation retenue, moyennée sur les 5 plis. Pour chaque pli, c'est la perte
  sur les sujets de validation (10 % des sujets d'entraînement) à l'époque gardée par
  l'early stopping. Plus elle est basse, mieux c'est.
La précision test est affichée POUR INFORMATION SEULEMENT : elle ne sert jamais à choisir.

Usage :
  python comparer_variantes.py --out-dir <dossier des résultats> --classes 4
Écrit aussi le tableau en Markdown dans <out-dir>/variantes_<n>cl.md.
"""

import argparse
import json
import os

import numpy as np

# (famille, nom de la variante, dossier). La référence = protocole de l'article, sans réglage.
VARIANTES = [
    ("CD-CNN", "référence (lr 0,001)", "cd_cnn_{n}cl"),
    ("CD-CNN", "(a) --clip 1,0", "cd_cnn_{n}cl_clip"),
    ("CD-CNN", "(c) lr 5e-4", "cd_cnn_{n}cl_lr0.0005"),
    ("CD-CNN", "(d) --norm-essai", "cd_cnn_{n}cl_normessai"),
    ("CNN-LSTM", "référence (lr 0,001)", "cnn_lstm32_{n}cl"),
    ("CNN-LSTM", "(a) --clip 1,0", "cnn_lstm32_{n}cl_clip"),
    ("CNN-LSTM", "(b) cnn_lstm_moy (moyenne des états)", "cnn_lstm_moy32_{n}cl"),
    ("CNN-LSTM", "(c) lr 5e-4", "cnn_lstm32_{n}cl_lr0.0005"),
    ("CNN-LSTM", "(d) --norm-essai", "cnn_lstm32_{n}cl_normessai"),
]


def lire_variante(dossier, n_folds=5):
    plis = [os.path.join(dossier, f"pli{k}_global.json") for k in range(1, n_folds + 1)]
    faits = [p for p in plis if os.path.exists(p)]
    if not faits:
        return None
    G = [json.load(open(p, encoding="utf-8")) for p in faits]
    pv = [g["historique"]["val_loss"][g["meilleure_epoque"] - 1] for g in G]
    r = {"n_plis": len(G), "pv": pv, "pv_moy": np.mean(pv), "pv_std": np.std(pv),
         "acc_moy": np.mean([g["test_acc"] for g in G]), "acc_std": np.std([g["test_acc"] for g in G]),
         "epoques": np.mean([g["epoques_faites"] for g in G]),
         "meilleure": [g["meilleure_epoque"] for g in G],
         "s_epoque": np.mean([np.mean(g["historique"]["temps_s"]) for g in G]),
         "device": G[0]["device"]}
    for mode in ("complet", "train"):
        A = [os.path.join(dossier, f"pli{k}_adapt_{mode}.json") for k in range(1, n_folds + 1)]
        if all(os.path.exists(a) for a in A):
            r[f"adapt_{mode}"] = np.mean([json.load(open(a, encoding="utf-8"))["apres"] for a in A])
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="resultats")
    p.add_argument("--classes", type=int, default=4, choices=[2, 4])
    a = p.parse_args()

    lignes = [f"### Variantes, {a.classes} classes (critère : perte de validation retenue, la plus basse gagne)",
              "",
              "| Modèle | Variante | Plis faits | **Perte val. retenue** (moy ± std) | Précision test (info) | "
              "Plis meilleurs que la réf. (val.) | Adapt. complet | Adapt. train | Époques (moy) | "
              "Meilleures époques | s/époque | Device |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    meilleures, refs = {}, {}
    for famille, nom, motif in VARIANTES:
        dossier = os.path.join(a.out_dir, motif.format(n=a.classes))
        r = lire_variante(dossier)
        if r is None:
            lignes.append(f"| {famille} | {nom} | 0/5 | non exécutée | | | | | | | | |")
            continue
        if nom.startswith("référence"):
            refs[famille] = r["pv"]
        # Comparaison pli à pli : les sujets de validation sont les mêmes pour toutes les variantes.
        if famille in refs and not nom.startswith("référence"):
            n = min(len(r["pv"]), len(refs[famille]))
            gagnes = f"{sum(r['pv'][i] < refs[famille][i] for i in range(n))}/{n}"
        else:
            gagnes = "—"
        complet = r["n_plis"] == 5
        ad = lambda m: f"{r[m]:.2%}" if m in r else "non calculée"
        lignes.append(
            f"| {famille} | {nom} | {r['n_plis']}/5 | **{r['pv_moy']:.4f} ± {r['pv_std']:.4f}** | "
            f"{r['acc_moy']:.2%} ± {r['acc_std']:.2%} | {gagnes} | {ad('adapt_complet')} | {ad('adapt_train')} | "
            f"{r['epoques']:.1f} | {r['meilleure']} | {r['s_epoque']:.2f} | {r['device']} |")
        if complet and (famille not in meilleures or r["pv_moy"] < meilleures[famille][1]):
            meilleures[famille] = (nom, r["pv_moy"])

    lignes.append("")
    for famille, (nom, pv) in meilleures.items():
        lignes.append(f"- Plus basse perte de validation, {famille} : **{nom}** ({pv:.4f}). "
                      "Seules les variantes à 5 plis complets comptent.")
    lignes.append("- La précision test est donnée pour information seulement. Elle ne sert pas au choix.")
    texte = "\n".join(lignes)
    print(texte)
    chemin = os.path.join(a.out_dir, f"variantes_{a.classes}cl.md")
    with open(chemin, "w", encoding="utf-8") as f:
        f.write(texte + "\n")
    print(f"\nTableau écrit : {chemin}")


if __name__ == "__main__":
    main()
