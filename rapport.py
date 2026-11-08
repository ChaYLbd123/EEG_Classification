"""Figures (PNG) : courbes de perte et test de sur-apprentissage.

Conventions graphiques (les mêmes partout) :
  - une couleur par modèle : CD-CNN = bleu, CNN-LSTM = orange ;
  - train = trait plein, validation = pointillés. L'identité ne repose donc pas
    seulement sur la couleur ;
  - grille discrète, une seule échelle y par graphique, légende + étiquettes directes.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")                     # pas de fenêtre : on écrit directement des fichiers
import matplotlib.pyplot as plt
import numpy as np

COULEURS = {"cd_cnn": "#2a78d6", "cnn_lstm32": "#eb6834", "cnn_lstm64": "#eb6834"}
NOMS = {"cd_cnn": "CD-CNN", "cnn_lstm32": "CNN-LSTM-32", "cnn_lstm64": "CNN-LSTM-64"}
TEXTE, TEXTE2, GRILLE, FOND = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(FOND)
    ax.grid(True, color=GRILLE, linewidth=0.8)
    ax.set_axisbelow(True)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color(TEXTE2)
    ax.tick_params(colors=TEXTE2, labelsize=9)


def courbes_perte(dossiers, chemin_png, titre):
    """Perte train (plein) et validation (pointillés), moyenne sur les plis, un panneau par modèle.

    dossiers : {cle_modele: dossier_resultats}. Les plis peuvent s'arrêter à des époques
    différentes (early stopping) : la moyenne à l'époque e porte sur les plis encore actifs.
    """
    fig, axes = plt.subplots(1, len(dossiers), figsize=(5.2 * len(dossiers), 3.8),
                             sharey=True, facecolor=FOND)
    axes = np.atleast_1d(axes)
    for ax, (cle, dossier) in zip(axes, dossiers.items()):
        plis = sorted(f for f in os.listdir(dossier) if f.endswith("_global.json"))
        H = [json.load(open(os.path.join(dossier, f), encoding="utf-8"))["historique"] for f in plis]
        coul = COULEURS[cle]
        for nom, style in (("train_loss", "-"), ("val_loss", "--")):
            n_max = max(len(h[nom]) for h in H)
            for h in H:   # chaque pli, en fin et transparent
                ax.plot(range(1, len(h[nom]) + 1), h[nom], style, color=coul, lw=0.8, alpha=0.25)
            moy = [np.mean([h[nom][e] for h in H if len(h[nom]) > e]) for e in range(n_max)]
            ax.plot(range(1, n_max + 1), moy, style, color=coul, lw=2,
                    label="train (moyenne)" if nom == "train_loss" else "validation (moyenne)")
            ax.annotate("train" if nom == "train_loss" else "val", (n_max, moy[-1]),
                        xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=9, color=TEXTE2)
        ax.axhline(np.log(4), color=TEXTE2, lw=1, ls=":")
        ax.annotate("hasard (ln 4)", (1, np.log(4)), xytext=(2, 3), textcoords="offset points",
                    fontsize=8, color=TEXTE2)
        _style(ax)
        ax.set_title(NOMS[cle], color=TEXTE, fontsize=11, loc="left")
        ax.set_xlabel("Époque", color=TEXTE2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXTE2, loc="lower left")
    axes[0].set_ylabel("Entropie croisée", color=TEXTE2, fontsize=9)
    fig.suptitle(titre, color=TEXTE, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(chemin_png, dpi=150, facecolor=FOND)
    plt.close(fig)
    print(f"Figure écrite : {chemin_png}")


def courbe_surapprentissage(resultats, chemin_png):
    """resultats : {cle_modele: {"pas": [...], "perte_eval": [...]}}."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=FOND)
    for cle, r in resultats.items():
        ax.plot(r["pas"], r["perte_eval"], color=COULEURS[cle], lw=2,
                ls="-" if cle.startswith("cnn_lstm") else "--", label=NOMS[cle])
        ax.annotate(f"{NOMS[cle]} : {r['perte_eval'][-1]:.3f}", (r["pas"][-1], r["perte_eval"][-1]),
                    xytext=(-4, 8), textcoords="offset points", ha="right", fontsize=9, color=TEXTE2)
    ax.axhline(np.log(4), color=TEXTE2, lw=1, ls=":")
    ax.annotate("hasard (ln 4 = 1,386)", (0, np.log(4)), xytext=(2, 3), textcoords="offset points",
                fontsize=8, color=TEXTE2)
    _style(ax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Pas d'optimisation (batch de 16)", color=TEXTE2, fontsize=9)
    ax.set_ylabel("Entropie croisée sur les 64 fenêtres\n(mode eval)", color=TEXTE2, fontsize=9)
    ax.set_title("Test de sur-apprentissage : 64 fenêtres, 4 classes", color=TEXTE, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXTE2)
    fig.tight_layout()
    fig.savefig(chemin_png, dpi=150, facecolor=FOND)
    plt.close(fig)
    print(f"Figure écrite : {chemin_png}")
