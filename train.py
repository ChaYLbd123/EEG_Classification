"""Point d'entrée : entraîne un modèle avec le protocole commun (voir training.py).

Exemples :
  # Essai rapide (10 sujets, 5 époques), en local sur CPU : CD-CNN et CNN-LSTM (H=32)
  python train.py --model cd_cnn cnn_lstm --classes 2 4 --quick

  # Options, seulement sur demande : CNN-LSTM avec 64 unités cachées,
  # ou variante cnn_lstm_moy (moyenne des 120 états cachés au lieu du dernier)
  python train.py --model cnn_lstm --hidden 64 --classes 2 4 --quick
  python train.py --model cnn_lstm --pool moyenne --classes 2 4 --quick

  # Entraînement complet sur Colab (résultats sur Google Drive) :
  python train.py --model cd_cnn cnn_lstm --classes 2 4 \
      --data-dir /content/eegmmidb-1.0.0/files --out-dir /content/drive/MyDrive/eeg_resultats

  # Afficher l'architecture et le nombre de paramètres, sans entraîner :
  python train.py --model cd_cnn cnn_lstm --classes 2 4 --resume-modele
"""

import argparse

from models import MODELES, construire_modele, nom_modele, resume_modele
from training import Config, lancer_validation_croisee


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", nargs="+", default=["cd_cnn"], choices=sorted(MODELES))
    p.add_argument("--hidden", type=int, default=32, choices=[32, 64],
                   help="unités cachées du LSTM (32 par défaut ; 64 = option, seulement sur demande)")
    p.add_argument("--pool", default="dernier", choices=["dernier", "moyenne"],
                   help="CNN-LSTM : résumé de la séquence. 'dernier' = dernier état caché (modèle validé) ; "
                        "'moyenne' = moyenne des 120 états cachés (variante cnn_lstm_moy, seulement sur demande)")
    p.add_argument("--classes", type=int, nargs="+", default=[2, 4], choices=[2, 4])
    p.add_argument("--data-dir", default=Config.data_dir)
    p.add_argument("--out-dir", default=Config.out_dir)
    p.add_argument("--cache-dir", default=None, help="cache .npz des données (défaut : <out-dir>/cache)")
    p.add_argument("--quick", action="store_true", help="essai rapide : 10 sujets, 5 époques")
    p.add_argument("--n-sujets", type=int, default=None,
                   help="sujets 1..N seulement (essais plus longs sur CPU) ; défaut : 109 (soit 105 sujets)")
    p.add_argument("--epoques", type=int, default=None, help="nombre maximum d'époques (défaut : 50)")
    p.add_argument("--adapt-norm", default="both", choices=["complet", "train", "both"])
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--scheduler", action="store_true",
                   help="lr du notebook 0.01/0.001/0.0001 au lieu du lr fixe 0.001 (désactivé par défaut)")
    p.add_argument("--lr", type=float, default=Config.lr,
                   help="learning rate fixe de l'entraînement global (défaut 0.001 ; variante : 5e-4)")
    p.add_argument("--sans-adaptation", action="store_true",
                   help="entraînement global seulement (criblage des variantes) ; "
                        "relancer sans cette option ajoute l'adaptation aux plis existants")
    p.add_argument("--clip", action="store_const", const=1.0, default=None,
                   help="écrêtage de la norme du gradient à 1.0 (désactivé par défaut)")
    p.add_argument("--norm-essai", action="store_true",
                   help="z-score de chaque fenêtre, électrode par électrode (désactivé par défaut)")
    p.add_argument("--resume-modele", action="store_true", help="affiche les couches et quitte")
    a = p.parse_args()

    for modele in a.model:
        for n_classes in a.classes:
            if a.resume_modele:
                print(f"\n{nom_modele(modele, a.hidden, a.pool)}, {n_classes} classes")
                resume_modele(construire_modele(modele, n_classes, a.hidden, a.pool))
                continue
            extra = {}
            if a.n_sujets:
                extra["n_subjects"] = a.n_sujets
            if a.epoques:
                extra["max_epochs"] = a.epoques
            cfg = Config(**extra, model=modele, hidden=a.hidden, pool=a.pool, n_classes=n_classes, data_dir=a.data_dir, out_dir=a.out_dir,
                         cache_dir=a.cache_dir, quick=a.quick, adapt_norm=a.adapt_norm, seed=a.seed,
                         scheduler=a.scheduler, clip=a.clip, norm_essai=a.norm_essai,
                         lr=a.lr, sans_adaptation=a.sans_adaptation)
            lancer_validation_croisee(cfg)


if __name__ == "__main__":
    main()
