"""Point d'entrée : entraîne un modèle avec le protocole commun (voir training.py).

Exemples :
  # Essai rapide (10 sujets, 5 époques), en local sur CPU :
  python train.py --model cd_cnn --classes 2 4 --quick

  # Entraînement complet sur Colab (résultats sur Google Drive) :
  python train.py --model cd_cnn --classes 2 4 \
      --data-dir /content/eegmmidb-1.0.0/files --out-dir /content/drive/MyDrive/eeg_resultats

  # Afficher l'architecture et le nombre de paramètres, sans entraîner :
  python train.py --model cd_cnn --classes 2 4 --resume-modele
"""

import argparse

from models import MODELES, construire_modele, resume_modele
from training import Config, lancer_validation_croisee


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="cd_cnn", choices=sorted(MODELES))
    p.add_argument("--classes", type=int, nargs="+", default=[2, 4], choices=[2, 4])
    p.add_argument("--data-dir", default=Config.data_dir)
    p.add_argument("--out-dir", default=Config.out_dir)
    p.add_argument("--cache-dir", default=None, help="cache .npz des données (défaut : <out-dir>/cache)")
    p.add_argument("--quick", action="store_true", help="essai rapide : 10 sujets, 5 époques")
    p.add_argument("--adapt-norm", default="both", choices=["complet", "train", "both"])
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--resume-modele", action="store_true", help="affiche les couches et quitte")
    a = p.parse_args()

    for n_classes in a.classes:
        if a.resume_modele:
            print(f"\n{a.model}, {n_classes} classes")
            resume_modele(construire_modele(a.model, n_classes))
            continue
        cfg = Config(model=a.model, n_classes=n_classes, data_dir=a.data_dir, out_dir=a.out_dir,
                     cache_dir=a.cache_dir, quick=a.quick, adapt_norm=a.adapt_norm, seed=a.seed)
        lancer_validation_croisee(cfg)


if __name__ == "__main__":
    main()
