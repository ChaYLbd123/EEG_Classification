# Classification EEG d'imagerie motrice (PhysioNet EEGMMIDB), en PyTorch

Dataset : PhysioNet EEG Motor Movement/Imagery. 64 électrodes, 160 Hz, fenêtres de 3 s
(480 échantillons), 105 sujets (les 109 moins les sujets 88, 92, 100 et 104).
Tâches : 4 classes (repos, main gauche, main droite, pieds) et 2 classes (gauche, droite).

Les notebooks Keras d'origine (`notebooks/`) sont gardés tels quels, comme référence historique.
Ils ne sont pas exécutables de haut en bas (fonctions non définies, état caché du noyau),
donc leurs chiffres ne servent que de repère approximatif.

## Structure

| Fichier | Rôle |
|---|---|
| `data.py` | `load_full_dataset` copié de `cd-cnn.ipynb` (labels entiers, erreurs affichées), passage au format PyTorch `(N, 1, 64, 480)`, cache `.npz` |
| `models/cd_cnn.py` | CD-CNN (type EEGNet) porté de Keras |
| `training.py` | Boucle commune : GroupKFold, early stopping, adaptation au sujet, reprise après coupure |
| `train.py` | Ligne de commande |
| `colab_run.ipynb` | Lancement sur Google Colab (GPU) |

## Protocole

1. **GroupKFold à 5 plis par sujet.** Les sujets de test ne sont jamais vus à l'entraînement.
2. **Validation = 10 % des sujets d'entraînement** (GroupShuffleSplit). L'early stopping
   (patience 15, meilleurs poids restaurés) regarde cette validation, jamais le test.
3. Adam (eps 1e-7), batch 16, 100 époques maximum, learning rate 0.01 / 0.001 / 0.0001
   (changements aux époques 20 et 50), entropie croisée, sans augmentation.
4. **Adaptation au sujet.** Le modèle du pli k est adapté uniquement aux sujets de test du pli k.
   Pour chaque sujet : KFold 4 plis (75 % / 25 %), 5 époques, Adam lr 5e-4. On mesure la
   précision avant et après, sur les mêmes 25 %.
5. Normalisation : z-score par sujet et par électrode (comme dans le notebook). Les statistiques
   utilisent toutes les fenêtres du sujet, ce qui crée une **fuite légère** pendant
   l'adaptation (aucun label n'est utilisé, mais les fenêtres de test du sujet entrent dans sa
   moyenne et son écart-type). L'option `--adapt-norm train` calcule ces statistiques sur les
   75 % d'adaptation seulement. `both` (par défaut) rapporte les deux résultats.

### Corrections par rapport aux notebooks

- Plus de « meilleur pli choisi sur le test ». Avant, un seul modèle était adapté aux 105
  sujets, alors qu'il avait déjà appris sur environ 80 % d'entre eux.
- Plus d'early stopping sur le test.
- Plus de `except: continue` silencieux : le nombre de sujets chargés est affiché.

## Utilisation

```bash
python train.py --model cd_cnn --classes 2 4 --resume-modele   # couches + nombre de paramètres
python train.py --model cd_cnn --classes 2 4 --quick           # essai rapide : 10 sujets, 5 époques
python train.py --model cd_cnn --classes 2 4 --data-dir <...>/files --out-dir <dossier>
```

Les résultats (un fichier json par pli, les poids `.pt`, `resume.json`) sont écrits dans
`<out-dir>/<modèle>_<n>cl/`.
