# Classification EEG d'imagerie motrice (PhysioNet EEGMMIDB), en PyTorch

Dataset : PhysioNet EEG Motor Movement/Imagery. 64 électrodes, 160 Hz, fenêtres de 3 s
(480 échantillons), 105 sujets (les 109 moins les sujets 88, 92, 100 et 104).
Tâches : 4 classes (repos, main gauche, main droite, pieds) et 2 classes (gauche, droite).

Deux modèles sont comparés avec exactement les mêmes données, le même découpage et les mêmes
hyperparamètres : **CD-CNN** et **CNN-LSTM** (32 unités cachées). Une variante à 64 unités
(`--hidden 64`) existe dans le code comme option, lancée seulement sur demande.

Les notebooks Keras d'origine (`notebooks/`) sont gardés tels quels, comme référence historique.
Ils ne sont pas exécutables de haut en bas, et leurs chiffres contiennent des fuites de données.
Ce ne sont que des repères approximatifs. Voir [NOTES.md](NOTES.md) pour les écarts entre
l'article, les notebooks et ce code.

## Structure

| Fichier | Rôle |
|---|---|
| `data.py` | `load_full_dataset` copié de `cd-cnn.ipynb` (labels entiers, erreurs affichées), format PyTorch `(N, 1, 64, 480)`, z-score « strict », normalisation par essai, cache `.npz` |
| `models/cd_cnn.py` | CD-CNN (type EEGNet) porté de Keras |
| `models/cnn_lstm.py` | CNN-LSTM (bloc CNN du CD-CNN + LSTM) |
| `training.py` | Boucle commune : GroupKFold, early stopping, adaptation au sujet, reprise après coupure |
| `train.py` | Ligne de commande |
| `colab_run.ipynb` | Lancement sur Google Colab (GPU) |
| `NOTES.md` | Différences entre l'article et le code, incohérences de l'article, fuites dans les notebooks |

## Protocole (le même pour tous les modèles)

1. **GroupKFold à 5 plis par sujet.** Les sujets de test ne sont jamais vus à l'entraînement.
2. **Validation = 10 % des sujets d'entraînement** (GroupShuffleSplit). L'early stopping
   (patience 15, meilleurs poids restaurés) regarde cette validation, jamais le test.
3. **Protocole de l'article (§2.3.1)** : Adam, learning rate fixe 0,001 (eps 1e-7, comme Keras),
   batch 16, 50 époques maximum, entropie croisée, sans augmentation.
4. **Adaptation au sujet.** Le modèle du pli k est adapté uniquement aux sujets de test du pli k.
   Pour chaque sujet : KFold 4 plis (75 % / 25 %), 5 époques, Adam lr 5e-4. On mesure la
   précision avant et après, sur les mêmes 25 %.
5. **Normalisation** : z-score par sujet et par électrode (comme dans le notebook). Pour
   l'adaptation, on rapporte deux variantes :
   - `complet` : statistiques sur toutes les fenêtres du sujet, comme le notebook. C'est une
     fuite légère, car les fenêtres de test du sujet entrent dans sa moyenne et son écart-type
     (aucun label n'est utilisé) ;
   - `train` : statistiques sur les 75 % d'adaptation seulement, donc sans fuite.

Options **désactivées par défaut**, lancées seulement sur demande :

| Option | Effet |
|---|---|
| `--scheduler` | learning rate du notebook CD-CNN : 0,01 (époques 0-19), 0,001 (20-49), 0,0001 ensuite |
| `--clip` | écrêtage de la norme du gradient à 1,0 (`clip_grad_norm_`) |
| `--norm-essai` | z-score de chaque fenêtre, électrode par électrode |
| `--hidden 64` | CNN-LSTM avec 64 unités cachées au lieu de 32 |

Les options actives apparaissent dans le nom du dossier de résultats (par exemple `cd_cnn_4cl_clip`).

### Corrections par rapport aux notebooks

- Plus de « meilleur pli choisi sur le test ». Avant, un seul modèle était adapté aux 105
  sujets, alors qu'il avait déjà appris sur environ 80 % d'entre eux.
- Plus d'early stopping sur le test.
- Plus de `except: continue` silencieux : le nombre de sujets chargés est affiché.

## Les modèles

| Modèle | Bloc après la depthwise | Pas de temps en sortie du CNN | Paramètres (2 cl. / 4 cl.) |
|---|---|---|---|
| CD-CNN | pooling (1,8) → convolution séparable → pooling (1,8) → dense | 7 | 2 866 / 3 092 |
| CNN-LSTM-32 | pooling (1,4) → LSTM 32 → dense | 120 | 8 562 / 8 628 |
| CNN-LSTM-64 (option `--hidden 64`) | pooling (1,4) → LSTM 64 → dense | 120 | 23 218 / 23 348 |

Les nombres de paramètres sont ceux de PyTorch (paramètres entraînables). Keras ajouterait
les moyennes et variances de la BatchNorm : +80 pour le CD-CNN, +48 pour le CNN-LSTM.

**Différence de pooling, à connaître :**
- Le **CD-CNN** est porté depuis le notebook : noyau temporel 128, poolings (1,8) puis (1,8),
  BatchNorm, dropout. L'article décrit un noyau de 64 et des poolings (4) puis (8).
- Le **CNN-LSTM** garde le premier pooling **(1,4) de l'article**. Le LSTM remplace la
  convolution séparable et le 2e pooling, et il a besoin d'une séquence assez longue :
  480 / 4 = 120 pas, contre 7 seulement après les deux poolings du CD-CNN.

### Le CNN-LSTM, expliqué simplement

Une fenêtre EEG est un tableau de 64 électrodes × 480 instants. Le modèle la lit en deux étapes.

1. **Le CNN nettoie et résume le signal.**
   - La convolution temporelle (8 filtres de 0,8 s) agit comme une série de filtres en
     fréquence. Elle peut par exemple isoler les rythmes mu (8-13 Hz) et bêta (13-30 Hz),
     qui changent quand on imagine un mouvement.
   - La convolution depthwise combine les 64 électrodes en 16 « électrodes virtuelles ».
     C'est l'idée des filtres spatiaux comme CSP : par exemple, comparer C3 (cortex moteur
     gauche) et C4 (cortex moteur droit).
   - Un pooling fait la moyenne sur 4 instants. On obtient 120 pas de temps, chacun décrit
     par 16 nombres.
2. **Le LSTM suit l'évolution dans le temps.** Il lit les 120 pas dans l'ordre (un toutes les
   25 ms) et met à jour une mémoire. À chaque pas, des « portes » décident quoi oublier, quoi
   ajouter à la mémoire et quoi transmettre. Il peut ainsi remarquer, par exemple, que le
   rythme mu baisse sur l'hémisphère droit *et reste bas* pendant une seconde. C'est la
   signature typique d'une imagerie de la main gauche.
3. **La décision.** Après le dernier pas, l'état caché du LSTM (32 ou 64 nombres) résume toute
   la fenêtre. Une couche dense le transforme en un score par classe (les logits). Il n'y a
   pas de softmax dans le modèle, car `nn.CrossEntropyLoss` le calcule elle-même, de façon
   plus stable.

**Pourquoi le CNN avant le LSTM ?** Un LSTM qui lirait directement 480 instants de 64 électrodes
bruitées aurait beaucoup de poids à apprendre, et une séquence longue et difficile à mémoriser.
Le CNN réduit d'abord le bruit et la taille des données (64 × 480 → 16 × 120). Le LSTM n'a plus
qu'à modéliser la dynamique de ces 16 caractéristiques.

## Utilisation

```bash
# Couches et nombre de paramètres
python train.py --model cd_cnn cnn_lstm --classes 2 4 --resume-modele
# Essai rapide : 10 sujets, 5 époques
python train.py --model cd_cnn cnn_lstm --classes 2 4 --quick
# Entraînement complet
python train.py --model cd_cnn cnn_lstm --classes 2 4 --data-dir <...>/files --out-dir <dossier>
# Option, seulement sur demande : LSTM à 64 unités
python train.py --model cnn_lstm --hidden 64 --classes 2 4 --data-dir <...>/files --out-dir <dossier>
```

Les résultats (un fichier json par pli, les poids `.pt`, `resume.json`) sont écrits dans
`<out-dir>/<modèle>_<n>cl[_options]/`.
