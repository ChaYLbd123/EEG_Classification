# Notes : article, notebooks Keras et code PyTorch

Source : `Eeg_article.pdf`, « Comparative Analysis of Deep Learning Architectures for Motor
Imagery EEG Classification » (M. T. Kali, C. Laabad, 11 janvier 2026, 13 pages).
L'article n'est pas modifié. Ce fichier liste les écarts à présenter honnêtement.

Légende : **[A]** = article, **[NB]** = notebooks Keras, **[PT]** = code PyTorch de ce dépôt.

---

## 1. Différences entre l'article et le code PyTorch

| # | Point | [A] (section / tableau) | [PT] | Origine du choix [PT] |
|---|---|---|---|---|
| 1 | Noyau temporel du CD-CNN | ≈ 64 échantillons, « half the sampling rate » (§2.2.2, Tab. 2 : (64,1)) | **128** | `cd-cnn.ipynb`, cellule 9 |
| 2 | Poolings du CD-CNN | (4) puis (8) : 480 → 120 → 15 (§2.2.2, Tab. 2, Fig. 5) | **(8) puis (8)** : 480 → 60 → 7 | `cd-cnn.ipynb`, cellule 9 |
| 3 | BatchNorm, Dropout 0,5, MaxNorm | non mentionnés (§2.2.2, Tab. 2) | présents | `cd-cnn.ipynb`, cellule 9 |
| 4 | Optimisation | Adam, lr fixe 0,001, 50 époques, batch 16 (§2.3.1) | **identique** (lr 0,001, 50 époques max, batch 16) | décision du protocole |
| 5 | Early stopping | non mentionné (§2.3.1 : « trained for 50 epochs ») | patience 15 sur une **validation de 10 % des sujets d'entraînement**, meilleurs poids restaurés | ajout, pour corriger le choix des poids sur le test dans [NB] |
| 6 | Scheduler 0,01 / 0,001 / 0,0001 | absent | option `--scheduler`, désactivée | `cd-cnn.ipynb`, cellule 10 |
| 7 | Normalisation | non mentionnée (§2.1) | z-score par sujet et par électrode, deux variantes pour l'adaptation (voir §3) | `cd-cnn.ipynb`, cellule 3 |
| 8 | Nombre d'essais | « 21 trials per class » (§2.1), 2205 par classe (Tab. 5, Tab. 6) | sans plafond : environ 22 à 23 essais par classe et par sujet (3 runs × 7 ou 8 essais par classe), 20 fenêtres de repos | `load_full_dataset` (cd-cnn). Le plafond de 21 vient du chargeur du SS-CNN |
| 9 | Fenêtres de repos | « segments from baseline recordings » (§2.1) | run 1, fenêtres fixes de 3 s sans recouvrement | `load_full_dataset` |
| 10 | Adaptation : découpage | 4 plis, 75 % / 25 % (§2.3.2) | identique | protocole |
| 11 | Adaptation : learning rate | non précisé (§2.3.2) | Adam 5e-4 | décision du protocole |
| 12 | Adaptation : quel modèle global | « W_global » (§2.3.1), un seul jeu de poids | **le modèle du pli k, adapté seulement aux sujets de test du pli k** | correction de la fuite décrite au §3 |
| 13 | CNN-LSTM | absent de l'article | nouveau modèle. Il reprend le **pooling (1,4) de l'article** (§2.2.2), pas le (1,8) du notebook, pour garder 120 pas de temps | voir README |

## 2. Incohérences internes de l'article

| # | Où | Problème |
|---|---|---|
| 1 | Résumé | « 79.48%, 69 and 58.62% » : il y a un « 69 » en trop. |
| 2 | Résumé et Tab. 3 / Tab. 4 | Le résumé présente 86,39 % et 70,17 % comme des scores *subject-independent* du CD-CNN. Selon les Tab. 3 et 4, ce sont des scores **après adaptation**. Les scores globaux du CD-CNN sont 83,22 % et 64,87 %. |
| 3 | Résumé, §4.3 et Tab. 4 | Le résumé annonce un gain de « 4-6 points » et le §4.3 « jusqu'à +7 % ». Le Tab. 4 donne +7,80 pour le SS-CNN. |
| 4 | §2.1 | « four types of tasks », suivi d'une liste de cinq (L, R, B, F, repos). « three experimental configurations », suivi de deux seulement. |
| 5 | §2.1 | Le déroulement d'un essai (2 s de repos, 4 s d'imagerie, 2 s de repos) ne correspond pas aux fichiers PhysioNet. Les annotations alternent T1/T2 (imagerie, 4,1 s) et T0 (repos, 4,1 à 4,2 s), sans phase « 2 s avant / 2 s après ». Vérifié sur S001R04 et S050R04 : 15 T0 et 15 T1+T2 par run. |
| 6 | §3 (intro) et §3.1 | Le SS-CNN est appelé « proposed » et le CD-CNN « baseline », alors que la conclusion (§5) retient le CD-CNN. La phrase « the CD-CNN consistently outperforms the compact baseline » est fausse, puisque le CD-CNN *est* le modèle compact. |
| 7 | §3.5 et Tab. 4 | Le texte dit que l'adaptation fait passer le SS-CNN « de 59,71 % à 68,93 % ». Le Tab. 4 donne 58,62 % → 66,42 %. Or 0,5971 et 0,6893 sont les **valeurs « cible »** codées dans `ss-cnn.ipynb` (cellules 5 et 7 : `target`, « Cible Dose »). Ce sont donc les chiffres d'un article de référence, pas des résultats de ce travail. |
| 8 | Tab. 4 et [NB] | Le SS-CNN adapté en 4 classes vaut 66,42 % dans le Tab. 4, mais 70,31 % dans `ss-cnn.ipynb` (cellule 8). |
| 9 | Fig. 8 | La légende nomme les classes « '0' (Left Hand) » et « 'L' (Right Hand) ». C'est un effet de bord des étiquettes `['0','L','R','F'][:2]` de `ss-cnn.ipynb` (cellule 6). |
| 10 | §4.1 | La meilleure généralisation du CD-CNN est attribuée aux « larger kernel sizes used in the SS-CNN ». Or le noyau temporel du SS-CNN (30) est **plus petit** que celui du CD-CNN (64 dans l'article, 128 dans le notebook). |
| 11 | Tab. 1 et références | Le Tab. 1 cite « Dose et al. Implementation », mais Dose et al. n'est pas dans les références. Le CD-CNN correspond à EEGNet (Lawhern et al., 2018), qui n'est pas cité non plus. |
| 12 | §2.2.1 / Tab. 1 et [NB] | L'article décrit un SS-CNN avec carré puis log. `build_model` dans `ss-cnn.ipynb` n'a **aucune activation**, et les cellules d'entraînement appellent `build_dose_model`, qui n'est pas défini. L'architecture qui a réellement produit 79,48 % et 58,62 % n'est donc pas vérifiable. |

## 3. Fuites de données dans les scores rapportés (Tab. 3 et Tab. 4)

1. **Score global du CD-CNN (83,22 % et 64,87 %)** : `EarlyStopping(monitor='val_loss', restore_best_weights=True)`
   avec `validation_data` = le pli de **test** (`cd-cnn.ipynb`, cellule 10). L'époque est choisie sur le test, donc le score est optimiste.
2. **Scores adaptés (Tab. 3 et Tab. 4)** : un seul modèle global est adapté aux **105** sujets.
   - Pour le CD-CNN, c'est le pli 1 (cellule 10). Pour le SS-CNN, c'est le pli qui a le meilleur score **sur le test** (cellule 5).
   - Ce modèle avait déjà appris sur environ 80 % de ces sujets, donc sur les fenêtres ensuite utilisées pour évaluer l'adaptation.
3. **Adaptation du CD-CNN (86,39 % et 70,17 %)** : `KFold(n_splits=2)`, soit 50 % / 50 %, avec Adam 5e-4 (`cd-cnn.ipynb`, cellule 13). Le §2.3.2 annonce 4 plis 75 % / 25 %.
4. **z-score par sujet** : les statistiques utilisent toutes les fenêtres du sujet, y compris ses fenêtres de test pendant l'adaptation. Aucun label n'est utilisé, mais c'est une fuite légère. [PT] rapporte donc deux variantes : `complet` (comme le notebook) et `train` (statistiques sur les 75 % d'adaptation seulement).
5. **Notebooks non reproductibles** : fonctions non définies (`build_dose_model`, `build_eegnet`, `build_eegnet_optimized`, `lr_callback`), données chargées hors du notebook, erreur de syntaxe dans `cd-cnn.ipynb` (cellule 2).

**Conséquence :** les chiffres des Tab. 3 et 4 ne peuvent pas servir de référence directe. Les chiffres
PyTorch de ce dépôt (protocole corrigé) seront probablement plus bas, surtout après adaptation.

---

## 4. Journal des diagnostics et des modifications

Règle : tout réglage se juge sur la **validation** (10 % des sujets d'entraînement), jamais sur
le test. Toute modification est appliquée aux deux modèles et notée ici.

### 2026-11-08 : outils ajoutés (aucun hyperparamètre modifié)
- `train.py --n-sujets N --epoques E` : essais partiels sur CPU. Le dossier de résultats est
  suffixé `_{N}suj_{E}ep`.
- `diagnostic_surapprentissage.py` et `rapport.py` (figures).

### 2026-11-08 : test de sur-apprentissage (64 fenêtres, 4 classes, 600 pas, lr 0,001)
Perte sur les 64 fenêtres, mesurée en mode eval :
- CNN-LSTM-32 : 1,389 → 0,007, 100 % de précision ;
- CD-CNN : 1,387 → 0,158, 100 % de précision.

**Verdict :** le modèle et la boucle d'entraînement sont capables d'apprendre. Il n'y a pas de bug
de forme, de gradient ou d'initialisation. Figure : `resultats/diagnostics/surapprentissage.png`.

### 2026-11-08 : essai 10 sujets, 30 époques max, 4 classes (protocole par défaut)
| Modèle | Global (test) | Adapt. complet après | Adapt. train après | Meilleure époque (val) |
|---|---|---|---|---|
| CD-CNN | 57,55 % ± 9,99 | 61,21 % | 61,00 % | 9, 20, 30, 18, 27 |
| CNN-LSTM-32 | 30,15 % ± 4,95 | 31,61 % | 30,70 % | 6, 6, 8, 2, 6 |

**Observation :** pour le CNN-LSTM-32, la perte train baisse régulièrement (environ 1,40 → 0,95 à 1,1).
La perte de validation, elle, remonte dès l'époque 6 environ (jusqu'à 1,5 à 1,8, au-dessus du
hasard). Le modèle apprend donc des motifs propres aux sujets d'entraînement, qui ne se
transfèrent pas aux nouveaux sujets : c'est un **sur-apprentissage inter-sujets**, pas un problème
d'optimisation. Le CD-CNN, avec les mêmes données, garde une validation stable autour de 1,0.
Figure : `resultats/diagnostics/courbes_30ep_4cl.png`.

**Aucune modification n'a encore été appliquée.** Les pistes sont en attente de décision.
