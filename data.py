"""Chargement des données PhysioNet EEG Motor Movement/Imagery.

`load_full_dataset` est une COPIE de la fonction du même nom dans notebooks/cd-cnn.ipynb
(cellule 3, la version sans erreur de syntaxe). Seuls ces points changent :
  1. to_categorical (Keras) -> labels entiers np.int64.
     Raison : nn.CrossEntropyLoss de PyTorch attend l'indice de la classe, pas un vecteur one-hot.
  2. « except: continue » -> message d'erreur clair, liste des sujets en échec,
     et affichage du nombre de sujets réellement chargés.
  3. Nouveau paramètre `normalize` (True par défaut = comportement exact du notebook).
     normalize=False renvoie les signaux bruts (en volts). Ça sert à l'option de z-score
     « strict », qui calcule les statistiques sur la partie train du sujet seulement.
  4. Plus de chemin Kaggle par défaut : data_dir est obligatoire.

Le découpage (runs, fenêtres de 0 à 3 s, 480 échantillons) et la normalisation
(z-score par sujet et par électrode) ne changent pas.
"""

import os

import mne
import numpy as np
from sklearn.preprocessing import StandardScaler

mne.set_log_level("WARNING")


def load_full_dataset(data_dir, n_subjects=109, classes=4, normalize=True):
    exclude_subjects = [88, 92, 100, 104]
    subjects = [s for s in range(1, n_subjects + 1) if s not in exclude_subjects]

    X_list, y_list, groups_list = [], [], []
    failed = []  # sujets en échec (au lieu de les ignorer en silence)
    runs_lr = [4, 8, 12]   # Gauche/Droite
    runs_ff = [6, 10, 14]  # Pieds (Run 6, 10, 14)

    print(f"Chargement de {len(subjects)} sujets ({classes} classes)...")

    for subject_id in subjects:
        try:
            sub_data, sub_labels = [], []
            subject_str = f"S{subject_id:03d}"

            # --- 1. CLASSE REPOS ---
            # Mapping: Rest=0
            if classes == 4:
                file_rest = os.path.join(data_dir, subject_str, f"{subject_str}R01.edf")
                if os.path.exists(file_rest):
                    raw_rest = mne.io.read_raw_edf(file_rest, preload=True, verbose=False)
                    epochs_rest = mne.make_fixed_length_epochs(raw_rest, duration=3.0, preload=True, verbose=False)
                    data_rest = epochs_rest.get_data(verbose=False)[:, :, :480]
                    sub_data.append(data_rest)
                    sub_labels.append(np.zeros(len(data_rest)))  # Label 0

            # --- 2. GAUCHE / DROITE (POUR 2 ET 4 CLASSES) ---
            raws_lr = []
            for run_id in runs_lr:
                run_str = f"R{run_id:02d}"
                file_path = os.path.join(data_dir, subject_str, f"{subject_str}{run_str}.edf")
                if os.path.exists(file_path):
                    raws_lr.append(mne.io.read_raw_edf(file_path, preload=True, verbose=False))

            if len(raws_lr) > 0:
                raw_lr = mne.concatenate_raws(raws_lr)
                mne.datasets.eegbci.standardize(raw_lr)
                events_lr, event_id_lr = mne.events_from_annotations(raw_lr, verbose=False)
                t1, t2 = event_id_lr.get('T1'), event_id_lr.get('T2')
                ev_filt = np.array([e for e in events_lr if e[2] in [t1, t2]])
                epochs_lr = mne.Epochs(raw_lr, ev_filt, tmin=0, tmax=3.0, baseline=None, preload=True, verbose=False)
                data_lr = epochs_lr.get_data(verbose=False)[:, :, :480]
                sub_data.append(data_lr)

                if classes == 2:
                    # Mode 2 Classes : Gauche=0, Droite=1
                    sub_labels.append(np.array([0 if e == t1 else 1 for e in epochs_lr.events[:, -1]]))
                else:
                    # Mode 4 Classes : Gauche=1, Droite=2 (car Rest=0)
                    sub_labels.append(np.array([1 if e == t1 else 2 for e in epochs_lr.events[:, -1]]))

            # --- 3. PIEDS  ---
            # Mapping: Feet=3
            if classes == 4:
                raws_ff = []
                for run_id in runs_ff:
                    run_str = f"R{run_id:02d}"
                    file_path = os.path.join(data_dir, subject_str, f"{subject_str}{run_str}.edf")
                    if os.path.exists(file_path):
                        raws_ff.append(mne.io.read_raw_edf(file_path, preload=True, verbose=False))

                if len(raws_ff) > 0:
                    raw_ff = mne.concatenate_raws(raws_ff)
                    mne.datasets.eegbci.standardize(raw_ff)
                    events_ff, event_id_ff = mne.events_from_annotations(raw_ff, verbose=False)
                    t2_f = event_id_ff.get('T2')  # T2 = Pieds
                    ev_filt_ff = np.array([e for e in events_ff if e[2] == t2_f])
                    epochs_ff = mne.Epochs(raw_ff, ev_filt_ff, tmin=0, tmax=3.0, baseline=None, preload=True, verbose=False)
                    data_ff = epochs_ff.get_data(verbose=False)[:, :, :480]
                    sub_data.append(data_ff)
                    sub_labels.append(np.full(len(data_ff), 3))  # Label 3

            # --- NORMALISATION
            if len(sub_data) > 0:
                X_sub = np.concatenate(sub_data)
                y_sub = np.concatenate(sub_labels)

                if normalize:
                    # Standardisation Z-score par sujet
                    scaler = StandardScaler()
                    N, C, T = X_sub.shape
                    X_sub = scaler.fit_transform(X_sub.transpose(0, 2, 1).reshape(-1, C)).reshape(N, T, C).transpose(0, 2, 1)

                X_list.append(X_sub)
                y_list.append(y_sub)
                groups_list.append(np.full(len(y_sub), subject_id))
            else:
                raise RuntimeError("aucun fichier de données trouvé pour ce sujet")

        except Exception as e:
            # Avant : « except: continue » (le sujet disparaissait sans aucun message).
            print(f"  [ERREUR] Sujet {subject_id} ignoré -> {type(e).__name__}: {e}")
            failed.append(subject_id)

    if not X_list:
        raise ValueError("Erreur : Aucune donnée chargée.")

    n_ok = len(X_list)
    print(f"Sujets réellement chargés : {n_ok}/{len(subjects)}")
    if failed:
        print(f"  ATTENTION, sujets en échec : {failed}")

    X = np.concatenate(X_list)[..., np.newaxis]
    # Changement 1 : labels entiers au lieu de to_categorical.
    return X, np.concatenate(y_list).astype(np.int64), np.concatenate(groups_list)


# ---------------------------------------------------------------------------
# Option « z-score strict » pour l'adaptation au sujet.
# ---------------------------------------------------------------------------
def zscore_sujet(X_sub, fit_idx):
    """Z-score d'UN sujet, avec des statistiques calculées seulement sur les fenêtres fit_idx.

    Même calcul que le notebook (StandardScaler par électrode, sur tous les instants),
    mais la moyenne et l'écart-type viennent uniquement de la partie train du sujet.
    Les fenêtres de test du sujet ne servent donc plus à calculer ses statistiques.

    X_sub : (N, 1, 64, 480) signaux bruts d'un sujet.  Renvoie la même forme, en float32.
    """
    N, _, C, T = X_sub.shape
    x = X_sub[:, 0].astype(np.float64)                     # (N, 64, 480)
    scaler = StandardScaler()
    scaler.fit(x[fit_idx].transpose(0, 2, 1).reshape(-1, C))    # stats sur le train seulement
    x = scaler.transform(x.transpose(0, 2, 1).reshape(-1, C)).reshape(N, T, C).transpose(0, 2, 1)
    return x[:, np.newaxis].astype(np.float32)


def normaliser_par_essai(X, eps=1e-8):
    """Option (désactivée par défaut) : z-score de CHAQUE fenêtre, électrode par électrode.

    Pour chaque fenêtre et chaque électrode, on retire la moyenne sur les 480 instants
    et on divise par l'écart-type. Les statistiques viennent de la fenêtre elle-même,
    donc il n'y a aucune fuite entre train et test.
    Remarque : ça annule le z-score par sujet (qui est lui aussi une transformation
    affine par électrode), donc les deux modes d'adaptation deviennent identiques.

    X : (N, 1, 64, 480). Renvoie la même forme, en float32.
    """
    moy = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    return ((X - moy) / (std + eps)).astype(np.float32)


# ---------------------------------------------------------------------------
# Enveloppe : passage au format PyTorch + cache sur disque.
# ---------------------------------------------------------------------------
def charger_donnees(data_dir, classes, n_subjects=109, normalize=True, cache_dir=None):
    """Charge (ou relit depuis le cache) les données au format PyTorch.

    Renvoie :
      X      : (N, 1, 64, 480) float32. Format PyTorch NCHW : 1 « canal image »,
               hauteur = 64 électrodes, largeur = 480 instants.
               (Le notebook donnait (N, 64, 480, 1), le format Keras NHWC.)
      y      : (N,) int64, labels entiers.
      groups : (N,) identifiant du sujet de chaque fenêtre, pour GroupKFold.
    """
    cache_path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        tag = "norm" if normalize else "brut"
        cache_path = os.path.join(cache_dir, f"donnees_{classes}cl_{n_subjects}suj_{tag}.npz")
        if os.path.exists(cache_path):
            d = np.load(cache_path)
            print(f"Données relues depuis le cache : {cache_path}  X={d['X'].shape}")
            return d["X"], d["y"], d["groups"]

    X, y, groups = load_full_dataset(data_dir, n_subjects=n_subjects, classes=classes, normalize=normalize)
    # (N, 64, 480, 1) -> (N, 1, 64, 480). float32 : la précision utilisée par Keras et PyTorch.
    X = np.ascontiguousarray(X.transpose(0, 3, 1, 2), dtype=np.float32)
    print(f"Format PyTorch : X={X.shape}, y={y.shape}, sujets={len(np.unique(groups))}")

    if cache_path is not None:
        np.savez(cache_path, X=X, y=y, groups=groups)
    return X, y, groups
