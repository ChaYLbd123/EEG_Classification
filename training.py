"""Boucle d'entraînement commune à tous les modèles.

Protocole (le même pour tous les modèles) :
  1. GroupKFold à 5 plis PAR SUJET : aucun sujet de test n'est vu pendant l'entraînement.
  2. Dans chaque pli, 10 % des sujets d'ENTRAÎNEMENT forment la validation (GroupShuffleSplit).
     L'early stopping regarde la perte de validation, jamais le test.
  3. Protocole de l'article (section 2.3.1) : Adam, learning rate FIXE 0.001, batch 16,
     50 époques maximum. En plus : early stopping (patience 15, meilleurs poids restaurés).
     Options désactivées par défaut :
       --scheduler  : learning rate du notebook CD-CNN, 0.01 / 0.001 / 0.0001
       --clip       : écrêtage de la norme du gradient à 1.0
       --norm-essai : z-score de chaque fenêtre (en plus du z-score par sujet)
  4. Évaluation du modèle du pli k sur les sujets de test du pli k.
  5. Adaptation au sujet : le modèle du pli k est adapté UNIQUEMENT aux sujets de test
     du pli k (il ne les a jamais vus). Pour chaque sujet : KFold 4 plis (75 % / 25 %),
     5 époques, Adam lr=5e-4. On mesure la précision avant et après, sur les mêmes 25 %.

Tout est sauvegardé après chaque pli. Si on relance après une coupure, les plis déjà
finis sont relus au lieu d'être recalculés.
"""

import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from torch.utils.data import DataLoader, Subset, TensorDataset

from data import charger_donnees, normaliser_par_essai, zscore_sujet
from models import compter_parametres, construire_modele, nom_modele


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    model: str = "cd_cnn"          # "cd_cnn" ou "cnn_lstm"
    hidden: int = 32               # taille de l'état caché du LSTM (32 ; 64 en option)
    n_classes: int = 4
    data_dir: str = "C:/Users/DELL/Downloads/eegmmidb-1.0.0/files"
    out_dir: str = "resultats"
    cache_dir: str = None          # None -> <out_dir>/cache
    n_subjects: int = 109          # sujets 1..n_subjects (moins 88, 92, 100, 104)
    max_epochs: int = 50           # article : 50 époques
    lr: float = 1e-3               # article : Adam, lr fixe 0.001
    scheduler: bool = False        # option : scheduler du notebook (0.01 / 0.001 / 0.0001)
    clip: float = None             # option : écrêtage du gradient (None = désactivé)
    norm_essai: bool = False       # option : z-score par fenêtre
    patience: int = 15
    batch_size: int = 16
    seed: int = 42
    n_folds: int = 5
    val_frac: float = 0.10         # part des sujets d'entraînement mis en validation
    ft_epochs: int = 5
    ft_folds: int = 4
    ft_lr: float = 5e-4
    # Normalisation pendant l'adaptation :
    #   "complet" : z-score du notebook (statistiques sur toutes les fenêtres du sujet)
    #   "train"   : statistiques sur les 75 % d'adaptation seulement (pas de fuite)
    #   "both"    : on calcule les deux
    adapt_norm: str = "both"
    quick: bool = False


def lr_epoque(cfg, epoch):
    """Learning rate de l'époque `epoch` (qui commence à 0, comme en Keras)."""
    return lr_cd_cnn(epoch) if cfg.scheduler else cfg.lr


def lr_cd_cnn(epoch):
    """Le scheduler du notebook CD-CNN (lr_scheduler_wang). Option --scheduler."""
    if epoch < 20:
        return 0.01
    elif epoch < 50:
        return 0.001
    return 0.0001


# ---------------------------------------------------------------------------
# Outils
# ---------------------------------------------------------------------------
def choisir_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nom = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Device : {device} ({nom})")
    return device


def fixer_graine(seed):
    """Même graine pour Python, NumPy et PyTorch (CPU et GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def faire_loader(Xt, yt, idx, batch_size, shuffle, seed=0):
    """DataLoader sur un sous-ensemble (idx) des données, sans copier le tableau complet."""
    ds = Subset(TensorDataset(Xt, yt), np.asarray(idx))
    g = torch.Generator().manual_seed(seed)     # mélange reproductible
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def creer_optimiseur(model, lr):
    # Keras Adam : epsilon = 1e-7 (PyTorch : 1e-8 par défaut). beta1, beta2 identiques.
    return torch.optim.Adam(model.parameters(), lr=lr, eps=1e-7)


def entrainer_une_epoque(model, loader, optimizer, critere, device, clip=None):
    """Une passe sur les données d'entraînement. Renvoie (perte moyenne, précision)."""
    model.train()                       # active le dropout et la mise à jour des stats BN
    total_perte, total_ok, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()           # remet les gradients à zéro (PyTorch les accumule)
        logits = model(xb)              # (B, n_classes)
        perte = critere(logits, yb)     # log-softmax + entropie croisée, sur des logits
        perte.backward()                # rétropropagation
        if clip:                        # option --clip : la norme globale du gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)   # est ramenée à <= clip
        optimizer.step()                # mise à jour des poids
        if hasattr(model, "appliquer_contraintes"):
            model.appliquer_contraintes()   # MaxNorm, après la mise à jour (comme Keras)
        total_perte += perte.item() * len(yb)
        total_ok += (logits.argmax(1) == yb).sum().item()
        n += len(yb)
    return total_perte / n, total_ok / n


@torch.no_grad()                        # pas de calcul de gradient : plus rapide, moins de mémoire
def evaluer(model, Xt, yt, idx, device, critere=None):
    """Renvoie (perte, précision, prédictions) sur les fenêtres idx."""
    model.eval()                        # dropout coupé, BN utilise ses moyennes glissantes
    critere = critere or nn.CrossEntropyLoss()
    total_perte, preds = 0.0, []
    for xb, yb in faire_loader(Xt, yt, idx, 256, shuffle=False):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        total_perte += critere(logits, yb).item() * len(yb)
        preds.append(logits.argmax(1).cpu())
    preds = torch.cat(preds).numpy()
    y_vrai = yt[np.asarray(idx)].numpy()
    return total_perte / len(idx), float((preds == y_vrai).mean()), preds


def _ecrire_json(chemin, obj):
    """Écriture atomique : un fichier à moitié écrit (coupure Colab) n'est jamais relu."""
    tmp = chemin + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, chemin)


def _lire_json(chemin):
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phase globale : entraînement d'un pli
# ---------------------------------------------------------------------------
def entrainer_global(cfg, Xt, yt, idx_train, idx_val, device, seed):
    fixer_graine(seed)
    model = construire_modele(cfg.model, cfg.n_classes, cfg.hidden).to(device)
    optimizer = creer_optimiseur(model, lr_epoque(cfg, 0))
    critere = nn.CrossEntropyLoss()
    loader = faire_loader(Xt, yt, idx_train, cfg.batch_size, shuffle=True, seed=seed)

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "temps_s": []}
    meilleure_perte, meilleur_etat, meilleure_epoque, attente = float("inf"), None, -1, 0

    for epoch in range(cfg.max_epochs):
        t0 = time.time()
        lr = lr_epoque(cfg, epoch)
        for g in optimizer.param_groups:   # lr fixe, ou LearningRateScheduler (Keras) si --scheduler
            g["lr"] = lr
        tr_loss, tr_acc = entrainer_une_epoque(model, loader, optimizer, critere, device, cfg.clip)
        va_loss, va_acc, _ = evaluer(model, Xt, yt, idx_val, device, critere)
        dt = time.time() - t0
        for k, v in zip(hist, (tr_loss, tr_acc, va_loss, va_acc, lr, dt)):
            hist[k].append(v)
        print(f"    époque {epoch + 1:3d}/{cfg.max_epochs}  lr={lr:.0e}  "
              f"train perte={tr_loss:.4f} préc={tr_acc:.3f} | "
              f"val perte={va_loss:.4f} préc={va_acc:.3f}  ({dt:.1f}s)")

        # Early stopping sur la perte de VALIDATION (sujets d'entraînement mis de côté).
        if va_loss < meilleure_perte:
            meilleure_perte, meilleure_epoque, attente = va_loss, epoch, 0
            meilleur_etat = copy.deepcopy(model.state_dict())
        else:
            attente += 1
            if attente >= cfg.patience:
                print(f"    Early stopping à l'époque {epoch + 1}, "
                      f"meilleure époque : {meilleure_epoque + 1}")
                break

    model.load_state_dict(meilleur_etat)    # = restore_best_weights=True
    return model, hist, meilleure_epoque


# ---------------------------------------------------------------------------
# Adaptation au sujet
# ---------------------------------------------------------------------------
def adapter_sujets(cfg, etat_global, X, y, groups, sujets, device, X_brut=None, mode="complet"):
    """Adapte le modèle global à chaque sujet de `sujets` (sujets de test du pli).

    mode "complet" : on utilise X (z-score du notebook, statistiques sur tout le sujet).
    mode "train"   : on re-normalise X_brut avec les statistiques des 75 % d'adaptation.
    """
    model = construire_modele(cfg.model, cfg.n_classes, cfg.hidden).to(device)
    critere = nn.CrossEntropyLoss()
    res = {}
    for s in sujets:
        idx_s = np.where(groups == s)[0]
        y_s = torch.from_numpy(y[idx_s])
        kf = KFold(n_splits=cfg.ft_folds, shuffle=True, random_state=42)
        avant, apres = [], []
        for j, (t, v) in enumerate(kf.split(idx_s)):
            if mode == "train":
                X_s = torch.from_numpy(zscore_sujet(X_brut[idx_s], t))
            else:
                X_s = torch.from_numpy(X[idx_s])

            model.load_state_dict(etat_global)              # on repart du modèle global
            _, acc_avant, _ = evaluer(model, X_s, y_s, v, device, critere)

            graine = cfg.seed + int(s) * 10 + j
            fixer_graine(graine)
            optimizer = creer_optimiseur(model, cfg.ft_lr)  # Adam neuf, lr fixe 5e-4
            loader = faire_loader(X_s, y_s, t, cfg.batch_size, shuffle=True, seed=graine)
            for _ in range(cfg.ft_epochs):
                entrainer_une_epoque(model, loader, optimizer, critere, device, cfg.clip)

            _, acc_apres, _ = evaluer(model, X_s, y_s, v, device, critere)
            avant.append(acc_avant)
            apres.append(acc_apres)
        res[int(s)] = {"avant": float(np.mean(avant)), "apres": float(np.mean(apres))}
    return res


# ---------------------------------------------------------------------------
# Validation croisée complète
# ---------------------------------------------------------------------------
def lancer_validation_croisee(cfg):
    if cfg.quick:
        cfg.n_subjects, cfg.max_epochs = 10, 5
    # Les options actives apparaissent dans le nom du dossier : pas de mélange entre variantes.
    # Essai partiel (moins de sujets ou d'époques que le protocole complet) : visible dans le nom.
    if not cfg.quick and (cfg.n_subjects != Config.n_subjects or cfg.max_epochs != Config.max_epochs):
        options_essai = f"_{cfg.n_subjects}suj_{cfg.max_epochs}ep"
    else:
        options_essai = ""
    options = options_essai + "".join(suffixe for suffixe, actif in (
        ("_sched", cfg.scheduler), ("_clip", cfg.clip),
        ("_normessai", cfg.norm_essai), ("_essai", cfg.quick)) if actif)
    nom = f"{nom_modele(cfg.model, cfg.hidden)}_{cfg.n_classes}cl{options}"
    dossier = os.path.join(cfg.out_dir, nom)
    os.makedirs(dossier, exist_ok=True)
    cache_dir = cfg.cache_dir or os.path.join(cfg.out_dir, "cache")

    print("=" * 70)
    lr_txt = "scheduler 0.01/0.001/0.0001" if cfg.scheduler else f"fixe {cfg.lr}"
    print(f"{nom} : {cfg.n_classes} classes, {cfg.n_subjects} sujets max, "
          f"{cfg.max_epochs} époques max, lr {lr_txt}, clip {cfg.clip}, norm. par essai {cfg.norm_essai}")
    print("=" * 70)

    # Si la config a changé depuis le dernier lancement, on prévient (risque de mélange).
    chemin_cfg = os.path.join(dossier, "config.json")
    if os.path.exists(chemin_cfg):
        ancienne = _lire_json(chemin_cfg)
        defauts = asdict(Config())
        diff = {k: (ancienne.get(k, defauts[k]), v) for k, v in asdict(cfg).items()
                if k not in ("data_dir", "out_dir", "cache_dir") and ancienne.get(k, defauts[k]) != v}
        if diff:
            raise RuntimeError(f"La config diffère de celle déjà sauvegardée dans {dossier} : {diff}. "
                               "Change de dossier de sortie ou supprime les anciens résultats.")
    _ecrire_json(chemin_cfg, asdict(cfg))

    device = choisir_device()
    X, y, groups = charger_donnees(cfg.data_dir, cfg.n_classes, cfg.n_subjects, True, cache_dir)
    modes = {"both": ["complet", "train"], "complet": ["complet"], "train": ["train"]}[cfg.adapt_norm]
    if cfg.norm_essai:
        # Le z-score par fenêtre annule le z-score par sujet : les deux modes seraient identiques.
        X = normaliser_par_essai(X)
        modes = ["complet"]
        print("Normalisation par essai activée : un seul mode d'adaptation.")
    X_brut = None
    if "train" in modes:
        X_brut, y_b, g_b = charger_donnees(cfg.data_dir, cfg.n_classes, cfg.n_subjects, False, cache_dir)
        assert np.array_equal(y_b, y) and np.array_equal(g_b, groups)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)

    print(f"Fenêtres : {len(y)}, sujets : {len(np.unique(groups))}, "
          f"classes : {np.bincount(y).tolist()}")
    n_params = compter_parametres(construire_modele(cfg.model, cfg.n_classes, cfg.hidden))
    print(f"Paramètres : {n_params}")

    gkf = GroupKFold(n_splits=cfg.n_folds)
    for k, (idx_train_complet, idx_test) in enumerate(gkf.split(X, y, groups), start=1):
        f_global = os.path.join(dossier, f"pli{k}_global.json")
        f_poids = os.path.join(dossier, f"pli{k}_modele.pt")
        sujets_test = np.unique(groups[idx_test])

        if os.path.exists(f_global) and os.path.exists(f_poids):
            print(f"\n--- Pli {k}/{cfg.n_folds} : déjà fait, relu depuis {f_global}")
            r = _lire_json(f_global)
            etat = torch.load(f_poids, map_location=device)
        else:
            # Validation = 10 % des sujets d'entraînement (jamais de sujet de test).
            gss = GroupShuffleSplit(n_splits=1, test_size=cfg.val_frac, random_state=cfg.seed)
            rel_tr, rel_val = next(gss.split(idx_train_complet, groups=groups[idx_train_complet]))
            idx_train, idx_val = idx_train_complet[rel_tr], idx_train_complet[rel_val]
            sujets_val = np.unique(groups[idx_val])
            assert not set(sujets_test) & set(groups[idx_train_complet])   # aucune fuite de sujet
            print(f"\n--- Pli {k}/{cfg.n_folds} : train {len(idx_train)} fenêtres, "
                  f"val {len(idx_val)} (sujets {sujets_val.tolist()}), "
                  f"test {len(idx_test)} (sujets {sujets_test.tolist()})")

            t0 = time.time()
            model, hist, best_ep = entrainer_global(cfg, Xt, yt, idx_train, idx_val, device, cfg.seed + k)
            t_train = time.time() - t0
            te_loss, te_acc, preds = evaluer(model, Xt, yt, idx_test, device)
            print(f"    Précision test pli {k} : {te_acc:.2%}   (entraînement : {t_train:.0f}s)")

            r = {"pli": k, "test_acc": te_acc, "test_loss": te_loss,
                 "meilleure_epoque": best_ep + 1, "epoques_faites": len(hist["train_loss"]),
                 "temps_entrainement_s": t_train, "device": str(device),
                 "n_train": len(idx_train), "n_val": len(idx_val), "n_test": len(idx_test),
                 "sujets_val": sujets_val.tolist(), "sujets_test": sujets_test.tolist(),
                 "historique": hist, "y_vrai": y[idx_test].tolist(), "y_pred": preds.tolist()}
            etat = model.state_dict()
            torch.save(etat, f_poids)       # d'abord les poids, puis le json « pli fini »
            _ecrire_json(f_global, r)

        for mode in modes:
            f_adapt = os.path.join(dossier, f"pli{k}_adapt_{mode}.json")
            if os.path.exists(f_adapt):
                continue
            t0 = time.time()
            res = adapter_sujets(cfg, etat, X, y, groups, sujets_test, device, X_brut, mode)
            dt = time.time() - t0
            av = np.mean([v["avant"] for v in res.values()])
            ap = np.mean([v["apres"] for v in res.values()])
            print(f"    Adaptation [{mode}] pli {k} : avant {av:.2%} -> après {ap:.2%}  ({dt:.0f}s)")
            _ecrire_json(f_adapt, {"pli": k, "mode": mode, "avant": av, "apres": ap,
                                   "temps_s": dt, "par_sujet": res})

    return resumer(dossier, cfg, n_params, modes)


def resumer(dossier, cfg, n_params, modes):
    """Moyenne ± écart-type sur les plis (écart-type de population, np.std, ddof=0)."""
    G = [_lire_json(os.path.join(dossier, f"pli{k}_global.json")) for k in range(1, cfg.n_folds + 1)]
    acc = [g["test_acc"] for g in G]
    resume = {"modele": nom_modele(cfg.model, cfg.hidden), "n_classes": cfg.n_classes, "essai_rapide": cfg.quick,
              "n_sujets": sum(len(g["sujets_test"]) for g in G), "parametres": n_params,
              "global_par_pli": acc, "global_moy": float(np.mean(acc)), "global_std": float(np.std(acc)),
              "temps_entrainement_par_pli_s": [g["temps_entrainement_s"] for g in G],
              "epoques_par_pli": [g["epoques_faites"] for g in G],
              "device": G[0]["device"]}
    for mode in modes:
        A = [_lire_json(os.path.join(dossier, f"pli{k}_adapt_{mode}.json")) for k in range(1, cfg.n_folds + 1)]
        for cle in ("avant", "apres"):
            v = [a[cle] for a in A]
            resume[f"adapt_{mode}_{cle}_par_pli"] = v
            resume[f"adapt_{mode}_{cle}_moy"] = float(np.mean(v))
            resume[f"adapt_{mode}_{cle}_std"] = float(np.std(v))
        resume[f"adapt_{mode}_temps_par_pli_s"] = [a["temps_s"] for a in A]
    _ecrire_json(os.path.join(dossier, "resume.json"), resume)

    print("\n" + "=" * 70)
    print(f"RÉSUMÉ {os.path.basename(dossier)}  (moyenne ± écart-type sur {cfg.n_folds} plis)")
    print(f"  Global (test)              : {resume['global_moy']:.2%} ± {resume['global_std']:.2%}")
    for mode in modes:
        print(f"  Adaptation [{mode:7s}] avant : {resume[f'adapt_{mode}_avant_moy']:.2%} ± "
              f"{resume[f'adapt_{mode}_avant_std']:.2%}   après : "
              f"{resume[f'adapt_{mode}_apres_moy']:.2%} ± {resume[f'adapt_{mode}_apres_std']:.2%}")
    print("=" * 70)
    return resume
