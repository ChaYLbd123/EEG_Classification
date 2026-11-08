import os
DATA_DIR = "C:/Users/DELL/Downloads/eegmmidb-1.0.0/files"
exclus = {88, 92, 100, 104}
runs = [1, 4, 6, 8, 10, 12, 14]
sujets = [s for s in range(1, 110) if s not in exclus]
manquants = [f"S{s:03d}R{r:02d}" for s in sujets for r in runs
             if not os.path.exists(f"{DATA_DIR}/S{s:03d}/S{s:03d}R{r:02d}.edf")]
print(len(sujets), "sujets attendus,", len(manquants), "fichiers manquants")
print(manquants[:5])
