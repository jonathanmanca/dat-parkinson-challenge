"""Build the cached array of striatum crops used to train the CNN.

Calls exactly the same `ritaglia_striato` that inference uses, so training and
prediction cannot drift apart — they used to be two separate copies of the same code.

Usage: python build_crops.py <mm> <crop> <side> [folder]
"""
import sys
import numpy as np
import pandas as pd
from preprocess import carica_volume_mm, ritaglia_striato

if len(sys.argv) > 5:                                  # refuse rather than ignore a typo
    sys.exit("usage: python build_crops.py <mm> <crop> <side> [folder]")
MM = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
CROP = int(sys.argv[2]) if len(sys.argv) > 2 else 32
LATO = int(sys.argv[3]) if len(sys.argv) > 3 else 112
CARTELLA_DATI = sys.argv[4] if len(sys.argv) > 4 else "data_privata"
print(f"crop {CROP}^3 at {MM} mm = {CROP*MM:.0f} mm per side, from a {LATO}^3 volume")

labels = pd.read_csv(f"{CARTELLA_DATI}/labels.csv")
crops = []
for i, uid in enumerate(labels["uid"], 1):
    v = carica_volume_mm(f"{CARTELLA_DATI}/{uid}.nii.gz", mm=MM, lato=LATO)
    crops.append(ritaglia_striato(v, crop=CROP))
    if i % 100 == 0:
        print(f"{i}/{len(labels)}")

X = np.stack(crops)
nome = f"crop{CROP}_mm{MM}".replace(".", "_")
if CARTELLA_DATI != "data_privata":                    # a trial run never overwrites the real cache
    nome += "_" + CARTELLA_DATI
np.savez(f"{nome}.npz", X=X, y=labels["is_pathologic"].to_numpy(dtype=np.float32))
print(f"saved {nome}.npz:", X.shape)

# Aggregate sanity checks only. Min-max normalisation puts every crop in [0, 1] with
# both endpoints attained, so anything else here means a volume came through wrong.
print(f"\nvalues: min {X.min():.3f}  mean {X.mean():.3f}  median {np.median(X):.3f}  max {X.max():.3f}")
picchi = X.reshape(len(X), -1).max(1)
minimi = X.reshape(len(X), -1).min(1)
print(f"per-patient range: peaks in [{picchi.min():.3f}, {picchi.max():.3f}], "
      f"floors in [{minimi.min():.3f}, {minimi.max():.3f}]  (expected 1.0 and 0.0)")
print(f"flat crops (max == min, i.e. nothing found): {int((picchi - minimi < 1e-6).sum())}")
print(f"NaN or infinite: {bool(np.isnan(X).any() or np.isinf(X).any())}")
