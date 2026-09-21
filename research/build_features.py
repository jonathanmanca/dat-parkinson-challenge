"""Compute the 672 engineered features for every patient at one physical scale.

Run once per scale (1.5 mm at side 152, 2.5 mm at side 88). Each scan contributes two
sets of features: one at full resolution and one at half, so the model sees both fine
detail and context. The blocks are concatenated in a fixed order and the first 518
columns are what the linear branches use.

This is the slow step — a couple of hours for 1362 scans — and the reason the features
are cached rather than recomputed.

Usage: python build_features.py <mm> <side>
"""
import sys
import numpy as np
import pandas as pd
from features_volume import feature_volume
from features_aligned import feature_allineate
from features_lobes import feature_lobi
from features_clinical import feature_v6
from preprocess import carica_volume_mm

MM = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
LATO = int(sys.argv[2]) if len(sys.argv) > 2 else 152
CARTELLA_DATI = "data_privata"
print(f"scale {MM} mm, volume {LATO}^3")

labels = pd.read_csv(f"{CARTELLA_DATI}/labels.csv")
feat = []
for i, uid in enumerate(labels["uid"], 1):
    v = carica_volume_mm(f"{CARTELLA_DATI}/{uid}.nii.gz", mm=MM, lato=LATO)
    h = LATO // 2
    vg = v.reshape(h, 2, h, 2, h, 2).mean(axis=(1, 3, 5))
    feat.append(np.concatenate([
        feature_volume(v, griglia=4), feature_volume(vg, griglia=4),
        feature_allineate(v), feature_allineate(vg),
        feature_lobi(v), feature_lobi(vg),
        feature_v6(v), feature_v6(vg)]))
    if i % 100 == 0:
        print(f"{i}/{len(labels)}")

F = np.stack(feat).astype(np.float32)
nome = f"feat6_mm{MM}".replace(".", "_")
np.savez(f"{nome}.npz", FEAT=F, y=labels["is_pathologic"].to_numpy(dtype=np.float32))
print(f"saved {nome}.npz:", F.shape)
