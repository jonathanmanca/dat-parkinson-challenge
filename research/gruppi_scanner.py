"""Is cross-validation inflated by scanner groups being split across folds?

Two questions, both answered from NIfTI headers only — never a voxel — and reported as
aggregate numbers.

1. Can the label be predicted from scanner metadata alone? If different hospitals had
   noticeably different rates of pathology, any model could exploit that shortcut.
   Answer: no, AUROC 0.53, chance.

2. Does a real branch collapse when whole scanner groups are held out? Answer: yes,
   dramatically. The linear branch goes from 0.3575 on random folds to 0.7560 on
   grouped folds while AUROC only drops from 0.92 to 0.82 — on an unseen scanner it
   still ranks patients but its probabilities fall apart. That is covariate shift (the
   features have scanner-dependent scales), not label leakage.

Writes gruppi.npz with a group id per patient, used by the other analysis scripts.

Usage: python gruppi_scanner.py [folder]
"""
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score

CARTELLA = sys.argv[1] if len(sys.argv) > 1 else "data_privata"
lab = pd.read_csv(f"{CARTELLA}/labels.csv")
y = lab["is_pathologic"].to_numpy(dtype=float)

meta, firme = [], []
for u in lab["uid"]:
    img = nib.load(f"{CARTELLA}/{u}.nii.gz")               # lazy: voxels stay on disk
    f = tuple(img.shape[:3])
    s = tuple(round(float(z), 2) for z in img.header.get_zooms()[:3])
    meta.append(list(f) + list(s))                         # six numbers, all scanner metadata
    firme.append((f, s))
M = np.array(meta, dtype=np.float32)
chiavi = {k: i for i, k in enumerate(sorted(set(firme), key=str))}
g = np.array([chiavi[k] for k in firme])
np.savez("gruppi.npz", g=g, meta=M, y=y)
print(f"{len(y)} patients | {len(chiavi)} groups | positives {100*y.mean():.1f}%")

# The row order here must match the order of the out-of-fold files, or every downstream
# comparison silently pairs the wrong patients.
if os.path.exists("oof_cnn48max_medio.npz"):
    yo = np.load("oof_cnn48max_medio.npz")["y"]
    print(f"aligned with the OOF files: {len(yo) == len(y) and np.array_equal(y, yo)}")
print(f"largest group sizes: {sorted(np.bincount(g), reverse=True)[:10]}\n")


def valuta(X, yy, cv, nome, modello):
    o = np.zeros(len(yy))
    for tr, va in cv:
        o[va] = modello().fit(X[tr], yy[tr]).predict_proba(X[va])[:, 1]
    o = np.clip(o, 1e-6, 1-1e-6)
    print(f"  {nome:34s} log loss {log_loss(yy, o, labels=[0.,1.]):.4f}   AUROC {roc_auc_score(yy, o):.4f}")


gb = lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=0)
cas = list(StratifiedKFold(5, shuffle=True, random_state=7).split(M, y))
gru = list(StratifiedGroupKFold(5, shuffle=True, random_state=7).split(M, y, groups=g))

print("--- TEST 1: predict the label from scanner metadata alone ---")
print("  (no image at all: only volume shape and voxel spacing)")
valuta(M, y, cas, "random folds (what we normally use)", gb)
valuta(M, y, gru, "grouped folds (whole groups held out)", gb)
print("  an AUROC well above 0.50 on the first line would mean the shortcut exists\n")

print("--- TEST 2: a real feature branch, with and without grouping ---")
if not os.path.exists("feat6_mm2_5.npz"):
    print("  (feat6_mm2_5.npz missing: skipped)")
    raise SystemExit(0)
X = np.load("feat6_mm2_5.npz")["FEAT"][:, :518].astype(np.float32)
if len(X) != len(y):
    print(f"  (features have {len(X)} rows but {len(y)} patients here: skipped)")
    raise SystemExit(0)
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
valuta(X, y, cas, "lr25 - random folds", lr)
valuta(X, y, gru, "lr25 - grouped folds", lr)
print("  the gap between those two lines is how optimistic our cross-validation is")
print("  because scanner groups are split between training and validation.")
