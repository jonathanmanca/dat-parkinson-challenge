"""Generate the synthetic NIfTI volumes used to develop and test the pipeline.

The competition scans could not be inspected, so all code was written and debugged
against random volumes with realistic headers: varying shapes, varying voxel spacing,
uint16 values, and a labels.csv in the same format. Predictions on them are meaningless
by construction — the point is that the pipeline runs end to end without the real data.

Writes to synthetic_data/ (committed in this repository as synthetic_data/).
"""
import os
import random
import numpy as np
import nibabel as nib
import pandas as pd

os.makedirs("synthetic_data", exist_ok=True)

# (filename, 3D shape, voxel spacing in mm, label)
finti = [
    ("fake_0001", (128, 128, 128), 3.895, 0.0),
    ("fake_0002", (142, 142, 112), 2.46,  1.0),
    ("fake_0003", (100, 100, 100), 4.5,   0.0),
]

random.seed(0)
for k in range(4, 13):
    lato = random.choice([96, 112, 128, 140])
    shape = (lato, lato, random.choice([96, 112]))       # deliberately not cubic
    spacing = random.choice([2.46, 3.0, 3.895])
    label = float(random.randint(0, 1))
    finti.append((f"fake_{k:04d}", shape, spacing, label))

righe = []
for uid, shape, spacing, label in finti:
    dati = np.random.randint(0, 65535, size=shape, dtype=np.uint16)
    affine = np.diag([spacing, spacing, spacing, 1.0])
    nib.save(nib.Nifti1Image(dati, affine), f"synthetic_data/{uid}.nii.gz")
    righe.append({"uid": uid, "is_pathologic": label})

pd.DataFrame(righe).to_csv("synthetic_data/labels.csv", index=False)
print(f"done: {len(finti)} volumes written to synthetic_data/")
