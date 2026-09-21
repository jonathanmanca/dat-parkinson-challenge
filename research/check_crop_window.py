"""Is the preprocessing window throwing away the brain on some scanners?

This is the script that found the largest defect in the project. It replays the exact
sequence of carica_volume_mm on a sample of patients per scanner group, and reports the
geometric-centre crop and the head-centred crop side by side, on the same patients:

  - how far the brain is from the geometric centre of the array
  - how much of the total signal survives the 112-voxel window
  - how often the striatum ends up so close to the edge that the CNN's 48^3 crop is
    pushed flat against the border

Results that motivated the fix: one group of 70 kept 49.5% of its signal with 100% of
crops against the edge, and the largest group (460 cases) kept 72.5% with 60% against
the edge. After centring on the head: 95.3%/0% and 97.7%/0%. About 530 of 1362 patients
had been getting a crop taken in the wrong place, and no metric had ever pointed at it.

Prints group averages only, never anything patient-level.
Usage: python check_crop_window.py [patients_per_group]
"""
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import zoom, center_of_mass
from features_aligned import trova_hotspot
from preprocess import centro_testa

CARTELLA, MM, LATO, CROP = "data_privata", 2.0, 112, 48
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
GRUPPI = [8, 15, 103, 6, 130]              # two suspects and three groups that looked fine

lab = pd.read_csv(f"{CARTELLA}/labels.csv")
g = np.load("gruppi.npz")["g"]
rng = np.random.default_rng(0)
print(f"sample of {N} patients per group | window: {LATO} voxels x {MM} mm = {int(LATO*MM)} mm\n")
print(f"{'':>10s} {'---- OLD (geometric centre) ----':>38s}  {'---- NEW (centred on head) ----':>38s}")
print(f"{'gr':>4s} {'cases':>5s} {'off-centre':>14s} {'kept':>8s} {'edge':>7s}   "
      f"{'kept':>8s} {'edge':>7s}")


def finestra(v, cm):
    """The LATO^3 window centred on `cm`, clamped inside the volume."""
    t = []
    for a in range(3):
        if v.shape[a] > LATO:
            i = int(round(cm[a] - LATO / 2))
            i = max(0, min(i, v.shape[a] - LATO))
            t.append(slice(i, i + LATO))
        else:
            t.append(slice(None))
    return v[tuple(t)]


def al_bordo(rit):
    """True if the striatum sits so near the margin that the CNN crop hits the clamp."""
    out = np.zeros((LATO, LATO, LATO), dtype=np.float32)
    o = [(LATO - rit.shape[a]) // 2 for a in range(3)]
    out[o[0]:o[0]+rit.shape[0], o[1]:o[1]+rit.shape[1], o[2]:o[2]+rit.shape[2]] = rit
    mn, mx = out.min(), out.max()
    out = (out - mn) / (mx - mn + 1e-8)
    c, _, _, _ = trova_hotspot(out, q=99.5)
    h = CROP // 2
    return bool(np.any(c.astype(int) != np.clip(c.astype(int), h, LATO - h)))


for k in GRUPPI:
    idx = np.where(g == k)[0]
    idx = rng.choice(idx, min(N, len(idx)), replace=False)
    lontananza, tenuto, in_battuta = [], [], 0
    tenuto_n, in_battuta_n = [], 0
    for i in idx:
        img = nib.load(f"{CARTELLA}/{lab['uid'][i]}.nii.gz")
        sp = np.sqrt((img.affine[:3, :3] ** 2).sum(0))
        sp = np.where(sp > 0, sp, MM)
        pieno = np.maximum(zoom(img.get_fdata(), sp / MM, order=1), 0)

        cm = np.array(center_of_mass(pieno))
        geo = (np.array(pieno.shape) - 1) / 2
        lontananza.append(float(np.linalg.norm(cm - geo) * MM))

        tot = pieno.sum() + 1e-9
        vecchio = finestra(pieno, geo)
        tenuto.append(float(vecchio.sum() / tot))
        in_battuta += al_bordo(vecchio)

        nuovo = finestra(pieno, centro_testa(pieno))
        tenuto_n.append(float(nuovo.sum() / tot))
        in_battuta_n += al_bordo(nuovo)

    print(f"{k:4d} {len(idx):5d} {np.mean(lontananza):11.0f} mm "
          f"{100*np.mean(tenuto):7.1f}% {100*in_battuta/len(idx):6.0f}%   "
          f"{100*np.mean(tenuto_n):7.1f}% {100*in_battuta_n/len(idx):6.0f}%")

print("\n'kept' = share of total signal surviving the crop (near 100% is good)")
print("'edge' = how often the CNN crop is pushed against the border (0% is good)")
print("\nThe fix works if, in the NEW columns, kept rises towards 100% and edge falls to")
print("0% for every group, without making the groups that were already fine any worse.")
