"""Load a DaT SPECT volume and resample it onto a fixed physical grid.

Scans come from 10 hospitals with voxel sizes between 1.37 and 4.42 mm, so the raw
arrays are not comparable to one another. Everything downstream assumes a common
millimetre grid, which this module produces.
"""
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, center_of_mass


def centro_testa(v):
    """Locate the head inside the volume.

    Field of view varies enormously between scanners: one group covers 629 mm per side,
    another 499 mm, while a head is about 200 mm. Cropping at the geometric centre of
    the array therefore discarded up to half the signal for those groups and left the
    striatum against the edge of the window.

    The centroid is taken over a binary mask rather than weighted by intensity, so that
    a single bright voxel cannot drag it; the threshold uses the 99.5th percentile
    instead of the maximum for the same reason.
    """
    rif = np.percentile(v, 99.5)
    mask = v > 0.3 * rif
    if mask.sum() < 100:                                   # empty or malformed volume
        return (np.array(v.shape) - 1) / 2
    return np.array(center_of_mass(mask))


def carica_volume_mm(percorso, mm=2.0, lato=112):
    """Resample to `mm` per voxel, then return a fixed `lato`^3 cube centred on the head."""
    img = nib.load(percorso)
    dati = img.get_fdata()
    spacing = np.sqrt((img.affine[:3, :3] ** 2).sum(0))
    spacing = np.where(spacing > 0, spacing, mm)           # fall back if the header is broken
    dati = zoom(dati, spacing / mm, order=1)

    out = np.zeros((lato, lato, lato), dtype=np.float32)
    dati = np.maximum(dati, 0)                             # negatives would corrupt the centroid
    cm = centro_testa(dati)
    tagli = []
    for a in range(3):
        if dati.shape[a] > lato:
            i = int(round(cm[a] - lato / 2))
            i = max(0, min(i, dati.shape[a] - lato))
            tagli.append(slice(i, i + lato))
        else:
            tagli.append(slice(None))
    dati = dati[tuple(tagli)]                              # sliced once, so `cm` stays valid on every axis
    i = [(lato - dati.shape[a]) // 2 for a in range(3)]
    out[i[0]:i[0]+dati.shape[0], i[1]:i[1]+dati.shape[1], i[2]:i[2]+dati.shape[2]] = dati

    mn, mx = out.min(), out.max()
    return ((out - mn) / (mx - mn + 1e-8)).astype(np.float32)
