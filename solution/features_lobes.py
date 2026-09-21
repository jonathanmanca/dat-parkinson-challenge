"""Intensity profiles along each axis, and a comparison of the two striatal lobes.

In a healthy scan uptake runs the length of the striatum, giving it a comma shape; as
the posterior putamen loses signal the shape collapses towards a dot. A profile along
an axis captures that gradient without needing to know which axis is which.
"""
import numpy as np
from features_aligned import trova_hotspot, ritaglia


def profilo(r, asse, bin=8):
    m = r.mean(axis=tuple(a for a in range(3) if a != asse))
    idx = np.linspace(0, len(m), bin + 1).astype(int)
    return [float(m[idx[i]:idx[i+1]].mean()) for i in range(bin)]


def feature_lobi(v):
    n = v.shape[0]
    centro, _, idx, _ = trova_hotspot(v, q=99.5)
    r = ritaglia(v, centro, n // 2)
    fondo = np.percentile(v, 50) + 1e-6
    f = []

    for asse in range(3):
        p = np.array(profilo(r, asse)) / fondo
        f += list(p)
        f += [float(p[:4].sum() / (p[4:].sum() + 1e-6))]   # front half over back half

    # The axis along which the bright voxels are most spread is the one separating the
    # two lobes; splitting at its median gives one lobe on each side.
    sep = idx.std(0).argmax()
    taglio = np.median(idx[:, sep])
    A, B = idx[idx[:, sep] < taglio], idx[idx[:, sep] >= taglio]
    val = []
    for L in (A, B):
        if len(L) < 5:
            val.append([0.0, 0.0, 0.0, 0.0])
            continue
        vi = v[L[:, 0], L[:, 1], L[:, 2]]
        val.append([float(vi.mean() / fondo), float(vi.max() / fondo),
                    len(L) / v.size, float(L.std(0).mean() / n)])
    a, b = np.array(val[0]), np.array(val[1])
    # Ordering by strength rather than by side makes the features invariant to which
    # hemisphere is affected, which we cannot read from the array anyway.
    forte, debole = np.maximum(a, b), np.minimum(a, b)
    f += list(forte) + list(debole)
    f += list((forte - debole) / (forte + debole + 1e-6))
    return np.array(f, dtype=np.float32)
