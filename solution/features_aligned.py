"""Features measured on a window aligned to the striatum rather than to the array.

The whole-volume features assume the striatum sits in the middle, but the 10 centres
position their scans differently. Locating the bright region first and measuring around
it was worth 0.033 in log loss — the second largest single gain in the project.
"""
import numpy as np


def trova_hotspot(v, q=99.0):
    """Centroid, spread and coordinates of the brightest voxels, i.e. the striatum."""
    soglia = np.percentile(v, q)
    idx = np.argwhere(v >= soglia)
    centro = idx.mean(0)
    disp = idx.std(0)
    return centro, disp, idx, soglia


def ritaglia(v, centro, lato):
    n = v.shape[0]
    h = lato // 2
    c = np.clip(centro.astype(int), h, n - h)
    return v[c[0]-h:c[0]+h, c[1]-h:c[1]+h, c[2]-h:c[2]+h]


def feature_allineate(v):
    n = v.shape[0]
    centro, disp, idx, soglia = trova_hotspot(v)
    f = list(centro / n) + list(disp / n)
    f += [len(idx) / v.size, float(v[v >= soglia].mean())]

    fondo = np.percentile(v, 50) + 1e-6                # background reference
    for lato in [n // 4, n // 3, n // 2]:              # three window sizes around the striatum
        r = ritaglia(v, centro, lato)
        p = np.percentile(r, [50, 75, 90, 95, 99])
        f += list(p / fondo)                           # ratios = binding ratio, measured in place
        f += [r.mean() / fondo, r.std() / fondo, float(r.max()) / fondo]

    r = ritaglia(v, centro, n // 2)
    m = r.shape[0] // 2
    for asse in range(3):
        a = np.take(r, range(0, m), axis=asse)
        b = np.take(r, range(m, 2*m), axis=asse)
        for q in [50, 90, 99]:
            x, yq = np.percentile(a, q), np.percentile(b, q)
            f.append(abs(x - yq) / (x + yq + 1e-6))
        f.append(abs(a.mean() - b.mean()) / (a.mean() + b.mean() + 1e-6))
    g = r.shape[0] // 4
    q = r[:4*g, :4*g, :4*g]                            # trim to a multiple of 4 so the grid is always 4x4x4
    f += list((q.reshape(4, g, 4, g, 4, g).mean(axis=(1, 3, 5))).flatten() / fondo)
    return np.array(f, dtype=np.float32)
