"""Radiomic features computed on the whole resampled volume.

The clinically meaningful quantity in a DaT scan is the striatal binding ratio: how much
brighter the striatum is than surrounding tissue. Most features here are therefore
ratios of a high percentile to a low one rather than absolute intensities, which keeps
them comparable across scanners.
"""
import numpy as np


def stat(v):
    p = np.percentile(v, [10, 50, 90, 99])
    return [p[0], p[1], p[2], p[3], v.mean(), v.std(), p[3] / (p[1] + 1e-6)]


def feature_volume(v, griglia=4):
    """Features of a cubic normalised volume, at whatever side length it has."""
    n = v.shape[0]
    f = list(stat(v))
    p = np.percentile(v, [50, 75, 90, 95, 99, 99.5, 99.9])
    f += list(p) + list(p[1:] / (p[0] + 1e-6))

    z = (v - v.mean()) / (v.std() + 1e-6)
    f += [float((z**3).mean()), float((z**4).mean())]      # skewness and kurtosis

    a, b = n // 4, 3 * n // 4
    c = v[a:b, a:b, a:b]
    f += list(stat(c))
    f += [c.mean() / (v.mean() + 1e-6), np.percentile(c, 99) / (np.percentile(v, 50) + 1e-6)]

    # Parkinson's onset is asymmetric, so left-right differences carry signal. The
    # anatomical orientation of the array is not assumed: all three axes are measured.
    m = c.shape[0] // 2
    for asse in range(3):
        s1 = np.take(c, range(0, m), axis=asse)
        s2 = np.take(c, range(m, 2*m), axis=asse)
        for q in [50, 90, 99]:
            x, yq = np.percentile(s1, q), np.percentile(s2, q)
            f.append(abs(x - yq) / (x + yq + 1e-6))

    while n % griglia != 0:                                # the grid must divide the side
        griglia -= 1
    g = n // griglia
    G = v.reshape(griglia, g, griglia, g, griglia, g).mean(axis=(1, 3, 5))
    f += list(G.flatten() / (v.mean() + 1e-6))
    for q in [95, 99]:                                     # where the bright signal sits
        idx = np.argwhere(v >= np.percentile(v, q))
        f += list(idx.mean(0) / n) + [float(idx.std(0).mean()) / n]
    return np.array(f, dtype=np.float32)


if __name__ == "__main__":
    d = np.load("cache.npz")
    F = np.stack([feature_volume(v) for v in d["X"]])
    np.savez("feat_cliniche2.npz", FEAT=F, y=d["y"])
    print("features:", F.shape)
