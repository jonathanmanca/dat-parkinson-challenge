"""Clinically motivated features: binding ratio, shape, sub-regions, texture.

These follow how a nuclear medicine reader describes a DaT scan. The reference is the
median of brain tissue rather than air, because the quantity of interest is striatum
against surrounding tissue, and the sub-regions follow the caudate-to-putamen axis,
along which the posterior putamen is the first part to lose uptake.
"""
import numpy as np
from features_aligned import trova_hotspot, ritaglia


def fondo_cerebrale(v):
    """Background level: the median of tissue, not of air.

    Air occupies most of the field of view, so a plain median of the volume would
    measure nothing. The threshold separates head from air roughly, with two fallbacks
    for volumes where it cuts too much.
    """
    soglia_aria = np.percentile(v, 50) * 0.5 + v.mean() * 0.25
    dentro = v[v > soglia_aria]
    if dentro.size < 100:
        dentro = v[v > np.percentile(v, 70)]
    if dentro.size < 10:
        dentro = v.ravel()
    return float(np.median(dentro)) + 1e-6, dentro


def sbr(regione, fondo):
    """Striatal binding ratio, the standard clinical quantity."""
    return float((regione.mean() - fondo) / fondo)


def forma(idx, n):
    """Shape of a suprathreshold region from the eigenvalues of its covariance.

    A healthy striatum is elongated (comma-shaped); as uptake is lost it contracts
    towards a dot, so sphericity and elongation carry the same information a reader
    takes from the outline.
    """
    if len(idx) < 8:
        return [0.0] * 7
    c = idx - idx.mean(0)
    ev = np.sort(np.linalg.eigvalsh(np.cov(c.T)))[::-1]
    ev = np.maximum(ev, 1e-9)
    volume = len(idx)
    raggio = (3 * volume / (4 * np.pi)) ** (1 / 3)
    return [volume / n**3,
            float(np.sqrt(ev[0]) / n), float(np.sqrt(ev[2]) / n),
            float(ev[2] / ev[0]),                      # sphericity: 1 sphere, 0 filament
            float(ev[1] / ev[0]),                      # flatness
            float(raggio / n),
            float(np.sqrt(ev.sum()) / n)]


def feature_v6(v):
    n = v.shape[0]
    fondo, dentro = fondo_cerebrale(v)
    f = [fondo, float(dentro.mean()), float(dentro.std()), float(dentro.size) / v.size]

    centro, _, idx, _ = trova_hotspot(v, q=99.5)
    r = ritaglia(v, centro, n // 2)
    for q in [97, 98, 99, 99.5, 99.9]:                 # binding ratio at several thresholds
        m = v >= np.percentile(v, q)
        f.append(sbr(v[m], fondo))
        f += forma(np.argwhere(m), n)
    f += [sbr(r, fondo), float(r.max() / fondo), float(np.percentile(r, 99) / fondo)]
    f += sotto_regioni(v, idx, fondo, n)
    f += texture(r)
    # A single NaN would poison the whole feature matrix and every model downstream.
    return np.nan_to_num(np.array(f, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def texture(r, livelli=16):
    q = np.clip((r / (r.max() + 1e-9) * (livelli - 1)).astype(np.int16), 0, livelli - 1)
    p = np.bincount(q.flatten(), minlength=livelli).astype(float)
    p /= p.sum()
    entropia = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    g = [np.diff(r, axis=a) for a in range(3)]
    out = [entropia, float((p ** 2).sum())]
    for gi in g:
        out += [float(np.abs(gi).mean()), float(gi.std())]
    diff = np.abs(np.diff(q, axis=0)).flatten()        # neighbour contrast, GLCM-like
    out += [float(diff.mean()), float(diff.std()), float((diff == 0).mean())]
    return out


def sotto_regioni(v, idx, fondo, n):
    """Split each lobe into thirds along its own long axis and compare them.

    The long axis runs caudate to putamen, so the three thirds approximate caudate,
    anterior putamen and posterior putamen. They are ordered by intensity rather than
    by position, because the anatomical direction of the array is not known.
    """
    sep = idx.std(0).argmax()                          # axis separating the two lobes
    taglio = np.median(idx[:, sep])
    out = []
    profili = []
    for L in (idx[idx[:, sep] < taglio], idx[idx[:, sep] >= taglio]):
        if len(L) < 12:
            out += [0.0] * 8
            profili.append([0.0, 0.0, 0.0])
            continue
        c = L - L.mean(0)
        vec = np.linalg.eigh(np.cov(c.T))[1][:, -1]
        t = c @ vec
        terzi = np.percentile(t, [33.3, 66.7])
        parti = [L[t < terzi[0]], L[(t >= terzi[0]) & (t < terzi[1])], L[t >= terzi[1]]]
        val = [float(v[p[:, 0], p[:, 1], p[:, 2]].mean() / fondo) if len(p) > 3 else 0.0
               for p in parti]
        profili.append(val)
        est, med, altro = sorted(val)
        out += val + [est / (altro + 1e-6),            # weakest over strongest third
                      med / (altro + 1e-6),
                      float(np.std(val)),
                      len(L) / v.size,
                      float(np.sqrt(np.cov(c.T).trace()) / n)]
    a, b = np.array(profili[0]), np.array(profili[1])
    out += list(np.abs(a - b) / (a + b + 1e-6))        # asymmetry, third by third
    return out
