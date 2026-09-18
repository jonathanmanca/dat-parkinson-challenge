"""3D augmentation for the striatum crops, at three intensities.

"base" is mirrors, small shifts and mild intensity jitter. "forte" adds rotations and
scaling and was worth 0.016 on its own — with only 1362 cases the base level was
leaving the model short of variety. "extra" adds Gaussian noise and was not adopted.

Mirroring on all three axes is deliberate even though Parkinson's is asymmetric: the
anatomical orientation of the array is not something the pipeline knows, so the model
is made insensitive to it rather than allowed to rely on it.
"""
import numpy as np
from scipy.ndimage import rotate, zoom


def aumenta(v, livello="base", rng=np.random):
    if livello == "base":
        for a in range(3):
            if rng.rand() < 0.5:
                v = np.flip(v, a)
        v = np.roll(v, rng.randint(-2, 3, 3), axis=(0, 1, 2))
        return np.ascontiguousarray(v * rng.uniform(0.9, 1.1), dtype=np.float32)

    for a in range(3):
        if rng.rand() < 0.5:
            v = np.flip(v, a)
    v = np.ascontiguousarray(v, dtype=np.float32)

    if rng.rand() < 0.7:                                 # small rotation in a random plane
        assi = [(0, 1), (0, 2), (1, 2)][rng.randint(3)]
        v = rotate(v, rng.uniform(-12, 12), axes=assi, reshape=False, order=1, mode="nearest")

    if rng.rand() < 0.5:                                 # scaling: heads differ in size
        f = rng.uniform(0.9, 1.1)
        n = v.shape[0]
        z = zoom(v, f, order=1)
        if z.shape[0] >= n:
            i = (z.shape[0] - n) // 2
            v = z[i:i+n, i:i+n, i:i+n]
        else:
            out = np.zeros((n, n, n), dtype=np.float32)
            i = (n - z.shape[0]) // 2
            out[i:i+z.shape[0], i:i+z.shape[1], i:i+z.shape[2]] = z
            v = out

    v = np.roll(v, rng.randint(-3, 4, 3), axis=(0, 1, 2))
    v = v * rng.uniform(0.85, 1.15)

    if livello == "extra":
        v = v + rng.normal(0, 0.02, v.shape)             # simulates noisier scanners
    return np.ascontiguousarray(v, dtype=np.float32)


if __name__ == "__main__":
    import time
    v = np.random.rand(48, 48, 48).astype(np.float32)
    for liv in ["base", "forte", "extra"]:
        t = time.time()
        for _ in range(50):
            out = aumenta(v.copy(), liv)
        print(f"{liv:6s}: {out.shape} NaN={bool(np.isnan(out).any())} "
              f"range[{out.min():.2f},{out.max():.2f}] | {(time.time()-t)/50*1000:.1f} ms/volume")
