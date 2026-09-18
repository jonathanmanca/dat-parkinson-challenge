"""Average the out-of-fold predictions of several seeds of the same configuration.

Averaging seeds is not just noise reduction, it also removes a bias: picking the best
of three seeds is still selection, and the number you get that way is not available to
you at submission time. The average is what you actually have.

The tag is explicit and required. An earlier version globbed for a pattern that had
since been renamed, which would have silently averaged two seeds instead of three and
overwritten the output file with the wrong thing.

Usage: python media_semi.py <tag> [output.npz]
"""
import sys
import glob
import numpy as np
from sklearn.metrics import log_loss

TAG = sys.argv[1] if len(sys.argv) > 1 else "48a"
USCITA = sys.argv[2] if len(sys.argv) > 2 else f"oof_cnn{TAG}_medio.npz"
file = sorted(glob.glob(f"oof_cnn{TAG}_s*.npz"))
if not file:
    print(f"no files for tag '{TAG}' (looking for oof_cnn{TAG}_s*.npz)")
    sys.exit(1)

P, y = [], None
for f in file:
    d = np.load(f)
    p = np.clip(d["prob"], 1e-6, 1-1e-6)
    y = d["y"]
    P.append(p)
    print(f"{f}: {log_loss(y, p, labels=[0., 1.]):.4f}")

media = np.mean(P, axis=0)
print(f"\nmean of {len(P)} seeds: {log_loss(y, media, labels=[0., 1.]):.4f}")
np.savez(USCITA, prob=media, y=y)
print(f"saved {USCITA}")
