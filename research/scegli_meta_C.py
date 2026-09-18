"""How strongly should the meta-model be regularised?

At C=0.1 it put large opposing weights on lr15 (-0.26) and lr25 (+0.35). Those are the
same linear model at two correlated scales, so the meta was effectively using their
difference — a pattern that scores well in-sample and tends to be brittle.

The decision is made on two criteria rather than one: random folds, and folds grouped
by scanner. C=0.03 costs 0.0008 on random folds (noise) but is the minimum on grouped
folds and halves the opposing weights, so that is what was submitted. Choosing on the
random-fold score alone is what had already misled us twice.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import log_loss
from config_submission import RAMI, ORDINE, CLIP

g = np.load("gruppi.npz")["g"]
P, y = [], None
for n in ORDINE:
    f = "oof_cnn48new_s0.npz" if RAMI[n]["tipo"] == "cnn" else RAMI[n]["oof"]
    d = np.load(f)
    p = np.clip(d["prob"], 1e-6, 1-1e-6)
    y = d["y"]
    P.append(np.log(p / (1 - p)))
M = np.column_stack(P)


def valuta(C, fold):
    o = np.zeros(len(y))
    for tr, va in fold:
        o[va] = LogisticRegression(C=C, max_iter=2000).fit(M[tr], y[tr]).predict_proba(M[va])[:, 1]
    return log_loss(y, np.clip(o, *CLIP), labels=[0., 1.])


cas = list(StratifiedKFold(5, shuffle=True, random_state=7).split(M, y))
gru = list(StratifiedGroupKFold(5, shuffle=True, random_state=7).split(M, y, groups=g))
print(f"{'C':>7s} {'random':>9s} {'grouped':>11s}   weights")
for C in [0.01, 0.03, 0.1, 0.3, 1.0]:
    w = LogisticRegression(C=C, max_iter=2000).fit(M, y).coef_[0]
    print(f"{C:7} {valuta(C, cas):9.4f} {valuta(C, gru):11.4f}   "
          f"{dict(zip(ORDINE, np.round(w, 2)))}")
print("\nLooking for the C that keeps both columns low without large opposing weights.")
