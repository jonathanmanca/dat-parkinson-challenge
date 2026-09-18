"""Full ensemble, old pipeline versus new (head-centred crop).

Like-for-like: one CNN seed on both sides, same folds, same meta-model. Comparing a
three-seed average against a single seed would have credited the preprocessing fix with
a gain that came from seed averaging.

Reported on random folds and on scanner-grouped folds, because a change that improves
robustness shows up much more clearly in the second. The result was 0.2511 -> 0.2421 on
random folds and 0.2518 -> 0.2458 on grouped folds, with the CNN branch itself slightly
worse: the whole gain came from the feature branches, which had been computing their
alignment-sensitive features on badly positioned volumes.

Needs the .vecchio backups written by aggiorna_oof.py.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from config_submission import RAMI, ORDINE, META_C, CLIP

g = np.load("gruppi.npz")["g"]
FEAT = [n for n in ORDINE if RAMI[n]["tipo"] == "feat"]


def colonne(suffisso, file_cnn):
    P, y = [], None
    for n in FEAT:
        d = np.load(RAMI[n]["oof"] + suffisso)
        p = np.clip(d["prob"], 1e-6, 1-1e-6)
        y = d["y"]
        P.append(np.log(p / (1 - p)))
    d = np.load(file_cnn)
    p = np.clip(d["prob"], 1e-6, 1-1e-6)
    P.append(np.log(p / (1 - p)))
    return np.column_stack(P), y, log_loss(y, p, labels=[0., 1.])


def stack(M, y, fold):
    o = np.zeros(len(y))
    for tr, va in fold:
        o[va] = LogisticRegression(C=META_C, max_iter=2000).fit(M[tr], y[tr]).predict_proba(M[va])[:, 1]
    o = np.clip(o, *CLIP)
    return log_loss(y, o, labels=[0., 1.]), roc_auc_score(y, o)


for nome, suff, cnn in [("OLD", ".vecchio", "oof_cnn48max_s0.npz"),
                        ("NEW", "",         "oof_cnn48new_s0.npz")]:
    M, y, ll_cnn = colonne(suff, cnn)
    cas = list(StratifiedKFold(5, shuffle=True, random_state=7).split(M, y))
    gru = list(StratifiedGroupKFold(5, shuffle=True, random_state=7).split(M, y, groups=g))
    a, auc = stack(M, y, cas)
    b, _ = stack(M, y, gru)
    meta = LogisticRegression(C=META_C, max_iter=2000).fit(M, y)
    print(f"{nome}: CNN alone {ll_cnn:.4f} | STACK random {a:.4f} (AUROC {auc:.4f}) | "
          f"grouped {b:.4f}")
    print(f"     meta weights: {dict(zip(FEAT + ['cnnA'], np.round(meta.coef_[0], 3)))}")

print("\nWhat matters is the difference between the two rows, not the absolute value:")
print("a single seed scores a little worse than the three-seed average used in submission.")
