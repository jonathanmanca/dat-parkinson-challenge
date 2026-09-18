"""Single source of truth for the submitted configuration.

`addestra_finale.py` (training) and `main.py` (inference) both read from here, so the
two can never disagree about which models exist, in what order, or on which inputs.
That ordering matters: the meta-model is a logistic regression over branch logits, and
feeding it the columns in a different order would produce a submission that runs
cleanly and predicts nonsense. `main.py` asserts on it before predicting.
"""
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

L = dict(num_leaves=15, reg_lambda=10.0, colsample_bytree=0.2, subsample=0.7,
         subsample_freq=1, min_child_samples=40, random_state=0, verbose=-1)


def _gb(d, i):
    return HistGradientBoostingClassifier(max_depth=d, learning_rate=0.05,
                                          max_iter=i, l2_regularization=1.0, random_state=0)


def _lr(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


# One entry per branch: how to build it, which inputs it needs, and the out-of-fold file
# the meta-model learns from. "feat" branches run on engineered features, "cnn" on crops.
# Feature sets are nested prefixes of one 672-column vector, so each scale is computed once.
RAMI = {
    "lgb25":  dict(tipo="feat", scala=2.5, n_feature=672, oof="oof_lgb25_C1.npz",
                   crea=lambda: lgb.LGBMClassifier(n_estimators=2400, learning_rate=0.0075, **L)),
    "lgb15":  dict(tipo="feat", scala=1.5, n_feature=672, oof="oof_lgb15_B.npz",
                   crea=lambda: lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, **L)),
    "lr15":   dict(tipo="feat", scala=1.5, n_feature=518, oof="oof_mm15_LR_C0.1.npz",
                   crea=lambda: _lr(0.1)),
    "lr25":   dict(tipo="feat", scala=2.5, n_feature=518, oof="oof_mm25_LR_C0.1.npz",
                   crea=lambda: _lr(0.1)),
    # `arch` is passed straight to the CNNCrop2 constructor and must match what was
    # trained, or the weights will not load. The three "seeds" are the three CV fold
    # models: using them rather than networks retrained on all data keeps inference on
    # the same scale the meta-model was fitted on.
    # tta=True averages the 8 axis-mirrored views of each patient at inference. Every
    # patient stays independent of the others, as the competition rules require.
    "cnnA":   dict(tipo="cnn", crop=48, mm=2.0, lato=112, semi=[0, 1, 2], aug="forte", tta=True,
                   oof="oof_cnn48new_s0.npz",
                   arch=dict(doppio_pool=True, residuo=True, canali=(24, 48, 96, 96))),
}
ORDINE = list(RAMI)
SCALE_FEAT = {1.5: 152, 2.5: 88}            # millimetres per voxel -> cube side

# 0.03 rather than 0.1: it costs 0.0008 on random folds (noise) but is the minimum on
# scanner-grouped folds, and it halves the large opposing weights the meta-model
# otherwise puts on lr15 and lr25, which are the same model at two correlated scales.
META_C = 0.03
CLIP = (0.01, 0.99)                         # insurance against a confident mistake
