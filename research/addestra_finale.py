"""Fit the feature branches on all the data, and the meta-model on the out-of-fold files.

The branches are refitted on all 1362 cases because that is the best estimate available
at submission time. The meta-model cannot be: it has to learn from predictions made on
data the branch had not seen, which is exactly what the out-of-fold files hold.

The column order comes from ORDINE and is stored alongside the meta-model, so that
main.py can assert on it before predicting.
"""
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from config_submission import RAMI, ORDINE, META_C

FEAT = {1.5: np.load("feat6_mm1_5.npz"), 2.5: np.load("feat6_mm2_5.npz")}
y = FEAT[1.5]["y"]

for nome in ORDINE:
    r = RAMI[nome]
    if r["tipo"] != "feat":                                # CNNs come from train_crop2.py
        continue
    X = FEAT[r["scala"]]["FEAT"][:, :r["n_feature"]]
    m = r["crea"]()
    m.fit(X, y)
    joblib.dump({"modello": m, "scala": r["scala"], "n_feature": r["n_feature"]}, f"finale_{nome}.joblib")
    print(f"saved finale_{nome}.joblib (scale {r['scala']} mm, {r['n_feature']} features)")

cols = []
for nome in ORDINE:
    p = np.clip(np.load(RAMI[nome]["oof"])["prob"], 1e-6, 1 - 1e-6)
    cols.append(np.log(p / (1 - p)))                       # logits, not probabilities
meta = LogisticRegression(C=META_C, max_iter=2000).fit(np.column_stack(cols), y)
joblib.dump({"meta": meta, "rami": ORDINE}, "finale_meta.joblib")
print("\nsaved finale_meta.joblib")
print("weights:", dict(zip(ORDINE, np.round(meta.coef_[0], 3))))
