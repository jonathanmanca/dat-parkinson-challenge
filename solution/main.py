"""Inference entry point: read the NIfTI scans and write submission.csv.

This is the file the competition container executes. It computes engineered features at
two physical scales, a striatum crop for the CNN branch, runs the five branches, and
combines their logits with the meta-model.
"""
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import torch
from joblib import Parallel, delayed
from preprocess_mm import carica_volume_mm
from feature_cliniche2 import feature_volume
from feature_cliniche3 import feature_allineate, trova_hotspot
from feature_cliniche4 import feature_lobi
from feature_cliniche6 import feature_v6
from modello_crop2 import CNNCrop2
from tta import predici_tta
from config_submission import RAMI, ORDINE, SCALE_FEAT, CLIP

RADICE = os.environ.get("CODE_EXEC", "/code_execution")
NIFTI = os.path.join(RADICE, "data", "niftis")
FORMATO = os.path.join(RADICE, "data", "submission_format.csv")
USCITA = os.path.join(RADICE, "submission.csv")

# Logging anything about the hidden test set is forbidden and the platform warns that it
# can lead to disqualification. An early submission of ours tripped that filter by
# printing the number of cases and array shapes. So output is split in two: `fase` emits
# fixed strings and values that come from our own configuration, never from the data,
# while `diag` prints anything data-dependent and only runs locally with DAT_DEBUG=1.
DIAGNOSTICA = os.environ.get("DAT_DEBUG") == "1"


def fase(testo):
    print(f"fase: {testo}", flush=True)


def diag(testo):
    if DIAGNOSTICA:
        print(testo, flush=True)


def feature_scala(percorso, mm, lato):
    """672 engineered features at one physical scale, plus the same at half resolution.

    The feature blocks are nested prefixes: the first 518 columns are what the linear
    branches consume, all 672 go to the gradient-boosted ones.
    """
    v = carica_volume_mm(percorso, mm=mm, lato=lato)
    h = lato // 2
    vg = v.reshape(h, 2, h, 2, h, 2).mean(axis=(1, 3, 5))
    return np.concatenate([feature_volume(v, griglia=4), feature_volume(vg, griglia=4),
                           feature_allineate(v), feature_allineate(vg),
                           feature_lobi(v), feature_lobi(vg),
                           feature_v6(v), feature_v6(vg)])


def crop_paziente(percorso, mm, lato, crop):
    """Cube centred on the striatum, min-max normalised.

    Normalising on the crop's own range acts as automatic contrast stretching and makes
    the shape maximally visible, which is what this branch contributes to the ensemble.
    Dividing by the brain background instead — clinically the more meaningful quantity —
    was tested and made the CNN clearly worse: the absolute ratio is already measured by
    the feature branches, and the contrast stretch is not.
    """
    v = carica_volume_mm(percorso, mm=mm, lato=lato)
    centro, _, _, _ = trova_hotspot(v, q=99.5)
    h = crop // 2
    c = np.clip(centro.astype(int), h, lato - h)
    r = v[c[0]-h:c[0]+h, c[1]-h:c[1]+h, c[2]-h:c[2]+h]
    return ((r - r.min()) / (r.max() - r.min() + 1e-8)).astype(np.float32)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(False)             # 3D convolutions lack deterministic kernels
    qui = os.path.dirname(os.path.abspath(__file__))
    uids = (pd.read_csv(FORMATO)["uid"].astype(str).tolist() if os.path.exists(FORMATO)
            else sorted(f[:-7] for f in os.listdir(NIFTI) if f.endswith(".nii.gz")))
    fase("elenco casi letto")
    diag(f"casi da predire: {len(uids)}")
    ncore = max(1, (os.cpu_count() or 2) - 1)
    perc = [os.path.join(NIFTI, f"{u}.nii.gz") for u in uids]

    scale_feat = {r["scala"] for r in RAMI.values() if r["tipo"] == "feat"}
    X = {}
    for mm in sorted(scale_feat):
        X[mm] = np.stack(Parallel(n_jobs=ncore)(
            delayed(feature_scala)(p, mm, SCALE_FEAT[mm]) for p in perc)).astype(np.float32)
        fase(f"feature a {mm} mm calcolate")
        diag(f"  forma: {X[mm].shape}")

    crops = {}
    for nome in ORDINE:
        r = RAMI[nome]
        if r["tipo"] != "cnn":
            continue
        chiave = (r["crop"], r["mm"], r["lato"])
        if chiave in crops:
            continue
        crops[chiave] = np.stack(Parallel(n_jobs=ncore)(
            delayed(crop_paziente)(p, r["mm"], r["lato"], r["crop"]) for p in perc))
        fase(f"crop {r['crop']} a {r['mm']} mm calcolati")
        diag(f"  forma: {crops[chiave].shape}")

    pred = {}
    for nome in ORDINE:
        r = RAMI[nome]
        if r["tipo"] == "feat":
            d = joblib.load(os.path.join(qui, f"finale_{nome}.joblib"))
            pred[nome] = d["modello"].predict_proba(X[d["scala"]][:, :d["n_feature"]])[:, 1]
        else:
            x = torch.from_numpy(crops[(r["crop"], r["mm"], r["lato"])]).unsqueeze(1)
            out = []
            for s in r["semi"]:
                rete = CNNCrop2(**r["arch"])
                rete.load_state_dict(torch.load(os.path.join(qui, f"finale_{nome}_s{s}.pt"),
                                                map_location="cpu"))
                # eval mode is mandatory, not cosmetic: BatchNorm in training mode would
                # use statistics of the current batch, which is information taken from
                # other test patients and explicitly against the rules.
                rete.eval()
                if r.get("tta"):
                    out.append(predici_tta(rete, x))
                else:
                    with torch.no_grad():
                        out.append(torch.sigmoid(rete(x)).squeeze(1).numpy())
            pred[nome] = np.mean(out, axis=0)
        fase(f"ramo {nome} applicato")
        diag(f"  media prob {pred[nome].mean():.3f}")

    m = joblib.load(os.path.join(qui, "finale_meta.joblib"))
    # If the branch order ever drifted from training, the submission would run cleanly
    # and predict nonsense. Fail loudly instead.
    assert m["rami"] == ORDINE, "BRANCH ORDER DOES NOT MATCH TRAINING"
    M = np.column_stack([logit(pred[n]) for n in m["rami"]])
    prob = np.clip(m["meta"].predict_proba(M)[:, 1], *CLIP)

    pd.DataFrame({"uid": uids, "is_pathologic": prob}).to_csv(USCITA, index=False)
    fase("submission.csv scritto")
    diag(f"  {USCITA}: {len(uids)} righe")


if __name__ == "__main__":
    main()
