"""Cross-validated training of the 3D CNN, with snapshot averaging.

Every architectural option is a flag so they can be tested one at a time. The submitted
configuration is `pool2 res big aug_forte salva`.

Two choices here are about honest measurement rather than accuracy:

Snapshot averaging. The reported score is the mean of the predictions of the last 15
epochs, never the best epoch. Picking the best epoch on the validation fold was
inflating every CNN result by about 0.010, which is larger than most of the differences
we were trying to detect. The "best" value is still printed, but only as a reminder of
how much that bias was worth.

`salva` writes each fold's network, weight-averaged over the same last 15 epochs. Those
are the networks that produced the out-of-fold predictions, so using them at inference
keeps the meta-model on the scale it was fitted on, instead of retraining on all data
and handing it sharper logits than it has ever seen.

Usage: python train_crop2.py <epochs> <folds> <seed> <crop_file> <tag> [options]
Options: group | se | res | pool2 | mix | big | xl | aug_forte | aug_extra | salva
"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from modello_crop2 import CNNCrop2
from augment import aumenta

EPOCHE = int(sys.argv[1]) if len(sys.argv) > 1 else 100
K = int(sys.argv[2]) if len(sys.argv) > 2 else 3
SEME = int(sys.argv[3]) if len(sys.argv) > 3 else 0
FILE = sys.argv[4] if len(sys.argv) > 4 else "crop48_mm2_0.npz"
TAG = sys.argv[5] if len(sys.argv) > 5 else "c2"
OPZ = set(sys.argv[6:])
BATCH, LR, ULTIME = 16, 0.001, 15
MIX_ALFA = 0.4

CFG = dict(tipo_norma="group" if "group" in OPZ else "batch",
           usa_se="se" in OPZ, residuo="res" in OPZ, doppio_pool="pool2" in OPZ,
           canali=((32, 64, 128, 128) if "xl" in OPZ else
                   (24, 48, 96, 96) if "big" in OPZ else (16, 32, 64, 64)))
LIV_AUG = "extra" if "aug_extra" in OPZ else ("forte" if "aug_forte" in OPZ else "base")
torch.manual_seed(SEME)
np.random.seed(SEME)
torch.set_num_threads(os.cpu_count())
d = np.load(FILE)
X, y = d["X"], d["y"]
print(f"{FILE}: {X.shape} | epochs={EPOCHE} folds={K} seed={SEME} | options={sorted(OPZ) or 'none'}")
print(f"config: {CFG} | mixup={'yes' if 'mix' in OPZ else 'no'} | augmentation={LIV_AUG}")


class Dati(Dataset):
    def __init__(self, idx, aug):
        self.idx, self.aug = idx, aug

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        v = X[j].copy()
        if self.aug:
            v = aumenta(v, LIV_AUG)
        v = np.ascontiguousarray(v, dtype=np.float32)
        return torch.from_numpy(v).unsqueeze(0), torch.tensor([y[j]], dtype=torch.float32)


def fold(k, tr_idx, va_idx):
    tr = DataLoader(Dati(tr_idx, True), batch_size=BATCH, shuffle=True, drop_last=True)
    va = DataLoader(Dati(va_idx, False), batch_size=BATCH)
    rete = CNNCrop2(**CFG)
    opt = torch.optim.Adam(rete.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHE)
    lossf = torch.nn.BCEWithLogitsLoss()
    snapshot, best = [], 9.0
    pesi_medi = None
    for ep in range(EPOCHE):
        rete.train()
        for xb, yb in tr:
            if "mix" in OPZ:                               # mixup: blend two patients and their labels
                lam = np.random.beta(MIX_ALFA, MIX_ALFA)
                perm = torch.randperm(xb.size(0))
                xb = lam * xb + (1 - lam) * xb[perm]
                yb = lam * yb + (1 - lam) * yb[perm]
            opt.zero_grad()
            lossf(rete(xb), yb).backward()
            opt.step()
        sched.step()
        rete.eval()
        veri, prob = [], []
        with torch.no_grad():
            for xb, yb in va:
                prob.extend(torch.sigmoid(rete(xb)).squeeze(1).tolist())
                veri.extend(yb.squeeze(1).tolist())
        ll = log_loss(veri, prob, labels=[0., 1.])
        best = min(best, ll)
        if ep >= EPOCHE - ULTIME:
            snapshot.append(np.array(prob))
            if "salva" in OPZ:                             # weight averaging over the same epochs
                sd = {n: v.detach().clone() for n, v in rete.state_dict().items()}
                pesi_medi = ({n: v.float() for n, v in sd.items()} if pesi_medi is None
                             else {n: pesi_medi[n] + sd[n].float() for n in sd})
        if (ep + 1) % 10 == 0:
            print(f"  fold {k} epoch {ep+1}: {ll:.4f} (best {best:.4f})")
    media = np.mean(snapshot, axis=0)
    onesta = log_loss(veri, np.clip(media, 1e-6, 1-1e-6), labels=[0., 1.])
    print(f"  fold {k}: honest={onesta:.4f} ('best' would have claimed {best:.4f})")
    if "salva" in OPZ:
        rif = rete.state_dict()                            # restore integer dtypes of BatchNorm counters
        uscita = f"fold_{TAG}_s{SEME}_f{k}.pt"
        torch.save({n: (v / ULTIME).to(rif[n].dtype) for n, v in pesi_medi.items()}, uscita)
        print(f"  saved {uscita}")
    return onesta, media


oof = np.zeros(len(y))
ris = []
for k, (tr_idx, va_idx) in enumerate(StratifiedKFold(K, shuffle=True, random_state=0).split(X, y), 1):
    b, p = fold(k, tr_idx, va_idx)
    oof[va_idx] = p
    ris.append(b)
np.savez(f"oof_cnn{TAG}_s{SEME}.npz", prob=oof, y=y)
print(f"MEAN = {np.mean(ris):.4f} | OOF = {log_loss(y, np.clip(oof,1e-6,1-1e-6), labels=[0.,1.]):.4f}")
