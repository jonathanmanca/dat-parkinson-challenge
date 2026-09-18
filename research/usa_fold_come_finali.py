"""Promote the cross-validation fold networks to the submission's CNN weights.

These are the same networks that produced the out-of-fold predictions the meta-model
learned from, so at inference it receives logits on the scale it was fitted on. The
alternative — retraining on all 1362 cases — gives individually stronger networks but
sharper logits than the meta has ever seen, and it costs another full training run.

Every file is verified against the configured architecture before anything is copied,
and the previous weights are kept with a .vecchio suffix.

Usage: python usa_fold_come_finali.py <TAG> [seed]
"""
import os
import sys
import glob
import shutil
import torch
from modello_crop2 import CNNCrop2
from config_submission import RAMI

TAG = sys.argv[1] if len(sys.argv) > 1 else "48max"
SEME = sys.argv[2] if len(sys.argv) > 2 else "*"
r = RAMI["cnnA"]

file = sorted(glob.glob(f"fold_{TAG}_s{SEME}_f*.pt"))
print(f"found {len(file)} networks: {file}")
if len(file) != len(r['semi']):
    sys.exit(f"STOP: config_submission expects {len(r['semi'])} networks (seeds {r['semi']}), "
             f"found {len(file)}")

for f in file:                                             # check all of them before copying any
    CNNCrop2(**r["arch"]).load_state_dict(torch.load(f, map_location="cpu"))
print("all compatible with the configured architecture")

for s, f in zip(r["semi"], file):
    dest = f"finale_cnnA_s{s}.pt"
    if os.path.exists(dest) and not os.path.exists(dest + ".vecchio"):
        shutil.copy2(dest, dest + ".vecchio")
    shutil.copy2(f, dest)
    print(f"  {f}  ->  {dest}  ({os.path.getsize(dest)/1e6:.2f} MB)")
print("\ndone. Previous weights kept with a .vecchio suffix.")
