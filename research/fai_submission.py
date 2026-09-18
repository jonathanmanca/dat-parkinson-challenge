"""Build submission.zip, with the checks that matter run before anything is written.

The file list is not maintained by hand. Modules are discovered by walking the imports
of main.py, and model files are derived from config_submission.py — the same file
main.py reads — so the package cannot drift out of step with the code. When a new
dependency was added late in the competition it was picked up automatically.

Checks, in order: every file exists; every model actually loads, with the CNN weights
matched against the configured architecture; and the branch order stored in the
meta-model equals ORDINE, which is the one mismatch that would produce a submission
that runs perfectly and predicts nonsense. The previous archive is kept as a backup,
and the finished zip is reopened and verified to be flat with main.py at the root.

Usage: python fai_submission.py             build it
       python fai_submission.py --controlla check only, write nothing
"""
import ast
import os
import sys
import shutil
import zipfile
import joblib
import torch
from modello_crop2 import CNNCrop2
from config_submission import RAMI, ORDINE

SOLO_CONTROLLO = "--controlla" in sys.argv
USCITA, INGRESSO = "submission.zip", "main.py"


def moduli_locali(f, visti=None):
    """Follow a file's imports recursively, keeping only modules that live here."""
    visti = set() if visti is None else visti
    if f in visti:
        return visti
    visti.add(f)
    for n in ast.walk(ast.parse(open(f, encoding="utf-8").read())):
        nomi = ([a.name for a in n.names] if isinstance(n, ast.Import) else
                [n.module] if isinstance(n, ast.ImportFrom) and n.module else [])
        for m in nomi:
            g = m.split(".")[0] + ".py"
            if os.path.exists(g):
                moduli_locali(g, visti)
    return visti


file = sorted(moduli_locali(INGRESSO))
modelli = []
for nome in ORDINE:
    r = RAMI[nome]
    modelli += ([f"finale_{nome}.joblib"] if r["tipo"] == "feat"
                else [f"finale_{nome}_s{s}.pt" for s in r["semi"]])
modelli.append("finale_meta.joblib")
tutti = file + sorted(modelli)

print(f"--- 1) required files ({len(tutti)}) ---")
mancanti = [f for f in tutti if not os.path.exists(f)]
for f in tutti:
    if os.path.exists(f):
        print(f"  OK      {f:24s} {os.path.getsize(f)/1024:8.0f} KB")
    else:
        print(f"  MISSING {f}")
if mancanti:
    sys.exit(f"\nSTOP: missing {mancanti}")

print("\n--- 2) do the models actually load? ---")
for nome in ORDINE:
    r = RAMI[nome]
    if r["tipo"] == "cnn":
        for s in r["semi"]:
            CNNCrop2(**r["arch"]).load_state_dict(torch.load(f"finale_{nome}_s{s}.pt", map_location="cpu"))
            print(f"  {nome} seed {s}: weights match the configured architecture")
    else:
        d = joblib.load(f"finale_{nome}.joblib")
        print(f"  {nome}: {type(d['modello']).__name__}, scale {d['scala']} mm, {d['n_feature']} features")

meta = joblib.load("finale_meta.joblib")
print(f"\n--- 3) branch order ---\n  meta trained on: {meta['rami']}\n  config ORDINE:   {ORDINE}")
if meta["rami"] != ORDINE:
    sys.exit("STOP: branch order differs -> the submission would predict nonsense")
print("  match")

if os.path.exists(USCITA):
    with zipfile.ZipFile(USCITA) as z:
        vecchi = set(z.namelist())
    nuovi = set(tutti)
    print("\n--- 4) differences from the previous archive ---")
    print(f"  added:   {sorted(nuovi - vecchi) or 'none'}")
    print(f"  removed: {sorted(vecchi - nuovi) or 'none'}")

if SOLO_CONTROLLO:
    print("\n(--controlla: all good, nothing written)")
    raise SystemExit(0)

if os.path.exists(USCITA):
    shutil.copy2(USCITA, "submission_precedente.zip")
    print("\n  backup -> submission_precedente.zip")

with zipfile.ZipFile(USCITA, "w", zipfile.ZIP_DEFLATED) as z:
    for f in tutti:
        z.write(f, arcname=os.path.basename(f))            # flat archive, as the runtime expects

with zipfile.ZipFile(USCITA) as z:                         # reopen and verify what was written
    dentro = sorted(z.namelist())
    assert dentro == sorted(tutti), "the archive does not contain what was intended"
    assert "main.py" in dentro, "main.py must sit at the root of the archive"
    assert not any("/" in n or "\\" in n for n in dentro), "the archive must be flat"
print(f"\nwrote {USCITA}: {len(dentro)} files, {os.path.getsize(USCITA)/1e6:.2f} MB")
