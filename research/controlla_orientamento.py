"""Count how many distinct anatomical orientations and fields of view the dataset contains.

Reads only the NIfTI header and affine matrix, never a voxel, and prints aggregate
counts only — no identifiers, nothing patient-level.

The affine says how the array is laid out relative to anatomy: R/L right-left,
A/P anterior-posterior, S/I superior-inferior. ('R','A','S') means axis 0 increases
towards the right, axis 1 towards the front, axis 2 upwards.

The hypothesis being tested was that different centres store their volumes in different
orientations, which would make every asymmetry feature incomparable across patients.
It was wrong: all 1362 scans are RAS. Ten minutes to rule out, instead of three days of
building around it. The output is still useful for what it says about field of view and
about how many distinct scanner signatures the dataset contains.

Usage: python controlla_orientamento.py [folder]
"""
import sys
from collections import Counter
import pandas as pd
import nibabel as nib

CARTELLA = sys.argv[1] if len(sys.argv) > 1 else "data_privata"
uids = pd.read_csv(f"{CARTELLA}/labels.csv")["uid"]
print(f"folder: {CARTELLA} | files: {len(uids)}\n")

orient, perm, forme, spaziature, scanner = Counter(), Counter(), Counter(), Counter(), Counter()
for u in uids:
    img = nib.load(f"{CARTELLA}/{u}.nii.gz")               # lazy: get_fdata is never called
    cod = nib.aff2axcodes(img.affine)
    orient[cod] += 1
    # Dropping the direction leaves only which anatomical axis is which, separating
    # permutations (a transpose, which no augmentation covers) from reflections.
    perm[tuple({"R": "DS", "L": "DS", "A": "AD", "P": "AD", "S": "SI", "I": "SI"}[c] for c in cod)] += 1
    f = img.shape
    s = tuple(round(float(z), 2) for z in img.header.get_zooms()[:3])
    forme[f] += 1
    spaziature[s] += 1
    scanner[(f, s, cod)] += 1


def mostra(titolo, c, nota=""):
    print(f"--- {titolo}: {len(c)} distinct --- {nota}")
    for k, n in c.most_common():
        print(f"   {str(k):28s} {n:5d} cases  ({100*n/len(uids):4.1f}%)")
    print()


mostra("ORIENTATIONS (direction included)", orient)
mostra("AXIS PERMUTATIONS", perm, "<- if >1, axis 0 does not mean the same thing for everyone")
mostra("volume shapes", forme)
mostra("voxel spacings (mm)", spaziature)

canonico = orient.get(("R", "A", "S"), 0)
print(f"already canonical RAS: {canonico}/{len(uids)} ({100*canonico/len(uids):.1f}%)")
print(f"would need reorienting: {len(uids)-canonico} ({100*(len(uids)-canonico)/len(uids):.1f}%)")
print(f"\ndistinct scanner signatures (shape + spacing + orientation): {len(scanner)}")
print("  (also feeds the other question: if random folds split these groups,")
print("   cross-validation is optimistic)")
print("  group sizes:", sorted((n for n in scanner.values()), reverse=True)[:12])
