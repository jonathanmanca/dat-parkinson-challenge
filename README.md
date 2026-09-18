# DaT Parkinson's Challenge — 53rd of 378

My solution for the [DrivenData DaT Parkinson's Challenge](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/):
predict, from a 3D DaT SPECT brain scan, the probability that it is pathological rather
than normal. Scored by log loss.

**Final: 53rd of 378 ranked participants — private log loss 0.3008, AUROC 0.9410**
(best public 0.2691, from 4 submissions). Top 14%. For reference, 1st place scored
0.2154 and 3rd 0.2286.

Two constraints shaped everything below. All training ran **on 8 CPU cores** — no GPU,
no cloud — which came to roughly 100 hours of compute and about one experiment per
night. And I **never looked at a single scan**. The rules forbid sending competition
data to third-party services, so rather than police which tool might touch what, I
developed entirely against synthetic volumes and left the real ones in a folder I never
opened. Every check in this repository reports aggregate statistics only. That
constraint turned out to matter more than the missing GPU, and there is a section below
on why.

---

## The approach

Five branches, combined by a logistic regression on their logits.

**1. Resample to a common physical grid.** Voxel sizes range from 1.37 to 4.42 mm
across the 10 contributing hospitals, so raw arrays are not comparable. Every scan is
resampled to 2 mm per voxel and cropped to a fixed cube **centred on the head**, located
from the centroid of a suprathreshold mask.

**2. Engineered features (4 branches).** 672 features per scan at each of two scales,
built around what a nuclear medicine reader looks at: the striatal binding ratio at
several thresholds, left-right asymmetry on all three axes, the shape of the
suprathreshold region, a caudate-to-putamen intensity profile, and texture. Two
LightGBM models and two regularised linear models consume nested prefixes of that
vector.

**3. A 3D CNN (1 branch).** A 48³ crop at 2 mm centred on the striatum, min-max
normalised. Four convolutional blocks with residual connections, ~988k parameters,
global average **and** max pooling concatenated before the head. Strong augmentation
(mirrors, ±12° rotations, ±10% scaling, intensity jitter). Three fold models are
averaged, and each patient is predicted over its 8 mirrored views.

**4. Stacking.** A logistic regression on the five branch logits, fitted on out-of-fold
predictions. Final weights: cnnA 0.60, lr25 0.22, lgb15 0.14, lgb25 0.09, lr15 −0.11.

Inference over the full test set takes about 5 minutes on the competition container.

---

## Submission history

| date | change | internal OOF | public | private |
|---|---|---|---|---|
| 30 Aug | first submission — 5-branch stack | 0.2639 | 0.2877 | — |
| 7 Sep | larger CNN + stronger augmentation | 0.2499 | 0.2859 | — |
| 8 Sep | + test-time augmentation | 0.2499 | 0.2856 | — |
| 13 Sep | head-centred crop, features recomputed, fold models, meta regularisation | 0.2429 | **0.2691** | **0.3008** |

Only the last submission's private score is visible, so the per-change private effects
below cannot be separated.

Internal progression, from the start: constant prediction 0.688 → first CNN 0.566 →
clinical features 0.500 → physical resampling 0.367 → stacking 0.288 → 0.2429.

---

## What worked

| change | measured (OOF) | on the leaderboard |
|---|---|---|
| resampling from voxel spacing | −0.053 | predates the first submission |
| features aligned to the striatum | −0.033 | predates the first submission |
| larger CNN + stronger augmentation | −0.0140 | −0.0018 |
| test-time augmentation | −0.0075 | −0.0003 |
| head-centred crop (bundled with 3 other changes) | −0.0070 | −0.0165 |

The last three rows are what I take away from this competition, with one caveat stated
up front: the third is a bundle of four changes, so I cannot attribute its gain to any
one of them. What is measured is that two changes which **refined a working model**
delivered 12% and 4% of what cross-validation promised, while a bundle containing a
**data-correctness fix** delivered more.

The proposed explanation — and it is an explanation, not a measurement — is that
internal validation cannot see the benefit of reducing scanner dependence, because in
random cross-validation the same scanners always appear on both sides of the split.
Consistent with that, the gap between out-of-fold and leaderboard narrowed from +0.036
to +0.026 with that submission, and its grouped-fold score improved as well. Three
observations, one of them bundled, are not enough to make this a rule.

The working heuristic I would use next time: if a change only lowers your
cross-validated score, discount it heavily; if it also narrows the gap between
random-fold and grouped-fold validation, take it more seriously.

## What didn't

| idea | result |
|---|---|
| normalise the crop by brain background instead of min-max | CNN 0.2552 → 0.2739 |
| a second CNN on a different representation in the ensemble | +0.0007, not worth the extra pipeline |
| non-linear meta-model (LightGBM over branch logits) | 0.289 → 0.302, overfits |
| a larger CNN (1.76M parameters) | 0.2528 → 0.2587, starts memorising |
| GroupNorm instead of BatchNorm | 0.3109 → 0.3399 |
| scanner metadata as a moderator in the meta-model | worse on both validation schemes |
| ImageNet-pretrained 2D branch on projections | worse; not included here |

The background-normalisation failure is the interesting one. Dividing by the brain
background gives the CNN the actual clinical quantity, which sounded strictly better.
It was not: min-max normalisation stretches each crop to full range and acts as
automatic contrast enhancement, which is what makes the *shape* legible — and the
absolute ratio was already measured by the feature branches. In an ensemble the
branches need different jobs.

---

## The bug that mattered

For two months the preprocessing cropped its window at the **geometric centre of the
array**. That is fine when the field of view is tight around the head, and wrong when it
is not: one scanner group covers 629 mm per side, another 499 mm, and a head is about
200 mm, so it does not sit in the middle of half a metre of scan.

Measured on a sample, before and after the fix:

| scanner group | cases | signal retained | striatum against the window edge |
|---|---|---|---|
| A | 70 | 49.5% → 95.3% | 100% → 0% |
| B | 460 | 72.5% → 97.7% | 60% → 0% |
| C | 143 | 99.9% → 100% | 0% → 0% |

About 530 of 1362 patients were getting a crop taken in the wrong place, and the group
with 460 cases had a perfectly healthy-looking log loss of 0.2387, so no metric ever
pointed at it.

Normally you catch this in thirty seconds by opening three scans in a viewer. I could
not, so it survived until I went looking at the geometry through aggregate statistics.
The real lesson is not "look at your data" but its corollary: **when you cannot look,
instrument.** Write the checks that report field of view, orientation and how much
signal survives each step, and write them in week one. A data defect does not just cost
its own margin — it silently corrupts every architecture decision you make downstream
of it, and all of mine were made on top of this one.

---

## Measurement notes

Things that cost me real score, recorded so I do not repeat them.

**Measure a change in the configuration where it will be used.** I measured test-time
augmentation on a single network and got 0.016. The submission already averaged three
networks, and seed averaging removes much of the same noise. On the leaderboard TTA was
worth 0.0003.

**Do not bundle.** The final submission changed four things at once. It gained 0.0165 and
I cannot tell you which change earned it.

**Random cross-validation flatters a multi-centre dataset.** Grouped by scanner
signature (shape + spacing + orientation), the same linear branch goes from 0.3575 to
0.7560: on an unseen scanner the model still ranks patients well (AUROC 0.82) but its
probabilities fall apart. Scanner metadata alone does not predict the label — AUROC 0.53
— so this is covariate shift, not label leakage.

**Snapshot averaging instead of best-epoch selection.** Picking the best validation
epoch was inflating every CNN result by 0.010. The fix is to average the predictions of
the last 15 epochs and never select.

**Public and private leaderboards differ.** 0.2691 public, 0.3008 private. With four
submissions there is no room to overfit a public split, so this is mostly the two
subsets not being equally hard.

**Where the loss lives.** 100 of 1362 cases carry half the total log loss. On the other
1262 the model is already close to perfect, so average-case improvements stopped paying
long before the competition ended.

---

## Repository layout

```
solution/         the 9 files that were executed by the competition container
research/         data preparation, training, diagnostics, evaluation
```

`research/` scripts import the shared modules from `solution/`, so run them from the
repository root with that directory on the path:

```bash
PYTHONPATH=solution python research/controlla_ritaglio.py 20
```

Worth reading first, in order: `controlla_orientamento.py` and `gruppi_scanner.py`
(what the dataset actually looks like), `controlla_ritaglio.py` (how the crop defect was
found and measured), `confronta_stack.py` and `scegli_meta_C.py` (how the last
submission was decided), `train_crop2.py` (snapshot averaging, and why the fold networks
are kept), `fai_submission.py` (the checks that run before an archive is written).

`solution/` is byte-for-byte the submitted code except for comments and a handful of
translated strings. Both were verified: the abstract syntax trees match with string
literals normalised, and running the rewritten code against the submitted weights
reproduces the submitted predictions exactly.

Identifiers and some inline notes are in Italian, which is the language the project was
written in.

## Running it

Neither the competition scans nor the trained weights are in this repository (see
below), but the inference path runs end to end on synthetic volumes. They are not
committed — incompressible random noise would be 41 MB against 150 KB of code — so
generate them first:

```bash
pip install -r requirements.txt
python research/genera_dati_finti.py          # 12 NIfTI volumes + a labels.csv
mkdir -p run/data && cp -r synthetic_data run/data/niftis
CODE_EXEC=$(pwd)/run DAT_DEBUG=1 python solution/main.py
```

`main.py` expects `data/niftis/*.nii.gz` under `CODE_EXEC` and writes `submission.csv`
there. It also needs the `finale_*` weight files next to itself, which are not
published, so this exercises preprocessing and feature extraction rather than the full
prediction path. Predictions on random noise would be meaningless anyway.
`DAT_DEBUG=1` enables diagnostic output; in competition the run is silent.

## Data and weights

Code, configuration and whole-dataset aggregate statistics only.

No scans, no labels and no trained weights are published here. The competition
prohibits redistributing the data, and the weights are derived from it closely enough
that I would rather not test where the line falls. Publishing the code is explicitly
permitted. The organisers have said the dataset will be released as open data, so the
pipeline can be retrained from scratch.

The numbers quoted throughout this README — voxel spacings, fields of view, group
sizes, how much signal a crop retains — are aggregates over the whole dataset or over
scanner groups. Nothing here describes an individual patient, and no voxel value or
label is reproduced. The same holds for every diagnostic script in `research/`: they
were written that way because the scans were never opened in the first place.

## Licence

MIT — see [LICENSE](LICENSE).
