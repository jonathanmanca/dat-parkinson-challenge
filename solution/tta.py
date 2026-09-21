"""Test-time augmentation: average a patient's prediction over the 8 mirrored views.

Only mirrors are used. They are exact — no interpolation, no invented voxels — and the
network was trained with random flips, so it is already insensitive to them by
construction. Rotations would blur the volume and can remove more than they add.

Averaging happens in probability space rather than in logits: averaging logits gives
extra weight to the most extreme views, which is the opposite of what log loss wants.

Caveat worth recording: measured on a single network this was worth 0.016, but the
submitted branch already averages three networks, and seed averaging removes much of
the same noise. On the leaderboard it was worth 0.0003. Measure an improvement in the
configuration where it will actually be used.
"""
import itertools
import numpy as np
import torch

# A batch of volumes has shape (N, 1, D, H, W), so the spatial axes are 2, 3 and 4.
# Every subset of them: () is the original, (2,) one mirror, (2,3,4) all three -> 8 views.
COMBINAZIONI = [c for r in range(4) for c in itertools.combinations((2, 3, 4), r)]


def predici_tta(rete, x, lotto=64, combinazioni=None):
    """Average the predictions of `rete` over the mirrored views of `x`.

    `rete` must already be in eval mode; `x` is a (N, 1, D, H, W) tensor. Returns an
    array of N probabilities.
    """
    combo = COMBINAZIONI if combinazioni is None else combinazioni
    out = []
    with torch.no_grad():
        for c in combo:
            xv = torch.flip(x, dims=c) if c else x
            p = [torch.sigmoid(rete(xv[i:i+lotto])).squeeze(1).numpy()
                 for i in range(0, len(xv), lotto)]
            out.append(np.concatenate(p))
    return np.mean(out, axis=0)


if __name__ == "__main__":
    from cnn import CNNCrop2
    print(f"views: {len(COMBINAZIONI)} -> {COMBINAZIONI}")
    torch.manual_seed(0)
    rete = CNNCrop2(doppio_pool=True, residuo=True, canali=(24, 48, 96, 96)).eval()
    x = torch.rand(5, 1, 48, 48, 48)
    semplice = torch.sigmoid(rete(x)).squeeze(1).detach().numpy()
    media = predici_tta(rete, x)
    print(f"single view : {np.round(semplice, 4)}")
    print(f"with TTA    : {np.round(media, 4)}")
    print(f"largest shift: {np.abs(semplice - media).max():.4f}  (zero would mean TTA has nothing to average)")
