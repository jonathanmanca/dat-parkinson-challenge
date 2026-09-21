"""Configurable 3D CNN for the striatum crops.

Each architectural option is a separate flag so that they can be tested one at a time.
The submitted configuration is `canali=(24, 48, 96, 96), doppio_pool=True, residuo=True`
(about 988k parameters). Capacity was tuned and has a clear optimum on 1362 cases:
439k -> 0.2854, 988k -> 0.2528, 1.76M -> 0.2587, the largest model starting to memorise.
"""
import torch
import torch.nn as nn


def norma(tipo, canali):
    if tipo == "group":
        g = 8 if canali % 8 == 0 else (4 if canali % 4 == 0 else 1)   # groups must divide channels
        return nn.GroupNorm(g, canali)
    return nn.BatchNorm3d(canali)


class SE(nn.Module):
    """Squeeze-and-Excitation: learn a per-channel gain from the channel averages."""

    def __init__(self, canali, riduzione=8):
        super().__init__()
        nascosti = max(1, canali // riduzione)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(canali, nascosti), nn.ReLU(),
                                nn.Linear(nascosti, canali), nn.Sigmoid())

    def forward(self, x):
        p = self.gap(x).flatten(1)
        w = self.fc(p).view(x.size(0), -1, 1, 1, 1)
        return x * w


class Blocco(nn.Module):
    def __init__(self, ci, co, tipo_norma="batch", usa_se=False, residuo=False, pool=True):
        super().__init__()
        self.residuo = residuo
        self.conv1 = nn.Conv3d(ci, co, 3, padding=1, bias=False)
        self.n1 = norma(tipo_norma, co)
        self.conv2 = nn.Conv3d(co, co, 3, padding=1, bias=False) if residuo else None
        self.n2 = norma(tipo_norma, co) if residuo else None
        # the skip connection needs matching channel counts, hence the 1x1 projection
        self.scorciatoia = (nn.Sequential(nn.Conv3d(ci, co, 1, bias=False), norma(tipo_norma, co))
                            if residuo and ci != co else None)
        self.se = SE(co) if usa_se else None
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(2) if pool else None

    def forward(self, x):
        y = self.relu(self.n1(self.conv1(x)))
        if self.residuo:
            y = self.n2(self.conv2(y))
            s = self.scorciatoia(x) if self.scorciatoia is not None else x
            y = self.relu(y + s)
        if self.se is not None:
            y = self.se(y)
        return self.pool(y) if self.pool is not None else y


class CNNCrop2(nn.Module):
    """Four convolutional blocks, then global pooling and a linear head.

    `doppio_pool` concatenates global average and global max pooling. It costs 64
    parameters and was the single best change tested: the average says how bright the
    region is overall, the maximum says whether a peak survives anywhere. In an
    asymmetric DaT scan the average dilutes the affected side and the maximum does not.
    """

    def __init__(self, canali=(16, 32, 64, 64), tipo_norma="batch", usa_se=False,
                 residuo=False, doppio_pool=False, dropout=0.4):
        super().__init__()
        blocchi, ci = [], 1
        for i, co in enumerate(canali):                    # the last block keeps the resolution
            blocchi.append(Blocco(ci, co, tipo_norma, usa_se, residuo, pool=(i < len(canali) - 1)))
            ci = co
        self.features = nn.Sequential(*blocchi)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.gmp = nn.AdaptiveMaxPool3d(1) if doppio_pool else None
        d = ci * (2 if doppio_pool else 1)
        self.testa = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, x):
        f = self.features(x)
        p = self.gap(f).flatten(1)
        if self.gmp is not None:
            p = torch.cat([p, self.gmp(f).flatten(1)], dim=1)
        return self.testa(p)


if __name__ == "__main__":
    for cfg in [dict(), dict(tipo_norma="group"), dict(usa_se=True), dict(residuo=True),
                dict(doppio_pool=True), dict(tipo_norma="group", usa_se=True, residuo=True,
                                             doppio_pool=True, canali=(24, 48, 96, 96))]:
        r = CNNCrop2(**cfg)
        n = sum(p.numel() for p in r.parameters())
        for lato in (48, 64):
            out = r(torch.rand(2, 1, lato, lato, lato))
            assert out.shape == (2, 1), f"wrong shape: {out.shape}"
        print(f"{str(cfg)[:70]:72s} parameters={n:,}")
