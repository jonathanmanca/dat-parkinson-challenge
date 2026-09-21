"""Generate the README figure: what cross-validation promised versus what the
leaderboard delivered, for the three changes measured on both.

Light and dark variants come from this one source so they cannot drift apart.
Colours are slots 1 and 2 of the reference categorical palette; both clear 3:1
against their own surface (4.30 and 3.12 light, 4.79 and 4.48 dark).
"""
import os

DATI = [
    ("Larger CNN + stronger augmentation", 14.0, 1.8),
    ("Test-time augmentation",              7.5, 0.3),
    ("Head-centred crop *",                 7.0, 16.5),
]
MASSIMO, TACCHE = 18.0, [0, 5, 10, 15]

TEMA = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", second="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834"),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", second="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926"),
}

W, H = 800, 366
X0, X1 = 290, 770                     # plot area; the label column left of X0 must fit the
                                      # longest label, which is ~221px at 13px
Y0 = 118                              # first group
ALTEZZA, INTERNO, PASSO = 18, 2, 74   # bar height, gap within a pair, group pitch
SCALA = (X1 - X0) / MASSIMO
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def barra(x, y, larghezza, h, colore):
    """Rounded only at the value end, square against the baseline."""
    r = min(4, max(0.1, larghezza / 2), h / 2)
    x1 = x + larghezza
    return (f'<path d="M{x},{y} H{x1-r} A{r},{r} 0 0 1 {x1},{y+r} '
            f'V{y+h-r} A{r},{r} 0 0 1 {x1-r},{y+h} H{x} Z" fill="{colore}"/>')


def figura(modo):
    t = TEMA[modo]
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family=\'{FONT}\' role="img" aria-label="Log loss improvement predicted by '
         f'cross-validation versus delivered on the leaderboard, for three changes.">',
         f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>',
         f'<text x="28" y="38" font-size="17" font-weight="600" fill="{t["ink"]}">'
         f'What cross-validation promised, and what arrived</text>',
         f'<text x="28" y="60" font-size="13" fill="{t["second"]}">'
         f'Log loss improvement per change, in thousandths. Longer is better.</text>']

    for i, (etichetta, colore) in enumerate([("predicted by cross-validation", t["s1"]),
                                             ("delivered on the leaderboard", t["s2"])]):
        x = 28 + i * 236
        o.append(f'<rect x="{x}" y="80" width="10" height="10" rx="2" fill="{colore}"/>')
        o.append(f'<text x="{x+16}" y="89" font-size="12" fill="{t["second"]}">{etichetta}</text>')

    for v in TACCHE:                                          # recessive grid, behind the bars
        x = X0 + v * SCALA
        o.append(f'<line x1="{x}" y1="{Y0-10}" x2="{x}" y2="{Y0 + PASSO*len(DATI) - 24}" '
                 f'stroke="{t["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{x}" y="{Y0 + PASSO*len(DATI) - 8}" font-size="11" fill="{t["muted"]}" '
                 f'text-anchor="middle">{v}</text>')
    o.append(f'<line x1="{X0}" y1="{Y0-10}" x2="{X0}" y2="{Y0 + PASSO*len(DATI) - 24}" '
             f'stroke="{t["axis"]}" stroke-width="1"/>')

    for g, (nome, atteso, reso) in enumerate(DATI):
        y = Y0 + g * PASSO
        o.append(f'<text x="{X0-16}" y="{y+23}" font-size="13" fill="{t["ink"]}" '
                 f'text-anchor="end">{nome}</text>')
        for k, (valore, colore) in enumerate([(atteso, t["s1"]), (reso, t["s2"])]):
            yb = y + k * (ALTEZZA + INTERNO)
            larghezza = valore * SCALA
            o.append(barra(X0, yb, larghezza, ALTEZZA, colore))
            o.append(f'<text x="{X0 + larghezza + 9}" y="{yb + 13}" font-size="12" '
                     f'font-weight="600" fill="{t["second"]}">{valore:.1f}</text>')

    o.append(f'<text x="28" y="{H-14}" font-size="11" fill="{t["muted"]}">'
             f'* bundled with three other changes, so its gain cannot be attributed to the crop fix alone.'
             f'</text>')
    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    os.makedirs("docs", exist_ok=True)
    for modo in TEMA:
        percorso = os.path.join("docs", f"transfer-{modo}.svg")
        open(percorso, "w", encoding="utf-8", newline="\n").write(figura(modo))
        print(f"wrote {percorso}")
