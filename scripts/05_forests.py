"""Act 1: two maps of British conifer forests as two "catches".

Counts come from the author's conifer-forest project (2026-10-18): named places of
100 ha or more that pass an NFI 2023 conifer rule, named by OpenStreetMap (OSM),
Ordnance Survey Open Names (OS), or both. Only these aggregate counts are published.

Data sources: National Forest Inventory 2023 (Forestry Commission, OGL v3),
OS Open Names (Ordnance Survey, OGL v3), OpenStreetMap contributors (ODbL).

Usage:  python scripts/05_forests.py      -> prints the estimate, writes figures/fig1_two_maps.png
"""
from pathlib import Path

import matplotlib.pyplot as plt

from crc import chapman

ROOT = Path(__file__).resolve().parent.parent

OSM_ONLY, BOTH, OS_ONLY = 325, 727, 463
SIZE_BANDS = {  # band: (OSM only, both, OS only)
    "100-300 ha": (169, 443, 284),
    "300-1,000 ha": (90, 189, 119),
    "1,000+ ha": (66, 95, 60),
}


def main():
    n1, n2, m = OSM_ONLY + BOTH, OS_ONLY + BOTH, BOTH
    e = chapman(n1, n2, m)
    print(f"OSM {n1}, OS {n2}, both {m}")
    print(f"found {e.found}, estimated {e.N:.0f} ({e.lo:.0f}-{e.hi:.0f}), missed by both ~{e.missed:.0f} "
          f"({e.missed / e.N:.0%})")

    total = 0.0
    for band, (a, b, c) in SIZE_BANDS.items():
        eb = chapman(a + b, c + b, b)
        total += eb.N
        print(f"  {band:<13} in both {b / (a + b + c):.0%}  estimated {eb.N:.0f}  missed {eb.missed / eb.N:.0%}")
    print(f"  stratified total {total:.0f}")

    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=150)
    parts = [(OSM_ONLY, "OSM only", "#3a6d8c"), (BOTH, "found by both", "#1f5a42"),
             (OS_ONLY, "OS only", "#8a9f2e"), (e.missed, "missed by both\n(estimated)", "#d5ddd6")]
    left = 0
    for v, label, col in parts:
        ax.barh(0, v, left=left, color=col, edgecolor="#5d6a62" if "missed" in label else col,
                hatch="//" if "missed" in label else None, height=0.55)
        ax.text(left + v / 2, 0, f"{label}\n{v:,.0f}", ha="center", va="center", fontsize=8.5,
                color="black" if "missed" in label else "white")
        left += v
    ax.set_xlim(0, left)
    ax.set_yticks([])
    ax.set_xlabel("named conifer places of 100 ha or more")
    ax.set_title(f"Two maps found {e.found:,}. The overlap says about {e.N:,.0f} exist.", loc="left", fontsize=11)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out = ROOT / "figures" / "fig1_two_maps.png"
    fig.savefig(out)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
