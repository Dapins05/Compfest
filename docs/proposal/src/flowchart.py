"""Generator diagram alir untuk proposal VisionQC.

Bentuk mengikuti konvensi flowchart ISO 5807: terminator (kapsul), proses
(persegi), keputusan (belah ketupat), data (jajaran genjang), dan basis
dokumen. Warna sengaja dijaga netral supaya tetap terbaca saat dicetak
hitam-putih.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.textpath import TextPath  # noqa: F401  (memaksa font ter-cache)

FONT = "Times New Roman"
INK = "#1a1a1a"
LINE = "#33415c"

PALETTE = {
    "terminator": ("#e8edf5", "#33415c"),
    "process": ("#ffffff", "#33415c"),
    "ai": ("#eef4ec", "#4a6741"),
    "decision": ("#fdf3e3", "#a9762a"),
    "data": ("#f2f0f7", "#5a4a7a"),
    "user": ("#fdeeee", "#9a3b3b"),
    "accent": ("#dce6f5", "#1f3a68"),
}

plt.rcParams["font.family"] = FONT
plt.rcParams["font.size"] = 10


def _wrap(text: str) -> str:
    return text.replace("|", "\n")


def box(ax, x, y, w, h, text, kind="process", fs=10, bold=False, radius=0.06):
    face, edge = PALETTE[kind]
    if kind == "terminator":
        patch = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=h / 2),
            facecolor=face, edgecolor=edge, linewidth=1.4,
        )
    elif kind == "decision":
        patch = mpatches.Polygon(
            [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
            closed=True, facecolor=face, edgecolor=edge, linewidth=1.4,
        )
    elif kind == "data":
        sk = w * 0.13
        patch = mpatches.Polygon(
            [(x - w / 2 + sk, y + h / 2), (x + w / 2, y + h / 2),
             (x + w / 2 - sk, y - h / 2), (x - w / 2, y - h / 2)],
            closed=True, facecolor=face, edgecolor=edge, linewidth=1.4,
        )
    else:
        patch = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=radius),
            facecolor=face, edgecolor=edge, linewidth=1.4,
        )
    ax.add_patch(patch)
    ax.text(
        x, y, _wrap(text), ha="center", va="center", fontsize=fs,
        color=INK, fontweight="bold" if bold else "normal",
        fontfamily=FONT, linespacing=1.35, zorder=5,
    )
    return (x, y, w, h)


def arrow(ax, p0, p1, label=None, style="-", rad=0.0, fs=8.5, lw=1.3, ls="-"):
    ax.annotate(
        "", xy=p1, xytext=p0,
        arrowprops=dict(
            arrowstyle="-|>", color=LINE, linewidth=lw, linestyle=ls,
            shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}",
            mutation_scale=13,
        ),
        zorder=1,
    )
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(
            mx, my, label, ha="center", va="center", fontsize=fs, color=LINE,
            fontfamily=FONT,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.95),
            zorder=6,
        )


def elbow(ax, p0, p1, label=None, first="v", fs=8.5, ls="-"):
    """Sambungan siku-siku: turun/naik dulu lalu mendatar, atau sebaliknya."""
    mid = (p0[0], p1[1]) if first == "v" else (p1[0], p0[1])
    ax.plot(
        [p0[0], mid[0]], [p0[1], mid[1]], color=LINE, linewidth=1.3,
        linestyle=ls, zorder=1, solid_capstyle="round",
    )
    arrow(ax, mid, p1, ls=ls)
    if label:
        ax.text(
            (p0[0] + mid[0]) / 2 if first == "h" else mid[0],
            (p0[1] + mid[1]) / 2 if first == "v" else mid[1],
            label, ha="center", va="center", fontsize=fs, color=LINE,
            fontfamily=FONT,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.95),
            zorder=6,
        )


def lane(ax, x, y, w, h, title, color="#8896ab"):
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=mpatches.BoxStyle("Round", pad=0, rounding_size=0.12),
            facecolor="#fbfcfe", edgecolor=color, linewidth=1.1,
            linestyle=(0, (5, 3)), zorder=0,
        )
    )
    ax.text(
        x + 0.18, y + h - 0.22, title, ha="left", va="center", fontsize=9.5,
        color=color, fontfamily=FONT, fontweight="bold", zorder=1,
    )


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.12,
                facecolor="white")
    plt.close(fig)
    print("tersimpan:", path)
