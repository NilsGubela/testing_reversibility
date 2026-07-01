import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
import numpy as np
from pathlib import Path
from util import * 

ROOT = Path(__file__).resolve().parents[2]
fig_dir = ROOT / "figs"

COLORS = {
        r"$\pi(0)$": {"face": "#B5D4F4", "edge": "#185FA5", "text": "#0C447C"},
        r"$\pi(t_1)$": {"face": "#9FE1CB", "edge": "#0F6E56", "text": "#085041"},
        r"$\pi(t_2)$": {"face": "#9FE1CB", "edge": "#0F6E56", "text": "#085041"},
        r"$\pi(t)$": {"face": "#9FE1CB", "edge": "#0F6E56", "text": "#085041"},

    }
LABELS = {r"$\pi(0)$": "ancestral sequence",
          r"$\pi(t_1)$": "evolved sequence 1",
          r"$\pi(t_2)$": "evolved sequence 2",
          r"$\pi(t)$": "evolved sequence"}

BOX_W, BOX_H = 1.6, 0.9
ARROW_KW = dict(arrowstyle="-|>", color="#555555",
                    mutation_scale=14, lw=1.2,
                    connectionstyle="arc3,rad=0.0")
FAINT_KW = dict(arrowstyle="-|>", color="#999999",
                    mutation_scale=11, lw=0.9,
                    connectionstyle="arc3,rad=0.0")

pos = {r"$\pi(0)$": (2.0, 12.5),
       r"$\pi(t_1)$": (1.0, 10.0),
       r"$\pi(t_2)$": (3.0, 10.5),
       r"$\pi(t)$": (2.0, 10.5)}

def left(lbl):  return pos[lbl][0] - BOX_W / 2, pos[lbl][1]
def right(lbl): return pos[lbl][0] + BOX_W / 2, pos[lbl][1]
def top(lbl):   return pos[lbl][0], pos[lbl][1] + BOX_H / 2
def bot(lbl):   return pos[lbl][0], pos[lbl][1] - BOX_H / 2

def draw_box(ax, label, cx, cy):
    x, y = cx - BOX_W / 2, cy - BOX_H / 2
    box = mpatches.FancyBboxPatch(
        (x, y), BOX_W, BOX_H,
        boxstyle="round,pad=0.06",
        facecolor=COLORS[label]["face"],
        edgecolor=COLORS[label]["edge"],
        linewidth=1.0, zorder=3,
    )
    ax.add_patch(box)
    ax.text(cx, cy, label,
            ha="center", va="center", fontsize=8, fontweight="bold",
            color=COLORS[label]["text"], zorder=4)

def overview_cherry(ax):
    ax.set_xlim(0, 4)
    ax.set_ylim(9, 14)
    ax.axis('off')

    pos = {r"$\pi(0)$": (2.0, 12.5),
        r"$\pi(t_1)$": (1.0, 10.0),
        r"$\pi(t_2)$": (3.0, 10.5)}

    for lbl, (cx, cy) in pos.items():
        draw_box(ax, lbl, cx, cy)

    transitions = [
        (bot(r"$\pi(0)$"), top(r"$\pi(t_1)$"), 0,0, 5,"black"),
        (bot(r"$\pi(0)$"), top(r"$\pi(t_2)$"), 0,0, 5,"black"),
    ]

    for src, dst, rad, shrinkA, shrinkB, color in transitions:
        x0, y0 = src
        x1, y1 = dst
        arc = FancyArrowPatch(
            posA=(x0, y0), posB=(x1, y1),
            arrowstyle="-|>",
            connectionstyle="arc3,rad="+str(rad),
            color=color, lw=0.9,
            mutation_scale=11, 
            zorder=2,
            shrinkA=shrinkA,
            shrinkB=shrinkB,   
            )
        ax.add_patch(arc)


    ax.text(0.6, 10.9, r"$P(t_1)$", ha="center", va="bottom", fontsize=6, color="black")
    ax.text(3.3, 11.35, r"$P(t_2)$", ha="center", va="bottom", fontsize=6, color="black")

def overview_linear(ax):
    ax.set_xlim(0, 4)
    ax.set_ylim(9, 14)
    ax.axis('off')

    pos = {r"$\pi(0)$": (1.8, 12.5),
        r"$\pi(t)$": (1.8, 10.5)}

    for lbl, (cx, cy) in pos.items():
        draw_box(ax, lbl, cx, cy)

    transitions = [
        (bot(r"$\pi(0)$"), top(r"$\pi(t)$"), 0,0, 5,"black"),
    ]

    for src, dst, rad, shrinkA, shrinkB, color in transitions:
        x0, y0 = src
        x1, y1 = dst
        arc = FancyArrowPatch(
            posA=(x0, y0), posB=(x1, y1),
            arrowstyle="-|>",
            connectionstyle="arc3,rad="+str(rad),   # negative rad = arc upward
            color=color, lw=0.9,
            mutation_scale=11, #linestyle="dashed",
            zorder=2,
            shrinkA=shrinkA,   # pull back from start
            shrinkB=shrinkB,   # pull back from end
            )
        ax.add_patch(arc)


    ax.text(1.4, 11.5, r"$P(t)$", ha="center", va="bottom", fontsize=6, color="black")



with plt.rc_context(STYLE2):
    n_grid_rows = 4
    n_cols = 5

    fig = plt.figure(figsize=(6.8, 4*1.3))
    width_ratios = [0.7, 1, 1, 1, 1]
    gs = GridSpec(
        n_grid_rows, n_cols, figure=fig,
        width_ratios=width_ratios,
        hspace=0.4, wspace=0.4,
    )

    ax_col0_row0 = fig.add_subplot(gs[0:2, 0])
    ax_col0_row1 = fig.add_subplot(gs[2:4, 0])

    overview_linear(ax_col0_row0)
    overview_cherry(ax_col0_row1)

    plt.savefig(fig_dir / "figure1_sketch.svg")
