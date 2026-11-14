"""Draw the figures for the technical document and the pipeline notes.

    python docs/figures/make_figures.py

matplotlib only, deterministic, no match data needed: every number below is either read from a
committed report or quoted from `docs/pipeline.md` with its provenance in a comment. The output is
SVG so the figures stay sharp on GitHub and inside the PDF that `docs/build_pdf.py` produces.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REPORT = os.path.join(ROOT, "evaluation", "reports", "nexus", "nexus_final.json")

INK = "#1f2933"
MUTED = "#5f6b7a"
GRID = "#d6dde5"
PAPER = "#ffffff"
FILL = "#f1f5f9"
ACCENT = "#0b6e5f"
ACCENT_SOFT = "#b9d9d1"
WARN = "#b45309"


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "svg.fonttype": "path",
        "svg.hashsalt": "nexus",  # stable element ids, so regenerating gives a clean diff
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
    })


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.12,
                metadata={"Date": None})
    plt.close(fig)
    print("wrote", os.path.relpath(path, ROOT))


def clean_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# ------------------------------------------------------------------------------------------------
# Figure 1: the pipeline
# ------------------------------------------------------------------------------------------------
def pipeline():
    fig, ax = plt.subplots(figsize=(7.4, 6.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 100)
    ax.axis("off")

    def box(x, y, w, h, title, lines, highlight=False, fs=8.2):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.4,rounding_size=1.4",
            linewidth=1.15, edgecolor=ACCENT if highlight else INK,
            facecolor=ACCENT_SOFT if highlight else FILL, zorder=2,
        ))
        ax.text(x + w / 2, y + h - 1.9, title, ha="center", va="top",
                fontsize=10.0, weight="bold", zorder=3)
        for i, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 5.6 - 2.6 * i, line, ha="center", va="top",
                    fontsize=fs, color=MUTED, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, linewidth=1.0, color=INK,
                                     shrinkA=0, shrinkB=0, zorder=1))

    def note(y, text):
        ax.plot([48.6, 52.4], [y, y], color=GRID, linewidth=0.9, zorder=0)
        ax.text(53.2, y, text, ha="left", va="center", fontsize=7.4, color=MUTED)

    box(4, 86, 44, 12, "Gated match bundle",
        ["tactical ATD · Smart Tagging", "homography · 199,319 rows"])
    box(4, 68, 44, 12, "features.npz · 199 channels",
        ["one row per video frame (8 s)", "nothing is labelled"])
    box(3, 46, 22, 12, "dead head",
        ["ball gap · out of play", "still ball at a restart"], fs=7.4)
    box(27, 46, 22, 12, "possession head",
        ["3 nearest per team", "formation fallback"], fs=7.4)
    box(4, 28, 44, 11, "Viterbi over {H, A, D}",
        ["switch H↔A 6.0 · switch alive↔D 2.5", "the ratio is the whole idea"])
    box(4, 13, 44, 11, "structural cleanup",
        ["possession ≥ 3 s · stoppage ≥ 1 s", "never merges across a stoppage"])
    box(4, 1, 44, 9, "team-nexus.csv",
        ["199,319 rows · 1.7 MB · 2 s"], highlight=True)

    # down the chain
    arrow(26, 86, 26, 80.5)
    # split into the two heads
    ax.plot([26, 26], [68, 63], color=INK, linewidth=1.0, zorder=1)
    ax.plot([14, 38], [63, 63], color=INK, linewidth=1.0, zorder=1)
    arrow(14, 63, 14, 58.4)
    arrow(38, 63, 38, 58.4)
    ax.text(26, 60.8, "two sub-problems, one decision each", ha="center", va="center",
            fontsize=7.8, color=MUTED, style="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=PAPER, edgecolor="none"))
    # merge into the decoder
    ax.plot([14, 14], [46, 44], color=INK, linewidth=1.0, zorder=1)
    ax.plot([38, 38], [46, 44], color=INK, linewidth=1.0, zorder=1)
    ax.plot([14, 38], [44, 44], color=INK, linewidth=1.0, zorder=1)
    arrow(26, 44, 26, 39.4)
    ax.text(26, 45.0, "log-odds", ha="center", va="center",
            fontsize=7.8, color=MUTED, style="italic",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=PAPER, edgecolor="none"))
    # decoder to output
    arrow(26, 28, 26, 24.5)
    arrow(26, 13, 26, 10.5)

    note(92, "all automatic, all uncorrected\nthe ball is missing on 41% of graded frames")
    note(74, "no video decoded\nno per-player identity: 46 anonymous slots")
    note(52, "the ball is a good dead-ball sensor\nand a poor possession sensor")
    note(33.5, "changing team is the possession event;\nstoppages are frequent and brief")
    note(18.5, "H H D D H H is two possessions,\nnot one — the metric counts spells")
    note(5.5, "same bundle in, same md5 out")

    save(fig, "pipeline.svg")


# ------------------------------------------------------------------------------------------------
# Figure 2: the three states, and what counts as one possession
# ------------------------------------------------------------------------------------------------
def decode():
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8.2, 3.6), gridspec_kw={"width_ratios": [1.0, 1.4]}
    )

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    nodes = {"H": (0.23, 0.74), "A": (0.77, 0.74), "D": (0.5, 0.26)}
    for label, (x, y) in nodes.items():
        ax1.add_patch(Circle((x, y), 0.088, facecolor=FILL, edgecolor=INK, linewidth=1.2, zorder=2))
        ax1.text(x, y, label, ha="center", va="center", fontsize=13, weight="bold", zorder=3)

    def edge(a, b, cost, tx=0.0, ty=0.0, emphasis=False):
        (x1, y1), (x2, y2) = nodes[a], nodes[b]
        ax1.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="<|-|>", mutation_scale=17,
            linewidth=1.7 if emphasis else 1.1,
            color=INK if emphasis else MUTED,
            shrinkA=27, shrinkB=27, zorder=1,
        ))
        ax1.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, cost, ha="center", va="center",
                 fontsize=10 if emphasis else 9, color=INK if emphasis else MUTED,
                 weight="bold" if emphasis else "normal",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor=PAPER, edgecolor="none"))

    edge("H", "A", "6.0", ty=0.075, emphasis=True)
    edge("H", "D", "2.5", tx=-0.075, ty=0.0)
    edge("A", "D", "2.5", tx=0.075, ty=0.0)
    ax1.set_title("switch costs, in log-odds", fontsize=10, pad=8)
    ax1.text(0.5, 0.03, "changing team is expensive; a stoppage is cheap",
             ha="center", va="bottom", fontsize=8, color=MUTED, style="italic")

    ax2.set_xlim(0, 10)
    ax2.set_ylim(0.2, 4.6)
    ax2.axis("off")

    def bars(y, labels, width, colours):
        x = 0.15
        for lab, col in zip(labels, colours):
            ax2.add_patch(Rectangle((x, y), width, 0.62, facecolor=col, edgecolor=PAPER,
                                    linewidth=1.8, zorder=2))
            ax2.text(x + width / 2, y + 0.31, lab, ha="center", va="center", fontsize=11,
                     weight="bold", color=PAPER if col in (ACCENT, WARN) else INK, zorder=3)
            x += width

    def bracket(x0, x1, y, text):
        ax2.plot([x0, x0, x1, x1], [y - 0.14, y - 0.24, y - 0.24, y - 0.14],
                 color=MUTED, linewidth=0.9)
        ax2.text((x0 + x1) / 2, y - 0.46, text, ha="center", va="top",
                 fontsize=8.2, color=MUTED)

    w = 0.75
    y1 = 3.15
    bars(y1, ["H", "H", "H", "D", "D", "H", "H", "H"], w,
         [ACCENT, ACCENT, ACCENT, MUTED, MUTED, ACCENT, ACCENT, ACCENT])
    bracket(0.15, 0.15 + 3 * w, y1, "possession 1")
    bracket(0.15 + 3 * w, 0.15 + 5 * w, y1, "stoppage")
    bracket(0.15 + 5 * w, 0.15 + 8 * w, y1, "possession 2")
    ax2.text(0.15, y1 + 0.98, "the same team on both sides of a stoppage", fontsize=9, color=INK)
    ax2.text(0.15, y1 + 0.72, "two possessions, and the metric counts two", fontsize=8.2,
             color=MUTED, style="italic")

    w2 = 1.5
    y2 = 1.15
    bars(y2, ["H", "H", "A", "A"], w2, [ACCENT, ACCENT, WARN, WARN])
    bracket(0.15, 0.15 + 2 * w2, y2, "possession 1")
    bracket(0.15 + 2 * w2, 0.15 + 4 * w2, y2, "possession 2")
    ax2.text(0.15, y2 + 0.98, "possession changes hands with no stoppage", fontsize=9, color=INK)
    ax2.text(0.15, y2 + 0.72, "still two possessions, and still a 6.0 switch",
             fontsize=8.2, color=MUTED, style="italic")

    plt.tight_layout()
    save(fig, "decode.svg")


# ------------------------------------------------------------------------------------------------
# Figure 3: what each step was worth on the public window
# ------------------------------------------------------------------------------------------------
def results():
    # provenance: "submitted" is read back from evaluation/reports/nexus/nexus_final.json; the
    # baselines and the per-step values are the archived public-window runs tabulated in
    # docs/pipeline.md.
    with open(REPORT) as fh:
        report = json.load(fh)
    rows = [
        ("published\nfloor", 0.6207, 0.3256),
        ("best baseline\non this window", 0.5812, 0.4242),
        ("Step 2\ndecoder", 0.6439, 0.4343),
        ("Step 3\nownership", 0.6511, 0.4471),
        ("Step 4\nstill ball", 0.7335, 0.4571),
        ("Step 5\nrecalibration", 0.7666, 0.4800),
        ("submitted\nStep 7", report["macro_f1"], report["spell_f1_mean"]),
    ]
    names = [r[0] for r in rows]
    macro = np.array([r[1] for r in rows])
    spell = np.array([r[2] for r in rows])

    fig, ax = plt.subplots(figsize=(8.0, 3.9))
    x = np.arange(len(rows))
    width = 0.38
    bars1 = ax.bar(x - width / 2, macro, width, label="macroF1 · per frame (tie-break)",
                   color=ACCENT_SOFT, edgecolor=ACCENT, linewidth=0.9, zorder=3)
    bars2 = ax.bar(x + width / 2, spell, width, label="spellF1 · per possession (the ranking)",
                   color=ACCENT, edgecolor=ACCENT, linewidth=0.9, zorder=3)
    for bars, vals in ((bars1, macro), (bars2, spell)):
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.4f}",
                    ha="center", va="bottom", fontsize=7.4, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8.4)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("score on the 5 public minutes")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    clean_axes(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=8.6)
    ax.set_title("Each step measured on the 17-possession window: a sanity check, not a ranking",
                 fontsize=9.8, loc="left", pad=10)
    plt.tight_layout()
    save(fig, "results.svg")


# ------------------------------------------------------------------------------------------------
# Figure 4: where the per-frame errors are, on the public window
# ------------------------------------------------------------------------------------------------
def confusion():
    with open(REPORT) as fh:
        report = json.load(fh)
    cm = np.array(report["confusion"], dtype=float)[:3, :3]  # drop the X row and column
    labels = ["H", "A", "D"]
    share = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "nexus", ["#ffffff", ACCENT_SOFT, ACCENT]
    )
    ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{int(cm[i, j]):,}\n{share[i, j]*100:.0f}%", ha="center", va="center",
                    fontsize=9.2, color=PAPER if share[i, j] > 0.45 else INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"pred {l}" for l in labels])
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"true {l}" for l in labels])
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which="minor", color=PAPER, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title(f"5 public minutes · {report['n_graded']:,} graded frames\n"
                 f"macroF1 {report['macro_f1']:.4f} · κ {report['kappa']:.3f}",
                 fontsize=9.4, loc="left", pad=8)
    plt.tight_layout()
    save(fig, "confusion.svg")


# ------------------------------------------------------------------------------------------------
# Figure 5: label-free structure, one row per decision
# ------------------------------------------------------------------------------------------------
def ablation():
    # provenance: `python -m pipeline.cli.ablate`, whole match, no labels involved; the same table
    # is written out in docs/pipeline.md.
    rows = [
        ("argmax, no decoder", 774, 4.30),
        ("no multi-scale context", 734, 4.56),
        ("unnormalised ownership", 597, 5.46),
        ("raw (unsigned) tag prior", 528, 6.24),
        ("switch cost 2.0", 514, 6.50),
        ("no tag prior", 495, 6.63),
        ("switch cost 4.0", 477, 6.99),
        ("no confidence / gate terms", 452, 7.38),
        ("minimum possession 1 s", 446, 7.49),
        ("switch cost 10.0", 442, 7.54),
        ("minimum possession 3 s", 438, 7.61),
        ("submitted configuration", 438, 7.61),
        ("still-ball weight 3.0", 424, 8.05),
        ("restart prior 2.5 (rejected)", 411, 7.21),
        ("no still-ball duration", 394, 9.14),
    ]
    names = [r[0] for r in rows]
    poss = np.array([r[1] for r in rows])
    mean = np.array([r[2] for r in rows])
    y = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    colours = [ACCENT if n == "submitted configuration" else ACCENT_SOFT for n in names]
    edges = [ACCENT if n == "submitted configuration" else "#7fa8a0" for n in names]
    ax.barh(y, poss, color=colours, edgecolor=edges, linewidth=0.9, height=0.68, zorder=3)

    ax.axvspan(400, 440, color=WARN, alpha=0.10, zorder=0)
    ax.axvline(400, color=WARN, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.axvline(440, color=WARN, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)

    for yi, p, m in zip(y, poss, mean):
        ax.text(p + 6, yi, f"{p}   ({m:.2f} s mean)", va="center", fontsize=7.8, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.4)
    ax.set_xlim(0, 960)
    ax.set_ylim(-1.8, len(rows) - 0.3)
    ax.set_xlabel("possessions over the whole match (no labels used)")
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    clean_axes(ax)
    ax.text(420, -1.25, "target band from the football prior (400–440)",
            ha="center", va="center", fontsize=8.2, color=WARN)
    ax.set_title("Every decoder decision, measured on the 154 minutes that have no ground truth",
                 fontsize=9.8, loc="left", pad=10)
    plt.tight_layout()
    save(fig, "ablation.svg")


# ------------------------------------------------------------------------------------------------
# Figure 6: the honest one — dead share grows as the ball disappears
# ------------------------------------------------------------------------------------------------
def deadshare():
    regions = ["first 5 minutes\n(the public window)", "rest of first half", "second half"]
    dead = [0.176, 0.478, 0.507]
    ball = [0.882, 0.563, 0.593]

    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    x = np.arange(3)
    bars = ax.bar(x, dead, 0.5, color=[ACCENT, WARN, WARN], edgecolor=PAPER, linewidth=1.2,
                  zorder=3, label="share of frames we label dead")
    for bar, d, b in zip(bars, dead, ball):
        ax.text(bar.get_x() + bar.get_width() / 2, d - 0.03, f"{d*100:.1f}% dead",
                ha="center", va="top", fontsize=9.5, color=PAPER, weight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 0.03, f"ball visible\n{b*100:.0f}% of frames",
                ha="center", va="bottom", fontsize=8.2, color=PAPER)

    truth = ax.axhline(0.196, color=INK, linewidth=1.0, linestyle=(0, (5, 3)), zorder=4,
                       label="truth in the public window: 19.6% dead")
    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=8.8)
    ax.set_ylim(0, 0.66)
    ax.set_ylabel("share of frames we label dead")
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    clean_axes(ax)
    handles = [Line2D([], [], marker="s", linestyle="none", markersize=9, color=ACCENT,
                      label="share of frames we label dead"),
               Line2D([], [], color=INK, linestyle=(0, (5, 3)), label="truth in the public window: 19.6% dead")]
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8.4)
    ax.set_title("The largest open risk: our dead-ball evidence leans on ball absence,\n"
                 "and the ball is visible on 88% of the frames we are allowed to look at",
                 fontsize=9.4, loc="left", pad=10)
    plt.tight_layout()
    save(fig, "deadshare.svg")


def main():
    style()
    os.makedirs(HERE, exist_ok=True)
    pipeline()
    decode()
    results()
    confusion()
    ablation()
    deadshare()


if __name__ == "__main__":
    main()
