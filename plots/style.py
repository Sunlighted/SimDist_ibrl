"""Shared plotting style for the paper figures.

One place for the palette, the matplotlib rcParams and the two mark helpers, so
every figure in the paper reads as one system. Import it, do not re-declare
colours in individual scripts:

    from style import PALETTE as P, setup, rounded_bar, save

Palette design
--------------
Chart colour has one job per figure, and in this paper it is almost always
"which single thing is the finding". So the default is a SINGLE accent on a
neutral ground rather than a categorical rainbow: greying out the bars that are
not significant is itself the message. Categorical slots exist for the few
figures that genuinely encode identity (e.g. two value choices in the
proposal-conditional plot), and are taken in the fixed order below -- never
cycled, never reassigned by rank.

The categorical hues are a colourblind-safe order (blue -> orange -> aqua ->
yellow); adjacent pairs stay separable under deuteranopia/protanopia. If you add
a 5th series, do not invent a hue: facet the figure instead.

Print notes: PDF/PS font type 42 (real embedded fonts, not type-3, which some
venues reject); every figure is authored at its FINAL printed width so the font
sizes are literal points on the page -- never scale a figure in LaTeX.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PALETTE = {
    # --- ink & structure -------------------------------------------------
    "ink":       "#0b0b0b",   # primary text: values, emphasised labels
    "ink2":      "#52514e",   # secondary text: axis labels, tick labels
    "muted":     "#898781",   # tertiary: annotations, units
    "axis":      "#c3c2b7",   # gridlines, spines, connector lines
    "surface":   "#ffffff",   # figure/axes background (white for print)
    # --- emphasis (the default single-accent scheme) ---------------------
    "accent":    "#2a78d6",   # the ONE thing the figure is about
    "neutral":   "#d8d7d1",   # everything that is not significant
    "anchor":    "#6f6d68",   # start/end reference bars (lighter than ink2:
                              # an anchor frames the story, it is not the story)
    # --- categorical slots, in fixed order (identity encoding only) ------
    "cat1":      "#2a78d6",   # blue
    "cat2":      "#eb6834",   # orange
    "cat3":      "#1baf7a",   # aqua
    "cat4":      "#eda100",   # yellow
    # --- semantic --------------------------------------------------------
    "negative":  "#c0392b",   # a swap that HURTS (used with a label, never alone)
    "reference": "#898781",   # oracle / ceiling reference lines
}

# Column widths of a two-column paper, in inches. Author at the final size.
WIDTH_1COL, WIDTH_2COL = 3.35, 7.0


# ICLR's style file does \usepackage{times}, which resolves to URW Nimbus Roman
# -- the same metric clone of Times Roman that is installed here. So figure text
# set in Nimbus Roman is literally the body font of the paper, not an
# approximation. Liberation Serif / FreeSerif are the fallbacks; DejaVu Serif is
# last because it is NOT Times-metric and will look wrong next to the body text.
SERIF = ["Nimbus Roman", "Times New Roman", "Liberation Serif", "FreeSerif",
         "DejaVu Serif"]


def setup(base=8, family="serif"):
    """Apply the shared rcParams. `base` is the body font size in points.

    Default is Times (see SERIF) to match the ICLR body font. Pass
    family="sans-serif" only for figures that are not going into the paper.
    """
    plt.rcParams.update({
        "font.family": family,
        "font.serif": SERIF,
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        # stix is the Times-compatible math set: $\lambda$ in a label then
        # matches the surrounding roman text instead of switching to DejaVu.
        "mathtext.fontset": "stix",
        "font.size": base,
        "axes.labelsize": base,
        "xtick.labelsize": base,
        "ytick.labelsize": base,
        "legend.fontsize": base - 1,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": PALETTE["surface"],
        "figure.facecolor": PALETTE["surface"],
        "axes.facecolor": PALETTE["surface"],
    })


def rounded_bar(ax, x, lo, hi, color, width=0.62, radius=0.012, aspect=0.35, z=3):
    """A bar between two y values with softly rounded ends.

    Rounded data-ends read as a deliberate mark rather than a raw rectangle; the
    radius is in DATA units on y, so pass something small relative to the range.
    """
    ax.add_patch(FancyBboxPatch(
        (x - width / 2, lo), width, hi - lo,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, facecolor=color, mutation_aspect=aspect, zorder=z))


def clean_axes(ax, grid_axis="y"):
    """Recessive grid, no top/right spines, muted ticks."""
    P = PALETTE
    getattr(ax, f"{grid_axis}axis").grid(True, color=P["axis"], lw=0.5, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(P["axis"])
    ax.tick_params(colors=P["muted"])
    for lb in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        lb.set_color(P["ink2"])


def save(fig, stem, outdir="figs"):
    """Write both a vector PDF (for LaTeX) and a PNG (for quick viewing)."""
    import os
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{outdir}/{stem}.pdf", bbox_inches="tight",
                facecolor=PALETTE["surface"])
    fig.savefig(f"{outdir}/{stem}.png", bbox_inches="tight",
                facecolor=PALETTE["surface"], dpi=300)
    print(f"wrote {outdir}/{stem}.{{pdf,png}}")


# ---------------------------------------------------------------------------
# Family palette (navy-anchored)
# ---------------------------------------------------------------------------
# For figures that encode a METHOD FAMILY plus its variants: one hue per family,
# three lightness tiers per hue (deep = the strongest / final variant, light =
# the ablation). Use this instead of PALETTE["cat*"] whenever a figure has more
# than one variant inside a family.
#
# Built around the paper's main-method navy #1D3765 (L*23, h=283). Design notes,
# all measured in CIELAB, all verified under deuteranope/protanope simulation:
#
#   * blue/orange/green are the three method families and are mutually safe:
#     dE76 = 82 / 61 / 53 normal, 90 / 55 / 36 under deuteranopia.
#   * orange's deep tier is capped at L*52 -- below that, h=68 reads as brown.
#     It is the one family whose deep tier cannot match navy's L*23.
#   * purple sits at h=320, NOT the more natural h=308: navy's own hue is 283,
#     so a violet purple collides with it (dE 16 vs 31). The cost is that purple
#     and rose are then only dE 21 apart -- do not put them in the same panel.
#   * DO NOT co-locate: rose+red (dE 35 but only 11 deg of hue apart in the
#     originals), green+red or green+rose (dE 8 / 9 under deuteranopia).
#   * The light tier has contrast 1.6-2.2 against white. It is fine for CI
#     bands and fills; as a standalone thin line it is at the legibility floor,
#     so give it >= 1.6pt.
FAMILIES = {
    "blue":   {"deep": "#1D3765", "mid": "#487DB2", "light": "#AABCDB"},
    "orange": {"deep": "#AC6E2A", "mid": "#E19D49", "light": "#EFC99B"},
    "green":  {"deep": "#376439", "mid": "#669877", "light": "#A4CBB7"},
    "purple": {"deep": "#885D94", "mid": "#A582AE", "light": "#C1A8C7"},
    "rose":   {"deep": "#742A60", "mid": "#A6608D", "light": "#DDA1BE"},
    "red":    {"deep": "#AA3A49", "mid": "#BE6C6D", "light": "#D0908F"},
}

# Neutral reference lines (oracle ceiling, source zero-shot). Never spend a
# family hue on a reference -- a reference is not a method.
REF = {"oracle": "#6B7280", "zeroshot": "#9CA3AF"}

# The main sim2sim figure's family assignment, so every panel agrees.
METHOD_FAMILY = {"planning": "blue", "dyna": "orange", "model_free": "green"}
