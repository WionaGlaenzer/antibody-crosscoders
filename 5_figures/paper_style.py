"""Original paper figure style (from make_category_figures.py) — reused so rebuild95
histogram/line figures match the paper exactly: step-histograms lw=1.8, framed box
legend (LEG), dashed y-grid (hgrid), pdf.fonttype=42, left-aligned titles."""

import matplotlib as mpl
import matplotlib.pyplot as plt

LEG = dict(
    frameon=True, fancybox=False, edgecolor="#666", framealpha=0.95, fontsize=7, borderpad=0.6
)


def paper_style():
    tab = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    mpl.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.titlelocation": "left",
            "legend.fontsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    return tab


def hgrid(ax):
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#cccccc", lw=0.6, ls=(0, (4, 3)), zorder=0)
