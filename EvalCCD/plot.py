# ==============================
# Plotting helpers
# ===============================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from astropy.stats import sigma_clipped_stats
from scipy.stats import gaussian_kde

def plot_syle_init():
    plt.rcParams["figure.figsize"] = [7, 5]
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["font.size"] = 14
    plt.rcParams["text.usetex"] = False
    plt.rcParams["axes.linewidth"] = 1.5
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.major.size"] = 7
    plt.rcParams["ytick.major.size"] = 7
    plt.rcParams["xtick.minor.size"] = 3.5
    plt.rcParams["ytick.minor.size"] = 3.5
    plt.rcParams["xtick.major.width"] = 1
    plt.rcParams["ytick.major.width"] = 1
    plt.rcParams["xtick.minor.width"] = 1
    plt.rcParams["ytick.minor.width"] = 1
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.top"] = True
    plt.rcParams["legend.frameon"] = False

plot_style_init = plot_syle_init
def show_frame(data: np.ndarray, return_mean: bool = False):
    """
    Display a frame with sigma-clipped column/row profiles.
    """
    Nrow, Ncol = data.shape
    xs, ys = np.arange(Ncol), np.arange(Nrow)

    column_val, _, _ = sigma_clipped_stats(data, axis=0, sigma=3)
    row_val, _, _ = sigma_clipped_stats(data, axis=1, sigma=3)

    gs = GridSpec(
        nrows=2,
        ncols=2,
        hspace=0,
        wspace=0,
        height_ratios=(1, 3),
        width_ratios=(2.5, 1),
    )

    fig = plt.figure(figsize=(5.4, 5))

    ax0 = plt.subplot(gs[1, 0])
    im = ax0.imshow(
        data,
        cmap="gray",
        origin="lower",
        extent=(0, Ncol, 0, Nrow),
        aspect="auto",
    )
    ax0.set_xlabel("x [px]")
    ax0.set_ylabel("y [px]")

    ax1 = plt.subplot(gs[0, 0])
    ax1.step(xs, column_val)
    ax1.set_xlim([0, Ncol])
    ax1.set_xticklabels("")
    ax1.set_ylabel("Avg Signal\n[ADU]", fontsize=9)

    ax2 = plt.subplot(gs[1, 1])
    ax2.step(row_val, ys)
    ax2.set_ylim([0, Nrow])
    ax2.tick_params(axis="x", rotation=45)
    ax2.set_yticklabels("")
    ax2.set_xlabel("Avg Signal\n[ADU]", fontsize=9)

    fig.colorbar(im, ax=ax2, location="right", pad=0.08, label="Counts [ADU]")

    if return_mean:
        return row_val, column_val


def draw_rdn_plot(x, y, label, fm=0.94, fM=1.003, std_bound=(0, 0)):
    """
    x: mean difference
    y: standard deviation
    """
    mean_max = round(float(np.max(x)), 0) + 1

    std_min, std_max = std_bound
    if std_bound[0] == 0:
        std_min = round(float(np.min(y)), 1) - 0.03
    if std_bound[1] == 0:
        std_max = round(float(np.max(y)), 1) + 0.03

    gs = GridSpec(
        nrows=2,
        ncols=2,
        hspace=0,
        wspace=0,
        height_ratios=(1, 3),
        width_ratios=(3, 1),
    )
    fig = plt.figure(figsize=(7, 7))

    # point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # histogram - mean deviation
    ax1 = plt.subplot(gs[0, 0])
    mcounts, mbins = np.histogram(x, bins=30)
    mbinc = 0.5 * (mbins[:-1] + mbins[1:])
    mfreq = mcounts / len(x)
    ax1.fill_between(mbinc, mfreq, step="pre", alpha=0.6)
    ax1.set_xlim(0, mean_max)
    ax1.set_yscale("log")
    ax1.set_xticklabels("")
    ax1.set_ylabel("Frequency")

    # histogram - stdev(RDN)
    ax2 = plt.subplot(gs[1, 1])
    counts, bins = np.histogram(y, bins=30)
    freq = counts / len(y)
    binc = 0.5 * (bins[:-1] + bins[1:])
    ax2.fill_between(freq, binc, step="pre", alpha=0.6)
    ax2.set_yticklabels("")
    ax2.set_ylim(std_min, std_max)
    ax2.axhline(np.median(y), color="gray", ls="dotted")
    ax2.text(
        0.13,
        np.median(y) + 0.01,
        s="$\\overline{\\sigma_{RDN}}=$" + f"{np.mean(y):.1f}",
        horizontalalignment="right",
        fontsize=12,
    )
    ax2.set_xscale("log")
    ax2.set_xlim(5e-4, 0.2)
    ax2.set_xlabel("Frequency")

    # scatter plot
    ax = plt.subplot(gs[1, 0])
    scat = ax.scatter(
        x, y, c=np.log10(z), s=10, edgecolor="k", lw=0.2, cmap="RdYlBu_r"
    )
    fig.colorbar(scat, ax=ax2, label="$\\rm log_{10}$(Density)")
    ax.set_xlabel("Mean Difference [ADU]")
    ax.set_ylabel("$\\sigma_{RDN}$ [ADU]")
    ax.text(mean_max * fm, std_min * fM, s=label, horizontalalignment="right", fontsize=18)
    ax.set_xlim(0, mean_max)
    ax.set_ylim(std_min, std_max)
    return freq
