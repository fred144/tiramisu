import sys

sys.path.append("../../")
import matplotlib.pyplot as plt
import numpy as np
from tools.cosmo import t_myr_from_z, z_from_t_myr
from matplotlib import cm
import matplotlib
from scipy import interpolate
from tools import plotstyle
import os
from tools.fscanner import filter_snapshots
import h5py as h5
from matplotlib import colors
import matplotlib as mpl
import cmasher as cmr
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from outflow_rates import read_outflow_rates
import seaborn as sns

from scipy.stats import binned_statistic


cc_times, cc_m_out, cc_mz_out, cc_m_in, cc_mz_in = read_outflow_rates(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
    173,
    466,
    boundary="CGM",
)

f7_times, f7_m_out, f7_mz_out, f7_m_in, f7_mz_in = read_outflow_rates(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    100,
    1570,
    step=6,
    boundary="CGM",
)

f3_times, f3_m_out, f3_mz_out, f3_m_in, f3_mz_in = read_outflow_rates(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    100,
    1606,
    step=6,
    boundary="CGM",
)


cmap = sns.color_palette(palette="cubehelix")
fig, ax = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(12, 8),
    dpi=300,
    sharex=True,
    sharey="row",
    gridspec_kw={"height_ratios": [3, 3, 3]},
)

ax = ax.ravel()
plt.subplots_adjust(hspace=0.0, wspace=0)

for i in range(0, 8, 3):
    ax[i].axvspan(465, 485, facecolor="grey", alpha=0.3)
    ax[i].axvspan(565, 588, facecolor="grey", alpha=0.3)

for i in range(1, 9, 3):
    ax[i].axvspan(440, 470, facecolor="grey", alpha=0.3)
    ax[i].axvspan(660, 678, facecolor="grey", alpha=0.3)

for i in range(2, 10, 3):
    ax[i].axvspan(410, 440, facecolor="grey", alpha=0.3)
    ax[i].axvspan(480, 500, facecolor="grey", alpha=0.3)
    ax[i].axvspan(540, 560, facecolor="grey", alpha=0.3)
    ax[i].axvspan(590, 610, facecolor="grey", alpha=0.3)


ax[0].plot(cc_times, cc_mz_out, label="outflowing", color=cmap[2], lw=3)
ax[1].plot(f7_times, f7_mz_out, color=cmap[2], lw=3)
ax[2].plot(f3_times, f3_mz_out, color=cmap[2], lw=3)

ax[0].plot(cc_times, -cc_mz_in, label="inflowing", color=cmap[3], lw=3)
ax[1].plot(f7_times, -f7_mz_in, color=cmap[3], lw=3)
ax[2].plot(f3_times, -f3_mz_in, color=cmap[3], lw=3)

ax[3].plot(cc_times, cc_m_out, color=cmap[2], lw=3, label="outflowing")
ax[4].plot(f7_times, f7_m_out, color=cmap[2], lw=3)
ax[5].plot(f3_times, f3_m_out, color=cmap[2], lw=3)

ax[3].plot(cc_times, -cc_m_in, label="inflowing", color=cmap[3], lw=3)
ax[4].plot(f7_times, -f7_m_in, color=cmap[3], lw=3)
ax[5].plot(f3_times, -f3_m_in, color=cmap[3], lw=3)

ax[3].set(
    ylabel=r"$|\dot{M}_{\rm gas}|\: \left[{\rm M_\odot \: yr^{-1}}\right]$",
    yscale="log",
)

ax[6].plot(cc_times, cc_mz_out / cc_m_out, color=cmap[2], label="outflowing", lw=3)
ax[7].plot(f7_times, f7_mz_out / f7_m_out, color=cmap[2], lw=3)
ax[8].plot(f3_times, f3_mz_out / f3_m_out, color=cmap[2], lw=3)

ax[6].plot(cc_times, cc_mz_in / cc_m_in, label="inflowing", color=cmap[3], lw=3)
ax[7].plot(f7_times, f7_mz_in / f7_m_in, color=cmap[3], lw=3)
ax[8].plot(f3_times, f3_mz_in / f3_m_in, color=cmap[3], lw=3)


ax[6].set(
    ylabel=r"$\langle Z \rangle \: [\mathrm{Z_\odot}$]",
    yscale="log",
    xlim=(345, 735),
)
ax[0].set(
    ylabel=r"$|\dot{M}_{\rm metal}|\: \left[{\rm M_\odot \: yr^{-1}}\right]$",
    yscale="log",
)


ax[0].legend(frameon=False)
# ax[3].legend(frameon=False, ncols=2)
ax[0].minorticks_on()
ax[7].set(xlabel=r"time [Myr]")
ax[7].locator_params(axis="x", nbins=12)

ax[0].text(
    0.05, 0.90, "VSFE", ha="left", va="top", fontsize=9, transform=ax[0].transAxes
)
ax[1].text(
    0.05, 0.90, "HSFE", ha="left", va="top", fontsize=9, transform=ax[1].transAxes
)
ax[2].text(
    0.05, 0.90, "LSFE", ha="left", va="top", fontsize=9, transform=ax[2].transAxes
)

for i in range(0, 3):
    redshft_ax = ax[i].twiny()
    redshft_ax.locator_params(axis="x", nbins=12)
    if i == 1:
        redshft_ax.set_xlabel("$z$")
    redshft_ax.set(xlim=(345, 735))

    redshft_ax.set_xticklabels(
        list(np.round(z_from_t_myr(redshft_ax.get_xticks()), 1).astype("str"))
    )


plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/cgm_outflow_rates.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()
