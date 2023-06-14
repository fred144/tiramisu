import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib import cm
import matplotlib
from matplotlib.ticker import MaxNLocator
import matplotlib.lines as mlines
import os
import matplotlib as mpl

#!!! to do generalize so that you can plot multiple simulations by looping
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 12,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
    }
)


def n_crit(tem, z):
    return 5.0e4 * ((1.0 + z) / 10.0) ** 2 * (tem / 100.0)


def fac(tem, z, n_H):
    return n_H / (5.0e3 * ((1.0 + z) / 10.0) ** 2 * (tem / 100.0))


def radius(n_H, tem, ra):
    m_H = 1.6e-24
    G = 6.7e-8
    cs = 1.0e6 * np.sqrt(tem / 1.0e4)
    # x=1.0/ra
    x = 1.0
    n_0 = n_H / (10.0 * x**3 + 3.0 - 3 * x)
    return 2.0 * cs / np.sqrt(2.0 * np.pi * G * m_H * n_0) / 3.0e18


def sfc_temperature(n_h, redshifts, ra, ncut=10):
    """
    temperature of a cloud given its hydrogen density,
    redshift of formation, current temp

    """
    # x = 1.0 / ra
    # n_0=10.*n_H/(10.*x**3+3-3*x)
    # !!! needs to depend on ncut
    # n_h = 0.26 n_crit

    # n_h = (3*ncut**0.5 - 2)*ncut**-1.5 * n_crit
    n_crit = ((3 * ncut**0.5 - 2) * ncut**-1.5) ** -1 * n_h
    # n_crit = 3.84 * n_h
    return 100.0 * (n_crit / 5e4) * ((1.0 + redshifts) / 10.0) ** -2


# log_sfc = np.loadtxt("../sim_log_files/fs07_refine/logSFC")

run = "../../container_tiramisu/sim_log_files/cc-kazu-run"
run_name = run.split("/")[-1]

log_sfc = np.loadtxt(os.path.join(run, "logSFC"))

redshft = log_sfc[:, 2]
r_pc_cloud = log_sfc[:, 4]
m_sun_cloud = log_sfc[:, 5]
n_hydrogen = log_sfc[:, 8]
metal_zun_cloud = log_sfc[:, 9]  # metalicity is normalized to z_sun


cmap = cm.get_cmap("Set2")
cmap = cmap(np.linspace(0, 1, 8))
hist_color = cmap[1]
cmap = cm.get_cmap("Set3")
cmap = cmap(np.linspace(0, 1, 11))
cvals = [0.1, 3]
colors = ["orangered", "cyan"]
norm = plt.Normalize(min(cvals), max(cvals))
tuples = list(zip(map(norm, cvals), colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)
hist_num_bins = 25

fig, ax = plt.subplots(1, 2, figsize=(10, 4.20), dpi=400)
plt.subplots_adjust(wspace=0.55)
temp = sfc_temperature(n_hydrogen, redshft, r_pc_cloud, ncut=100)

im_n = ax[0].scatter(
    metal_zun_cloud,
    n_hydrogen,
    c=redshft,
    cmap="autumn_r",
    marker="o",
    edgecolors="black",
    linewidth=0.5,
    s=40,
    alpha=0.8,
)
cbar_ax = ax[0].inset_axes([0, 1.02, 1, 0.05])
bar = fig.colorbar(im_n, cax=cbar_ax, pad=0, orientation="horizontal")
# bar .ax.xaxis.set_tick_params(pad=2)
bar.set_label(r"$z_{\mathrm{formation}}$", labelpad=6)
bar.ax.xaxis.set_label_position("top")
bar.ax.xaxis.set_ticks_position("top")
bar.ax.locator_params(nbins=6)
ax[0].set(
    xlim=[0.5 * metal_zun_cloud.min(), 1.75 * metal_zun_cloud.max()],
    ylim=[n_hydrogen.min() * 0.65, n_hydrogen.max() * 1.5],
    xscale="log",
    yscale="log",
    xlabel=r"$Z_{\rm MC} (\mathrm{Z }_{\odot})$",
    ylabel=r"$\overline{n_\mathrm{H}} \: \left( \mathrm{cm} ^{-3} \right)$",
)
bar.ax.invert_xaxis()

sfc = mlines.Line2D(
    [], [], color="k", marker="o", ls="", label=r"${{\rm {}}}$".format(run_name)
)
ax[0].legend(
    # title="$\mathrm{SFE} \: (f_{*})$",
    loc="upper right",
    handles=[sfc],
)

hist_ax = ax[0].inset_axes([1.02, 0, 0.30, 1], sharey=ax[0])
bins = np.geomspace(n_hydrogen.min() * 0.65, n_hydrogen.max() * 1.5, hist_num_bins)
hist_ax.hist(
    n_hydrogen,
    bins=bins,
    color=hist_color,
    histtype="step",
    hatch="\\\\\\\\",
    edgecolor=hist_color,
    alpha=0.9,
    linewidth=1.25,
    density=True,
    orientation="horizontal",
)
hist_ax.set_xlabel(r"$\mathrm{PDF} (\overline{n_{\rm H}})$")
hist_ax.tick_params(axis="both", labelleft=False, labelsize=6)

im = ax[1].scatter(
    metal_zun_cloud,
    temp,
    c=r_pc_cloud,
    label=r"0.70",
    cmap=cmap,
    marker="o",
    edgecolors="black",
    linewidth=0.5,
    s=40,
    alpha=0.8,
    # vmax=2.8,
)

cbar_ax = ax[1].inset_axes([0, 1.02, 1, 0.05])
bar = fig.colorbar(im, cax=cbar_ax, pad=0, orientation="horizontal")
# bar .ax.xaxis.set_tick_params(pad=2)
bar.set_label(r"$\mathrm{R_{MC}} (\mathrm{pc})$", labelpad=6)
bar.ax.xaxis.set_label_position("top")
bar.ax.xaxis.set_ticks_position("top")
bar.ax.locator_params(nbins=6)

ax[1].set(
    xlim=[0.5 * metal_zun_cloud.min(), 1.75 * metal_zun_cloud.max()],
    ylim=[temp.min() * 0.65, temp.max() * 1.5],
    xscale="log",
    yscale="log",
    xlabel=r"$Z_{\rm MC} (\mathrm{Z }_{\odot})$",
    ylabel=r"$T_{\rm MC}\:({\rm K})$",
)


hist_ax = ax[1].inset_axes([1.02, 0, 0.30, 1], sharey=ax[1])
bins = np.geomspace(temp.min() * 0.65, temp.max() * 1.5, hist_num_bins)
hist_ax.hist(
    temp,
    bins=bins,
    color=hist_color,
    histtype="step",
    hatch="\\\\\\\\",
    edgecolor=hist_color,
    alpha=0.9,
    linewidth=1.25,
    density=True,
    orientation="horizontal",
)
hist_ax.set_xlabel(r"$\mathrm{PDF} (T_{\rm MC})$")
hist_ax.tick_params(axis="both", labelleft=False, labelsize=6)

plt.show()
# plt.savefig(
#     "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/lowres/sfc_metal_temp.png",
#     dpi=300,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
# plt.savefig(
#     "../../g_drive/Research/AstrophysicsSimulation/sci_plots/final/sfc_metal_temp.png",
#     dpi=400,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
