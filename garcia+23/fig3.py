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


def plotting_interface(run_logpath, simulation_name, marker, hist_color):
    # create a custom color map
    cvals = [0.1, 3]
    colors = ["orangered", "cyan"]
    norm = plt.Normalize(min(cvals), max(cvals))
    tuples = list(zip(map(norm, cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.20), dpi=400)
    plt.subplots_adjust(wspace=0.55)
    hist_ax_left = ax[0].inset_axes([1.02, 0, 0.30, 1], sharey=ax[0])
    hist_ax_left.set_xlabel(r"$\mathrm{PDF} (\overline{n_{\rm H}})$")
    hist_ax_left.tick_params(axis="both", labelleft=False, labelsize=6)

    hist_ax_right = ax[1].inset_axes([1.02, 0, 0.30, 1], sharey=ax[1])
    hist_ax_right.set_xlabel(r"$\mathrm{PDF} (T_{\rm MC})$")
    hist_ax_right.tick_params(axis="both", labelleft=False, labelsize=6)

    xlims = [1e-4, 9e-3]  # zsun
    nh_lims = [2e3, 8e4]  # mean hydrogen number density
    temp_lims = [1.5e2, 3e3]  # Kelvin
    hist_num_bins = 25
    temp_bins = np.geomspace(temp_lims[0], temp_lims[1], hist_num_bins)
    nh_bins = bins = np.geomspace(nh_lims[0], nh_lims[1], hist_num_bins)

    ax[0].set(
        xlim=xlims,
        ylim=nh_lims,
        xscale="log",
        yscale="log",
        xlabel=r"$Z_{\rm MC} (\mathrm{Z }_{\odot})$",
        ylabel=r"$\overline{n_\mathrm{H}} \: \left( \mathrm{cm} ^{-3} \right)$",
    )

    ax[1].set(
        xlim=xlims,
        ylim=temp_lims,
        xscale="log",
        yscale="log",
        xlabel=r"$Z_{\rm MC} (\mathrm{Z }_{\odot})$",
        ylabel=r"$T_{\rm MC}\:({\rm K})$",
    )

    h = []

    for n, m in zip(simulation_name, marker):
        sim_label = mlines.Line2D(
            [],
            [],
            color="k",
            marker=m,
            ls="",
            label=n,
        )
        h.append(sim_label)

    ax[0].legend(loc="upper right", handles=h)

    hatches = ["\\\\\\\\\\", "/////"]

    for i, r in enumerate(run_logpath):
        run_name = os.path.basename(os.path.normpath(r))
        log_sfc = np.loadtxt(os.path.join(r, "logSFC"))
        redshft = log_sfc[:, 2]
        r_pc_cloud = log_sfc[:, 4]
        m_sun_cloud = log_sfc[:, 5]
        n_hydrogen = log_sfc[:, 8]
        metal_zun_cloud = log_sfc[:, 9]  # metalicity is normalized to z_sun
        temp = sfc_temperature(n_hydrogen, redshft, r_pc_cloud, ncut=100)

        im_n = ax[0].scatter(
            metal_zun_cloud,
            n_hydrogen,
            c=redshft,
            cmap="autumn_r",
            marker=marker[i],
            edgecolors="black",
            linewidth=0.5,
            s=40,
            alpha=0.6,
            vmax=13,
            vmin=8,
        )

        hist_ax_left.hist(
            n_hydrogen,
            bins=bins,
            color=hist_color[i],
            edgecolor=hist_color[i],
            histtype="step",
            hatch=hatches[i % 2],
            alpha=0.9,
            linewidth=1.25,
            density=True,
            orientation="horizontal",
            label=simulation_name[i],
        )

        im_t = ax[1].scatter(
            metal_zun_cloud,
            temp,
            c=r_pc_cloud,
            label=r"0.70",
            cmap=cmap,
            marker=marker[i],
            edgecolors="black",
            linewidth=0.5,
            s=40,
            alpha=0.6,
        )

        hist_ax_right.hist(
            temp,
            bins=temp_bins,
            color=hist_color[i],
            edgecolor=hist_color[i],
            histtype="step",
            hatch=hatches[i % 2],
            alpha=0.9,
            linewidth=1.25,
            density=True,
            orientation="horizontal",
        )

    cbar_ax_left = ax[0].inset_axes([0, 1.02, 1, 0.05])
    form_bar = fig.colorbar(im_n, cax=cbar_ax_left, pad=0, orientation="horizontal")
    form_bar.set_label(r"$z_{\mathrm{formation}}$", labelpad=6)
    form_bar.ax.xaxis.set_label_position("top")
    form_bar.ax.xaxis.set_ticks_position("top")
    form_bar.ax.locator_params(nbins=6)
    form_bar.ax.invert_xaxis()

    cbar_ax_right = ax[1].inset_axes([0, 1.02, 1, 0.05])
    radius_bar = fig.colorbar(im_t, cax=cbar_ax_right, pad=0, orientation="horizontal")
    radius_bar.set_label(r"$\mathrm{R_{MC}} (\mathrm{pc})$", labelpad=6)
    radius_bar.ax.xaxis.set_label_position("top")
    radius_bar.ax.xaxis.set_ticks_position("top")
    radius_bar.ax.locator_params(nbins=6)

    hist_ax_left.legend(fontsize=7, loc="upper center")


if __name__ == "__main__":
    # log_sfc = np.loadtxt("../sim_log_files/fs07_refine/logSFC")
    cmap = matplotlib.colormaps["Set2"]
    cmap = cmap(np.linspace(0, 1, 8))

    runs = [
        "../../container_tiramisu/sim_log_files/fs07_refine",
        # "../../container_tiramisu/sim_log_files/fs035_ms10",
        "../../container_tiramisu/sim_log_files/CC-fiducial",
    ]
    names = [
        "$f_* = 0.70$",
        # "$f_* = 0.35$",
        r"${\rm He+19}$",
    ]
    markers = [
        "o",
        # "P",
        "v",
    ]
    colors = [
        cmap[0],
        # cmap[1],
        cmap[2],
    ]

    plotting_interface(
        run_logpath=runs,
        simulation_name=names,
        marker=markers,
        hist_color=colors,
    )

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
