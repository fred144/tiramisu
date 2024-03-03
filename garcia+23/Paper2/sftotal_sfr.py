"""
This shows the pop2 SFR
"""

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


def match_dmhalo_logsfc(finder_time: float, log_time: float, finder_mass: float):
    # look_up_lumi = data[:, column_idx]
    residuals = np.abs(finder_time - log_time[:, np.newaxis])
    closest_match_idxs = np.argmin(residuals, axis=1)
    log_sfc_dm_mass = finder_mass[closest_match_idxs]
    return log_sfc_dm_mass


def plotting_interface(run_logpath, simulation_name, color):
    sfr_binwidth_myr = 1

    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(5, 4),
        dpi=300,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    redshft_ax = ax[0].twiny()
    plt.subplots_adjust(hspace=0, wspace=0)

    earliest_times = []
    latest_times = []
    max_galsfe = []

    for i, r in enumerate(run_logpath):
        print(os.path.join(r, "logSFC"))
        log_sfc = np.loadtxt(os.path.join(r, "logSFC"))

        # get star formation properties as a function of time and redshift
        redshift = log_sfc[:, 2]
        t_myr = t_myr_from_z(redshift)
        mass_in_star = log_sfc[:, 7]
        running_total_mass = np.cumsum(mass_in_star)

        # interpolate points so that SFR is not infinity, since SF is a step function
        t_interp_points = np.arange(t_myr.min(), t_myr.max(), sfr_binwidth_myr)
        total_mass_interpolator = interpolate.interp1d(
            x=t_myr, y=running_total_mass, kind="previous"
        )
        total_mass = total_mass_interpolator(t_interp_points)
        # calculate the sfr in msun / yr
        sfr = np.gradient(total_mass) / (sfr_binwidth_myr * 1e6)
        print(
            "sfr peaks at",
            t_interp_points[t_interp_points < 550][
                np.argmax(sfr[t_interp_points < 550])
            ],
        )
        earliest_times.append(t_myr.min())
        latest_times.append(t_myr.max())

        ax[0].plot(
            t_interp_points,
            total_mass,
            label=simulation_name[i],
            color=color[i],
            linewidth=4,
            # alpha=0.8,
        )
        ax[1].plot(
            t_interp_points,
            sfr,
            label=simulation_name[i],
            color=color[i],
            linewidth=1.5,
            # alpha=0.8,
        )
        if i == 0:
            # ax[1].axvspan(465, 485, facecolor=color[i], alpha=0.3)

            # snapshot 374 - 399, VSFE
            ax[1].axvspan(565, 590, facecolor=color[i], alpha=0.3)  # Myr
        elif i == 1:
            # snapshot 333 - 428, high SFE
            # ax[1].axvspan(445, 465, facecolor=color[i], alpha=0.3)

            # snapshot 1300 - 1426, high SFE
            ax[1].axvspan(658, 685, facecolor=color[i], alpha=0.3)
        elif i == 2:
            # snapshot 359 - 458, low SFE
            ax[1].axvspan(416, 438, facecolor=color[i], alpha=0.3)

        redshft_ax.plot(t_interp_points, total_mass, linewidth=0)
    ax[0].set(
        yscale="log",
        ylabel=r"$\mathrm{\log \:M_{\star}}\: \left[ \mathrm{M}_{\odot} \right] $",
    )
    ax[0].legend(fontsize=11, loc="lower right", frameon=False)
    ax[1].set(
        # ylim=(0, 0.12),
        ylabel=r"$\mathrm{SFR} \:\left[ \mathrm{M}_{\odot} \:\mathrm{yr}^{-1}\right]$",
        xlabel=r"time $\left[ \mathrm{ Myr} \right]$",
        xlim=(np.min(earliest_times) - 1, np.max(latest_times)),
    )
    # ax[1].xaxis.get_ticklocs(minor=True)
    ax[1].minorticks_on()

    redshft_ax.set(xlim=(np.min(earliest_times), np.min(latest_times)), xlabel="$z$")
    redshft_ax.set_xticklabels(
        list(np.round(z_from_t_myr(redshft_ax.get_xticks()), 1).astype("str"))
    )
    ax[1].locator_params(axis="x", nbins=8)

    # ax[0].grid(ls="--", which="both")
    # ax[1].grid(ls="--", which="both")
    # ax[2].grid(ls="--", which="both")


if __name__ == "__main__":
    cmap = matplotlib.colormaps["Dark2"]
    cmap = cmap(np.linspace(0, 1, 8))

    runs = [
        "../../../container_tiramisu/sim_log_files/CC-Fiducial",
        "../../../container_tiramisu/sim_log_files/fs07_refine",
        "../../../container_tiramisu/sim_log_files/fs035_ms10",
        # "../../container_tiramisu/sim_log_files/haloD_varSFE_Lfid_Salp_ks20231024",
        # "../../container_tiramisu/sim_log_files/CC_2x_Salp",
    ]

    names = [
        "VSFE",
        "high SFE",
        "low SFE",
        # r"$2.0 \times$ Lfid",
    ]

    colors = [
        # cmap[0],
        cmap[0],
        cmap[1],
        cmap[2],
    ]

    plotting_interface(
        run_logpath=runs,
        simulation_name=names,
        color=colors,
    )

    plt.savefig(
        "../../../gdrive_columbia/research/massimo/paper2/SF_history.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()
