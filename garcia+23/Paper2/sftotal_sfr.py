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
        nrows=4,
        ncols=1,
        sharex=True,
        figsize=(5, 6),
        dpi=300,
        gridspec_kw={"height_ratios": [4, 2, 2, 2]},
    )
    redshft_ax = ax[0].twiny()

    plt.subplots_adjust(hspace=0, wspace=0)

    earliest_times = []
    latest_times = []
    max_galsfe = []
    sfrs = []
    times = []

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
            t_interp_points[t_interp_points < 700][
                np.argmax(sfr[t_interp_points < 700])
            ],
            "with value",
            np.max(sfr[t_interp_points < 700]),
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
        ax[i + 1].plot(
            t_interp_points,
            sfr,
            label=simulation_name[i],
            color=color[i],
            linewidth=1.5,
            alpha=1,
        )
        ax[i + 1].minorticks_on()
        ax[i + 1].locator_params(axis="y", nbins=4)

        sf_times = t_interp_points[np.argwhere(sfr > 0.001)]
        # trange = t_interp_points[-1] - t_interp_points[0]
        print(sf_times.size)
        print(t_interp_points.size)
        fduty = sf_times.size / t_interp_points.size
        # if i == 0:
        label = r"$f_{{\rm duty}} = {:.2f}$".format(fduty)
        # else:
        #     label = r"${:.2f}$".format(fduty)

        ax[i + 1].text(
            0.02,
            0.90,
            label,
            ha="left",
            va="top",
            transform=ax[i + 1].transAxes,
            fontsize=9,
        )

        sfrs.append(sfr)
        times.append(t_interp_points)

        sf_mask = np.argwhere(sfr > 1e-3)
        sf_times = t_interp_points[sf_mask].flatten()
        diff_between_peaks = np.diff(sf_times)
        quiscent_mask = diff_between_peaks > 1
        quiscent_times = sf_times[1:][quiscent_mask]
        print(quiscent_times)
        leftedges = quiscent_times[::2]
        right_edges = quiscent_times[1::2]
        # print(quiscent_times[::2])
        # for q, right in enumerate(right_edges):
        #     ax[i + 1].axvspan(leftedges[q], right, facecolor="grey", alpha=0.3)

    ax[1].set(ylim=(-0.01, 0.14))
    ax[2].set(ylim=(-0.01 * 0.5, 0.14 * 0.5))
    ax[3].set(ylim=(-0.01 * 0.2, 0.14 * 0.2))

    ax[0].set(
        yscale="log",
        ylabel=r"$\mathrm{M_{\star}}\: \left[ \mathrm{M}_{\odot} \right] $",
        ylim=(3e2, 3e6),
    )
    ax[0].legend(fontsize=11, loc="lower right", frameon=False)

    ax[2].set(
        ylabel=r"$\mathrm{SFR} \:\left[ \mathrm{M}_{\odot} \:\mathrm{yr}^{-1}\right]$",
    )

    ax[3].set(
        xlabel=r"time $\left[ \mathrm{ Myr} \right]$",
        xlim=(np.min(earliest_times) - 1, np.max(latest_times)),
    )

    # ax[1].xaxis.get_ticklocs(minor=True)
    redshft_ax.locator_params(axis="x")
    redshft_ax.set(
        xlim=(np.min(earliest_times) - 1, np.max(latest_times)), xlabel="$z$"
    )
    redshft_ax.set_xticklabels(
        list(np.round(z_from_t_myr(redshft_ax.get_xticks()), 1).astype("str"))
    )

    # event labels

    ax[1].axvspan(465, 485, facecolor="grey", alpha=0.3)
    ax[1].axvspan(565, 585, facecolor="grey", alpha=0.3)
    ax[1].text(465, 0.1, "(a)", ha="right", va="center", fontsize=9)
    ax[1].text(565, 0.1, "(b)", ha="right", va="center", fontsize=9)

    # ax[1].axvline(585)
    ax[2].axvspan(440, 470, facecolor="grey", alpha=0.3)
    ax[2].axvspan(660, 678, facecolor="grey", alpha=0.3)
    ax[2].text(440, 0.05, "(c)", ha="right", va="center", fontsize=9)
    ax[2].text(660, 0.05, "(d)", ha="right", va="center", fontsize=9)

    ax[3].axvspan(410, 440, facecolor="grey", alpha=0.3)
    ax[3].axvspan(480, 500, facecolor="grey", alpha=0.3)
    ax[3].axvspan(540, 560, facecolor="grey", alpha=0.3)
    ax[3].axvspan(590, 610, facecolor="grey", alpha=0.3)
    ax[3].text(405, 0.01, "(e)", ha="right", va="center", fontsize=9)
    ax[3].text(475, 0.01, "(f)", ha="right", va="center", fontsize=9)
    ax[3].text(535, 0.01, "(g)", ha="right", va="center", fontsize=9)
    ax[3].text(585, 0.01, "(h)", ha="right", va="center", fontsize=9)
    # ax[3].axvspan(660, 678, facecolor="grey", alpha=0.3)

    # ax[0].grid(ls="--", which="both")
    # ax[1].grid(ls="--", which="both")
    # ax[2].grid(ls="--", which="both")
    return sfrs, times


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

    sfr, time = plotting_interface(
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
    # # %%
    # x = time[2]
    # y = sfr[2]
    # sf_mask = np.argwhere(y > 1e-3)
    # sf_times = x[sf_mask].flatten()
    # diff_between_peaks = np.diff(sf_times)
    # quescent_mask = diff_between_peaks > 10
    # quiscent_times = sf_times[1:][quescent_mask]
    # print(quiscent_times)

    # plt.plot(x, y, color="r")
    # plt.scatter(x[sf_mask], y[sf_mask])
    # plt.show()
