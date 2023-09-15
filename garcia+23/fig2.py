import sys

sys.path.append("..")
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


def plotting_interface(run_logpath, dm_path, simulation_name, color):
    sfr_binwidth_myr = 1

    fig, ax = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        figsize=(5, 4.5),
        dpi=300,
        gridspec_kw={"height_ratios": [3, 2, 2]},
    )
    redshft_ax = ax[0].twiny()
    plt.subplots_adjust(hspace=0, wspace=0)

    earliest_times = []
    latest_times = []
    max_galsfe = []

    for i, (dm, r) in enumerate(zip(dm_path, run_logpath)):
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

        # get the main halo (galaxy halo)
        main_halo = np.loadtxt(os.path.join(dm, "main_halo.txt"))
        halo_t = main_halo[:, 1]
        halo_mvir = main_halo[:, 3]  # msun
        halo_mstar = main_halo[:, 5]  # stars within the box
        #!!! should compute stars within the virial radius in the future
        earliest_times.append(t_myr.min())
        latest_times.append(t_myr.max())

        # compute galaxy wide SFE
        gal_sfe = halo_mstar / halo_mvir
        max_galsfe.append(gal_sfe.max())
        # dm_mass = match_halofinder_logsfc(
        #     f3_halo[:, 1], fs035_interp_points, f3_halo[:, 3]
        # )[f3_mask]

        ax[0].plot(
            t_interp_points,
            total_mass,
            label=simulation_name[i],
            color=color[i],
            linewidth=4,
            alpha=0.8,
        )
        ax[1].plot(
            t_interp_points,
            sfr,
            label=simulation_name[i],
            color=color[i],
            linewidth=1.5,
            alpha=0.8,
        )
        ax[2].plot(
            halo_t,
            gal_sfe,
            label=simulation_name[i],
            color=color[i],
            linewidth=4,
            alpha=0.8,
        )
        redshft_ax.plot(t_interp_points, total_mass, linewidth=0)
    ax[0].set(
        yscale="log",
        ylabel=r"$\mathrm{\log_{10}\:M_{*}}\:(\mathrm{M}_{\odot})$",
    )
    ax[0].legend()
    ax[1].set(
        ylim=(0, 0.12),
        ylabel=r"$\mathrm{SFR} \:\left( \mathrm{M}_{\odot} \:\mathrm{yr}^{-1}\right)$",
    )
    ax[1].locator_params(nbins=6)
    ax[2].set(
        yscale="log",
        xlabel="$\mathrm{t } \:(\mathrm{Myr})$",
        ylabel=r"$\mathrm{M_{*} \; M_{halo}^{-1}}$   ",
        xlim=(np.min(earliest_times), np.max(latest_times)),
    )
    ax[2].set_ylim(top=np.max(max_galsfe) * 4)

    redshft_ax.set(xlim=(np.min(earliest_times), np.max(latest_times)), xlabel="$z$")
    redshft_ax.set_xticklabels(
        list(np.round(z_from_t_myr(redshft_ax.get_xticks()), 1).astype("str"))
    )
    ax[2].locator_params(axis="x", nbins=8)

    # ax[0].grid(ls="--", which="both")
    # ax[1].grid(ls="--", which="both")
    # ax[2].grid(ls="--", which="both")


if __name__ == "__main__":
    cmap = matplotlib.colormaps["Set2"]
    cmap = cmap(np.linspace(0, 1, 8))

    runs = [
        "../../container_tiramisu/sim_log_files/fs07_refine",
        # "../../container_tiramisu/sim_log_files/fs035_ms10",
        "../../container_tiramisu/sim_log_files/CC-Fiducial",
    ]
    dm_paths = [
        "../../container_tiramisu/post_processed/dm_hop/fs07_refine",
        # "../../container_tiramisu/post_processed/dm_hop/fs035_ms10",
        "../../container_tiramisu/post_processed/dm_hop/CC-Fiducial",
    ]
    names = [
        "$f_* = 0.70$",
        # "$f_* = 0.35$",
        r"${\rm He+19}$",
    ]

    colors = [
        cmap[0],
        # cmap[1],
        cmap[2],
    ]

    plotting_interface(
        run_logpath=runs,
        dm_path=dm_paths,
        simulation_name=names,
        color=colors,
    )

    plt.savefig(
        "../../gdrive_columbia/research/massimo/fig2.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()
