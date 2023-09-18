import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from tools.cosmo import t_myr_from_z, z_from_t_myr
from matplotlib import cm
import matplotlib
import matplotlib.lines as mlines
from scipy import interpolate
from tools import plotstyle
import os


def plotting_interface(run_path, simulation_name, color):
    fig, ax = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(5, 4.5),
        dpi=300,
        gridspec_kw={"height_ratios": [4, 2]},
    )
    plt.subplots_adjust(hspace=0, wspace=0)
    redshft_ax = ax[0].twiny()

    earliest_times = []
    latest_times = []

    for i, r in enumerate(run_path):
        data = np.loadtxt(r)
        t_myr = data[:, 1]
        redshift = data[:, 2]
        clumped_mass = data[:, 3]
        bsc_mass = data[:, 4]
        disrupted_mass = data[:, 5]
        unbound_mass = data[:, 6]

        clumped_lum = data[:, 7]
        total_lum = data[:, 8]
        disrupted_lum = data[:, 9]
        unbound_lum = data[:, 10]

        earliest_times.append(t_myr.min())
        latest_times.append(t_myr.max())
        # plot the total mass
        ax[0].plot(
            t_myr,
            clumped_lum + unbound_lum,
            color=color[i],
            linewidth=2,
            alpha=0.5,
            label=simulation_name[i],
        )
        ax[0].plot(
            t_myr,
            clumped_lum,
            color=color[i],
            ls="--",
            linewidth=2,
            alpha=1,
        )

        ax[1].plot(
            t_myr,
            clumped_lum / (clumped_lum + unbound_lum),
            color=color[i],
            linewidth=3,
            alpha=0.8,
        )

        ax[1].plot(
            t_myr,
            clumped_lum / (clumped_lum + unbound_lum),
            color=color[i],
            linewidth=3,
            alpha=0.8,
        )
        ax[1].axhline(y=0.5, ls="--", c="grey", alpha=0.8)

        redshft_ax.plot(t_myr, clumped_mass + unbound_mass, linewidth=0)

    # ax[0].set(yscale="log")
    ax[0].set(
        yscale="log",
        ylabel=r"$\mathrm{M_{*} \; M_{halo}^{-1}}$   ",
        xlim=(np.min(earliest_times), np.max(latest_times)),
        ylim=(2e35, 2e39),
    )
    ax[0].legend()
    ax[0].set_ylabel(
        r"$L_{\rm \lambda = 1500 \: \mathrm{\AA}} \:$"
        r"$(\mathrm{erg} \:\mathrm{s}^{-1} \:\mathrm{\AA}^{-1})$",
        labelpad=10,
    )
    ax[1].set_ylabel(r"${L_{\rm clumped}} / L_{\rm total}$", labelpad=5)
    ax[1].set_xlabel("$\mathrm{t } \:(\mathrm{Myr})$")

    redshft_ax.set(xlim=(np.min(earliest_times), np.max(latest_times)), xlabel="$z$")
    redshft_ax.set_xticklabels(
        list(np.around(z_from_t_myr(redshft_ax.get_xticks()), 1).astype("str"))
    )

    # print(np.around(z_from_t_myr(redshft_ax.get_xticks()), 2))

    bsc = mlines.Line2D([], [], color="k", ls="-", label=r"${\rm Total}$")
    unbound = mlines.Line2D([], [], color="k", ls="--", label=r"${\rm Unbound}$")

    ax[1].legend(
        bbox_to_anchor=(0.0, 3),
        loc="upper left",
        handles=[bsc, unbound],
        fontsize=10,
        frameon=False,
    )


if __name__ == "__main__":
    cmap = matplotlib.colormaps["Set2"]
    cmap = cmap(np.linspace(0, 1, 8))
    parent = "../../container_tiramisu/post_processed/bsc_catalogues/"
    runs = [
        os.path.join(parent, "fs07_refine/fs07_refine_timeseries-00113-01570.txt"),
        # "../../container_tiramisu/sim_log_files/fs035_ms10",
        os.path.join(parent, "CC-Fiducial/CC-Fiducial_timeseries-00304-00405.txt"),
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
        run_path=runs,
        simulation_name=names,
        color=colors,
    )

    plt.savefig(
        "../../gdrive_columbia/research/massimo/fig13.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()
