"""
SFE
"""

import sys

sys.path.append("..")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.optimize import curve_fit
import matplotlib.lines as mlines
import matplotlib
import os
from tools import plotstyle
from labellines import labelLines
import cmasher as cmr


def star_formation_efficiency(n_h: float, mass: float, metallicity: float):
    """
    derive expected SFE for constant SFE runs

    Parameters
    ----------
    n_h : float
        mean hydrogen number density for a cloud
    mass : float
        mass of the cloud
    metallicity : float
        metalicity of the cloud

    Returns
    -------
    efficiency : TYPE
        gas-to-star conversion efficiency

    """
    # n_crit=n_H*0+1.e3/12.0
    # with shell 6 times less dense. At t_relax 12 times less dense
    # f_s reduced by 5 for low met.
    # f_s increases with stronger B-field
    n_crit = 100.0  # (n_h * 0 + 1.0e3) / (4.0 * 2)

    efficiency = (
        (2.0e-2 / 5.0)
        * (mass / 1.0e4) ** 0.4
        * (n_h / n_crit + 1.0) ** (0.91)
        * (metallicity / 1e-3) ** 0.25
    )
    # f_s=4.e-3*(mass/1.e4)**0.4*(n_H/n_crit+1.0)**(0.91)
    efficiency = np.where(efficiency < 0.9, efficiency, 0.9)
    return efficiency


def plotting_interface(run_logpath, simulation_name, marker, hist_color, sfe: str):
    """


    Parameters
    ----------
    run_logpath : str, list
        path to the logfiles, can be obtained by running sim_scraper.py
    simulation_name : str, list
        what to name the simulations
    marker : TYPE
        marker shape
    hist_color : TYPE
        color
    sfe : str
        type of sim run, "variable" SFE or "constant"

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    None.

    """
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(5, 4),
        gridspec_kw={"width_ratios": [3, 1]},
        dpi=400,
        sharey=True,
    )

    # hist_ax_right = ax.inset_axes([1.02, 0, 0.30, 1], sharey=ax)
    ax[1].set_xlabel(r"$\mathrm{PDF (SFE)}$")
    ax[1].tick_params(axis="both", labelleft=False, labelsize=6)
    plt.subplots_adjust(wspace=0.05)
    latest_redshift = 5
    hatches = ["\\\\\\\\\\", "/////"]

    sfe_bins = np.geomspace(1, 100, 25)
    for i, r in enumerate(run_logpath):
        run_name = os.path.basename(os.path.normpath(r))
        print(os.path.join(r, "logSFC"))
        log_sfc = np.loadtxt(os.path.join(r, "logSFC"))

        redshft = log_sfc[:, 2]
        mask = redshft > latest_redshift
        redshft = redshft[mask]
        r_pc_cloud = log_sfc[:, 4][mask]
        m_sun_cloud = log_sfc[:, 5][mask]
        m_sun_stars = log_sfc[:, 7][mask]
        n_hydrogen = log_sfc[:, 8][mask]
        metal_zsun_cloud = log_sfc[:, 9][mask]  # metalicity is normalized to z_sun

        if sfe[i] == "constant":
            sfe_val = (
                star_formation_efficiency(n_hydrogen, m_sun_cloud, metal_zsun_cloud)
                * 100
            )
        elif sfe[i] == "variable":
            sfe_val = (m_sun_stars / m_sun_cloud) * 100
        else:
            print("sfe is ether constant or variable")
            raise ValueError

        sfe_scatter = ax[0].scatter(
            m_sun_cloud,
            sfe_val,
            c=np.log10(n_hydrogen),
            label=simulation_name[i],
            cmap=cmr.tropical,
            marker=marker[i],
            edgecolors="black",
            linewidth=0.5,
            s=40,
            alpha=0.7,
        )
        print(np.max(sfe_val))
        ax[1].hist(
            sfe_val,
            bins=sfe_bins,
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

    ax[1].legend(fontsize=8, loc="upper center", frameon=False)
    cbar_ax = ax[0].inset_axes([0, 1.02, 1, 0.05])
    dens_bar = fig.colorbar(sfe_scatter, cax=cbar_ax, pad=0, orientation="horizontal")

    dens_bar.set_label(
        label=(r"$\log_{10}\:\overline{n_\mathrm{H}}\:\left[\mathrm{cm}^{-3} \right]$"),
        fontsize=12,
        labelpad=6,
    )
    dens_bar.ax.xaxis.set_label_position("top")
    dens_bar.ax.xaxis.set_ticks_position("top")
    dens_bar.ax.locator_params(nbins=6)

    ax[0].set(
        xlabel=r"Cloud Mass $\left[\mathrm{M}_{\odot}\right]$",
        ylabel=r"$\mathrm{SFE}\:\left[\% \right]$",
        xscale="log",
        yscale="log",
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

    ax[0].legend(loc="lower right", handles=h, frameon=False)

    ax[0].set_ylim(top=120)
    xmin, _ = ax[0].get_xlim()
    ax[0].axhline(y=80, color="grey", ls="--", zorder=-1)
    ax[0].annotate("$80 \%$", (xmin * 1.2, 65), color="grey")

    ax[0].axhline(y=10, color="grey", ls="--", zorder=-1)
    ax[0].annotate("$10 \%$", (xmin * 1.2, 8), color="grey")


if __name__ == "__main__":
    cmap = matplotlib.colormaps["Dark2"]
    cmap = cmap(np.linspace(0, 1, 8))

    colors = [
        # cmap[0],
        cmap[0],
        cmap[1],
    ]
    runs = [
        "../../container_tiramisu/sim_log_files/fs07_refine",
        # "../../container_tiramisu/sim_log_files/fs035_ms10",
        "../../container_tiramisu/sim_log_files/CC-Fiducial",
    ]
    names = [
        "high SFE",
        # "$f_* = 0.35$",
        "VSFE",
    ]
    markers = [
        "o",
        # "P",
        "v",
    ]
    calc_type = [
        "constant",
        # "constant",
        "variable",
    ]

    plotting_interface(
        run_logpath=runs,
        simulation_name=names,
        hist_color=colors,
        marker=markers,
        sfe=calc_type,
    )
    plt.savefig(
        "../../gdrive_columbia/research/massimo/paper2/SFE.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()
