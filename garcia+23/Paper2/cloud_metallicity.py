"""
SFE
"""

import sys

sys.path.append("../../")
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
from tools.cosmo import t_myr_from_z


def log_data_function(data: float, num_bins: int, bin_range: tuple):
    """
    makes metallicity/ mass function.
    makes sure that the area under the histogram is 1.
    given data, bins data into num_bins from bin_range[0] to bin_range[1]

    Parameters
    ----------
    data : float
        data to bin, usually unlogged values for a given BSC
    num_bins : int
        number of bins to use.
    bin_range : tuple
        range of the bins.

    Returns
    -------
    TYPE
        DESCRIPTION.
    counts_per_log_bin : TYPE
        DESCRIPTION.

    """
    bin_range = np.log10(bin_range)
    log_data = np.log10(data)
    count, bin_edges = np.histogram(log_data, num_bins, bin_range)
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]

    bin_ctrs = 0.5 * (left_edges + right_edges)

    # normalize with width of the bins
    counts_per_log_bin = count / (right_edges - left_edges)

    return 10**bin_ctrs, counts_per_log_bin


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
        3,
        1,
        figsize=(3.5, 9),
        # gridspec_kw={"width_ratios": [3, 3, 3]},
        dpi=400,
        sharey=True,
        sharex=True,
    )
    ax = ax.ravel()

    ax[2].tick_params(axis="both", labelbottom=False)
    plt.subplots_adjust(hspace=0.0)
    # latest_redshift = 9.12 # before second starburst
    latest_redshift = 5
    hatches = ["\\\\\\\\\\", "/////"]

    hist_ax = ax[2].inset_axes([0, -0.52, 1, 0.5], sharex=ax[2])
    hist_ax.set_ylabel(
        r"$\mathrm{d} N / {\rm d}\log \:\left({ M_{\rm cloud}}/\rm{M}_{\odot}\right)$",
        loc="bottom",
    )
    hist_ax.sharex(ax[2])
    hist_ax.set(
        xlabel=r"$M_{\rm cloud}\:\left[\mathrm{M}_{\odot}\right]$",
        yscale="log",
        ylim=(3, 5e3),
    )

    sfe_bins = np.geomspace(1, 100, 25)
    for i, r in enumerate(run_logpath):
        run_name = os.path.basename(os.path.normpath(r))
        print(os.path.join(r, "logSFC"))
        log_sfc = np.loadtxt(os.path.join(r, "logSFC"))

        redshft = log_sfc[:, 2]
        t_form = t_myr_from_z(redshft)
        delta_t_form = t_form - t_form.min()
        mask = redshft > latest_redshift
        redshft = redshft[mask]
        r_pc_cloud = log_sfc[:, 4][mask]
        m_sun_cloud = log_sfc[:, 5][mask]
        m_sun_stars = log_sfc[:, 7][mask]
        n_hydrogen = log_sfc[:, 8][mask]
        metal_zsun_cloud = log_sfc[:, 9][mask]  # metalicity is normalized to z_sun

        print(delta_t_form.max())
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

        normalize = matplotlib.colors.Normalize(vmin=250, vmax=685)
        sfe_scatter = ax[i].scatter(
            m_sun_cloud,
            metal_zsun_cloud,
            c=t_form,
            # label=simulation_name[i],
            cmap=cmr.tropical_r,
            marker=marker[i],
            edgecolors="black",
            linewidth=0.5,
            s=40,
            alpha=0.8,
            norm=normalize,
        )

        metal_ax = ax[i].inset_axes([1.02, 0, 0.4, 1], sharey=ax[i])

        metal_ax.tick_params(axis="both", labelleft=False)
        metal, dn_dlogmetal = log_data_function(metal_zsun_cloud, 20, (1.5e-4, 0.012))
        metal_ax.plot(
            dn_dlogmetal,
            metal,
            drawstyle="steps-mid",
            linewidth=2,
            color=hist_color[i],
            label=simulation_name[i],
        )
        metal_ax.fill_between(
            dn_dlogmetal,
            metal,
            step="mid",
            alpha=0.4,
            color=hist_color[i],
        )
        metal_ax.set(xscale="log", xlim=(3, 5e3))

        mass, dn_dlogmass = log_data_function(m_sun_cloud, 20, (2e2, 2e5))
        hist_ax.plot(
            mass,
            dn_dlogmass,
            drawstyle="steps-mid",
            linewidth=2,
            color=hist_color[i],
            label=simulation_name[i],
        )
        hist_ax.fill_between(
            mass,
            dn_dlogmass,
            step="mid",
            alpha=0.4,
            color=hist_color[i],
        )

        ax[i].set(
            xscale="log",
            yscale="log",
        )

        ax[i].text(
            0.95,
            0.05,
            simulation_name[i],
            ha="right",
            va="bottom",
            color="black",
            transform=ax[i].transAxes,
            # fontsize=10,
        )

    metal_ax.set(
        xlabel=r"$\mathrm{dN / d\log}$"
        "\n"
        r"$\left( Z_{\rm cloud}/ \rm{Z}_{\odot}\right)$",
    )
    ax[1].set(
        ylabel=r"$Z_{\rm cloud}\:\left[\mathrm{Z}_\odot \right]$",
    )

    cbar_ax = ax[0].inset_axes([0, 1.02, 1.42, 0.05])
    dens_bar = fig.colorbar(sfe_scatter, cax=cbar_ax, pad=0, orientation="horizontal")
    cbar_ax.minorticks_on()
    dens_bar.set_label(
        label=(r"$t_{\rm form}$ [Myr]"),
        # label=(r"$\log\:\overline{n_\mathrm{H}}\:\left[\mathrm{cm}^{-3} \right]$"),
        fontsize=10,
        labelpad=6,
    )
    dens_bar.ax.xaxis.set_label_position("top")
    dens_bar.ax.xaxis.set_ticks_position("top")
    dens_bar.ax.locator_params(nbins=12)


if __name__ == "__main__":
    cmap = matplotlib.colormaps["Dark2"]
    cmap = cmap(np.linspace(0, 1, 8))

    colors = [
        cmap[0],
        cmap[2],
        cmap[1],
    ]
    runs = [
        "../../../container_tiramisu/sim_log_files/CC-Fiducial",
        "../../../container_tiramisu/sim_log_files/fs035_ms10",
        "../../../container_tiramisu/sim_log_files/fs07_refine",
    ]
    names = [
        "VSFE",
        "low SFE",
        "high SFE",
    ]
    markers = [
        "o",
        "o",
        "o",
    ]
    calc_type = [
        "variable",
        "constant",
        "constant",
    ]

    plotting_interface(
        run_logpath=runs,
        simulation_name=names,
        hist_color=colors,
        marker=markers,
        sfe=calc_type,
    )
    plt.savefig(
        "../../../gdrive_columbia/research/massimo/paper2/CloudMetals_v2.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()
