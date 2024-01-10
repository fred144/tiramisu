"""
This shows the star forming cloud property distributions
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


def gauss(x, amp, mean, sigma):
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def bimodal(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
    return amp1 * np.exp(-0.5 * ((x - mean1) / sigma1) ** 2) + amp2 * np.exp(
        -0.5 * ((x - mean2) / sigma2) ** 2
    )


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


def plotting_interface(run_logpath, simulation_name, hist_color):
    bns = 15
    mass_xrange = (2e2, 2e5)
    metal_xrange = (1.5e-4, 0.015)
    radius_xrange = np.arange(0.6, 4, 0.3)
    latest_redshift = 5
    youngest_cloud_redshift = []

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3), dpi=500)
    plt.subplots_adjust(hspace=0, wspace=0.32)
    for i, r in enumerate(run_logpath):
        run_name = os.path.basename(os.path.normpath(r))
        print(os.path.join(r, "logSFC"))
        log_sfc = np.loadtxt(os.path.join(r, "logSFC"))

        redshft = log_sfc[:, 2]
        mask = redshft > latest_redshift
        redshft = redshft[mask]
        r_pc_cloud = log_sfc[:, 4][mask]
        m_sun_cloud = log_sfc[:, 5][mask]
        metal_zsun_cloud = log_sfc[:, 9][mask]  # metalicity is normalized to z_sun
        youngest_cloud = np.min(redshft)
        youngest_cloud_redshift.append(youngest_cloud)

        #                           cloud mass function
        mass, dn_dlogm = log_data_function(m_sun_cloud, bns, mass_xrange)
        #                           fit with a log normal
        params, _ = curve_fit(
            f=gauss, xdata=np.nan_to_num(np.log10(mass), neginf=0), ydata=dn_dlogm
        )
        mass_xfit = np.log10(np.geomspace(mass.min(), mass.max(), 100))
        mass_yfit = gauss(mass_xfit, *params)
        #                           cloud metallicity function
        metal, dn_dlogmetal = log_data_function(metal_zsun_cloud, bns, metal_xrange)
        #                           cloud radius counts
        count, bin_edges = np.histogram(r_pc_cloud, bins=radius_xrange, density=True)
        right_edges = bin_edges[1:]
        left_edges = bin_edges[:-1]
        bin_ctrs = 0.5 * (left_edges + right_edges)

        ax[0].plot(
            mass,
            dn_dlogm,
            drawstyle="steps-mid",
            linewidth=2.5,
            alpha=0.8,
            color=hist_color[i],
        )
        ax[0].fill_between(mass, dn_dlogm, step="mid", alpha=0.4, color=hist_color[i])
        # plot the fits
        ax[0].plot(
            10**mass_xfit,
            mass_yfit,
            ls="--",
            linewidth=2,
            alpha=1,
            color=hist_color[i],
            label=(r"$ \log_{{10}} ( \mu = {:.3f}, \Sigma =  {:.3f} )$").format(
                params[1], np.abs(params[2])
            ),
        )

        ax[1].plot(
            metal,
            dn_dlogmetal,
            drawstyle="steps-mid",
            linewidth=2.5,
            alpha=0.8,
            color=hist_color[i],
            label=(simulation_name[i] + r" ${{(z={:.2f})}}$".format(youngest_cloud)),
        )
        ax[1].fill_between(
            metal,
            dn_dlogmetal,
            step="mid",
            alpha=0.4,
            color=hist_color[i],
        )

        ax[2].plot(
            bin_ctrs,
            count,
            drawstyle="steps-mid",
            linewidth=2.5,
            alpha=0.8,
            color=hist_color[i],
            label=r"$\mu = {:.2f}$".format(np.mean(r_pc_cloud)),
        )
        ax[2].fill_between(
            bin_ctrs,
            count,
            step="mid",
            alpha=0.4,
            color=hist_color[i],
        )

    ax[0].set(
        ylabel=r"$\mathrm{dN / d\log} \:\:\left(\mathrm{M_{MC}}/\mathrm{M}_{\odot}\right)$",
        xscale="log",
        yscale="log",
        ylim=(5, np.max(dn_dlogmetal) * 8),
    )
    ax[0].set_xlabel(
        xlabel=r"$  \mathrm{M_{MC}}\:\:\left( \mathrm{M}_{\odot} \right) $", labelpad=2
    )
    ax[0].legend(
        # title=r"$\log_{{10}}\:(\mu,\:\Sigma)$",
        loc="upper center",
        fontsize=10,
        frameon=False,
    )

    ax[1].set(
        ylabel=r"$\mathrm{dN / d\log} \:\: \left(\mathrm{Z_{MC}}/\mathrm{Z}_{\odot}\right )$",
        xscale="log",
        yscale="log",
        ylim=(5, np.max(dn_dlogmetal) * 10),
    )

    ax[1].set_xlabel(
        xlabel=r"$\mathrm{Z_{MC}} \:\:  \left( \mathrm{Z}_{\odot} \right) $", labelpad=2
    )
    ax[1].legend(fontsize=10, loc="upper center")
    ax[2].set(xlabel=r"$\mathrm{R_{MC} \: (pc)}$", ylabel=r"$\mathrm{PDF \: (R_{MC})}$")
    ax[2].set_ylim(bottom=0)
    ax[2].legend(fontsize=10, frameon=False)
    # labelLines(ax[0].get_lines(), align=True, fontsize=8, xvals=[2e4, 3e4], color="k")


if __name__ == "__main__":
    cmap = matplotlib.colormaps["Set2"]
    cmap = cmap(np.linspace(0, 1, 8))

    runs = [
        "../../container_tiramisu/sim_log_files/fs07_refine",
        # "../../container_tiramisu/sim_log_files/fs035_ms10",
        "../../container_tiramisu/sim_log_files/CC-Fiducial",
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
        hist_color=colors,
    )

    plt.savefig(
        "../../gdrive_columbia/research/massimo/fig4.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()
