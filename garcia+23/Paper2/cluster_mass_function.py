"""
Cluster mass functions
"""

import sys

sys.path.append("../../")
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
from tools.fscanner import filter_snapshots
from tools.cosmo import t_myr_from_z
import matplotlib.lines as mlines
import glob
import matplotlib
from tools import plotstyle
from labellines import labelLine, labelLines
from astropy import units as u
from astropy import constants as const

# from modules.match_t_sims import find_matching_time, get_snapshots
from scipy.optimize import curve_fit


def pwr_law(x, a, coeff):
    return coeff * x**a


def gauss(x, amp, mean, sigma):
    return amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def lgauss(x, amp, mean, sigma):
    return np.log10(amp * np.exp(-0.5 * ((x - mean) / sigma) ** 2))


def log_data_function(
    data: float, num_bins: int, bin_range: tuple, func_type: str = "PDF"
):
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
    func_type : str
        if "PDF", returns the probability distribution function, if "counts", returns a
        more traditional histogram.

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

    if func_type == "PDF":
        return 10**bin_ctrs, counts_per_log_bin
    elif func_type == "counts":
        return 10**bin_ctrs, count
    else:
        print("Type not valid")
        return ValueError


def snapshot_from_time(snapshots, time, split_sym="-", snap_idx=1, time_idx=2):
    """
    Given a list of postprocesed pop ii snapshot files, get the corresponding time

    Parameters
    ----------
    time : TYPE
        DESCRIPTION.
    snapshots : TYPE
        DESCRIPTION.
    split_sym : TYPE, optional
        DESCRIPTION. The default is "-".
    snap_idx : TYPE, optional
        DESCRIPTION. The default is 1.
    time_idx : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    None.

    """
    uni_age = []
    snapshot = []
    snap_list = snapshots
    for f in snapshots:
        name = os.path.basename(os.path.normpath(f))
        sn_numbers = float(name.split(split_sym)[snap_idx])
        tmyr = float(name.split(split_sym)[time_idx].replace("_", "."))

        uni_age.append(tmyr)
        snapshot.append(sn_numbers)

    uni_age = np.array([uni_age])
    snapshots = np.array(snapshot)
    residuals = np.abs(uni_age - np.array(time)[:, np.newaxis])
    closest_match_idxs = np.argmin(residuals, axis=1).astype(int)

    matching_snaps = snapshots[closest_match_idxs]
    matching_files = list(np.take(snap_list, closest_match_idxs))

    return matching_snaps, matching_files


def take_snapshot(directory, to_take):
    wanted_dirs = []
    files = sorted(os.listdir(directory))
    for f in files:
        if "info_" in f:
            snap_num = int(f.split("_")[-1])
            # print(snap_num)
            if np.isin(snap_num, to_take):
                wanted_dirs.append(f)

    return [os.path.join(directory, file) for file in wanted_dirs]


def fit_mfunc(func, xdata, ydata, weights=None):
    params, pcov = curve_fit(func, xdata, ydata, sigma=weights, absolute_sigma=True)
    theory_x = np.linspace(xdata.min(), xdata.max(), 100)
    theory_y = func(theory_x, *params)

    return params, theory_x, theory_y


def plotting_interface(bsc_dirs, pop2_dirs, times, log_sfc, simulation_name, color):
    minimum_bsc_mass = 250  # minimum solar mass
    maximum_bsc_mass = 2e5
    x_range = (60, 5e6)
    bns = 15
    max_alpha = 10
    fig, ax = plt.subplots(
        nrows=2, ncols=4, sharex=True, sharey=True, figsize=(11, 6), dpi=300
    )
    # ax = ax.ravel()
    plt.subplots_adjust(hspace=0, wspace=0)

    hatches = ["\\\\\\\\", "////"]
    # print(times)
    for n, r in enumerate(bsc_dirs):
        # for the given times, compute the snapshot numbers
        snap_nums, pop2_dat = snapshot_from_time(pop2_dirs[n], times)

        catalog_dirs = take_snapshot(directory=bsc_dirs[n], to_take=snap_nums)

        logsfc_data = np.loadtxt(os.path.join(log_sfc[n], "logSFC"))
        logsfc_ftime = t_myr_from_z(logsfc_data[:, 2])
        logsfc_formation_mass = logsfc_data[:, 7]
        # log_mask =
        print(catalog_dirs)

        for i, cat in enumerate(catalog_dirs):  # loop over the times
            actual_time = float(
                os.path.basename(pop2_dat[i]).split("-")[2].replace("_", ".")
            )

            pop2_masses = np.sum(np.loadtxt(pop2_dat[i])[:, -1])

            # clumps we were succesful in profiling
            cat_file = glob.glob(os.path.join(cat, "profiled_*"))[0]
            catalogue = np.loadtxt(cat_file)
            # print(catalogue.shape())
            clump_masses = catalogue[:, 8]  # msun
            clump_alphas = catalogue[:, 14]
            clump_ids = catalogue[:, 0]
            sx, sy, sz = catalogue[:, 9:12].T * 1e3 * (u.m / u.s)
            sigma_3d_squared = sx**2 + sy**2 + sz**2  # m/s this is 1D

            r_half = catalogue[:, 21] * u.parsec  # half light radius
            r = catalogue[:, 4] * u.parsec
            r_core = catalogue[:, 12] * u.parsec  # pc
            half_masses = 0.5 * clump_masses * u.Msun
            m_core = catalogue[:, -1]

            print(
                actual_time,
                pop2_masses,
                clump_masses.max(),
                clump_masses.max() / pop2_masses,
            )

            ke_avg_perparticle = (1 / 2) * sigma_3d_squared * half_masses.to(u.kg)
            pot_energy = (
                (3 / 5) * (const.G * half_masses.to(u.kg) ** 2) / r_half.to(u.m)
            )
            # vir_parameter = (5 * sigma_3d_squared * r_half.to(u.m)) / (
            #     const.G * (half_masses).to(u.kg)
            # )

            vir_parameter = 2 * ke_avg_perparticle / pot_energy

            # plt.figure()
            # plt.hist(vir_parameter)

            mask_all = (
                (clump_masses > minimum_bsc_mass)
                & (clump_alphas < max_alpha)
                & (clump_masses < maximum_bsc_mass)
            )

            mask_vir = (vir_parameter <= 10) & (clump_masses < maximum_bsc_mass)

            for h, mask in enumerate([mask_all, mask_vir]):
                bsc_masses = clump_masses[mask]

                bulge_masses = clump_masses[clump_masses > maximum_bsc_mass]

                # central_mass =
                # bsc mass function
                # print(i)
                mass_bins, dn_dlogm = log_data_function(
                    bsc_masses, bns, x_range, func_type="counts"
                )
                bulge_mass_bins, bulge_dn_dlogm = log_data_function(
                    bulge_masses, bns, x_range, func_type="counts"
                )

                mass_weight = np.where(mass_bins > 700, 0.01, 100)
                # mass_weight = np.where(mass_bins > 700, 1, 1)
                pwr_law_params, mass_bins_theory, dn_dlogm_theory = fit_mfunc(
                    pwr_law, mass_bins, dn_dlogm, weights=mass_weight
                )

                # at formation mass function
                logsfc_mask = logsfc_ftime < times[i]
                imf_bins, imf_counts = log_data_function(
                    logsfc_formation_mass[logsfc_mask], bns, x_range, func_type="counts"
                )
                lognrml_parms, _, _ = fit_mfunc(gauss, np.log10(imf_bins), imf_counts)
                imf_bins_theory = np.geomspace(imf_bins.min(), imf_bins.max(), 100)
                imf_theory = gauss(np.log10(imf_bins_theory), *lognrml_parms)

                ax[h, i].plot(
                    bulge_mass_bins,
                    bulge_dn_dlogm,
                    drawstyle="steps-mid",
                    linewidth=2,
                    alpha=1,
                    color="crimson",
                )
                ax[h, i].fill_between(
                    bulge_mass_bins,
                    bulge_dn_dlogm,
                    step="mid",
                    facecolor="none",
                    edgecolor="crimson",
                    hatch=hatches[n % 2],
                    # label=simulation_name[n],
                )

                ax[h, i].plot(
                    imf_bins,
                    imf_counts,
                    drawstyle="steps-mid",
                    alpha=1,
                    lw=2,
                    color=color[n],
                )
                ax[h, i].fill_between(
                    imf_bins,
                    imf_counts,
                    step="mid",
                    facecolor=color[n],
                    edgecolor=color[n],
                    alpha=0.2,
                    zorder=-1
                    # label=simulation_name[n],
                )

                ax[h, i].plot(
                    imf_bins_theory,
                    imf_theory,
                    ls=":",
                    alpha=1,
                    lw=2,
                    color="k",
                    # label=r"$ \mu = {:.1f}$, $\Sigma = {:.1f}$".format(
                    #     lognrml_parms[1], np.abs(lognrml_parms[2])
                    # ),
                )
                if h == 0:
                    ax[h, i].text(
                        0.55,
                        0.6,
                        r"$ \mu = {:.1f}, \sigma = {:.1f}$".format(
                            lognrml_parms[1], np.abs(lognrml_parms[2])
                        ),
                        transform=ax[h, i].transAxes,
                        fontsize=8,
                        ha="left",
                    )
                    ax[h, i].text(
                        0.95,
                        0.95,
                        r"${{\rm t = {:.0f}\:{{\rm Myr }}}}$".format(actual_time),
                        transform=ax[h, i].transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        horizontalalignment="right",
                        clip_on=False,
                    )

                ax[h, i].plot(
                    mass_bins,
                    dn_dlogm,
                    drawstyle="steps-mid",
                    linewidth=2,
                    alpha=1,
                    color=color[n],
                )
                ax[h, i].fill_between(
                    mass_bins,
                    dn_dlogm,
                    step="mid",
                    facecolor="none",
                    edgecolor=color[n],
                    hatch=hatches[n % 2],
                    # label=simulation_name[n],
                )
                ax[h, i].plot(
                    mass_bins_theory,
                    dn_dlogm_theory,
                    color="k",
                    ls="--",
                    label=r"${{\Gamma = {:.1f}}}$".format(pwr_law_params[0]),
                )
                ax[h, i].text(
                    0.5,
                    0.45,
                    r"${{\Gamma = {:.1f}}}$".format(pwr_law_params[0]),
                    transform=ax[h, i].transAxes,
                    fontsize=8,
                    ha="right",
                )

                ax[h, i].set(
                    xscale="log", yscale="log", xlim=(80, 8e5), ylim=(6e-1, 500)
                )

            # if i == 3:
            #     yoff = -500
            # else:
            #     yoff= -10

            # labelLines(
            #     ax[i].get_lines(),
            #     zorder=2.5,
            #     fontsize=8,
            #     # color="k",
            #     backgroundcolor="none",
            #     # ha="right",
            #     # va="bottom",
            #     align=False,
            #     xvals=(100, 2e4),
            #     yoffsets=yoff,
            # )

        # for t, _ in enumerate(times):

        #     labelLines(
        #         ax[t].get_lines(),
        #         fontsize=8,
        #         color="k",
        #         backgroundcolor="none",
        #         ha="right",
        #         va="bottom",
        #         align=True,
        #         xvals=(1000, 1e4),
        #         yoffsets=-10,
        #     )

    # sim_handls = []
    # for s, name in enumerate(simulation_name):
    #     sim_legend = mlines.Line2D([], [], color=color[s], ls="-", label=name)
    #     sim_handls.append(sim_legend)

    # ax[0].legend(
    #     bbox_to_anchor=(0.0, 1),
    #     loc="upper left",
    #     handles=sim_handls,
    #     fontsize=10,
    #     frameon=True,
    # )

    cmf = mlines.Line2D([], [], color="k", ls="--", label=r"${\rm CMF}$")
    icmf = mlines.Line2D([], [], color="k", ls=":", label=r"${\rm ICMF}$")

    ax[0, 0].legend(
        bbox_to_anchor=(0.0, 1),
        loc="upper left",
        handles=[cmf, icmf],
        fontsize=10,
        frameon=False,
        ncols=1,
    )
    # ax[0].ax.locator_params(nbins=12)
    ax[0, 0].minorticks_on()
    fig.text(
        0.5,
        0.05,
        r"$M_{\rm star \: cluster}\:\left[ \mathrm{M}_{\odot} \right] $",
        ha="center",
    )

    fig.text(0.09, 0.5, r"$N_{\rm star \: cluster}$", va="center", rotation="vertical")

    ax[1, 0].text(
        0.05,
        0.95,
        r"$\alpha_{\rm vir} \leq 10$, bound",
        transform=ax[1, 0].transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        clip_on=False,
    )

    # ax[0, 0].set(ylabel=r"$N_{\rm star \: cluster}\:$")


if __name__ == "__main__":
    cmap = matplotlib.colormaps["Set2"]
    cmap = cmap(np.linspace(0, 1, 8))
    colors = [
        # cmap[1],
        # cmap[2],
        cmap[0],
    ]
    pop2_dirs = [
        # "../../../container_tiramisu/post_processed/pop2/fs07_refine",
        # "../../../container_tiramisu/post_processed/pop2/fs035_ms10",
        "../../../container_tiramisu/post_processed/pop2/CC-Fiducial",
    ]
    bsc_dirs = [
        # "../../../container_tiramisu/post_processed/bsc_catalogues/fs07_refine",
        # "../../../container_tiramisu/post_processed/bsc_catalogues/fs035_ms10",
        "../../../container_tiramisu/post_processed/bsc_catalogues/CC-Fiducial",
    ]
    logs = [
        # "../../../container_tiramisu/sim_log_files/fs07_refine",
        # "../../../container_tiramisu/sim_log_files/fs035_ms10",
        "../../../container_tiramisu/sim_log_files/CC-Fiducial",
    ]
    names = [
        # "$f_* = 0.70$",
        # r"${\rm He+19}$",
    ]
    pop2_files = [
        # filter_snapshots(pop2_dirs[0], 113, 1570, 1, snapshot_type="pop2_processed"),
        # filter_snapshots(pop2_dirs[0], 140, 1606, 1, snapshot_type="pop2_processed"),
        filter_snapshots(pop2_dirs[0], 304, 466, 1, snapshot_type="pop2_processed"),
    ]
    # print(pop2_files)
    wanted_times = [495, 512, 595, 700]  # myr

    plotting_interface(
        bsc_dirs=bsc_dirs,
        pop2_dirs=pop2_files,
        times=wanted_times,
        log_sfc=logs,
        simulation_name=names,
        color=colors,
    )

    plt.savefig(
        "../../../gdrive_columbia/research/massimo/paper2/cluster_mass_function_virialized.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()
