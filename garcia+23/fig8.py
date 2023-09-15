import sys

sys.path.append("..")
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
    matching_files = list(np.take(snapshots, closest_match_idxs))

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
    x_range = (60, 5e5)
    bns = 14
    max_alpha = 10
    fig, ax = plt.subplots(
        nrows=2, ncols=2, sharex=True, sharey=True, figsize=(6, 5), dpi=300
    )
    ax = ax.ravel()
    plt.subplots_adjust(hspace=0, wspace=0)

    hatches = ["\\\\\\\\", "////"]

    for n, r in enumerate(bsc_dirs):
        # for the given times, compute the snapshot numbers
        snap_nums, _ = snapshot_from_time(pop2_dirs[n], times)
        catalog_dirs = take_snapshot(directory=bsc_dirs[n], to_take=snap_nums)
        # print(catalog_dirs)
        logsfc_data = np.loadtxt(os.path.join(log_sfc[n], "logSFC"))
        logsfc_ftime = t_myr_from_z(logsfc_data[:, 2])
        logsfc_formation_mass = logsfc_data[:, 7]
        # log_mask =
        for i, cat in enumerate(catalog_dirs):
            cat_file = glob.glob(os.path.join(cat, "profiled_*"))[0]
            catalogue = np.loadtxt(cat_file)
            clump_masses = catalogue[:, 8]  # msun
            clump_alphas = catalogue[:, 14]
            mask = (clump_masses > minimum_bsc_mass) & (clump_alphas < max_alpha)
            clump_masses = clump_masses[mask]

            # bsc mass function
            mass_bins, dn_dlogm = log_data_function(
                clump_masses, bns, x_range, func_type="counts"
            )
            mass_weight = np.where(mass_bins > 700, 0.01, 100)
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

            ax[i].plot(
                imf_bins,
                imf_counts,
                drawstyle="steps-mid",
                ls="-",
                alpha=0.8,
                lw=2,
                color=color[n],
            )
            ax[i].plot(
                imf_bins_theory,
                imf_theory,
                ls=":",
                alpha=0.8,
                lw=2,
                color=color[n],
                label=r"$ \mu = {:.2f}$, $\Sigma = {:.2f}$".format(
                    lognrml_parms[1], np.abs(lognrml_parms[2])
                ),
            )

            ax[i].plot(
                mass_bins,
                dn_dlogm,
                drawstyle="steps-mid",
                linewidth=2,
                alpha=0.8,
                color=color[n],
            )
            ax[i].fill_between(
                mass_bins,
                dn_dlogm,
                step="mid",
                facecolor="none",
                edgecolor=color[n],
                hatch=hatches[n % 2],
                label=simulation_name[n],
            )
            ax[i].plot(
                mass_bins_theory,
                dn_dlogm_theory,
                color=color[n],
                ls="--",
                label=r"${{\Gamma = {:.2f}}}$".format(pwr_law_params[0]),
            )

            ax[i].text(
                0.95,
                0.95,
                r"${{\rm t = {:.0f}\:{{\rm Myr }}}}$".format(times[i]),
                transform=ax[i].transAxes,
                fontsize=10,
                verticalalignment="top",
                horizontalalignment="right",
                clip_on=False,
            )

        for t, _ in enumerate(times):
            labelLines(
                ax[t].get_lines(),
                fontsize=8,
                color="k",
                backgroundcolor="none",
                ha="right",
                va="bottom",
                align=False,
                xvals=(1e3, 1e4),
            )

    ax[0].set(xscale="log", yscale="log", xlim=x_range, ylim=(6e-1, 500))
    sim_handls = []
    for s, name in enumerate(simulation_name):
        sim_legend = mlines.Line2D([], [], color=color[s], ls="-", label=name)
        sim_handls.append(sim_legend)

    ax[0].legend(
        bbox_to_anchor=(0.0, 1),
        loc="upper left",
        handles=sim_handls,
        fontsize=10,
        frameon=True,
    )

    cmf = mlines.Line2D([], [], color="k", ls="--", label=r"${\rm CMF}$")
    icmf = mlines.Line2D([], [], color="k", ls=":", label=r"${\rm ICMF}$")

    ax[1].legend(
        bbox_to_anchor=(0.0, 1),
        loc="upper left",
        handles=[cmf, icmf],
        fontsize=10,
        frameon=False,
        ncols=1,
    )

    fig.text(
        0.5, 0.05, r"$M_{\rm BSC}\:\left( \mathrm{M}_{\odot} \right) $", ha="center"
    )

    fig.text(0.05, 0.5, r"$N_{\rm BSC}$", va="center", rotation="vertical")


if __name__ == "__main__":
    cmap = matplotlib.colormaps["Set2"]
    cmap = cmap(np.linspace(0, 1, 8))
    colors = [
        cmap[0],
        # cmap[1],
        cmap[2],
    ]
    pop2_dirs = [
        # "../../container_tiramisu/post_processed/pop2/fs07_refine",
        "../../container_tiramisu/post_processed/pop2/CC-Fiducial",
    ]
    bsc_dirs = [
        # "../../container_tiramisu/post_processed/bsc_catalogues/fs07_refine",
        "../../container_tiramisu/post_processed/bsc_catalogues/CC-Fiducial",
    ]
    logs = [
        # "../../container_tiramisu/sim_log_files/fs07_refine",
        "../../container_tiramisu/sim_log_files/CC-Fiducial",
    ]
    names = [
        # "$f_* = 0.70$",
        r"${\rm He+19}$",
    ]
    pop2_files = [
        # filter_snapshots(pop2_dirs[0], 113, 1570, 1, snapshot_type="pop2_processed"),
        filter_snapshots(pop2_dirs[0], 304, 405, 1, snapshot_type="pop2_processed"),
    ]

    wanted_times = [480, 525, 588, 596]  # myr

    plotting_interface(
        bsc_dirs=bsc_dirs,
        pop2_dirs=pop2_files,
        times=wanted_times,
        log_sfc=logs,
        simulation_name=names,
        color=colors,
    )

    plt.savefig(
        "../../gdrive_columbia/research/massimo/fig8.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()
