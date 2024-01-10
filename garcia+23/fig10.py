"""
PDFs of cluster size and ratio of original cluster mass and bound star cluster mass
"""

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
    counts_per_log_bin : ndarray
        counts per log bin

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


def plotting_interface(bsc_dirs, pop2_dirs, times, log_sfc, simulation_name, color):
    minimum_bsc_mass = 250  # minimum solar mass
    bns = 14
    max_alpha = 5
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(5, 5.3), dpi=300)
    plt.subplots_adjust(wspace=0.25, hspace=0.35)
    ax = ax.ravel()

    hatches = ["\\\\\\\\", "////"]

    for n, r in enumerate(bsc_dirs):
        # for the given times, compute the snapshot numbers
        snap_nums, _ = snapshot_from_time(pop2_dirs[n], times)
        catalog_dirs = take_snapshot(directory=bsc_dirs[n], to_take=snap_nums)
        # print(catalog_dirs)

        logsfc_data = np.loadtxt(os.path.join(log_sfc[n], "logSFC"))

        logsfc_ftime = t_myr_from_z(logsfc_data[:, 2])
        logsfc_formation_mass = logsfc_data[:, 7]

        cat_file = glob.glob(os.path.join(catalog_dirs[0], "profiled_catalogue-*"))[0]
        z = os.path.basename(os.path.normpath(cat_file)).split("-")[-1].split(".")[0]

        catalogue = np.loadtxt(cat_file)
        clump_masses = catalogue[:, 8]  # msun
        clump_alphas = catalogue[:, 14]
        mask = (clump_masses > minimum_bsc_mass) & (clump_alphas < max_alpha)
        clump_masses = clump_masses[mask]
        clump_alphas = clump_alphas[mask]
        clump_corerad = catalogue[:, 12][mask]
        clump_sigma0 = catalogue[:, 16][mask]

        hist_val = [clump_corerad, clump_sigma0, clump_alphas]
        if n == 0:
            hist_bin_ranges = [
                np.geomspace(0.1, clump_corerad.max(), bns),
                np.geomspace(10, clump_sigma0.max(), bns),
                np.linspace(clump_alphas.min(), clump_alphas.max(), bns),
            ]
        xlabels = [
            r"$r_\mathrm{{core}}\:\mathrm{(pc)}$",
            r"$\Sigma_0\:\mathrm{\left(M_{\odot}\:pc^{-2}\right)}$",
            r"$\alpha$",
            r"$m\mathrm{_{BSC}}  \:/ \:  m_{\mathrm{SC}}$",
        ]

        for i, r in enumerate(hist_bin_ranges):
            # bsc mass function
            ax[i].hist(
                hist_val[i],
                bins=r,
                color=color[n],
                histtype="step",
                hatch=hatches[n % 2],
                alpha=1,
                linewidth=1,
                # label=r"$f_{{*}} = 0.35$",
            )
            if i != 2:
                ax[i].set(xlabel=xlabels[i], xscale="log")
            else:
                ax[i].set(xlabel=xlabels[i])

        fig.text(0.03, 0.5, r"$\mathrm{Counts}$", va="center", rotation="vertical")

    ax[3].axis("off")
    ax[2].locator_params(nbins=6)
    sim_handls = []
    for s, name in enumerate(simulation_name):
        sim_legend = mlines.Line2D([], [], color=color[s], ls="-", label=name)
        sim_handls.append(sim_legend)

    ax[3].legend(
        loc="upper left",
        handles=sim_handls,
        fontsize=12,
        frameon=False,
        title=r"$z = {:.2f}$".format(float(z.replace("_", "."))),
    )


if __name__ == "__main__":
    cmap = matplotlib.colormaps["Set2"]
    cmap = cmap(np.linspace(0, 1, 8))
    colors = [
        cmap[0],
        cmap[2],
    ]
    pop2_dirs = [
        "../../container_tiramisu/post_processed/pop2/CC-Fiducial",
        "../../container_tiramisu/post_processed/pop2/fs07_refine",
    ]
    bsc_dirs = [
        "../../container_tiramisu/post_processed/bsc_catalogues/CC-Fiducial",
        "../../container_tiramisu/post_processed/bsc_catalogues/fs07_refine",
    ]
    logs = [
        "../../container_tiramisu/sim_log_files/CC-Fiducial",
        "../../container_tiramisu/sim_log_files/fs07_refine",
    ]
    names = [
        r"${\rm He+19}$",
        "$f_* = 0.70$",
    ]
    pop2_files = [
        filter_snapshots(pop2_dirs[0], 304, 405, 1, snapshot_type="pop2_processed"),
        filter_snapshots(pop2_dirs[1], 113, 1570, 1, snapshot_type="pop2_processed"),
    ]

    wanted_times = [596]  # myr

    plotting_interface(
        bsc_dirs=bsc_dirs,
        pop2_dirs=pop2_files,
        times=wanted_times,
        log_sfc=logs,
        simulation_name=names,
        color=colors,
    )

    plt.savefig(
        "../../gdrive_columbia/research/massimo/fig10.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.show()
