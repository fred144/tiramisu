import sys

sys.path.append("../../")

import yt
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import cmasher as cmr
from matplotlib.colors import LogNorm
from astropy import units as u
from astropy import constants as const
from tools import plotstyle

from scipy.optimize import curve_fit
from astropy.modeling.models import Sersic1D
import matplotlib
import glob
from tools.fscanner import filter_snapshots
import os

plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "font.size": 16,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
    }
)


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
    filepaths = snapshots
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
    matching_files = list(np.take(filepaths, closest_match_idxs))

    return matching_snaps, matching_files


times = [576, 577, 595, 659]
paths = filter_snapshots(
    "../../../container_tiramisu/post_processed/pop2/CC-Fiducial",
    153,
    466,
    1,
    snapshot_type="pop2_processed",
)
snap_nums, files = snapshot_from_time(paths, times)

fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(8, 15), dpi=300, sharex="col")
plt.subplots_adjust(wspace=-0.03, hspace=0)
path = "../../../container_tiramisu/post_processed/bsc_catalogues/CC-Fiducial"

for i, pop2 in enumerate(files):
    snapshot = "info_00{:}".format(int(snap_nums[i]))
    clumped_cat = glob.glob(os.path.join(os.path.join(path, snapshot), "profiled*.txt"))
    clumped_dat = np.loadtxt(clumped_cat[0])
    clump_masses = clumped_dat[:, 8]
    # the bulge is the most massive
    bulge_id = clumped_dat[:, 0][np.argmax(clump_masses)]

    bulge_group_x = clumped_dat[:, 1][np.argmax(clump_masses)]
    bulge_group_y = clumped_dat[:, 2][np.argmax(clump_masses)]
    bulge_group_z = clumped_dat[:, 3][np.argmax(clump_masses)]

    # pop2 = "../../../container_tiramisu/post_processed/pop2/CC-Fiducial"
    # full_dat = np.loadtxt(glob.glob(os.path.join(pop2, "pop2-00{:}-*").format(i))[0])
    full_dat = np.loadtxt(pop2)

    tmyr, redshift = full_dat[0:2, 0]
    all_pop2_mass = full_dat[:, -1]
    all_ages = full_dat[:, 2]
    creation_time = tmyr - all_ages

    starting_point = 0.01
    prof_rad = 200
    pids = full_dat[:, 1]
    cmap = matplotlib.colormaps["Dark2"]
    cmap = cmap(np.linspace(0, 1, 8))
    color = cmap[0]

    age_mask = (creation_time > 575) & (
        creation_time < 588
    )  # all_ages < 10  # newly made
    bulg_particles = pids[age_mask]

    def surf_dense(x, y, m):
        all_positions = np.vstack((x, y)).T
        r = np.geomspace(starting_point, prof_rad, num=20, endpoint=True)

        distances = np.sqrt(np.sum(np.square(all_positions), axis=1))

        mass_per_bin, bin_edges = np.histogram(distances, bins=r, weights=m)
        count_per_bin, _ = np.histogram(distances, bins=r)
        mask = mass_per_bin > 0
        mass_per_bin = mass_per_bin[mask]

        # getting bin properties
        right_edges = bin_edges[1:]
        left_edges = bin_edges[:-1]
        bin_ctrs = 0.5 * (left_edges + right_edges)[mask]

        ring_areas = np.pi * (right_edges**2 - left_edges**2)[mask]
        surf_mass_density = mass_per_bin / ring_areas
        avg_star_masses = mass_per_bin / count_per_bin[mask]  # average star mass
        err_surf_mass_density = np.sqrt(count_per_bin[mask]) * (
            avg_star_masses / ring_areas
        )

        return bin_ctrs, surf_mass_density, err_surf_mass_density

    allx, ally, allz = full_dat[:, 4:7].T

    allx_recentered = allx - bulge_group_x
    ally_recentered = ally - bulge_group_y
    allz_recentered = allz - bulge_group_z

    bin_ctrsxy, sigma_xy, xy_err = surf_dense(
        allx_recentered[age_mask], ally_recentered[age_mask], all_pop2_mass[age_mask]
    )
    bin_ctrsxz, sigma_xz, xz_err = surf_dense(
        allx_recentered[age_mask], allz_recentered[age_mask], all_pop2_mass[age_mask]
    )
    bin_ctrsyz, sigma_yz, yz_err = surf_dense(
        ally_recentered[age_mask], allz_recentered[age_mask], all_pop2_mass[age_mask]
    )

    # without the mask
    # all_bin_ctrsxy, all_sigma_xy = surf_dense(
    #     allx_recentered, ally_recentered, all_pop2_mass
    # )
    # all_bin_ctrsxz, all_sigma_xz = surf_dense(
    #     allx_recentered, allz_recentered, all_pop2_mass
    # )
    # all_bin_ctrsyz, all_sigma_yz = surf_dense(
    #     ally_recentered, allz_recentered, all_pop2_mass
    # )

    # popt, pcov = curve_fit(sersic, bin_ctrsyz[fit_mask], sigma_yz[fit_mask])

    ax[i, 0].errorbar(
        bin_ctrsxy,
        sigma_xy,
        yerr=xy_err,
        color=color,
        label=r"$t_{\rm form} > 577 \: {\rm Myr}$",
        fmt=".",
        markersize=10
        # lw=3,
    )

    # ax[0].fill_between(
    #     bin_ctrsxy,
    #     sigma_xy - xy_err,
    #     sigma_xy + xy_err,
    #     color=color,
    #     alpha=0.2,
    #     # label=r"$t_{\rm form} > 577 \: {\rm Myr}$",
    #     # lw=2,
    # )

    # ax[0].scatter(bin_ctrsxz, sigma_xz, color=color)
    # ax[0].scatter(
    #     bin_ctrsyz, sigma_yz, color=color, label=r"$t_{\rm form} > 577 \: {\rm Myr}$"
    # )

    # ax[0].plot(all_bin_ctrsxy, all_sigma_xy, color="tab:red", alpha=0.4, lw=3)
    # ax[0].plot(all_bin_ctrsxz, all_sigma_xz, color="tab:red", alpha=0.4, lw=3)
    # ax[0].plot(all_bin_ctrsyz, all_sigma_yz, color="tab:red", label=r"all", alpha=0.4, lw=3)

    # ax.plot(
    #     big_profile[:, 0],
    #     big_profile[:, 1],
    #     color="k",
    #     label=r"$R_{\rm vir}$ cutoff",
    # )

    def sersic(r, i0, r_halflight, n, b):
        intensity = i0 * np.exp(-b * ((r / r_halflight) ** (1 / n) - 1))
        return intensity

    s1 = Sersic1D(amplitude=55, r_eff=18, n=2)
    r = np.arange(0.02, 100, 0.01)

    test_prof = sersic(r, 8e2, 1.5, 2, 1)

    # popt, pcov = curve_fit(sersic, bin_ctrsxy, sigma_xy, p0=[8e2, 1.5, 2, 1])

    ax[i, 0].plot(r, s1(r), ls="--", lw="3", color="grey", label="sersic, n = 2")
    # ax[0].plot(r, test_prof)

    ax[i, 0].set(
        xscale="log",
        yscale="log",
        # ylabel=r"Stellar Surface Density $\left[ {\rm M_\odot pc^{-2}} \right]$",
        # xlabel="Radial Distance [pc]",
        xlim=(0.04, 120),
        ylim=(2, 3e4),
    )

    ax[i, 0].axvspan(0.001, 0.1, alpha=0.2, facecolor="k")

    pw = 150
    star_bins = 2000
    pxl_size = (pw / star_bins) ** 2

    stellar_mass_dens, _, _ = np.histogram2d(
        allx,
        ally,
        bins=star_bins,
        weights=all_pop2_mass,
        range=[
            [-pw / 2, pw / 2],
            [-pw / 2, pw / 2],
        ],
    )
    stelllar_range = (20, 2e4)
    lum_alpha = 0.8
    stellar_mass_dens = stellar_mass_dens.T
    surface_dens = stellar_mass_dens / pxl_size
    sdense = ax[i, 1].imshow(
        surface_dens,
        cmap="cmr.ember",
        origin="lower",
        extent=[-pw / 2, pw / 2, -pw / 2, pw / 2],
        norm=LogNorm(vmin=stelllar_range[0], vmax=stelllar_range[1]),
        alpha=lum_alpha,
    )
    ax[i, 1].set(ylim=(-pw / 2, pw / 2), xlim=(-pw / 2, pw / 2))
    ax[i, 1].set_facecolor("k")

    ax[i, 1].spines["bottom"].set_color("white")
    ax[i, 1].spines["top"].set_color("white")
    ax[i, 1].spines["right"].set_color("white")
    ax[i, 1].spines["left"].set_color("white")
    ax[i, 1].tick_params(axis="x", colors="w", which="both")
    ax[i, 1].tick_params(axis="y", colors="w", which="both")

    ax[i, 1].text(
        0.05,
        0.95,
        r"${{\rm t = {:.0f}\:{{\rm Myr }}}}$".format(tmyr),
        transform=ax[i, 1].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        color="white",
        clip_on=False,
    )

    # ax[1].scatter(allx[~age_mask], ally[~age_mask], s=1, alpha=0.01, color="grey")
    # ax[1].scatter(allx[age_mask], ally[age_mask], s=1, alpha=0.01, color="red")
    ax[i, 1].scatter(
        bulge_group_x, bulge_group_y, s=50, marker="+", alpha=0.9, color="cyan"
    )

ax[0, 0].legend(frameon=False)

ax[3, 0].set(xlabel="r [pc]")
[t.set_color("black") for t in ax[3, 1].xaxis.get_ticklabels()]
ax[3, 1].set(xlabel="pc")
fig.text(
    0.03,
    0.5,
    r"$\Sigma_{\rm PopII} \left[ {\rm M_\odot pc^{-2}} \right]$",
    va="center",
    rotation="vertical",
)

plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/bulge_profile.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)

plt.show()
