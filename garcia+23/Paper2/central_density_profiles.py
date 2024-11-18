# %%
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
from labellines import labelLine, labelLines
from scipy.optimize import curve_fit
from astropy.modeling.models import Sersic1D
import matplotlib
import glob
from tools.fscanner import filter_snapshots
import os
import matplotlib.patches as patches

plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 11,
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


# %%

times = [595, 600, 627, 655]
# get all the snapshots
paths = filter_snapshots(
    "../../../container_tiramisu/post_processed/pop2/CC-Fiducial",
    153,
    466,
    1,
    snapshot_type="pop2_processed",
)
# filter them based on times=[]
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

    age_mask = (creation_time > 300) & (
        creation_time < 700
    )  # all_ages < 10  # newly made
    bulg_particles = pids[age_mask]

    def surf_dense(x, y, m):
        all_positions = np.vstack((x, y)).T
        r = np.geomspace(starting_point, prof_rad, num=30, endpoint=True)

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
        markersize=10,
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
        allx_recentered,
        allx_recentered,
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


# %%let's do it for  a bunch and take the average

paths = filter_snapshots(
    "../../../container_tiramisu/post_processed/pop2/CC-Fiducial",
    153,
    466,
    1,
    snapshot_type="pop2_processed",
)
# filter them based on times=[]

# times = np.arange(580, 800, 5)
# times = np.geomspace(580, 800, 8)
times = [580, 595, 629, 659]
double_sersic = [595, 659]
snap_nums, files = snapshot_from_time(paths, times)

path = "../../../container_tiramisu/post_processed/bsc_catalogues/CC-Fiducial"


def surf_dense(x, y, m):
    starting_point = 0.01  # start 0.01 at the resolution of the simulation
    prof_rad = 180  # the maximum radius in pc
    all_positions = np.vstack((x, y)).T
    r = np.geomspace(starting_point, prof_rad, num=30, endpoint=True)

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


def sersic_density(r, sigma0, r_eff, n):
    """
    https://www.aanda.org/articles/aa/full_html/2020/03/aa37202-19/aa37202-19.html
    """
    # Capaciolli (1989) approximation https://ui.adsabs.harvard.edu/abs/1989woga.conf..208C/abstract
    bn = 1.9992 * n - 0.3271
    prof = sigma0 * np.exp(-bn * ((r / r_eff) ** (1 / n)))
    return prof


fig, ax = plt.subplots(
    nrows=2, ncols=2, figsize=(6, 6), dpi=300, sharex=True, sharey=True
)
plt.subplots_adjust(wspace=0.0, hspace=0)
ax = ax.ravel()

inset = ax[1].inset_axes([1.02, -1, 2, 2])

for i, pop2 in enumerate(files):
    snapshot = "info_00{:}".format(int(snap_nums[i]))
    print(pop2)
    clumped_cat = glob.glob(os.path.join(os.path.join(path, snapshot), "profiled*.txt"))
    clumped_dat = np.loadtxt(clumped_cat[0])
    clump_masses = clumped_dat[:, 8]

    # the bulge is the most massive
    bulg_index = np.argmax(clump_masses)
    bulge_id = clumped_dat[:, 0][bulg_index]
    print(bulge_id)
    bulge_group_x = clumped_dat[:, 1][bulg_index]
    bulge_group_y = clumped_dat[:, 2][bulg_index]
    bulge_group_z = clumped_dat[:, 3][bulg_index]

    full_dat = np.loadtxt(pop2)

    tmyr, redshift = full_dat[0:2, 0]
    all_pop2_mass = full_dat[:, -1]
    all_ages = full_dat[:, 2]
    creation_time = tmyr - all_ages

    pids = full_dat[:, 1]
    cmap = matplotlib.colormaps["Dark2"]
    cmap = cmap(np.linspace(0, 1, 8))
    color = cmap[0]

    # mask for the age of the stars
    age_mask = (creation_time > 200) & (creation_time < 700)

    bulg_particles = pids[age_mask]

    allx = full_dat[:, 4]
    ally = full_dat[:, 5]
    allz = full_dat[:, 6]

    allx_recentered = allx - bulge_group_x
    ally_recentered = ally - bulge_group_y
    allz_recentered = allz - bulge_group_z

    star_bins = 2000
    pw = 400
    stellar_mass_dens, _, _ = np.histogram2d(
        allx_recentered,
        ally_recentered,
        bins=star_bins,
        weights=all_pop2_mass,
        range=[
            [-pw / 2, pw / 2],
            [-pw / 2, pw / 2],
        ],
    )
    bin_ctrsxy, sigma_xy, xy_err = surf_dense(
        allx_recentered[age_mask], ally_recentered[age_mask], all_pop2_mass[age_mask]
    )
    bin_ctrsxz, sigma_xz, xz_err = surf_dense(
        allx_recentered[age_mask], allz_recentered[age_mask], all_pop2_mass[age_mask]
    )
    bin_ctrsyz, sigma_yz, yz_err = surf_dense(
        ally_recentered[age_mask], allz_recentered[age_mask], all_pop2_mass[age_mask]
    )

    ax[i].errorbar(
        bin_ctrsxy,
        sigma_xy,
        yerr=xy_err,
        ecolor="gray",
        elinewidth=1,
        fmt="o",
        markerfacecolor="black",
        markeredgecolor="black",
        markersize=4,
    )

    if times[i] in double_sersic:
        print("fitting double sersic", pop2)
        inner_mask = bin_ctrsxy < 1
        outer_mask = bin_ctrsxy > 0 
        theory_x = np.geomspace(0.01, 200, 100)

        # popt_inner, pcov_inner = curve_fit(sersic_density, bin_ctrsxy[inner_mask], sigma_xy[inner_mask])

        # ax[i].plot(
        #     theory_x, sersic_density(theory_x, *popt_inner),
        #     ls="--",
        #     lw=2,
        #     color="grey",
        #     label=r"$r_{{\rm eff}} = {:.1f}$ pc""\n"r"$n = {:.1f}$".format(
        #     popt_inner[1],  popt_inner[2])
        # )
        # 10 msun/ ( 0.1 pc^2) in resolution
        # min_err = 10 / 0.1**2 
        min_err = 10 # mass resolution in msun
        xy_err[xy_err < min_err] = min_err
        popt, pcov = curve_fit(
            sersic_density, bin_ctrsxy[outer_mask], sigma_xy[outer_mask], sigma=xy_err[outer_mask]
        )
        err = np.sqrt(np.diag(pcov))
        ax[i].plot(
            theory_x,
            sersic_density(theory_x, *popt),
            lw=3,
            color=color,
            label=r"$r_{{\rm half-mass}} = {:.1f} \pm {:.1f}$ pc"
            "\n"
            r"$n = {:.1f} \pm {:.1f}$".format(popt[1], err[1], popt[2], err[2]),
        )

        # labelLines(ax[i].get_lines(), color="k", align=False, fontsize=10)
        ax[i].legend(frameon=False, loc="lower left", fontsize=10)

    else:
        popt, pcov = curve_fit(sersic_density, bin_ctrsxy, sigma_xy)
        theory_x = np.geomspace(0.01, 200, 100)
        err = np.sqrt(np.diag(pcov))
        ax[i].plot(
            theory_x,
            sersic_density(theory_x, *popt),
            lw=3,
            color=color,
            label=r"$r_{{\rm half-mass}} = {:.1f} \pm {:.1f}$ pc"
            "\n"
            r"$n = {:.1f} \pm {:.1f}$".format(popt[1], err[1], popt[2], err[2]),
        )
        ax[i].legend(frameon=False, loc="lower left", fontsize=10)

    ax[i].text(
        0.05,
        0.95,
        r"${{t = {:.0f}\:{{\rm Myr }}}}$".format(tmyr),
        transform=ax[i].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        color="black",
        clip_on=False,
    )

    ax[i].axvspan(0.001, 0.1, alpha=0.2, facecolor="k")

ax[0].set(xlabel="pc", xscale="log", yscale="log", xlim=(0.04, 120), ylim=(30, 3e4))
fig.text(
    0.04,
    0.5,
    r"$\Sigma_{\rm PopII} (r) \left[ {\rm M_\odot pc^{-2}} \right]$",
    va="center",
    rotation="vertical",
)
fig.text(0.5, 0.07, r"$r \: [{\rm pc}]$", va="center")

pxl_size = (pw / star_bins) ** 2
stelllar_range = (11, 2e3)
lum_alpha = 0.8
stellar_mass_dens = stellar_mass_dens.T
surface_dens = stellar_mass_dens / pxl_size
sdense = inset.imshow(
    surface_dens,
    cmap="cmr.ember",
    origin="lower",
    extent=[-pw / 2, pw / 2, -pw / 2, pw / 2],
    norm=LogNorm(vmin=stelllar_range[0], vmax=stelllar_range[1]),
    alpha=lum_alpha,
)
inset.scatter(0, 0, s=50, marker="+", alpha=0.9, color="cyan")

inset.set_facecolor("k")
inset.spines["bottom"].set_color("white")
inset.spines["top"].set_color("white")
inset.spines["right"].set_color("white")
inset.spines["left"].set_color("white")
inset.tick_params(axis="x", colors="w", which="both")
inset.tick_params(axis="y", colors="w", which="both")
inset.set_xticklabels([])
inset.set_yticklabels([])

# get a colorbar
with plt.style.context("dark_background"):
    cbar_ax = inset.inset_axes([0.05, 0.9, 0.5, 0.05])
    cbar = fig.colorbar(
        sdense,
        cax=cbar_ax,
        pad=0,
        orientation="horizontal",
    )
    cbar.ax.xaxis.set_tick_params(pad=-14, labelsize=10)
    cbar.set_label(
        r"$\Sigma_{\rm PopII} \left[ {\rm M_\odot pc^{-2}} \right]$", labelpad=1
    )
    cbar.ax.minorticks_on()

    # scale bar
    scale = patches.Rectangle(
        xy=(pw / 2 * -0.85, -pw / 2 * 0.80),
        width=pw / 2 * 0.5,
        height=0.020 * pw / 2,
        linewidth=0,
        edgecolor="white",
        facecolor="white",
        clip_on=False,
        alpha=0.8,
    )
    inset.text(
        pw / 2 * -0.60,
        -pw / 2 * 0.87,
        r"$\mathrm{{{:.0f}\:pc}}$".format(pw / 2 * 0.5),
        ha="center",
        va="center",
        color="white",
        fontsize=11,
    )
    inset.add_patch(scale)
 
    # timestamp
    inset.text(0.9, 0.9, r"$t \mathrm{ = 659\:Myr}$", color="white", transform=inset.transAxes, ha="right", va="top")
    inset.minorticks_off()
    
plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/nsc_densityprofile.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)

plt.show()
