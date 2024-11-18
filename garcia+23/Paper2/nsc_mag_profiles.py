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

from scipy.optimize import curve_fit
from astropy.modeling.models import Sersic1D
import matplotlib
import glob
from tools.fscanner import filter_snapshots
import os
from astropy.modeling import models, fitting
from labellines import labelLine, labelLines


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
        'xtick.minor.visible': True,
        "ytick.minor.visible": True,
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


times = [595, 627, 659]
paths = filter_snapshots(
    "../../../container_tiramisu/post_processed/pop2/CC-Fiducial",
    153,
    466,
    1,
    snapshot_type="pop2_processed",
)
snap_nums, files = snapshot_from_time(paths, times)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 6), dpi=300, sharex=True)

ax = ax.T
plt.subplots_adjust(wspace=0.12, hspace=0.04)
path = "../../../container_tiramisu/post_processed/bsc_catalogues/CC-Fiducial"

inner_rad = [0.4, 50, 1e2]
for i, pop2 in enumerate(files):
    snapshot = "info_00{:}".format(int(snap_nums[i]))
    clumped_cat = glob.glob(os.path.join(os.path.join(path, snapshot), "profiled*.txt"))
    clumped_dat = np.loadtxt(clumped_cat[0])
    clump_masses = clumped_dat[:, 8]
    # the bulge/ nuclear star cluster is the most massive
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
    lum_erg_per_s_ang = 10 ** full_dat[:, 3]
    creation_time = tmyr - all_ages

    starting_point = 0.01  # where the profile starts pc, 0.01 is the smallest value
    prof_rad = 200  # pc, where the profile ends
    pids = full_dat[:, 1]
    cmap = matplotlib.colormaps["Dark2"]
    cmap = cmap(np.linspace(0, 1, 8))
    color = cmap[0]

    age_mask = (creation_time > 570) & (creation_time < 659)
    # age_mask = creation_time < 800

    bulg_particles = pids[age_mask]

    allx, ally, allz = full_dat[:, 4:7].T

    allx_recentered = allx - bulge_group_x
    ally_recentered = ally - bulge_group_y
    allz_recentered = allz - bulge_group_z

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

    def surf_brightness(x, y, lums, redshift, highest_mag=40):
        all_positions = np.vstack((x, y)).T
        r = np.geomspace(starting_point, prof_rad, num=20, endpoint=True)
        distances = np.sqrt(np.sum(np.square(all_positions), axis=1))

        lum_per_bin, bin_edges = np.histogram(distances, bins=r, weights=lums)
        count_per_bin, _ = np.histogram(distances, bins=r)

        mask = lum_per_bin > 0
        lum_per_bin = lum_per_bin[mask]

        # getting bin properties
        right_edges = bin_edges[1:]
        left_edges = bin_edges[:-1]
        bin_ctrs = 0.5 * (left_edges + right_edges)[mask]
        ring_areas = np.pi * (right_edges**2 - left_edges**2)[mask]

        surface_brightness = lum_per_bin / ring_areas  # now has units of erg/s/AA/pc^2
        avg_star_lums = lum_per_bin / count_per_bin[mask]
        err_poisson = np.sqrt(count_per_bin[mask]) * (avg_star_lums / ring_areas)

        mu_ab = (
            23.4
            - 2.5
            * np.log10(
                surface_brightness / 1e36,
                out=np.full_like(surface_brightness, np.nan),
                where=(surface_brightness != 0),
            )
            + 10 * np.log10((1 + redshift) / 10)
        )  # mag/arcsec^2

        mu_ab_err = (
            23.4
            - 2.5 * np.log10(err_poisson / 1e36)
            + 10 * np.log10((1 + redshift) / 10)
        )

        print(np.nanmax(mu_ab))
        mu_ab[np.isnan(mu_ab)] = highest_mag  # for 0, just put to the floor fo 40 mags
        return bin_ctrs, mu_ab,mu_ab_err

    bin_ctrsxy, sigma_xy, xy_err = surf_dense(
        allx_recentered[age_mask], ally_recentered[age_mask], all_pop2_mass[age_mask]
    )
    bin_ctrsxz, sigma_xz, xz_err = surf_dense(
        allx_recentered[age_mask], allz_recentered[age_mask], all_pop2_mass[age_mask]
    )
    bin_ctrsyz, sigma_yz, yz_err = surf_dense(
        ally_recentered[age_mask], allz_recentered[age_mask], all_pop2_mass[age_mask]
    )

    r, muab_xy, muab_xy_err = surf_brightness(
        allx_recentered[age_mask],
        ally_recentered[age_mask],
        lum_erg_per_s_ang[age_mask],
        redshift,
    )

    ax[i, 0].errorbar(
        bin_ctrsxy,
        sigma_xy,
        yerr=xy_err,
        color=color,
        # label=r"$t_{\rm form} > 577 \: {\rm Myr}$",
        fmt=".",
        markersize=10,
    )

    ax[i, 1].errorbar(
        r, muab_xy, color=color,markersize=10, fmt=".",
    )
    ax[i, 1].set(xscale="log", ylim=(20.5, 32.5))
    ax[i, 1].invert_yaxis()

    def sersic(r, i0, r_halflight, n):
        """
        http://burro.case.edu/Academics/Astr323/Lectures/GalaxyPopulations.pdf
        """
        b= 1.9992*n - 0.3271 
        mu = i0 + (2.5 * b / np.log(10)) * ((r / r_halflight) ** (1 / n) - 1)
        return mu

    nsc_mask = r <= inner_rad[i]
    popt_nsc, pcov = curve_fit(
        sersic, r[nsc_mask], muab_xy[nsc_mask], p0=[22, 1.5, 2]
    )
    r_theory = np.arange(0.02, 100, 0.01)
    ax[i, 1].plot(
        r_theory,
        sersic(r_theory, *popt_nsc),
        ls="--",
        lw=2,
        color="grey",
        label=r"$r_{{\rm half}} = {:.1f}$ pc""\n"r"$n = {:.1f}$".format(
            popt_nsc[1], popt_nsc[2]
    ))
   
    if i == 0:
        gal_mask = r > 0.5
        popt_nsc, pcov = curve_fit(
            sersic, r[gal_mask ], muab_xy[gal_mask ], p0=[22, 1.5, 2]
        )
        ax[i, 1].plot(
            r_theory,
            sersic(r_theory, *popt_nsc),
            ls="--",
            lw=2,
            color="grey",
             label=r"$r_{{\rm half}} = {:.1f}$ pc""\n"r"$n = {:.1f}$".format(
            popt_nsc[1], popt_nsc[2])
        )
        labelLines(ax[i, 1].get_lines(), xvals=(5, 10), color="k",align=False, fontsize=11)
    else:
        ax[i, 1].legend(frameon=False)
    
    
    ax[i, 0].set(xscale="log", yscale="log", ylim=(5, 4e4))

    ax[i, 0].axvspan(0.001, 0.1, alpha=0.2, facecolor="k")
    ax[i, 1].axvspan(0.001, 0.1, alpha=0.2, facecolor="k")

    ax[i, 0].text(
        0.05,
        0.95,
        r"${{\rm t = {:.0f}\:{{\rm Myr }}}}$".format(tmyr),
        transform=ax[i, 0].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        # color="white",
        clip_on=False,
    )


ax[0, 0].legend(frameon=False)
ax[1, 1].set(xlabel="$r$ [pc]", xlim=(0.02, 200))
ax[0, 0].set(
    ylabel=r"$\log \: \Sigma_{\rm PopII} \left[ {\rm M_\odot pc^{-2}} \right]$",
)
ax[0, 1].set_ylabel(r"$\mathrm{UV}\: \mu_{\rm AB} \: {\rm [mag \: arcsec^{-2}]}$")
plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/nuclear_cluster_profiles.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.01,
)
plt.show()
#%%

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3), dpi=300, sharex=True, sharey=True)

plt.subplots_adjust(wspace=0.0, hspace=0.04)
path = "../../../container_tiramisu/post_processed/bsc_catalogues/CC-Fiducial"

inner_rad = [0.4, 50, 1e2]

for i, pop2 in enumerate(files):
    snapshot = "info_00{:}".format(int(snap_nums[i]))
    clumped_cat = glob.glob(os.path.join(os.path.join(path, snapshot), "profiled*.txt"))
    clumped_dat = np.loadtxt(clumped_cat[0])
    clump_masses = clumped_dat[:, 8]
    # the bulge/ nuclear star cluster is the most massive
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
    lum_erg_per_s_ang = 10 ** full_dat[:, 3]
    creation_time = tmyr - all_ages

    starting_point = 0.01  # where the profile starts pc, 0.01 is the smallest value
    prof_rad = 200  # pc, where the profile ends
    pids = full_dat[:, 1]
    cmap = matplotlib.colormaps["Dark2"]
    cmap = cmap(np.linspace(0, 1, 8))
    color = cmap[0]

    age_mask = (creation_time > 575) 
    # age_mask = creation_time < 800

    bulg_particles = pids[age_mask]

    allx, ally, allz = full_dat[:, 4:7].T

    allx_recentered = allx - bulge_group_x
    ally_recentered = ally - bulge_group_y
    allz_recentered = allz - bulge_group_z


    def surf_brightness(x, y, lums, redshift, bins=20, highest_mag=40):
        all_positions = np.vstack((x, y)).T
        r = np.geomspace(starting_point, prof_rad, num=bins, endpoint=True)
        distances = np.sqrt(np.sum(np.square(all_positions), axis=1))

        lum_per_bin, bin_edges = np.histogram(distances, bins=r, weights=lums)
        count_per_bin, _ = np.histogram(distances, bins=r)

        mask = lum_per_bin > 0
        lum_per_bin = lum_per_bin[mask]

        # getting bin properties
        right_edges = bin_edges[1:]
        left_edges = bin_edges[:-1]
        bin_ctrs = 0.5 * (left_edges + right_edges)[mask]
        ring_areas = np.pi * (right_edges**2 - left_edges**2)[mask]

        surface_brightness = lum_per_bin / ring_areas  # now has units of erg/s/AA/pc^2
        avg_star_lums = lum_per_bin / count_per_bin[mask]
        err_poisson = np.sqrt(count_per_bin[mask]) * (avg_star_lums / ring_areas)

        mu_ab = (
            23.4
            - 2.5
            * np.log10(
                surface_brightness / 1e36,
                out=np.full_like(surface_brightness, np.nan),
                where=(surface_brightness != 0),
            )
            + 10 * np.log10((1 + redshift) / 10)
        )  # mag/arcsec^2

        mu_ab_err = (
            23.4
            - 2.5 * np.log10(err_poisson / 1e36)
            + 10 * np.log10((1 + redshift) / 10)
        )

        print(np.nanmax(mu_ab))
        mu_ab[np.isnan(mu_ab)] = highest_mag  # for 0, just put to the floor fo 40 mags
        return bin_ctrs, mu_ab,mu_ab_err



    r, muab_xy, muab_xy_err = surf_brightness(
        allx_recentered[age_mask],
        ally_recentered[age_mask],
        lum_erg_per_s_ang[age_mask],
        redshift,
        bins=20
    )

   
    ax[i].errorbar(
        r, muab_xy, color=color,markersize=10, fmt=".",
    )
    ax[i].set(xscale="log", ylim=(20.5, 32.5))
    ax[i].invert_yaxis()

    def sersic(r, i0, r_halflight, n):
        """
        http://burro.case.edu/Academics/Astr323/Lectures/GalaxyPopulations.pdf
        """
        b= 2*n - 0.333 
        mu = i0 + (2.5 * b / np.log(10)) * ((r / r_halflight) ** (1 / n) - 1)
        return mu

    nsc_mask = r <= inner_rad[i]
    popt_nsc, pcov = curve_fit(
        sersic, r[nsc_mask], muab_xy[nsc_mask], p0=[22, 1.5, 2]
    )
    r_theory = np.arange(0.02, 100, 0.01)
    ax[i].plot(
        r_theory,
        sersic(r_theory, *popt_nsc),
        ls="--",
        lw=2,
        color="grey",
        label=r"$r_{{\rm half}} = {:.1f}$ pc""\n"r"$n = {:.1f}$".format(
            popt_nsc[1], popt_nsc[2]
    ))
   
    if i == 0:
        gal_mask = r > 0.5
        popt_nsc, pcov = curve_fit(
            sersic, r[gal_mask ], muab_xy[gal_mask ], p0=[22, 1.5, 2]
        )
        ax[i].plot(
            r_theory,
            sersic(r_theory, *popt_nsc),
            ls="--",
            lw=2,
            color="grey",
             label=r"$r_{{\rm half}} = {:.1f}$ pc""\n"r"$n = {:.1f}$".format(
            popt_nsc[1], popt_nsc[2])
        )
        labelLines(ax[i].get_lines(), xvals=(5, 10), color="k",align=False, fontsize=11)
    else:
        ax[i].legend(frameon=False)
    
    ax[i].axvspan(0.001, 0.1, alpha=0.2, facecolor="k")


    ax[i].text(
        0.05,
        0.95,
        r"${{\rm t = {:.0f}\:{{\rm Myr }}}}$".format(tmyr),
        transform=ax[i].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        # color="white",
        clip_on=False,
    )



# ax[0].legend(frameon=False)
ax[1].set(xlabel="$r$ [pc]", xlim=(0.02, 200))
# ax[0].set(
#     ylabel=r"$\log \: \Sigma_{\rm PopII} \left[ {\rm M_\odot pc^{-2}} \right]$",
# )
ax[0].set_ylabel(r"$\mathrm{UV}\: \mu_{\rm AB}(r) \: {\rm [mag \: arcsec^{-2}]}$")
plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/nuclear_cluster_mag_profile.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.1,
)
plt.show()