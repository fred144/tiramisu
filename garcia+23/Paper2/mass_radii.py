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

from labellines import labelLine, labelLines
from scipy.optimize import curve_fit
from astropy.modeling.models import Sersic1D
import matplotlib
import glob
from tools.fscanner import filter_snapshots
import os
import matplotlib.patches as patches
from tools.cosmo import t_myr_from_z, z_from_t_myr
from scipy.stats import binned_statistic_2d
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
from tools import plotstyle


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


def metal_lookup(log_sfc_path, starform_times):
    log = np.loadtxt(log_sfc_path)
    z_form = log[:, 2]
    t_form = t_myr_from_z(z_form)
    m_sun_form = log[:, 7]  # mass in stars
    z_sun = log[:, 9]  # solar metallicity

    residuals = np.abs(t_form - starform_times[:, np.newaxis])
    closest_match_idxs = np.argmin(residuals, axis=1)
    star_metals = z_sun[closest_match_idxs] * 3.81
    bsc_m_sun_form = m_sun_form[closest_match_idxs]
    return star_metals


times = [512, 659]  # [595, 600, 627, 655]
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

# fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(8, 15), dpi=300, sharex="col")
# plt.subplots_adjust(wspace=-0.03, hspace=0)
path = "../../../container_tiramisu/post_processed/bsc_catalogues/CC-Fiducial"
logsfc = os.path.expanduser("~/test_data/CC-Fiducial/logSFC")
logsfc_dat = np.loadtxt(logsfc)

# get data from neumayer 2020
reffvsmass = np.loadtxt("neumayer2020_reffvsM.txt", delimiter=",")
reffvsmass = 10**reffvsmass

sigmaeffvsmass = np.loadtxt("neumayer2020_SigmaEffvsM.txt", delimiter=",")
sigmaeffvsmass = 10**sigmaeffvsmass
# %%

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 6), dpi=300)
plt.subplots_adjust(hspace=0.25, wspace=0.25)


def create_extended_colormap(boundaries, cmap_name="viridis"):
    """
    Create a colormap that extends the given colormap with gray at the end.

    Parameters:
    - boundaries: List of boundaries defining the color segments (must be sorted).
    - cmap_name: Name of the base colormap (default: 'viridis').

    Returns:
    - custom_cmap: A ListedColormap with the extended segments.
    - norm: A BoundaryNorm for mapping data to the custom segments.
    """
    # Number of segments for the colormap
    n_segments = len(boundaries) - 1

    # Generate colors from the specified colormap
    base_cmap = plt.cm.get_cmap(cmap_name)
    cmap_colors = base_cmap(
        np.linspace(0, 1, n_segments - 1)
    )  # All but the last segment

    # Add gray color as the final segment
    gray_color = np.array([[0.5, 0.5, 0.5, 0.4]])  # RGBA for gray
    cmap_colors = np.vstack([cmap_colors, gray_color])

    # Create a ListedColormap
    custom_cmap = ListedColormap(cmap_colors)

    # Create a BoundaryNorm
    norm = BoundaryNorm(boundaries, len(cmap_colors))

    return custom_cmap, norm


# Example: Define boundaries

intervals = 0.5
cmap_end = 2.5
boundaries = np.arange(0, cmap_end + intervals * 2, intervals)

# Create custom colormap and norm
custom_cmap, norm = create_extended_colormap(boundaries, cmap_name="cmr.tropical")

for i, pop2 in enumerate(files):
    snapshot = "info_00{:}".format(int(snap_nums[i]))

    # bound and unbound popII stars
    # bound_star_path = glob.glob(
    #     os.path.join(os.path.join(path, snapshot), "clumped_*.txt")
    # )
    # field_star_path = glob.glob(
    #     os.path.join(os.path.join(path, snapshot), "field_*.txt")
    # )
    # bound_dat = np.loadtxt(bound_star_path[0])
    # unbound_dat = np.loadtxt(field_star_path[0])
    # bx, by, bz = bound_dat[:, 3:6].T
    # ux, uy, uz = unbound_dat[:, 3:6].T
    # or can read the pop II stars directly
    full_dat = np.loadtxt(pop2)
    tmyr, redshift = full_dat[0:2, 0]
    all_pop2_mass = full_dat[:, -1]
    all_ages = full_dat[:, 2]
    creation_time = tmyr - all_ages
    allx = full_dat[:, 4]
    ally = full_dat[:, 5]
    allz = full_dat[:, 6]

    # the terminology is that bound star clusters are profilled clumps and disrupted stars are not
    clumped_cat = glob.glob(os.path.join(os.path.join(path, snapshot), "profiled*.txt"))
    bsc_paths = glob.glob(os.path.join(os.path.join(path, snapshot), "bsc_0*.txt"))
    disrupted_paths = glob.glob(
        os.path.join(os.path.join(path, snapshot), "disrupted_*.txt")
    )

    # get the BSC catalogue
    clumped_cat = glob.glob(os.path.join(os.path.join(path, snapshot), "profiled*.txt"))
    clumped_dat = np.loadtxt(clumped_cat[0])

    clump_radii = clumped_dat[:, 4] * u.parsec
    clump_half_mass_radius = clumped_dat[:, 21] * u.parsec
    clump_masses = clumped_dat[:, 8] * u.Msun
    # central density
    Sigma0 = clumped_dat[:, 16]

    sx, sy, sz = clumped_dat[:, 9:12].T * 1e3 * (u.m / u.s)
    sigma_3d_squared = sx**2 + sy**2 + sz**2  # m/s this is 1D

    r_half = clumped_dat[:, 21] * u.parsec  # half light radius
    r = clumped_dat[:, 4] * u.parsec
    r_core = clumped_dat[:, 12] * u.parsec  # pc
    half_masses = 0.5 * clump_masses
    m_core = clumped_dat[:, -1]
    ke_avg_perparticle = (1 / 2) * sigma_3d_squared * clump_masses.to(u.kg)
    pot_energy = (3 / 5) * (const.G * clump_masses.to(u.kg) ** 2) / r_half.to(u.m)

    vir_parameter = 2 * ke_avg_perparticle / pot_energy

    # the bulge/nsc the most massive after it was seeded
    bulge_id = clumped_dat[:, 0][np.argmax(clump_masses)]
    # group properties
    bulge_group_x = clumped_dat[:, 1][np.argmax(clump_masses)]
    bulge_group_y = clumped_dat[:, 2][np.argmax(clump_masses)]
    bulge_group_z = clumped_dat[:, 3][np.argmax(clump_masses)]

    # this is the consituent stars of the bulge
    big_un = os.path.join(
        os.path.join(path, snapshot),
        "bsc_{}.txt".format(str(int(bulge_id)).zfill(4)),
    )
    big_bsc = np.loadtxt(big_un)
    big_ages = big_bsc[:, 1]
    bigx, bigy, bigz = big_bsc[:, 3:6].T
    big_mass = big_bsc[:, -1]
    vx, vy, vz = big_bsc[:, 6:9].T * 1e3  # km/s
    total_mass = np.sum(big_bsc[:, -1]) * u.Msun
    # print(total_mass.value)
    catalog_idx = int(np.argwhere(clumped_dat[:, 0] == int(bulge_id)))
    clump_radius = clumped_dat[:, 4][catalog_idx] * u.parsec
    half_mass_radius = clumped_dat[:, 21][catalog_idx] * u.parsec
    print("per the clump finder, the central object")
    print("has a mass of {:.2e}".format(total_mass))
    print("has a half mass radius of {:.2f}".format(half_mass_radius))
    print("and a clump radius of {:.2f}".format(clump_radius))
    print(
        "and an average density of {:.2e}".format(
            total_mass / (4 / 3 * np.pi * clump_radius**3)
        )
    )

    vir_mask = vir_parameter < cmap_end

    # plot mass of clump vs radius of clumps
    ax[i, 0].scatter(
        clump_masses[~vir_mask],
        clump_radii[~vir_mask],
        c="grey",
        s=40,
        alpha=0.3,
        edgecolors="none",
    )
    bound_scat = ax[i, 0].scatter(
        clump_masses[vir_mask],
        clump_half_mass_radius[vir_mask],
        c=vir_parameter[vir_mask],
        cmap=custom_cmap,
        norm=norm,
        s=40,
        edgecolors="k",
        alpha=0.9,
    )

    # plot central density as a function of object mass
    bulge_mask = clump_masses == clump_masses[np.argmax(clump_masses)]
    Sigma0[bulge_mask] = 1e3
    ax[i, 1].scatter(
        clump_masses[vir_mask],
        Sigma0[vir_mask],
        c=vir_parameter[vir_mask],
        s=40,
        edgecolors="k",
        cmap=custom_cmap,
        norm=norm,
        alpha=0.9,
    )
    ax[i, 1].scatter(
        clump_masses[~vir_mask],
        Sigma0[~vir_mask],
        c="grey",
        s=40,
        alpha=0.3,
        edgecolors="none",
    )

    ax[i, 0].set(
        xscale="log",
        yscale="log",
        ylabel=r"$r_{\rm half-mass}\: {\rm [pc]}$",
        xlabel=r"$m_{\rm star\: cluster} {[\rm M_\odot]}$",
        xlim=(2e2, 9e5),
        ylim=(0.1, 50),
    )

    ax[i, 1].set(
        xscale="log",
        yscale="log",
        ylabel=r"$\Sigma_0$ [M$_\odot$/pc$^2$]",
        xlabel=r"$m_{\rm star\: cluster} {[\rm M_\odot]}$",
        xlim=(2e2, 9e5),
        # xlim=(2e2, 9e7),
        ylim=(60, 3e4),
    )

    ax[i, 0].axhspan(0.01, 0.1, color="gray", alpha=0.8)
    # time

    ax[i, 0].text(
        0.1,
        0.9,
        r"$t = {:.0f}$ Myr".format(tmyr),
        transform=ax[i, 0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    # line for constant density
    mtheory = np.geomspace(1e2, 1e6, 100)
    rtheory = mtheory ** (1 / 3) * 0.06
    rtheory2 = mtheory ** (1 / 2) * 0.06

    if i == 1:
        ax[i, 0].scatter(reffvsmass[:, 0], reffvsmass[:, 1], s=2, color="k", alpha=0.8)
        ax[i, 1].scatter(
            sigmaeffvsmass[:, 0], sigmaeffvsmass[:, 1], s=2, color="k", alpha=0.5
        )

        # make a confidence band
        # power law in log space
        def power_law(x, a, b):
            return a * x**b

        popt, pcov = curve_fit(
            power_law, np.log10(reffvsmass[:, 0]), np.log10(reffvsmass[:, 1])
        )
        reffvsmass_err = np.sqrt(np.diag(pcov))

        x = np.geomspace(1e2, 1e8, 100)
        
        reffvsmass_theory = 10 ** power_law(np.log10(x), *popt)
        ax[i, 0].plot(x, reffvsmass_theory, lw=2, ls="-.", c="k")

        # confidence band with 1 sigma errors
        # ax[i, 0].fill_between(
        #     x,
        #     10 ** power_law(np.log10(x), popt[0] - reffvsmass_err[0], popt[1] - reffvsmass_err[1]),
        #     10 ** power_law(np.log10(x), popt[0] + reffvsmass_err[0], popt[1] + reffvsmass_err[1]),
        #     color="k",
        #     alpha=0.1,
        # )

        # condfidence with half a dex above and below
        ax[i, 0].fill_between(
            x,
            10 ** (power_law(np.log10(x), popt[0], popt[1]) - 0.6),
            10 ** (power_law(np.log10(x), popt[0], popt[1]) + 0.6),
            alpha=0.1,
            facecolor="k",
        )

        # now do the same for the surface density
        popt, pcov = curve_fit(
            power_law, np.log10(sigmaeffvsmass[:, 0]), np.log10(sigmaeffvsmass[:, 1])
        )
        sigmaeffvsmass_err = np.sqrt(np.diag(pcov))
        sigmaeffvsmass_theory = 10 ** power_law(np.log10(x), *popt)
        ax[i, 1].plot(x, sigmaeffvsmass_theory, lw=2, ls="-.", c="k", label=r"Neumayer et al. 2020, $z=0$")
       
        # confidence band with 1 sigma errors
        # ax[i, 1].fill_between(x, 10 ** (power_law(np.log10(x), popt[0] - sigmaeffvsmass_err[0], popt[1] - sigmaeffvsmass_err[1]), 10 ** (power_law(np.log10(x), popt[0] + sigmaeffvsmass_err[0], popt[1] + sigmaeffvsmass_err[1])), color="k", alpha=0.1))

        # condfidence with half a dex above and below
        ax[i, 1].fill_between(
            x,
            10 ** (power_law(np.log10(x), popt[0], popt[1]) - 0.6),
            10 ** (power_law(np.log10(x), popt[0], popt[1]) + 0.6),
            alpha=0.1,
            facecolor="k",
        )
        ax[i,1].legend(fontsize=10, loc='upper left', frameon=False, bbox_to_anchor=(-0.1, 2.45))

    # ax[i, 0].plot(
    #     mtheory,
    #     rtheory,
    #     lw=2,
    #     ls="--",
    #     alpha=0.2,
    #     c="k",
    #     label=r"$r \propto m_{\rm star\: cluster}^{1/3}$",
    # )

    # ax[i+2].plot(mtheory, rtheory2, lw=2, ls="--", alpha=0.2, c="k", label=r"$\Sigma_0 \propto m^{1/2}$")
    # labelLines(ax[i, 0].get_lines(), color="k", align=True, yoffsets=-1, fontsize=10)

ax[1, 0].annotate(
    "NSC",
    xy=(2e5, 10),
    xytext=(5e4, 20),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="grey"),
)

ax[1, 1].annotate(
    "NSC",
    xy=(2e5, 1e3),
    xytext=(5e4, 3e2),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="grey"),
)


# create a colorbar
cbar_ax = ax[0, 0].inset_axes([0.0, 1.1, 1, 0.08])

cbar = fig.colorbar(
    bound_scat,
    cax=cbar_ax,
    pad=0,
    orientation="horizontal",
    boundaries=boundaries,
    ticks=[0.5, 1, 1.5, 2, 2.5],
    extend="max",
)
# cbar.set_label(r"$\alpha_{\rm vir}$", fontsize=10,loc='left', labelpad=-20, x=0.9)
cbar.ax.xaxis.set_tick_params(labelsize=10)
cbar_ax.set_ylabel(r"$\alpha_{\rm vir}$", fontsize=10, rotation=0, labelpad=10, y=0.1)
cbar.ax.minorticks_on()
plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/cluster_populations.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.01,
)
plt.show()
