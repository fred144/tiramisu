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
from tools.cosmo import t_myr_from_z, z_from_t_myr
from scipy.stats import binned_statistic_2d
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm

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


times = [659]  # [595, 600, 627, 655]
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
    clump_masses = clumped_dat[:, 8]
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

    bulge_Z = metal_lookup(logsfc, tmyr - big_ages)
    all_Z = metal_lookup(logsfc, tmyr - all_ages)

    # make segmented color map
    # cmap = plt.get_cmap("viridis")

    # Create a colormap with 8 colors correspindig to the ages
    num_segments = 8
    bounds = np.linspace(80, 320, num_segments + 1)

    # Choose a base colormap and sample colors based on the number of segments
    base_cmap = cm.get_cmap("cmr.tropical_r")
    colors = base_cmap(np.linspace(0, 1, num_segments))

    # Create a ListedColormap and a BoundaryNorm
    segmented_cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, segmented_cmap.N)

    pw = 200

    pxl_size = (pw / star_bins) ** 2  # in parsec^2

    # age_masks = [big_ages > 300, (big_ages < 300) & (big_ages > 200), big_ages < 200]
    # for mask in age_masks:
    #     fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    #     ax.scatter(ally - bulge_group_y, allx - bulge_group_x, s=0.1, c="grey", alpha=0.1)

    #     metal_scatter = ax.scatter(
    #         bigy[mask] - bulge_group_y,
    #         bigx[mask] - bulge_group_x,
    #         s=0.1,
    #         c=big_ages[mask],
    #         cmap=segmented_cmap,
    #         alpha=0.5,
    #         norm=norm,
    #     )

    #     ax.set(xlim=[-pw / 2, pw / 2], ylim=[-pw / 2, pw / 2])

    #     plt.show()

    # metals, _, _ = np.histogram2d(
    # allx,
    # ally,
    # bins=star_bins,
    # weights=masses,
    # range=[
    #     [-wdth / 2, wdth / 2],
    #     [-wdth / 2, wdth / 2],
    # ],
    # )

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.scatter( allx - bulge_group_x,ally - bulge_group_y, s=0.1, c="grey", alpha=0.1, zorder=0)
    star_bins = 128

    mean_metal, x_edges, y_edges, binnumber = binned_statistic_2d(
        bigx - bulge_group_x,
        bigy - bulge_group_y,
        bulge_Z,
        statistic="max",
        bins=[star_bins, star_bins],
        range=[[-pw / 2, pw / 2], [-pw / 2, pw / 2]],
    )

    # mass per bin in Msun
    mass_per_bin, _, _ = np.histogram2d(
        bigx,
        bigy,
        bins=star_bins,
        weights=big_mass,
        range=[[-pw / 2, pw / 2], [-pw / 2, pw / 2]],
    )

    # mass weighted metallicity per bin
    metallicity_mass_weighted_sum, _, _ = np.histogram2d(
        bigx,
        bigy,
        bins=star_bins,
        weights=big_mass * bulge_Z,
        range=[[-pw / 2, pw / 2], [-pw / 2, pw / 2]],
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mass_weighted_mean_metallicity = np.divide(
            metallicity_mass_weighted_sum,
            mass_per_bin,
            out=np.ones_like(metallicity_mass_weighted_sum) * np.nan,
            where=mass_per_bin != 0,
        )

    surf_metal = ax.imshow(
        np.log10( mean_metal/pxl_size).T,
        cmap="cmr.tree",
        origin="lower",
        extent=[-pw / 2, pw / 2, -pw / 2, pw / 2],
        
    )

    # add colorbar in dark mode
    with plt.style.context("dark_background"):
        cbar_ax = ax.inset_axes([0.05, 0.9, 0.5, 0.05])
        cbar = fig.colorbar(
            surf_metal,
            cax=cbar_ax,
            pad=0,
            orientation="horizontal",
        )

        cbar.ax.xaxis.set_tick_params(pad=-14, labelsize=10)
        cbar.set_label(r"$\langle Z \rangle$", labelpad=1)
        cbar.ax.minorticks_on()

    ax.set(xlim=[-pw / 2, pw / 2], ylim=[-pw / 2, pw / 2])
    ax.set_facecolor("k")
    plt.show()
