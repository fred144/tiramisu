# %%
import sys

sys.path.append("../../")

import yt
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
# import cmasher as cmr
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

plt.rcParams.update(
    {
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
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


# times = [659]  # [595, 600, 627, 655]
times = np.linspace(585, 660, 5)
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
cmap = matplotlib.colormaps["Dark2"]
cmap = cmap(np.linspace(0, 1, 8))
color = cmap[0]
# fahrion data
fahrion2022 = np.loadtxt("./fahrion2022.txt")
fahrion2021 = np.loadtxt("./fahrion2021.txt")
#carlsten data
calstern_virgo = np.loadtxt("./carlsten2022.txt", delimiter=",")

fahrion_data = np.concatenate((fahrion2022, fahrion2021), axis=0)
galaxy_mass = []
nsc_mass = []
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
    all_masses = full_dat[:, -1]

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
    clump_masses = clumped_dat[:, 8]
    # central density
    Sigma0 = clumped_dat[:, 16]

    sx, sy, sz = clumped_dat[:, 9:12].T * 1e3 * (u.m / u.s)
    sigma_3d_squared = sx**2 + sy**2 + sz**2  # m/s this is 1D

    r_half = clumped_dat[:, 21] * u.parsec  # half light radius
    r = clumped_dat[:, 4] * u.parsec
    r_core = clumped_dat[:, 12] * u.parsec  # pc
    half_masses = 0.5 * clump_masses * u.Msun
    m_core = clumped_dat[:, -1]
    ke_avg_perparticle = (1 / 2) * sigma_3d_squared * half_masses.to(u.kg)
    pot_energy = (3 / 5) * (const.G * half_masses.to(u.kg) ** 2) / r_half.to(u.m)

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
    nsc_mass.append(big_mass.sum())
    galaxy_mass.append(np.sum(all_masses))

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4.5), dpi=300, sharex="col")
plt.subplots_adjust(hspace=0)
masses = np.geomspace(1e5, 1e12, 100)  # galaxy masses
# Neumauer et al. 2020
log_m_nsc = 0.48 * np.log10(masses / 1e9) + 6.51
# ax.scatter(np.sum(all_masses), big_mass.sum(), marker="*", s=100, color=color, edgecolors="black", label="This work")
ax.scatter(
    galaxy_mass[-1],
    nsc_mass[-1],
    marker="*",
    s=200,
    c=color,
    alpha=0.8,
    edgecolors="black",
    zorder=10,
    label=r"This work, $z = {:.1f}$".format(z_from_t_myr(tmyr)),
    # cmap="inferno"
)
ax.plot(masses, 10**log_m_nsc, color="k", lw=3, ls="-.",  alpha=0.8, zorder=10,)
# add 0.6 dex scatter
ax.fill_between(
    masses,
    10 ** (log_m_nsc - 0.6),
    10 ** (log_m_nsc + 0.6),
    facecolor="grey",
    alpha=0.2,
)
# ax.scatter(
#     10 ** fahrion2022[:, 0],
#     10 ** fahrion2022[:, 1],
#     label="Fahrion et al. 2020",
#     marker="d",
#     color="grey",
#     edgecolors="black",
#     alpha=0.8
# )
ax.scatter(
    10 ** fahrion_data[:, 0],
    10 ** fahrion_data[:, 1],
    label="Fahrion et al. 2020, 2021",
    color="orange",
    marker="d",
    # edgecolors="black",
    alpha=0.8,
)
ax.scatter(
    10 ** calstern_virgo[:, 0],
    10 ** calstern_virgo[:, 1],
    label="Carlsten et al. 2022",
    color="grey",
    marker="^",
    # edgecolors="black",
    alpha=0.8,
)

# ax.plot(masses, 0.3*masses, label="NSC mass relation")

# label the relation
ax.text(
    0.01,
    0.18,
    "Neumayer et al. 2020",
    transform=ax.transAxes,
    fontsize=9,
    rotation=20,
    color="k",
)
ax.set(
    yscale="log",
    xscale="log",
    xlim=(1e5, 2e10),
    ylim=(2e3, 2e8),
    xlabel="$M_\star \mathrm{[M_\odot]}$",
    ylabel="NSC mass [M$_\odot$]",
)
ax.legend(frameon=False, loc="upper left", fontsize=10, ncols=1)
plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/nsc_galmass_relation.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
