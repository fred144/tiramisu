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

path = "../../../container_tiramisu/post_processed/bsc_catalogues/CC-Fiducial"

snapshot = "info_00397"
bound_star_path = glob.glob(os.path.join(os.path.join(path, snapshot), "clumped_*.txt"))
field_star_path = glob.glob(os.path.join(os.path.join(path, snapshot), "field_*.txt"))
clumped_cat = glob.glob(os.path.join(os.path.join(path, snapshot), "profiled*.txt"))
bsc_paths = glob.glob(os.path.join(os.path.join(path, snapshot), "bsc_0*.txt"))
disrupted_paths = glob.glob(
    os.path.join(os.path.join(path, snapshot), "disrupted_*.txt")
)

# %%
bound_dat = np.loadtxt(bound_star_path[0])
unbound_dat = np.loadtxt(field_star_path[0])

bx, by, bz = bound_dat[:, 3:6].T
ux, uy, uz = unbound_dat[:, 3:6].T
stelllar_range = (20, 2e4)
wdth = 300
star_bins = 2000
pxl_size = (wdth / star_bins) ** 2

all_x = np.concatenate((bx, ux))
all_y = np.concatenate((by, uy))
all_z = np.concatenate((bz, uz))
masses = 10 * np.ones_like(all_x)
surface_dens, _, _ = np.histogram2d(
    all_x,
    all_y,
    bins=star_bins,
    weights=masses,
    range=[
        [-wdth / 2, wdth / 2],
        [-wdth / 2, wdth / 2],
    ],
)
surface_dens = surface_dens.T / pxl_size

lum_alpha = 1

clumped_dat = np.loadtxt(clumped_cat[0])

# # %%
# for bsc_f in sorted(bsc_paths):
#     bscnum = bsc_f.split("/")[-1].split(".")[0].split("_")[-1]
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=300)
#     print(bscnum)
#     # ax.scatter(ux, uy, color="grey", s=0.5, alpha=0.05)
#     # ax.scatter(bx, by, color="r", s=0.5, alpha=0.05)

#     ax.imshow(
#         surface_dens,
#         cmap="cmr.amethyst",
#         origin="lower",
#         extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
#         norm=LogNorm(vmin=stelllar_range[0], vmax=stelllar_range[1]),
#     )
#     ax.set(xlim=(-wdth / 2, wdth / 2), ylim=(-wdth / 2, wdth / 2))
#     ax.set_facecolor("k")

#     bsc_inset = ax.inset_axes([1.1, 0.5, 0.50, 0.5])

#     bsc_dat = np.loadtxt(bsc_f)
#     bsc_x, bsc_y, bsc_z = bsc_dat[:, 3:6].T

#     # com_x = bsc_x.mean()
#     # com_y = bsc_y.mean()

#     catalog_idx = int(np.argwhere(clumped_dat[:, 0] == int(bscnum)))
#     com_x = clumped_dat[:, 1][catalog_idx]
#     com_y = clumped_dat[:, 2][catalog_idx]

#     sx, sy, sz = clumped_dat[:, 9:12][catalog_idx] * 1e3 * (u.m / u.s)
#     sigma_3d_squared = sx**2 + sy**2 + sz**2  # m/s this is 1D

#     r_half = clumped_dat[:, 21][catalog_idx] * u.parsec  # half light radius
#     r = clumped_dat[:, 4][catalog_idx] * u.parsec
#     r_core = clumped_dat[:, 12][catalog_idx] * u.parsec  # pc

#     total_mass = clumped_dat[:, 8][catalog_idx] * u.Msun
#     half_mass = 0.5 * total_mass
#     m_core = clumped_dat[:, -1][catalog_idx] * u.Msun

#     ke_avg_perparticle = (1 / 2) * sigma_3d_squared * half_mass.to(u.kg)

#     pot_energy = (3 / 5) * (const.G * half_mass.to(u.kg) ** 2) / r_half.to(u.m)

#     vir_parameter = 2 * ke_avg_perparticle / pot_energy

#     # vir_parameter = (5 * sigma_3d_squared * r_half.to(u.m)) / (
#     #     const.G * total_mass.to(u.kg)
#     # )

#     bsc_inset.text(
#         0.1,
#         -0.6,
#         r"$M = {:.0f} \: {{\rm M_\odot}}$"
#         "\n"
#         r"$r_{{\rm core}} = {:.2f} \: {{\rm pc}}$"
#         "\n"
#         r"$r_{{\rm half}} = {:.2f} \: {{\rm pc}}$"
#         "\n"
#         r"$\alpha_{{\rm vir}} = {:.2f}$".format(
#             total_mass.value, r_core.value, r_half.value, vir_parameter
#         ),
#         transform=bsc_inset.transAxes,
#     )

#     ax.scatter(com_x, com_y, marker="x", c="red", s=2, alpha=0.5)

#     bsc_inset.scatter(ux, uy, color="grey", s=0.5, alpha=0.05)
#     bsc_inset.scatter(bsc_x, bsc_y, color="r", s=1, alpha=0.2, label="BSC " + bscnum)
#     bsc_inset.set(
#         xlim=(com_x - 20, com_x + 20),
#         ylim=(com_y - 20, com_y + 20),
#     )
#     bsc_inset.set_xticklabels([])
#     bsc_inset.set_yticklabels([])
#     bsc_inset.xaxis.set_ticks_position("none")
#     bsc_inset.yaxis.set_ticks_position("none")
#     bsc_inset.legend(loc="lower right")

#     rect, lines = ax.indicate_inset_zoom(bsc_inset)
#     rect.set_edgecolor("white")
#     plt.savefig(
#         "../../../container_tiramisu/post_processed/clump_finder_check/BSC_check_300Msunthresh/{}.png".format(
#             bscnum
#         ),
#         bbox_inches="tight",
#         dpi=300,
#     )
#     plt.close()

# %%

# for bsc_f in sorted(disrupted_paths):
#     bscnum = bsc_f.split("/")[-1].split(".")[0].split("_")[-1]
#     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=300)
#     print(bscnum)
#     # ax.scatter(ux, uy, color="grey", s=0.5, alpha=0.05)
#     # ax.scatter(bx, by, color="r", s=0.5, alpha=0.05)

#     # ax.set(xlim=(-150, 150), ylim=(-150, 150))

#     ax.imshow(
#         surface_dens,
#         cmap="cmr.amethyst",
#         origin="lower",
#         extent=[-wdth / 2, wdth / 2, -wdth / 2, wdth / 2],
#         norm=LogNorm(vmin=stelllar_range[0], vmax=stelllar_range[1]),
#     )
#     ax.set(xlim=(-wdth / 2, wdth / 2), ylim=(-wdth / 2, wdth / 2))
#     ax.set_facecolor("k")

#     bsc_inset = ax.inset_axes([1.1, 0.5, 0.50, 0.5])
#     bsc_dat = np.loadtxt(bsc_f)
#     bsc_x, bsc_y, bsc_z = bsc_dat[:, 3:6].T
#     com_x = bsc_x.mean()
#     com_y = bsc_y.mean()
#     ax.scatter(com_x, com_y, marker="x", c="red", s=2, alpha=0.5)
#     bsc_inset.scatter(ux, uy, color="grey", s=0.5, alpha=0.05)
#     bsc_inset.scatter(
#         bsc_x, bsc_y, color="purple", s=1, alpha=0.2, label="Disrupted " + bscnum
#     )
#     bsc_inset.set(
#         xlim=(bsc_x.mean() - 20, bsc_x.mean() + 20),
#         ylim=(bsc_y.mean() - 20, bsc_y.mean() + 20),
#     )
#     bsc_inset.set_xticklabels([])
#     bsc_inset.set_yticklabels([])
#     bsc_inset.legend(loc="lower right")

#     rect, lines = ax.indicate_inset_zoom(bsc_inset)
#     rect.set_edgecolor("white")
#     plt.savefig(
#         "../../../container_tiramisu/post_processed/clump_finder_check/HOP/{}.png".format(
#             bscnum
#         ),
#         bbox_inches="tight",
#         dpi=300,
#     )
#     plt.close()

# %%
big_un = os.path.join(os.path.join(path, snapshot), "bsc_0034.txt")
big_bsc = np.loadtxt(big_un)
big_profile = np.loadtxt(
    os.path.join(os.path.join(path, snapshot), "bsc_profile_0034.txt")
)

# pop2 data from the big BSC
# ID|CurrentAges[Myr]| |log10UV(150nm)Lum[erg/s]| |X[pc]|Y[pc]||Z[pc]| |Vx[km/s]|Vy[km/s]||Vz[km/s]| |mass[Msun]
big_ages = big_bsc[:, 1]
bigx, bigy, bigz = big_bsc[:, 3:6].T

vx, vy, vz = big_bsc[:, 6:9].T * 1e3  # km/s
total_mass = np.sum(big_bsc[:, -1]) * u.Msun

catalog_idx = int(np.argwhere(clumped_dat[:, 0] == int(34)))
radius = clumped_dat[:, 4][catalog_idx] * u.parsec

sx = np.std(vx) * (u.m / u.s)
sy = np.std(vy) * (u.m / u.s)
sz = np.std(vz) * (u.m / u.s)
sigma_3d_squared = sx**2 + sy**2 + sz**2  # (km/s)^2
sigma_3d = np.sqrt(sigma_3d_squared)

# %% graph the age distribution

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=300)
ax.hist(big_ages, bins=np.linspace(0, 230, 30))
ax.set(yscale="log", ylabel="Number of PopII Stars", xlabel="Age (Myr)")
plt.show()


# %% surface density profile with cutoff at virial radius of the clump

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=300)
ax.plot(big_profile[:, 0], big_profile[:, 1], label="bulge clump")
# ax.plot(big_profile[:, 0] ** (1 / 4), big_profile[:, 1])
# ax.plot(np.linspace(0.01, 15, 100), np.linspace(0.01, 15, 100) ** -(1 / 4))
ax.set(
    xscale="log",
    yscale="log",
    ylabel=r"Surface Density Msun/pc**2",
    xlabel="Radial Distance (pc)",
    # xlim=(0.1, 12),
)
ax.legend()
plt.show()

# %% profiling without the cutoff
import numpy as np
from pytreegrav import Accel, Potential

pop2 = "../../../container_tiramisu/post_processed/pop2/CC-Fiducial"
full_dat = np.loadtxt(os.path.join(pop2, "pop2-00397-588_12-myr-z-8_746.txt"))
allx, ally, allz = full_dat[:, 4:7].T
pots = Potential(big_bsc[:, 3:6], big_bsc[:, -1], method="bruteforce")
minimum_u_xyz_pos = np.argmin(pots)

min_x = bigx[minimum_u_xyz_pos]
min_y = bigy[minimum_u_xyz_pos]
min_z = bigz[minimum_u_xyz_pos]

avg_x = np.mean(bigx)
avg_y = np.mean(bigy)
avg_z = np.mean(bigz)

# name of BSC
bsc = 34
catalog_idx = int(np.argwhere(clumped_dat[:, 0] == int(bsc)))

# get's the center of the bulge
bsc_x = clumped_dat[:, 1][catalog_idx]
bsc_y = clumped_dat[:, 2][catalog_idx]
bsc_z = clumped_dat[:, 3][catalog_idx]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=300)

age_1 = big_ages < 50  # myr
age_2 = (big_ages >= 50) & (big_ages < 150)
age_3 = big_ages > 150


ax.scatter(
    ally,
    allz,
    s=1,
    alpha=0.01,
    color="grey",
    label=r"age $\le$ 50 Myr",
)

ax.scatter(
    bigy[age_1],
    bigz[age_1],
    s=2,
    alpha=0.3,
    color="blue",
    label=r"age $\le$ 50 Myr",
)

ax.scatter(
    bigy[age_2],
    bigz[age_2],
    s=2,
    alpha=0.3,
    color="green",
    label="50 $<$ age $<$ 150 Myr",
)
ax.scatter(
    bigy[age_3],
    bigz[age_3],
    s=2,
    alpha=0.3,
    color="red",
    label="age $>$ 150 Myr",
)

ax.scatter(min_y, min_z, c="magenta", s=10, label="minimum potential")
ax.scatter(avg_y, avg_z, c="cyan", s=10, label="centroid")
ax.scatter(bsc_y, bsc_z, c="k", s=10, label="fof")
ax.legend(ncols=2)
plt.show()
# %%
from scipy.optimize import curve_fit
from astropy.modeling.models import Sersic1D
import matplotlib
import glob
from tools.fscanner import filter_snapshots

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
    lum_alpha
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
