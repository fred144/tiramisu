#%%
# import sys

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
import matplotlib
from tools.cosmo import t_myr_from_z, z_from_t_myr


matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 10,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        # "xtick.major.size": 6,
        # "ytick.major.size": 6,
        # "xtick.minor.size": 4,
        # "ytick.minor.size": 4,
    }
)


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


path = "../../../container_tiramisu/post_processed/bsc_catalogues/CC-Fiducial"
logsfc = os.path.expanduser("~/test_data/CC-Fiducial/logSFC")
logsfc_dat = np.loadtxt(logsfc)
times = [576, 577, 595, 659]
bulge_clumpid = [4, 1, 15, 17]
snapshots = ["info_00385", "info_00386", "info_00404", "info_00466"]
cmap = matplotlib.colormaps["Dark2"]
cmap = cmap(np.linspace(0, 1, 8))
color = cmap[0]

# fig, ax = plt.subplots(
#     nrows=4, ncols=1, figsize=(8, 15.35), dpi=300, sharex=True, sharey=True
# )

plt.subplots_adjust(hspace=0)
for i, snapshot in enumerate(snapshots):
    bound_star_path = glob.glob(
        os.path.join(os.path.join(path, snapshot), "clumped_*.txt")
    )
    field_star_path = glob.glob(
        os.path.join(os.path.join(path, snapshot), "field_*.txt")
    )
    clumped_cat = glob.glob(os.path.join(os.path.join(path, snapshot), "profiled*.txt"))
    bsc_paths = glob.glob(os.path.join(os.path.join(path, snapshot), "bsc_0*.txt"))
    disrupted_paths = glob.glob(
        os.path.join(os.path.join(path, snapshot), "disrupted_*.txt")
    )

    # full_dat = np.loadtxt(
    #     os.path.join(
    #         "../../../container_tiramisu/post_processed/pop2/CC-Fiducial",
    #         "pop2-{}-*.txt".format(snapshot.split("_")[1]),
    #     )
    # )

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

    big_un = os.path.join(
        os.path.join(path, snapshot),
        "bsc_{}.txt".format(str(bulge_clumpid[i]).zfill(4)),
    )
    big_bsc = np.loadtxt(big_un)
    big_profile = np.loadtxt(
        os.path.join(
            os.path.join(path, snapshot),
            "bsc_profile_{}.txt".format(str(bulge_clumpid[i]).zfill(4)),
        )
    )

    # pop2 data from the big BSC
    # ID|CurrentAges[Myr]| |log10UV(150nm)Lum[erg/s]| |X[pc]|Y[pc]||Z[pc]| |Vx[km/s]|Vy[km/s]||Vz[km/s]| |mass[Msun]
    big_ages = big_bsc[:, 1]
    bigx, bigy, bigz = big_bsc[:, 3:6].T

    vx, vy, vz = big_bsc[:, 6:9].T * 1e3  # km/s
    total_mass = np.sum(big_bsc[:, -1]) * u.Msun
    print(total_mass.value)
    catalog_idx = int(np.argwhere(clumped_dat[:, 0] == int(bulge_clumpid[i])))
    radius = clumped_dat[:, 4][catalog_idx] * u.parsec

    sx = np.std(vx) * (u.m / u.s)
    sy = np.std(vy) * (u.m / u.s)
    sz = np.std(vz) * (u.m / u.s)
    sigma_3d_squared = sx**2 + sy**2 + sz**2  # (km/s)^2
    sigma_3d = np.sqrt(sigma_3d_squared)

    creation_time = times[i] - big_ages
    pop2_metallicity = metal_lookup(logsfc, creation_time)

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 5), dpi=300)
    plt.subplots_adjust(wspace=0.25, hspace=0.25)
    ax[0].hist(
        big_ages,
        color=color,
        bins=np.linspace(0, 340, 30),
        linewidth=0.8,
        edgecolor="k",
    )
    ax[0].text(
        0.95,
        0.95,
        r"${{\rm t = {:.0f}\:{{\rm Myr }}}}$".format(times[i]),
        transform=ax[0].transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        color="k",
        clip_on=False,
    )

    ax[0].text(
        0.05,
        0.95,
        r"NSC",
        transform=ax[0].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        color="k",
        clip_on=False,
    )

    ax[1].hist(
        np.log10(pop2_metallicity),
        color=color,
        bins=np.linspace(np.log10(6e-4), np.log10(4e-2), 30),
        linewidth=0.8,
        edgecolor="k",
    )

    ax[0].set(yscale="log", xlabel="PopII Ages [Myr]", ylim=(0.5, 8e4))
    ax[1].set(
        yscale="log", xlabel=r"log $Z_{\rm PopII}~[{\rm Z_\odot}]$", ylim=(0.5, 8e4)
    )
    fig.text(
        0.03,
        0.5,
        r"$N_{\rm PopII stars}$",
        va="center",
        rotation="vertical",
    )
    ax[0].minorticks_on()
    ax[1].minorticks_on()

plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/bulge_age_dist.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()


# %%
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
)
ax.minorticks_on()
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
bsc = 17
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
