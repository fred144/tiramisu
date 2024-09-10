import sys

sys.path.append("../../")
import matplotlib.pyplot as plt
import numpy as np
from tools.cosmo import t_myr_from_z, z_from_t_myr
from matplotlib import cm
import matplotlib
from scipy import interpolate
from tools import plotstyle
import os
from tools.fscanner import filter_snapshots
import h5py as h5
from matplotlib import colors
import matplotlib as mpl
import cmasher as cmr
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import pandas as pd


def snapshot_from_time(snapshots, time, split_sym="-", snap_idx=1, time_idx=2):
    """
    Given a list of postprocesed pop ii snapshot files,
    get the files for a corresponding time

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
    snapshots_nums = np.array(snapshot)
    residuals = np.abs(uni_age - np.array(time)[:, np.newaxis])
    closest_match_idxs = np.argmin(residuals, axis=1).astype(int)

    matching_snaps = snapshots_nums[closest_match_idxs]
    matching_files = list(np.take(snapshots, closest_match_idxs))

    return matching_snaps, matching_files


def running_zstellar_mstellar(
    logSFC,
    gas_prop_dir,
    start,
    end,
    sampling_tmyr=1,
):
    log_sfc = np.loadtxt(logSFC)
    redshift = log_sfc[:, 2]

    t_myr = t_myr_from_z(redshift)
    # print(t_myr)
    mass_in_star = np.cumsum(log_sfc[:, 7])

    cloud_zsun = log_sfc[:, 9] * 3.81
    running_avg = []

    # compute the running average such that as more and more clouds/ stars are made,
    # it gets more and more metallic
    for i, z in enumerate(cloud_zsun, start=1):
        running_avg.append(np.mean(cloud_zsun[0:i]))
    running_avg = np.array(running_avg)

    # sample evenly through time
    # start_time = t_myr[0]
    # sampled_times = []
    # sampled_times.append(start_time)

    # for i, t in enumerate(t_myr):
    #     if (t - start_time) > sampling_tmyr:
    #         # print(t)
    #         start_time = t
    #         sampled_times.append(t)

    # sample_mask = np.isin(t_myr, sampled_times)

    ##  get gas properties
    # fpaths, snums = filter_snapshots(
    #     gas_prop_dir,
    #     start,
    #     end,
    #     sampling=1,
    #     str_snaps=True,
    #     snapshot_type="pop2_processed",
    # )
    # _, files = snapshot_from_time(fpaths, sampled_times)
    # # print(files)
    # galaxy_gas_metal = []

    # for i, file in enumerate(files):
    #     f = h5.File(file, "r")
    #     galaxy_gas_metal.append(f["Galaxy/MeanMetallicity"][()])
    #     f.close()

    # metal = np.array(galaxy_gas_metal)

    running_avg_interpolator = interpolate.interp1d(
        x=t_myr, y=running_avg, kind="previous"
    )
    running_stellar_mass_interpolator = interpolate.interp1d(
        t_myr, mass_in_star, kind="previous"
    )
    t_myr_interp = np.arange(t_myr.min(), t_myr.max(), sampling_tmyr)
    running_avg_interpolated = running_avg_interpolator(t_myr_interp)
    total_mstar_interpolated = running_stellar_mass_interpolator(t_myr_interp)
    return total_mstar_interpolated, running_avg_interpolated

    # return mass_in_star[sample_mask], running_avg[sample_mask]


# %% running average

cc_mstar_running, cc_zstar_running = running_zstellar_mstellar(
    "/home/fabg/container_tiramisu/sim_log_files/CC-Fiducial/logSFC",
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial",
    356,
    466,
    sampling_tmyr=10,
)

f7_mstar_running, f7_zstar_running = running_zstellar_mstellar(
    "/home/fabg/container_tiramisu/sim_log_files/fs07_refine/logSFC",
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    115,
    1570,
    sampling_tmyr=10,
)

f3_mstar_running, f3_zstar_running = running_zstellar_mstellar(
    "/home/fabg/container_tiramisu/sim_log_files/fs035_ms10/logSFC",
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    100,
    1606,
    sampling_tmyr=10,
)


def local_mz_relation(logmass, b=-1.69, m=0.3):
    """
    from Kirby + 11
    """
    # return -2.5 + 0.333 * (logmass - 4.0)
    return b + m * (logmass - 6)


def local_berg_12(logmstar, m=0.3, b=5.43):
    twelve_log_oh = m * logmstar + b
    zstar = twelve_log_oh - 8.69
    return zstar


metal = np.geomspace(1e2, 2e7, 20)

cmap = matplotlib.colormaps["Dark2"]
cmap = cmap(np.linspace(0, 1, 8))
vsfe_clr = cmap[0]
high_clr = cmap[1]
low_clr = cmap[2]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4), dpi=300)
# ax.scatter(cc_mstar[3::], cc_zgas[3::], color=vsfe_clr, alpha=0.8)
# ax.scatter(f7_mstar, f7_zgas, color=high_clr, alpha=0.8)
# ax.scatter(f3_mstar, f3_zgas, color=low_clr, alpha=0.8)


kirby_dat = np.loadtxt("./kirby13.txt")

ax.errorbar(
    kirby_dat[:, 0],
    kirby_dat[:, 2],
    xerr=kirby_dat[:, 1],
    yerr=kirby_dat[:, 3],
    fmt=".",
    markersize=5,
    color="olive",
)
ax.plot(
    np.log10(metal),
    local_mz_relation(np.log10(metal)),
    color="olive",
    lw=2,
    alpha=0.5,
    zorder=0,
    ls="--",
)
ax.text(
    2.6,
    -2.65,
    "Kirby+ 13, $z \sim 0$",
    rotation=27,
    rotation_mode="anchor",
    color="olive",
    ha="left",
    va="bottom",
    fontsize=10,
)
# ax.fill_between(
#     np.log10(metal),
#     local_mz_relation(np.log10(metal), -1.69 - 0.04, 0.30 - 0.02),
#     local_mz_relation(np.log10(metal), -1.69 + 0.04, 0.30 + 0.02),
#     alpha=0.2,
#     color="grey",
#     linewidth=0.0,
#     zorder=0,
# )
# ax.fill_between(
#     np.log10(metal),
#     local_mz_relation(np.log10(metal)) + 2 * 0.17,
#     local_mz_relation(np.log10(metal)) - 2 * 0.17,
#     alpha=0.2,
#     color="grey",
#     linewidth=0.0,
#     zorder=0,
# )


# def heintz_23(logmstar):
#     twelve_log_oh = 0.33 * logmstar + 4.65
#     zstar = twelve_log_oh - 8.69
#     return zstar


# ax.plot(
#     np.log10(metal),
#     heintz_23(np.log10(metal)),
#     color="grey",
#     lw=2,
#     alpha=0.4,
#     zorder=0,
#     # label="Heintz+23, z = 7-10",
# )
# ax.plot(
#     np.log10(metal),
#     heintz_23(np.log10(metal)),
#     color="grey",
#     lw=2,
#     alpha=0.4,
#     zorder=0,
# )
# ax.fill_between(
#     np.log10(metal),
#     heintz_23(np.log10(metal)) - 0.4,
#     heintz_23(np.log10(metal)) + 0.4,
#     alpha=0.2,
#     facecolor="grey",
#     linewidth=0.0,
#     zorder=0,
# )
# ax.fill_between(
#     np.log10(metal),
#     heintz_23(np.log10(metal)) - 0.4,
#     heintz_23(np.log10(metal)) + 0.4,
#     alpha=0.2,
#     facecolor="grey",
#     linewidth=0.0,
#     zorder=0,
# )
# ax.text(
#     3,
#     -3.25,
#     "Heintz+ 24, $z$ = 7 - 10",
#     rotation=30,
#     rotation_mode="anchor",
#     color="grey",
#     ha="left",
#     va="bottom",
#     fontsize=10,
# )


# Berg
ax.plot(
    np.log10(metal),
    local_berg_12(np.log10(metal)),
    color="darkorange",
    lw=2,
    alpha=0.4,
    zorder=0,
)
ax.fill_between(
    np.log10(metal),
    local_berg_12(np.log10(metal), 0.30 + 0.05, 5.43 + 0.42),
    local_berg_12(np.log10(metal), 0.30 - 0.05, 5.43 - 0.42),
    alpha=0.2,
    color="darkorange",
    linewidth=0.0,
    zorder=0,
)
ax.text(
    4.5,
    -1.85,
    "Berg+ 12, $z \sim 0$",
    rotation=27,
    rotation_mode="anchor",
    color="darkorange",
    ha="left",
    va="bottom",
    fontsize=10,
)


# Our Data
ax.scatter(
    np.log10(cc_mstar_running),
    np.log10(cc_zstar_running),
    color=vsfe_clr,
    edgecolors="none",
    alpha=0.8,
    s=20,
    label="VSFE",
)
ax.scatter(
    np.log10(f7_mstar_running),
    np.log10(f7_zstar_running),
    color=high_clr,
    edgecolors="none",
    alpha=0.8,
    s=20,
    label="high SFE",
)
ax.scatter(
    np.log10(f3_mstar_running),
    np.log10(f3_zstar_running),
    color=low_clr,
    edgecolors="none",
    alpha=0.8,
    s=20,
    label="low SFE",
)


ax.set(
    xlabel=r"$M_{\rm \star} [{\rm M_\odot}]$",
    ylabel=r"$\langle Z_{\rm \star} \rangle \:  [{\rm Z_\odot}]$",
)
ax.legend(frameon=False, loc="lower right")
ax.set(
    # xlabel=r"$M_{\rm \star} [{\rm M_\odot}]$",
    ylabel=r"$\log \: \langle Z_\star \rangle \:  [{\rm Z_\odot}]$",
    # xscale="log",
    # yscale="log",
    ylim=(-3.38, -1.4),
    xlim=(2.5, 6.5),
)
ax.text(0.05, 0.95, "binned, cumulative", ha="left", va="top", transform=ax.transAxes)


obs2 = ax.twinx()
obs2.locator_params(axis="y")
obs2.set(ylim=(-3.38, -1.4), ylabel=r"log (O/H) + 12")
obs2.set_yticklabels(list(np.round(obs2.get_yticks() + 8.69, 1).astype("str")))
ax.minorticks_on()
obs2.minorticks_on()

plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/Meanmetal_vs_Mstar-binned_cumu.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()

# %% binned
from scipy.stats import binned_statistic


def zstellar_mstellar_binned(
    logSFC,
    gas_prop_dir,
    start,
    end,
    sampling_tmyr=1,
):
    log_sfc = np.loadtxt(logSFC)
    redshift = log_sfc[:, 2]

    t_myr = t_myr_from_z(redshift)
    # print(t_myr)

    # cumulative stellas mass in the SFC
    mass_stars = np.cumsum(log_sfc[:, 7])
    cloud_zsun = log_sfc[:, 9] * 3.81

    interp_tmyr = np.linspace(t_myr.min(), t_myr.max(), 500)

    cloud_metal_interp = interpolate.interp1d(x=t_myr, y=cloud_zsun, kind="previous")
    cloud_zsun = cloud_metal_interp(interp_tmyr)

    plot_myr_bins = np.arange(t_myr.min(), t_myr.max(), sampling_tmyr)
    bin_means, bin_edges, binnumber = binned_statistic(
        interp_tmyr, cloud_zsun, statistic="mean", bins=plot_myr_bins
    )
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]

    # the tmyr corresponding to each mean value is the bin center
    bin_centers = 0.5 * (left_edges + right_edges)

    print(bin_centers)

    # take the previous mass,
    # but interpolate it so that it can have multiple samplings matching up
    # with the binned statistics before
    stellar_mass_interp = interpolate.interp1d(x=t_myr, y=mass_stars, kind="previous")
    mstar = stellar_mass_interp(bin_centers)

    valid_indices = bin_means != 0
    # binned_x = bin_centers[valid_indices]
    # bin_averages = np.nan_to_num(bin_means[valid_indices], nan=0)
    # binned_x = binned_x[bin_averages != 0]
    # bin_averages = bin_averages[bin_averages != 0]

    # previous look up method using masses, not good

    # bins = np.geomspace(4e2, 3e6, 10)  # bin galaxy masses
    # within each bin, compute the mean falling with
    # mean metallicity within each bin
    # bin_means, bin_edges, binnumber = binned_statistic(
    #     logsfc_mass_in_star, cloud_zsun, statistic="mean", bins=bins
    # )

    # bin_width = bin_edges[1] - bin_edges[0]
    # bin_centers = (bin_edges[1:] - bin_width) / 2

    # valid_indices = bin_means != 0
    # binned_x = bin_centers[valid_indices]
    # bin_averages = np.nan_to_num(bin_means[valid_indices], nan=0)
    # binned_x = binned_x[bin_averages != 0]
    # bin_averages = bin_averages[bin_averages != 0]

    #  get gas properties from the correspodning bin centers
    fpaths, snums = filter_snapshots(
        gas_prop_dir,
        start,
        end,
        sampling=1,
        str_snaps=True,
        snapshot_type="pop2_processed",
    )

    # get sampled times based on the bin centers, whatever is the closest
    # residuals = np.abs(logsfc_mass_in_star - binned_x[:, np.newaxis])
    # closest_match_idxs = np.argmin(residuals, axis=1)
    # sampled_times = t_myr[closest_match_idxs]

    #  get gas properties from the correspodning bin centers
    _, files = snapshot_from_time(fpaths, bin_centers)
    print(files)
    galaxy_gas_metal = []

    for i, file in enumerate(files):
        f = h5.File(file, "r")
        galaxy_gas_metal.append(f["Galaxy/MeanMetallicity"][()] * 3.81)
        f.close()

    metal = np.array(galaxy_gas_metal)

    # print(files)
    # print(sampled_times)
    # print(metal)
    # print(bin_averages)
    return mstar, bin_means, metal


cc_mstar, cc_zstar, cc_zgas = zstellar_mstellar_binned(
    "/home/fabg/container_tiramisu/sim_log_files/CC-Fiducial/logSFC",
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial",
    173,
    466,
    sampling_tmyr=10,
)

f7_mstar, f7_zstar, f7_zgas = zstellar_mstellar_binned(
    "/home/fabg/container_tiramisu/sim_log_files/fs07_refine/logSFC",
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    115,
    1570,
    sampling_tmyr=10,
)

f3_mstar, f3_zstar, f3_zgas = zstellar_mstellar_binned(
    "/home/fabg/container_tiramisu/sim_log_files/fs035_ms10/logSFC",
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    100,
    1606,
    sampling_tmyr=10,
)
# %%
fig, ax = plt.subplots(
    nrows=3,
    ncols=1,
    sharex=True,
    figsize=(4.6, 9),
    dpi=300,
    gridspec_kw={"height_ratios": [4, 4, 2]},
)
plt.subplots_adjust(wspace=0, hspace=0)

# stelllar metallicities
ax[0].scatter(
    np.log10(cc_mstar),
    np.log10(cc_zstar),
    color=vsfe_clr,
    edgecolors="none",
    alpha=0.8,
    label="VSFE",
    s=20,
)
ax[0].scatter(
    np.log10(f7_mstar),
    np.log10(f7_zstar),
    color=high_clr,
    edgecolors="none",
    alpha=0.8,
    s=20,
    label="HSFE",
    zorder=2,
)
ax[0].scatter(
    np.log10(f3_mstar),
    np.log10(f3_zstar),
    color=low_clr,
    edgecolors="none",
    alpha=0.8,
    s=20,
    label="LSFE",
    zorder=2,
)

# gas metallicities
ax[1].scatter(
    np.log10(cc_mstar),
    np.log10(cc_zgas),
    color=vsfe_clr,
    edgecolors="none",
    alpha=0.8,
    s=20,
    zorder=2,
)
ax[1].scatter(
    np.log10(f7_mstar),
    np.log10(f7_zgas),
    color=high_clr,
    edgecolors="none",
    alpha=0.8,
    s=20,
    zorder=2,
)
ax[1].scatter(
    np.log10(f3_mstar),
    np.log10(f3_zgas),
    color=low_clr,
    edgecolors="none",
    alpha=0.8,
    s=20,
    zorder=2,
)

ax[2].scatter(
    np.log10(cc_mstar), np.log10(cc_zstar / cc_zgas), zorder=2, color=vsfe_clr, s=5
)
ax[2].scatter(
    np.log10(f7_mstar), np.log10(f7_zstar / f7_zgas), zorder=2, color=high_clr, s=5
)
ax[2].scatter(
    np.log10(f3_mstar), np.log10(f3_zstar / f3_zgas), zorder=2, color=low_clr, s=5
)
ax[2].axhline(y=0, lw=2, color="grey", alpha=0.2, ls="--")


# =============================================================================
# # comparing to observations
# =============================================================================
def heintz_23(logmstar):
    twelve_log_oh = 0.33 * logmstar + 4.65
    zstar = twelve_log_oh - 8.69
    return zstar


metal = np.geomspace(1e2, 2e7, 20)
# heintz
ax[0].plot(
    np.log10(metal),
    heintz_23(np.log10(metal)),
    color="grey",
    lw=2,
    alpha=0.4,
    zorder=0,
    # label="Heintz+23, z = 7-10",
)
ax[1].plot(
    np.log10(metal),
    heintz_23(np.log10(metal)),
    color="grey",
    lw=2,
    alpha=0.4,
    zorder=0,
)
ax[0].fill_between(
    np.log10(metal),
    heintz_23(np.log10(metal)) - 0.4,
    heintz_23(np.log10(metal)) + 0.4,
    alpha=0.2,
    facecolor="grey",
    linewidth=0.0,
    zorder=0,
)
ax[1].fill_between(
    np.log10(metal),
    heintz_23(np.log10(metal)) - 0.4,
    heintz_23(np.log10(metal)) + 0.4,
    alpha=0.2,
    facecolor="grey",
    linewidth=0.0,
    zorder=0,
)
ax[0].text(
    5.0,
    -2.7,
    "Heintz+ 24, $z$ = 7 - 10",
    rotation=27,
    rotation_mode="anchor",
    color="grey",
    ha="left",
    va="bottom",
    fontsize=10,
)


# Curti
def curti_23(mstar, m=0.11, b=7.65):
    twelve_log_oh = m * np.log10(mstar / 1e8) + b
    zstar = twelve_log_oh - 8.69
    return zstar


# ax[0].plot(
#     np.log10(metal),
#     curti_23(metal),
#     color=cmap[4],
#     lw=2,
#     alpha=0.4,
#     zorder=0,
#     # label="curti_23, z = 7-10",
# )


# nakajima 23
def nakajima_23(mstar, m=0.25, b=8.24):
    twelve_log_oh = m * np.log10(mstar / 1e10) + b
    zstar = twelve_log_oh - 8.69
    return zstar


ax[0].plot(
    np.log10(metal),
    nakajima_23(metal),
    color="darkorange",
    ls="--",
    lw=2,
    alpha=0.8,
    zorder=0,
    # label="nakajima_23, z = 7-10",
)
ax[1].plot(
    np.log10(metal),
    nakajima_23(metal),
    color="darkorange",
    ls="--",
    lw=2,
    alpha=0.8,
    zorder=0,
)
ax[0].text(
    3.0,
    -2.15,
    "Nakajima+ 23, $z$ = 4 - 10",
    rotation=20,
    rotation_mode="anchor",
    color="darkorange",
    ha="left",
    va="bottom",
    fontsize=10,
)

# ax[0].fill_between(
#     np.log10(metal),
#     nakajima_23(metal, m=0.25 - 0.03, b=8.24 - 0.05),
#     nakajima_23(metal, m=0.25 + 0.03, b=8.24 + 0.05),
#     facecolor="darkorange",
#     alpha=0.2,
#     zorder=0,
#     linewidth=0.0,
# )
# ax[1].fill_between(
#     np.log10(metal),
#     nakajima_23(metal, m=0.25 - 0.03, b=8.24 - 0.05),
#     nakajima_23(metal, m=0.25 + 0.03, b=8.24 + 0.05),
#     facecolor="darkorange",
#     alpha=0.2,
#     zorder=0,
#     linewidth=0.0,
# )


# Moroshita
def moroshita_24(mstar, alpha=0.27, Bz=7.73, a_z=0.01):
    z = 3
    twelve_log_oh = alpha * np.log10(mstar / 10**8.8) + Bz  # + a_z * np.log(1 + z)
    zstar = twelve_log_oh - 8.69
    return zstar


ax[0].plot(
    np.log10(metal),
    moroshita_24(metal),
    color="olive",
    lw=2,
    alpha=0.5,
    zorder=0,
)
ax[1].plot(
    np.log10(metal),
    moroshita_24(metal),
    color="olive",
    lw=2,
    alpha=0.5,
    zorder=0,
)


ax[0].fill_between(
    np.log10(metal),
    moroshita_24(metal) + 0.26,
    moroshita_24(metal) - 0.26,
    facecolor="olive",
    alpha=0.2,
    zorder=0,
    linewidth=0.0,
)
ax[1].fill_between(
    np.log10(metal),
    moroshita_24(metal) + 0.26,
    moroshita_24(metal) - 0.26,
    facecolor="olive",
    alpha=0.2,
    zorder=0,
    linewidth=0.0,
)
ax[0].text(
    3.0,
    -2.48,
    "Moroshita+ 24, $z$ = 3 - 9.5",
    rotation=22,
    rotation_mode="anchor",
    color="olive",
    ha="left",
    va="bottom",
    fontsize=10,
)

ax[0].set(
    ylabel=r"$\log \: \langle Z_{\rm \star, inst.} \rangle \:  [{\rm Z_\odot}]$",
    ylim=(-3.38, -1.4),
    xlim=(2.5, 7),
)
ax[0].legend(frameon=False, loc="lower right", fontsize=10)


obs = ax[0].twinx()
obs.locator_params(axis="y")
obs.set(ylim=(-3.38, -1.4), ylabel=r"log (O/H) + 12")
obs.set_yticklabels(list(np.round(obs.get_yticks() + 8.69, 1).astype("str")))
obs.minorticks_on()

ax[1].set(
    ylabel=r"$\log \: \langle Z_{\rm gas, inst.} \rangle \:  [{\rm Z_\odot}]$",
    ylim=(-3.38, -1.4),
    xlim=(2.5, 6.5),
)
ax[1].minorticks_on()

obs1 = ax[1].twinx()
obs1.locator_params(axis="y")
obs1.set(ylim=(-3.38, -1.4), ylabel=r"log (O/H) + 12")
obs1.set_yticklabels(list(np.round(obs1.get_yticks() + 8.69, 1).astype("str")))
obs1.minorticks_on()


ax[2].set(
    xlabel=r"$\log \:  M_{\rm \star} [{\rm M_\odot}]$",
    ylim=(-0.5, 0.5),
)
ax[2].set_ylabel(
    r"$\log (\langle Z_{\rm \star, inst.} \rangle / \langle Z_{\rm gas, inst.} \rangle)$",
    fontsize=9,
)
ax[2].minorticks_on()
ax[2].locator_params(axis="y", nbins=4)
ax[0].text(
    0.05, 0.95, "binned, instantaneous", ha="left", va="top", transform=ax[0].transAxes
)
ax[0].minorticks_on()
ax[1].minorticks_on()
# colors = ["lightgrey", "lightgrey"]
# lw = [0, 0]
# ls = ["s", "o"]
# s = [8, 8]
# labels = ["Pop II", "gas"]
# lines = [
#     Line2D(
#         [0],
#         [0],
#         color=c,
#         markersize=s[i],
#         linewidth=lw[i],
#         marker=ls[i],
#         label=labels[i],
#     )
#     for i, c in enumerate(colors)
# ]
# handles.extend(lines)
# ax[0].legend(handles=handles, ncols=2, frameon=False, loc="upper left", fontsize=9)

plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/Meanmetal_vs_Mstar-binned_inst.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()
