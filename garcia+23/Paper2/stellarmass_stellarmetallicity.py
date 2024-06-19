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


def zstellar_mstellar(
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

    cloud_zsun = log_sfc[:, 9]
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

cc_mstar, cc_zstar = zstellar_mstellar(
    "/home/fabg/container_tiramisu/sim_log_files/CC-Fiducial/logSFC",
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial",
    356,
    466,
    sampling_tmyr=20,
)

f7_mstar, f7_zstar = zstellar_mstellar(
    "/home/fabg/container_tiramisu/sim_log_files/fs07_refine/logSFC",
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    115,
    1570,
    sampling_tmyr=20,
)

f3_mstar, f3_zstar = zstellar_mstellar(
    "/home/fabg/container_tiramisu/sim_log_files/fs035_ms10/logSFC",
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    100,
    1606,
    sampling_tmyr=20,
)

cmap = matplotlib.colormaps["Dark2"]
cmap = cmap(np.linspace(0, 1, 8))
vsfe_clr = cmap[0]
high_clr = cmap[1]
low_clr = cmap[2]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 4), dpi=300)
# ax.scatter(cc_mstar[3::], cc_zgas[3::], color=vsfe_clr, alpha=0.8)
# ax.scatter(f7_mstar, f7_zgas, color=high_clr, alpha=0.8)
# ax.scatter(f3_mstar, f3_zgas, color=low_clr, alpha=0.8)

ax.scatter(
    cc_mstar,
    cc_zstar,
    color=vsfe_clr,
    edgecolors="black",
    alpha=0.9,
    s=50,
    label="VSFE",
)
ax.scatter(
    f7_mstar,
    f7_zstar,
    color=high_clr,
    edgecolors="black",
    s=50,
    alpha=0.9,
    label="high SFE",
)
ax.scatter(
    f3_mstar,
    f3_zstar,
    color=low_clr,
    edgecolors="black",
    alpha=0.9,
    s=50,
    label="low SFE",
)


ax.set(
    xlabel=r"$M_{\rm \star} [{\rm M_\odot}]$",
    ylabel=r"$\langle Z_{\rm \star} \rangle \:  [{\rm Z_\odot}]$",
    xscale="log",
    yscale="log",
    ylim=(2e-4, 5e-3),
    xlim=(2e2, 2e6),
)
ax.legend(frameon=False)

# plt.savefig(
#     "../../../gdrive_columbia/research/massimo/paper2/Meanmetal_vs_Mstar-interpolated.png",
#     dpi=300,
#     bbox_inches="tight",
#     pad_inches=0.05,
# )
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

    plot_myr_bins = np.arange(t_myr.min(), t_myr.max(), 10)
    bin_means, bin_edges, binnumber = binned_statistic(
        interp_tmyr, cloud_zsun, statistic="mean", bins=plot_myr_bins
    )
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]

    # the tmyr corresponding to each mean value is the bin center
    bin_centers = 0.5 * (left_edges + right_edges)

    print(bin_centers)
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
    153,
    466,
)

f7_mstar, f7_zstar, f7_zgas = zstellar_mstellar_binned(
    "/home/fabg/container_tiramisu/sim_log_files/fs07_refine/logSFC",
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    115,
    1570,
)

f3_mstar, f3_zstar, f3_zgas = zstellar_mstellar_binned(
    "/home/fabg/container_tiramisu/sim_log_files/fs035_ms10/logSFC",
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    100,
    1606,
)
# %%
fig, ax = plt.subplots(
    nrows=2,
    ncols=1,
    sharex=True,
    figsize=(5, 5),
    dpi=300,
    gridspec_kw={"height_ratios": [4, 2]},
)
plt.subplots_adjust(wspace=0, hspace=0)
ax[0].scatter(
    cc_mstar[3::], cc_zgas[3::], color=vsfe_clr, edgecolors="black", alpha=0.8
)
ax[0].scatter(f7_mstar, f7_zgas, color=high_clr, edgecolors="black", alpha=0.8)
ax[0].scatter(f3_mstar, f3_zgas, color=low_clr, edgecolors="black", alpha=0.8)

ax[0].scatter(
    cc_mstar,
    cc_zstar,
    color=vsfe_clr,
    edgecolors="black",
    alpha=0.9,
    # s=60,
    label="VSFE",
    marker="s",
)
ax[0].scatter(
    f7_mstar,
    f7_zstar,
    color=high_clr,
    edgecolors="black",
    # s=60,
    alpha=0.9,
    label="high SFE",
    marker="s",
)
ax[0].scatter(
    f3_mstar,
    f3_zstar,
    color=low_clr,
    edgecolors="black",
    alpha=0.9,
    # s=60,
    label="low SFE",
    marker="s",
)

# gas properties of the SF region
# note the gas properties are so low because the gas
ax[0].scatter(cc_mstar, cc_zgas, color=vsfe_clr, edgecolors="black", alpha=0.7, s=50)
ax[0].scatter(f7_mstar, f7_zgas, color=high_clr, edgecolors="black", alpha=0.7, s=50)
ax[0].scatter(f3_mstar, f3_zgas, color=low_clr, edgecolors="black", alpha=0.7, s=50)

x = np.geomspace(5e4, 2e6, 20)
ax[0].plot(x, 1e-6 * x**0.7, "--k")


ax[1].scatter(
    cc_mstar,
    np.log10(cc_zstar / cc_zgas),
    color=vsfe_clr,
    # edgecolors="black",
    s=5,
    # alpha=0.7,
)
ax[1].scatter(
    f7_mstar,
    np.log10(f7_zstar / f7_zgas),
    color=high_clr,
    # edgecolors="black",
    s=5,
    # alpha=0.7,
)
ax[1].scatter(
    f3_mstar,
    np.log10(f3_zstar / f3_zgas),
    color=low_clr,
    # edgecolors="black",
    s=5,
    # alpha=0.7,
)


ax[1].set(
    xlabel=r"$M_{\rm \star} [{\rm M_\odot}]$",
    ylabel=r"$\log (Z_{\rm \star} / Z_{\rm gas})$",
    xscale="log",
    # yscale="log"
    ylim=(-0.5, 0.5),
    # xlim=(2e2, 2e6),
)
ax[1].minorticks_on()
ax[1].locator_params(axis="y", nbins=4)

handles, labels = ax[0].get_legend_handles_labels()


ax[0].set(
    # xlabel=r"$M_{\rm \star} [{\rm M_\odot}]$",
    ylabel=r"$\langle Z \rangle \:  [{\rm Z_\odot}]$",
    xscale="log",
    yscale="log",
    ylim=(5e-4, 9e-2),
    xlim=(7e2, 2e6),
)
ax[0].legend(frameon=False)
colors = ["lightgrey", "lightgrey"]
lw = [0, 0]
ls = ["s", "o"]
s = [8, 8]
labels = ["Pop II", "gas"]
lines = [
    Line2D(
        [0],
        [0],
        color=c,
        markersize=s[i],
        linewidth=lw[i],
        marker=ls[i],
        label=labels[i],
    )
    for i, c in enumerate(colors)
]
handles.extend(lines)
ax[0].legend(handles=handles, ncols=2, frameon=False, loc="upper left", fontsize=10)

plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/Meanmetal_vs_Mstar-binned.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()
