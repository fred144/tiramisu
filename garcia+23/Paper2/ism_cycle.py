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
from outflow_rates import read_outflow_rates
import seaborn as sns

from scipy.stats import binned_statistic


def read_masses(path, start, stop, region="Galaxy", step=1):
    fpaths, snums = filter_snapshots(
        path,
        start,
        stop,
        sampling=step,
        str_snaps=True,
        snapshot_type="pop2_processed",
    )

    galaxy_gas_mass = []
    galaxy_metal_mass = []
    times = []
    for i, file in enumerate(fpaths):
        f = h5.File(file, "r")
        times.append(f["Header/time"][()])
        galaxy_gas_mass.append(f["{}/GasMass".format(region)][()])
        galaxy_metal_mass.append(f["{}/MetalMass".format(region)][()])
        f.close()

    return (
        np.array(times),
        np.array(galaxy_gas_mass),
        np.array(galaxy_metal_mass) * 3.81,
    )


def read_mean_metallicities(path, start, stop, step=1):
    fpaths, snums = filter_snapshots(
        # "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
        # 306,
        # 466,
        path,
        start,
        stop,
        sampling=step,
        str_snaps=True,
        snapshot_type="pop2_processed",
    )

    galaxy_metal = []
    cgm_metal = []
    igm_metal = []
    vir_metal = []
    times = []
    for i, file in enumerate(fpaths):
        f = h5.File(file, "r")
        galaxy_metal.append(f["Galaxy/MeanMetallicity"][()] * 3.81)
        cgm_metal.append(f["CGM/MeanMetallicity"][()] * 3.81)
        igm_metal.append(f["IGM/MeanMetallicity"][()] * 3.81)
        vir_metal.append(f["Halo/MeanMetallicity"][()] * 3.81)
        times.append(f["Header/time"][()])
        f.close()

    return times, cgm_metal, galaxy_metal, igm_metal, vir_metal


def read_cloud_properties(logsfc_path):
    # logsfc_path = "/home/fabg/container_tiramisu/sim_log_files/CC-Fiducial/logSFC"

    log_sfc = np.loadtxt(logsfc_path)
    redshift = log_sfc[:, 2]
    t_myr = t_myr_from_z(redshift)
    cloud_metal = log_sfc[:, 9] * 3.81
    cloud_dens = log_sfc[:, 8]
    return t_myr, cloud_metal, cloud_dens


t_myr, cloud_metal, cloud_nh = read_cloud_properties(
    "/home/fabg/container_tiramisu/sim_log_files/CC-Fiducial/logSFC"
)
f70_t_myr, f70_cloud_metal, f70_cloud_nh = read_cloud_properties(
    "/home/fabg/container_tiramisu/sim_log_files/fs07_refine/logSFC"
)
f35_t_myr, f35_cloud_metal, f35_cloud_nh = read_cloud_properties(
    "/home/fabg/container_tiramisu/sim_log_files/fs035_ms10/logSFC"
)


cc_times, cc_m_out, cc_mz_out, cc_m_in, cc_mz_in = read_outflow_rates(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
    173,
    466,
)

f7_times, f7_m_out, f7_mz_out, f7_m_in, f7_mz_in = read_outflow_rates(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    100,
    1570,
    step=6,
)

f3_times, f3_m_out, f3_mz_out, f3_m_in, f3_mz_in = read_outflow_rates(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    100,
    1606,
    step=6,
)


_, cc_mgas_cgm, cc_mzgas_cgm = read_masses(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
    173,
    466,
    region="CGM",
)

_, f7_mgas_cgm, f7_mzgas_cgm = read_masses(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    100,
    1570,
    region="CGM",
    step=6,
)

_, f3_mgas_cgm, f3_mzgas_cgm = read_masses(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    100,
    1606,
    region="CGM",
    step=6,
)

_, cc_mgas_igm, cc_mzgas_igm = read_masses(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
    173,
    466,
    region="IGM",
)

_, f7_mgas_igm, f7_mzgas_igm = read_masses(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    100,
    1570,
    region="IGM",
    step=6,
)

_, f3_mgas_igm, f3_mzgas_igm = read_masses(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    100,
    1606,
    region="IGM",
    step=6,
)


(
    cc_times,
    cc_cgm_metal,
    cc_galaxy_metal,
    cc_igm_metal,
    cc_vir_metal,
) = read_mean_metallicities(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
    173,
    466,
)
(
    f70_times,
    f70_cgm_metal,
    f70_galaxy_metal,
    f70_igm_metal,
    f70_vir_metal,
) = read_mean_metallicities(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    100,
    1570,
    step=6,
)
(
    f35_times,
    f35_cgm_metal,
    f35_galaxy_metal,
    f35_igm_metal,
    f35_vir_metal,
) = read_mean_metallicities(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    100,
    1606,
    step=6,
)

cc_times, cc_mgas, cc_mzgas = read_masses(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
    173,
    466,
)

f7_times, f7_mgas, f7_mzgas = read_masses(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    100,
    1570,
    step=6,
)

f3_times, f3_mgas, f3_mzgas = read_masses(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    100,
    1606,
    step=6,
)


# %%
def metal_ratio(logsntime, logsnmetal, time2, metal2):
    logsntime = np.array(logsntime)
    logsnmetal = np.array(logsnmetal)
    time2 = np.array(time2)
    metal2 = np.array(metal2)

    mask = logsntime > time2.min()

    logsntime = logsntime[mask]
    logsnmetal = logsnmetal[mask]

    time_mask = np.ones_like(time2)

    for i, t in enumerate(time2):
        closest_age_idx = np.argmin(np.abs(logsntime - t))

        time_mask[i] = closest_age_idx

    # downsample the logSN to match closer to the
    closest_matching_metal = logsnmetal[np.array(time_mask, dtype="int")]

    ratio = metal2 / closest_matching_metal
    return time2, ratio


def read_sne(logsfc_path, interuped=False):
    # logsfc_path = "/home/fabg/container_tiramisu/sim_log_files/CC-Fiducial/logSFC"
    log_sfc = np.loadtxt(logsfc_path)
    if interuped is True:
        log_sfc_1 = np.loadtxt(logsfc_path + "-earlier")
        log_sfc = np.concatenate((log_sfc_1, log_sfc), axis=0)

    redshift = log_sfc[:, 2]

    t_myr = t_myr_from_z(redshift)
    # SNe properties
    m_ejecta = log_sfc[:, 6]
    e_thermal_injected = log_sfc[:, 7]
    ejecta_zsun = log_sfc[:, 8] * 3.81

    # let's do the accumulation of metals produced
    mass_in_metals = m_ejecta * ejecta_zsun

    total_mass_in_metals = np.cumsum(mass_in_metals)

    return t_myr, total_mass_in_metals


cc_t_myr, cc_sn_metal = read_sne(
    "/home/fabg/container_tiramisu/sim_log_files/CC-Fiducial/logSN", interuped=True
)
f7_t_myr, f7_sn_metal = read_sne(
    "/home/fabg/container_tiramisu/sim_log_files/fs07_refine/logSN"
)
f3_t_myr, f3_sn_metal = read_sne(
    "/home/fabg/container_tiramisu/sim_log_files/fs035_ms10/logSN"
)

# mass in the sf region / mass produced by SNe
cc_t, cc_sf_ratio = metal_ratio(cc_t_myr, cc_sn_metal, cc_times, cc_mzgas)
f7_t, f7_sf_ratio = metal_ratio(f7_t_myr, f7_sn_metal, f7_times, f7_mzgas)
f3_t, f3_sf_ratio = metal_ratio(f3_t_myr, f3_sn_metal, f3_times, f3_mzgas)

cmap = matplotlib.colormaps["Dark2"]
cmap = sns.color_palette(palette="rocket")
vsfe_clr = cmap[0]
high_clr = cmap[1]
low_clr = cmap[2]
# %%
cmap = sns.color_palette(palette="rocket")
fig, ax = plt.subplots(
    nrows=4,
    ncols=3,
    figsize=(12, 10),
    dpi=300,
    sharex=True,
    sharey="row",
    gridspec_kw={"height_ratios": [3, 3, 3, 2]},
)
ax = ax.ravel()
plt.subplots_adjust(hspace=0.0, wspace=0)


ax[0].text(475, 3.5e-2, "(a)", ha="center", va="center", fontsize=9)
ax[0].text(578, 3.5e-2, "(b)", ha="center", va="center", fontsize=9)
ax[1].text(455, 3.5e-2, "(c)", ha="center", va="center", fontsize=9)
ax[1].text(669, 3.5e-2, "(d)", ha="center", va="center", fontsize=9)
ax[2].text(425, 3.5e-2, "(e)", ha="center", va="center", fontsize=9)
ax[2].text(490, 3.5e-2, "(f)", ha="center", va="center", fontsize=9)
ax[2].text(550, 3.5e-2, "(g)", ha="center", va="center", fontsize=9)
ax[2].text(600, 3.5e-2, "(h)", ha="center", va="center", fontsize=9)

for i in range(0, 10, 3):
    ax[i].axvspan(465, 485, facecolor="grey", alpha=0.3)
    ax[i].axvspan(565, 588, facecolor="grey", alpha=0.3)

for i in range(1, 11, 3):
    ax[i].axvspan(440, 470, facecolor="grey", alpha=0.3)
    ax[i].axvspan(660, 678, facecolor="grey", alpha=0.3)

for i in range(2, 12, 3):
    ax[i].axvspan(410, 440, facecolor="grey", alpha=0.3)
    ax[i].axvspan(480, 500, facecolor="grey", alpha=0.3)
    ax[i].axvspan(540, 560, facecolor="grey", alpha=0.3)
    ax[i].axvspan(590, 610, facecolor="grey", alpha=0.3)


ax[0].plot(cc_times, cc_galaxy_metal, color=cmap[1], lw=3, label="ISM")
ax[1].plot(f70_times, f70_galaxy_metal, color=cmap[1], lw=3)
ax[2].plot(f35_times, f35_galaxy_metal, color=cmap[1], lw=3)

ax[0].plot(cc_times, cc_cgm_metal, color=cmap[3], lw=3, label="CGM")
ax[1].plot(f70_times, f70_cgm_metal, color=cmap[3], lw=3)
ax[2].plot(f35_times, f35_cgm_metal, color=cmap[3], lw=3)

ax[0].plot(cc_times, cc_vir_metal, color=cmap[4], lw=3, ls=":", label="Halo")
ax[1].plot(f70_times, f70_vir_metal, color=cmap[4], lw=3, ls=":")
ax[2].plot(f35_times, f35_vir_metal, color=cmap[4], lw=3, ls=":")

nh_cmap = mpl.colors.ListedColormap(
    matplotlib.colormaps["viridis_r"](np.linspace(0, 1, 8))
)
norm = mpl.colors.BoundaryNorm(
    np.round(
        np.linspace(
            np.log10(cloud_nh.min()), np.log10(cloud_nh.max()), 8, endpoint=True
        ),
        1,
    ),
    nh_cmap.N,
)
# normalize = matplotlib.colors.Normalize(vmin=cloud_nh.min(), vmax=cloud_nh.max())
nh_scatter = ax[0].scatter(
    t_myr,
    cloud_metal,
    c=np.log10(cloud_nh),
    cmap=nh_cmap,
    norm=norm,
    alpha=0.5,
    s=30,
    marker="o",
    label="star forming cloud",
)

ax[1].scatter(
    f70_t_myr,
    f70_cloud_metal,
    c=np.log10(f70_cloud_nh),
    cmap=nh_cmap,
    norm=norm,
    alpha=0.5,
    s=30,
    marker="o",
)
ax[2].scatter(
    f35_t_myr,
    f35_cloud_metal,
    c=np.log10(f35_cloud_nh),
    cmap=nh_cmap,
    norm=norm,
    alpha=0.5,
    s=30,
    marker="o",
)

cbar_ax = ax[1].inset_axes([0.4, 0.15, 0.5, 0.05])
dens_bar = fig.colorbar(nh_scatter, cax=cbar_ax, pad=0, orientation="horizontal")
cbar_ax.minorticks_on()
cbar_ax.tick_params(labelsize=8)
cbar_ax.set_title(
    label=(r"$\log\:\overline{n_\mathrm{H}}\:\left[\mathrm{cm}^{-3} \right]$"),
    fontsize=10,
    # labelpad=6,
)


ax[3].plot(cc_t_myr, cc_sn_metal, lw=3, color=cmap[4], label="SNe", ls="--")
ax[4].plot(f7_t_myr, f7_sn_metal, color=cmap[4], lw=3, ls="--")
ax[5].plot(f3_t_myr, f3_sn_metal, color=cmap[4], lw=3, ls="--")

ax[3].plot(cc_times, cc_mzgas, lw=3, color=cmap[1], label="ISM")
ax[4].plot(f7_times, f7_mzgas, color=cmap[1], lw=3)
ax[5].plot(f3_times, f3_mzgas, color=cmap[1], lw=3)

ax[3].plot(cc_times, cc_mzgas_cgm, lw=3, color=cmap[3], alpha=0.8, label="CGM")
ax[4].plot(f7_times, f7_mzgas_cgm, color=cmap[3], lw=3, alpha=0.8)
ax[5].plot(f3_times, f3_mzgas_cgm, color=cmap[3], lw=3, alpha=0.8)

ax[3].plot(cc_times, cc_mzgas_igm, lw=3, color=cmap[2], alpha=0.8, label="IGM")
ax[4].plot(f7_times[::5], f7_mzgas_igm[::5], color=cmap[2], alpha=0.8, lw=3)
ax[5].plot(f3_times[::5], f3_mzgas_igm[::5], color=cmap[2], alpha=0.8, lw=3)


ax[3].set(
    ylim=(8e2, 3e6),
    yscale="log",
    ylabel=r"$M_{\rm metals} \: [\mathrm{M_\odot}]$",
)


# ax[6].plot(cc_t, cc_sf_ratio, lw=3, color=cmap[0])
# ax[7].plot(f7_t, f7_sf_ratio, color=cmap[0], lw=3)
# ax[8].plot(f3_t, f3_sf_ratio, color=cmap[0], lw=3)

ax[6].plot(cc_times, cc_mgas, color=cmap[1], lw=3, label="ISM")
ax[7].plot(f7_times, f7_mgas, color=cmap[1], lw=3)
ax[8].plot(f3_times, f3_mgas, color=cmap[1], lw=3)

ax[6].plot(cc_times, cc_mgas_cgm, color=cmap[3], lw=3, label="CGM")
ax[7].plot(f7_times, f7_mgas_cgm, color=cmap[3], lw=3)
ax[8].plot(f3_times, f3_mgas_cgm, color=cmap[3], lw=3)

ax[6].plot(cc_times, cc_mgas_igm, color=cmap[2], lw=3, label="IGM")
ax[7].plot(f7_times, f7_mgas_igm, color=cmap[2], lw=3)
ax[8].plot(f3_times, f3_mgas_igm, color=cmap[2], lw=3)


ax[6].set(
    yscale="log",
    ylabel=r" $M_{\rm gas} \: \left[{\rm M_\odot}\right]$",
    ylim=(8e4, 2e8),
)
ax[7].locator_params(axis="x", nbins=10)


ax[9].plot(cc_t, cc_sf_ratio, lw=3, color=cmap[0])
ax[10].plot(f7_t, f7_sf_ratio, color=cmap[0], lw=3)
ax[11].plot(f3_t, f3_sf_ratio, color=cmap[0], lw=3)

ax[9].set(ylim=(0, 0.85), ylabel="ISM metal retention")
ax[10].set(xlabel=r"time [Myr]")
ax[0].minorticks_on()
ax[9].minorticks_on()


for i in range(0, 3):
    redshft_ax = ax[i].twiny()
    redshft_ax.locator_params(axis="x", nbins=10)
    if i == 1:
        redshft_ax.set_xlabel("$z$", fontsize=10)
    redshft_ax.set(xlim=(350, 735))

    redshft_ax.set_xticklabels(
        list(np.round(z_from_t_myr(redshft_ax.get_xticks()), 1).astype("str"))
    )


ax[0].set(ylabel=r"$\langle Z \rangle \: [\mathrm{Z_\odot}$]", yscale="log")
ax[0].legend(frameon=False, ncols=1, fontsize=10, loc="lower right")
ax[0].set(xlim=(350, 735), ylim=(8e-4, 5e-2))


ax[3].legend(frameon=False, ncols=1, fontsize=10, loc="upper left")

ax[6].legend(frameon=False, ncols=1, fontsize=10, loc="lower right")


ax[0].text(
    0.05, 0.90, "VSFE", ha="left", va="top", fontsize=10, transform=ax[0].transAxes
)
ax[1].text(
    0.05, 0.90, "HSFE", ha="left", va="top", fontsize=10, transform=ax[1].transAxes
)
ax[2].text(
    0.05, 0.90, "LSFE", ha="left", va="top", fontsize=10, transform=ax[2].transAxes
)

plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/ism_metal_cycle.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()

# %%
# cc_times, cc_m_out, cc_mz_out, cc_m_in, cc_mz_in


def mass_loading(logsfc_path, outflow_times, outflow_rates, sampling_tmyr):
    r"""
    bin sfr to calculate mass loading
    """
    sfr_binwidth_myr = 1

    log_sfc = np.loadtxt(logsfc_path)
    redshift = log_sfc[:, 2]
    t_myr = t_myr_from_z(redshift)
    cloud_metal = log_sfc[:, 9] * 3.81
    cloud_dens = log_sfc[:, 8]
    mass_in_star = log_sfc[:, 7]

    running_total_mass = np.cumsum(mass_in_star)
    t_interp_points = np.arange(t_myr.min(), t_myr.max(), sfr_binwidth_myr)
    total_mass_interpolator = interpolate.interp1d(
        x=t_myr, y=running_total_mass, kind="previous"
    )
    total_mass = total_mass_interpolator(t_interp_points)
    # calculate the sfr in msun / yr
    sfr = np.gradient(total_mass) / (sfr_binwidth_myr * 1e6)

    # average_bins_myr = np.arange(
    #     outflow_times.min(), outflow_times.max(), sampling_tmyr
    # )

    average_bins_myr = np.arange(350, 750, sampling_tmyr)

    mean_sfr_binnedmyr, bin_edges, binnumber = binned_statistic(
        t_interp_points, sfr, statistic="mean", bins=average_bins_myr
    )
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]

    # the tmyr corresponding to each mean value is the bin center
    bin_centers = 0.5 * (left_edges + right_edges)

    mean_outflowrate_binnedmyr, _, _ = binned_statistic(
        outflow_times, outflow_rates, statistic="mean", bins=average_bins_myr
    )

    # mask = (np.isnan(mean_sfr_binnedmyr)) & (mean_sfr_binnedmyr> 0)

    # print(average_bins_myr[mask])
    # print(mean_sfr_binnedmyr[~mask])

    # return bin_centers, mean_outflowrate_binnedmyr[~mask] / mean_sfr_binnedmyr[~mask]
    return bin_centers, mean_outflowrate_binnedmyr / mean_sfr_binnedmyr


def metal_loading(logsfc_path, metal_times, metal_rates, ism_metal, sampling_tmyr):
    r"""
    bin sfr to calculate metallicty
    """
    sfr_binwidth_myr = 1

    log_sfc = np.loadtxt(logsfc_path)
    redshift = log_sfc[:, 2]
    t_myr = t_myr_from_z(redshift)
    cloud_metal = log_sfc[:, 9] * 3.81
    cloud_dens = log_sfc[:, 8]
    mass_in_star = log_sfc[:, 7]

    running_total_mass = np.cumsum(mass_in_star)
    t_interp_points = np.arange(t_myr.min(), t_myr.max(), sfr_binwidth_myr)
    total_mass_interpolator = interpolate.interp1d(
        x=t_myr, y=running_total_mass, kind="previous"
    )
    total_mass = total_mass_interpolator(t_interp_points)
    # calculate the sfr in msun / yr
    sfr = np.gradient(total_mass) / (sfr_binwidth_myr * 1e6)

    # average_bins_myr = np.arange(
    #     outflow_times.min(), outflow_times.max(), sampling_tmyr
    # )

    average_bins_myr = np.arange(350, 750, sampling_tmyr)

    mean_sfr_binnedmyr, bin_edges, binnumber = binned_statistic(
        t_interp_points, sfr, statistic="mean", bins=average_bins_myr
    )
    right_edges = bin_edges[1:]
    left_edges = bin_edges[:-1]

    # the tmyr corresponding to each mean value is the bin center
    bin_centers = 0.5 * (left_edges + right_edges)

    mean_outflowrate_binnedmyr, _, _ = binned_statistic(
        metal_times, metal_rates, statistic="mean", bins=average_bins_myr
    )

    mean_ism_metal, _, _ = binned_statistic(
        metal_times, ism_metal, statistic="mean", bins=average_bins_myr
    )

    return bin_centers, mean_outflowrate_binnedmyr / (
        mean_ism_metal * mean_sfr_binnedmyr
    )


# cc_times, cc_galaxy_metal

cc_t, cc_massloading = mass_loading(
    "/home/fabg/container_tiramisu/sim_log_files/CC-Fiducial/logSFC",
    cc_times,
    cc_m_out,
    50,
)

f7_t, f7_massloading = mass_loading(
    "/home/fabg/container_tiramisu/sim_log_files/fs07_refine/logSFC",
    f7_times,
    f7_m_out,
    50,
)

f3_t, f3_massloading = mass_loading(
    "/home/fabg/container_tiramisu/sim_log_files/fs035_ms10/logSFC",
    f3_times,
    f3_m_out,
    50,
)


cc_tmetal, cc_metalloading = metal_loading(
    "/home/fabg/container_tiramisu/sim_log_files/CC-Fiducial/logSFC",
    cc_times,
    cc_mz_out,
    cc_galaxy_metal,
    50,
)

f7_tmetal, f7_metalloading = metal_loading(
    "/home/fabg/container_tiramisu/sim_log_files/fs07_refine/logSFC",
    f7_times,
    f7_mz_out,
    f70_galaxy_metal,
    50,
)


f3_tmetal, f3_metalloading = metal_loading(
    "/home/fabg/container_tiramisu/sim_log_files/fs035_ms10/logSFC",
    f3_times,
    f3_mz_out,
    f35_galaxy_metal,
    50,
)


# %%

cmap = sns.color_palette(palette="cubehelix")
fig, ax = plt.subplots(2, 1, figsize=(4.5, 6), dpi=300, sharex=True)
plt.subplots_adjust(hspace=0.0, wspace=0)

ax[0].plot(cc_t, cc_massloading, c=cmap[1], lw=2, label="VSFE")
ax[0].plot(f7_t, f7_massloading, c=cmap[3], lw=2, label="HSFE")
ax[0].plot(f3_t, f3_massloading, c=cmap[4], lw=2, label="LSFE")
ax[0].set(ylabel=r"$\eta_{\rm gas}$", yscale="log", ylim=(5, 3e4))

ax[1].plot(cc_tmetal, cc_metalloading, c=cmap[1], lw=2, label="VSFE")
ax[1].plot(f7_tmetal, f7_metalloading, c=cmap[3], lw=2, label="HSFE")
ax[1].plot(f3_tmetal, f3_metalloading, c=cmap[4], lw=2, label="LSFE")
ax[1].set(
    ylabel=r"$\eta_{\rm metals}$",
    yscale="log",
    ylim=(5, 3e5),
    xlim=(345, 735),
    xlabel=r"time [Myr]",
)

ax[0].minorticks_on()

redshft_ax = ax[0].twiny()
redshft_ax.minorticks_on()
redshft_ax.set_xlabel("$z$")
redshft_ax.set(xlim=(345, 735))
redshft_ax.set_xticklabels(
    list(np.round(z_from_t_myr(redshft_ax.get_xticks()), 1).astype("str"))
)
redshft_ax.locator_params(axis="x", nbins=12)

ax[0].legend(frameon=False)
plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/massandmetal_loading.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()
# %%

cmap = sns.color_palette(palette="cubehelix")
fig, ax = plt.subplots(
    nrows=3,
    ncols=3,
    figsize=(12, 8),
    dpi=300,
    sharex=True,
    sharey="row",
    gridspec_kw={"height_ratios": [3, 3, 3]},
)

ax = ax.ravel()
plt.subplots_adjust(hspace=0.0, wspace=0)

for i in range(0, 8, 3):
    ax[i].axvspan(465, 485, facecolor="grey", alpha=0.3)
    ax[i].axvspan(565, 588, facecolor="grey", alpha=0.3)

for i in range(1, 9, 3):
    ax[i].axvspan(440, 470, facecolor="grey", alpha=0.3)
    ax[i].axvspan(660, 678, facecolor="grey", alpha=0.3)

for i in range(2, 10, 3):
    ax[i].axvspan(410, 440, facecolor="grey", alpha=0.3)
    ax[i].axvspan(480, 500, facecolor="grey", alpha=0.3)
    ax[i].axvspan(540, 560, facecolor="grey", alpha=0.3)
    ax[i].axvspan(590, 610, facecolor="grey", alpha=0.3)


ax[0].plot(cc_times, cc_mz_out, label="outflowing", color=cmap[2], lw=3)
ax[1].plot(f7_times, f7_mz_out, color=cmap[2], lw=3)
ax[2].plot(f3_times, f3_mz_out, color=cmap[2], lw=3)

ax[0].plot(cc_times, -cc_mz_in, label="inflowing", color=cmap[3], lw=3)
ax[1].plot(f7_times, -f7_mz_in, color=cmap[3], lw=3)
ax[2].plot(f3_times, -f3_mz_in, color=cmap[3], lw=3)

ax[3].plot(cc_times, cc_m_out, color=cmap[2], lw=3, label="outflowing")
ax[4].plot(f7_times, f7_m_out, color=cmap[2], lw=3)
ax[5].plot(f3_times, f3_m_out, color=cmap[2], lw=3)

ax[3].plot(cc_times, -cc_m_in, label="inflowing", color=cmap[3], lw=3)
ax[4].plot(f7_times, -f7_m_in, color=cmap[3], lw=3)
ax[5].plot(f3_times, -f3_m_in, color=cmap[3], lw=3)

ax[3].set(
    ylabel=r"$|\dot{M}_{\rm gas}|\: \left[{\rm M_\odot \: yr^{-1}}\right]$",
    yscale="log",
)

ax[6].plot(cc_times, cc_mz_out / cc_m_out, color=cmap[2], label="outflowing", lw=3)
ax[7].plot(f7_times, f7_mz_out / f7_m_out, color=cmap[2], lw=3)
ax[8].plot(f3_times, f3_mz_out / f3_m_out, color=cmap[2], lw=3)

ax[6].plot(cc_times, cc_mz_in / cc_m_in, label="inflowing", color=cmap[3], lw=3)
ax[7].plot(f7_times, f7_mz_in / f7_m_in, color=cmap[3], lw=3)
ax[8].plot(f3_times, f3_mz_in / f3_m_in, color=cmap[3], lw=3)


ax[6].set(
    ylabel=r"$\langle Z \rangle \: [\mathrm{Z_\odot}$]",
    yscale="log",
    xlim=(345, 735),
)
ax[0].set(
    ylabel=r"$|\dot{M}_{\rm metal}|\: \left[{\rm M_\odot \: yr^{-1}}\right]$",
    yscale="log",
)


ax[0].legend(frameon=False)
# ax[3].legend(frameon=False, ncols=2)
ax[0].minorticks_on()
ax[7].set(xlabel=r"time [Myr]")
ax[7].locator_params(axis="x", nbins=12)

ax[0].text(
    0.05, 0.90, "VSFE", ha="left", va="top", fontsize=9, transform=ax[0].transAxes
)
ax[1].text(
    0.05, 0.90, "HSFE", ha="left", va="top", fontsize=9, transform=ax[1].transAxes
)
ax[2].text(
    0.05, 0.90, "LSFE", ha="left", va="top", fontsize=9, transform=ax[2].transAxes
)

for i in range(0, 3):
    redshft_ax = ax[i].twiny()
    redshft_ax.locator_params(axis="x", nbins=12)
    if i == 1:
        redshft_ax.set_xlabel("$z$")
    redshft_ax.set(xlim=(345, 735))

    redshft_ax.set_xticklabels(
        list(np.round(z_from_t_myr(redshft_ax.get_xticks()), 1).astype("str"))
    )


plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/gas_outflow_rates.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()
