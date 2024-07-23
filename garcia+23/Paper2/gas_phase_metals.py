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

    metal_profile = []
    times = []
    metal_radii = []

    for i, file in enumerate(fpaths):
        f = h5.File(file, "r")
        metalbins = f["Profiles/MetalDensityWeighted"][:]
        metalbins_mask = metalbins > 0

        redshift = f["Header/redshift"][()]
        radius = f["Profiles/Radius"][:][:-1]
        radius = radius[metalbins_mask]

        metal_profile.append(metalbins[metalbins_mask])
        metal_radii.append(radius)
        times.append(f["Header/time"][()])

        f.close()

    temp_profile = []
    temp_radii = []
    for i, file in enumerate(fpaths):
        f = h5.File(file, "r")
        temp = f["Profiles/TempDensityWeighted"][:]
        temp_mask = temp > 0

        radius = f["Profiles/Radius"][:][:-1]
        radius = radius[temp_mask]

        temp_profile.append(temp[temp_mask])
        temp_radii.append(radius)

        f.close()

    velocity_profile = []
    velocity_radii = []
    for i, file in enumerate(fpaths):
        f = h5.File(file, "r")
        velocity = f["Profiles/RadialVelocity"][:]

        radius = f["Profiles/Radius"][:][:-1]

        velocity_profile.append(velocity)
        velocity_radii.append(radius)

        f.close()

    galaxy_metal = []
    cgm_metal = []
    igm_metal = []
    mean_metal = []
    times = []
    for i, file in enumerate(fpaths):
        f = h5.File(file, "r")
        galaxy_metal.append(f["Galaxy/MeanMetallicity"][()] * 3.81)
        cgm_metal.append(f["CGM/MeanMetallicity"][()] * 3.81)
        igm_metal.append(f["IGM/MeanMetallicity"][()] * 3.81)
        mean_metal.append(f["Halo/MeanMetallicity"][()] * 3.81)
        times.append(f["Header/time"][()])
        f.close()

    return times, cgm_metal, galaxy_metal, mean_metal


def read_cloud_properties(logsfc_path):
    # logsfc_path = "/home/fabg/container_tiramisu/sim_log_files/CC-Fiducial/logSFC"

    log_sfc = np.loadtxt(logsfc_path)
    redshift = log_sfc[:, 2]
    t_myr = t_myr_from_z(redshift)
    cloud_metal = log_sfc[:, 9] * 3.81
    return t_myr, cloud_metal


# wnm_mass = []
# hot_mass = []
# cnm_mass = []

# for i, file in enumerate(fpaths):
#     f = h5.File(file, "r")
#     wnm_mass.append(f["Galaxy/HotGasMass"][()])
#     hot_mass.append(f["Galaxy/ColdNeutralMediumMass"][()])
#     cnm_mass.append(f["Galaxy/WarmNeutralMediumMass"][()])
#     f.close()
# %%


cc_times, cc_mass, cc_metalmass, cc_mass_in, cc_metalmass_in = read_outflow_rates(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
    153,
    466,
)
f7_times, f7_mass, f7_metalmass, f7_mass_in, f7_metalmass_in = read_outflow_rates(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    115,
    1570,
    step=6,
)
(
    f3_times,
    f3_mass,
    f3_metalmass,
    f3_mass_in,
    f3_metalmass_in,
) = read_outflow_rates(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    200,
    1606,
    step=6,
)


cc_times, cc_cgm_metal, cc_galaxy_metal, cc_vir_metal = read_mean_metallicities(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
    153,
    466,
)
f70_times, f70_cgm_metal, f70_galaxy_metal, f70_vir_metal = read_mean_metallicities(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    115,
    1570,
    step=6,
)
f35_times, f35_cgm_metal, f35_galaxy_metal, f35_vir_metal = read_mean_metallicities(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    200,
    1606,
    step=6,
)

t_myr, cloud_metal = read_cloud_properties(
    "/home/fabg/container_tiramisu/sim_log_files/CC-Fiducial/logSFC"
)
f70_t_myr, f70_cloud_metal = read_cloud_properties(
    "/home/fabg/container_tiramisu/sim_log_files/fs07_refine/logSFC"
)
f35_t_myr, f35_cloud_metal = read_cloud_properties(
    "/home/fabg/container_tiramisu/sim_log_files/fs035_ms10/logSFC"
)


cmap = matplotlib.colormaps["Dark2"]
cmap = cmap(np.linspace(0, 1, 8))
vsfe_clr = cmap[0]
high_clr = cmap[1]
low_clr = cmap[2]


fig, ax = plt.subplots(3, 1, figsize=(5, 7), dpi=300, sharex=True, sharey=True)
redshft_ax = ax[0].twiny()
plt.subplots_adjust(hspace=0, wspace=0)
ax[0].plot(cc_times, cc_galaxy_metal, label="SF region", color="k", lw=2)
ax[0].plot(cc_times, cc_cgm_metal, label="CGM", color="grey", lw=2)
ax[0].plot(
    cc_times,
    cc_metalmass / cc_mass,
    label="outflows",
    color="tab:red",
    ls=":",
    lw=2,
)
ax[0].plot(cc_times, cc_metalmass_in / cc_mass_in, ls="--", label="inflows", lw=2)

ax[1].plot(f70_times, f70_galaxy_metal, label="SF region", color="k", lw=2)
ax[1].plot(f70_times, f70_cgm_metal, label="CGM", color="grey", lw=2)
ax[1].plot(f7_times, f7_metalmass / f7_mass, color="tab:red", ls=":", lw=2)
ax[1].plot(f7_times, f7_metalmass_in / f7_mass_in, ls="--", lw=2)

ax[2].plot(f35_times, f35_galaxy_metal, label="SF region", color="k", lw=2)
ax[2].plot(f35_times, f35_cgm_metal, label="CGM", color="grey", lw=2)
ax[2].plot(f3_times, f3_metalmass / f3_mass, color="tab:red", ls=":", lw=2)
ax[2].plot(f3_times, f3_metalmass_in / f3_mass_in, ls="--", lw=2)

ax[0].scatter(t_myr, cloud_metal, alpha=0.1, s=10, c=vsfe_clr, marker="o")
ax[1].scatter(f70_t_myr, f70_cloud_metal, alpha=0.1, s=10, c=high_clr, marker="o")
ax[2].scatter(f35_t_myr, f35_cloud_metal, alpha=0.1, s=10, c=low_clr, marker="o")

# ax.axvline(x=590)
# ax.axvline(x=575)
ax[0].legend(frameon=False, loc="lower right", ncols=2)

ax[0].set(
    xlim=(340, 718),
    ylim=(8e-4, 0.3),
)

ax[0].minorticks_on()
ax[0].text(
    0.05, 0.90, "VSFE", ha="left", va="top", fontsize=9, transform=ax[0].transAxes
)
ax[1].text(
    0.05, 0.90, "high SFE", ha="left", va="top", fontsize=9, transform=ax[1].transAxes
)
ax[2].text(
    0.05, 0.90, "low SFE", ha="left", va="top", fontsize=9, transform=ax[2].transAxes
)

ax[1].set(ylabel=r"$\langle  Z_{\rm gas}\rangle \: [\mathrm{Z_\odot}$]", yscale="log")
ax[2].set(xlabel="time [Myr]")

redshft_ax.locator_params(axis="x")
redshft_ax.set(xlim=(340, 718), xlabel="$z$")
redshft_ax.set_xticklabels(
    list(np.round(z_from_t_myr(redshft_ax.get_xticks()), 1).astype("str"))
)

plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/mean_metals.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)

plt.show()


# %%
def read_metal_masses(path, start, stop, step=1):
    """
    read the mass in metals in Msun in the
    """
    fpaths, snums = filter_snapshots(
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
    mean_metal = []
    times = []
    for i, file in enumerate(fpaths):
        f = h5.File(file, "r")
        galaxy_metal.append(f["Galaxy/MetalMass"][()])
        cgm_metal.append(f["CGM/MetalMass"][()])
        igm_metal.append(f["IGM/MetalMass"][()])
        mean_metal.append(f["Halo/MetalMass"][()])
        times.append(f["Header/time"][()])
        f.close()
    galaxy_metal = np.array(galaxy_metal)
    cgm_metal = np.array(cgm_metal)
    igm_metal = np.array(igm_metal)
    mean_metal = np.array(mean_metal)
    times = np.array(times)
    return times, cgm_metal, galaxy_metal, mean_metal, igm_metal


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
    ejecta_zsun = log_sfc[:, 8]

    # let's do the accumulation of metals produced
    mass_in_metals = m_ejecta * ejecta_zsun

    total_mass_in_metals = np.cumsum(mass_in_metals)

    return t_myr, total_mass_in_metals


def metal_ratio(logsntime, logsnmetal, time2, metal2):
    time_mask = np.ones_like(time2)
    for i, t in enumerate(time2):
        closest_age_idx = np.argmin(np.abs(logsntime - t))
        time_mask[i] = closest_age_idx

    # downsample the logSN to match closer to the
    closest_matching_metal = logsnmetal[np.array(time_mask, dtype="int")]

    ratio = metal2 / closest_matching_metal
    return time2, ratio


cc_times, cc_cgm_metal, cc_galaxy_metal, cc_vir_metal, cc_igm = read_metal_masses(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
    153,
    466,
    step=2,
)
f70_times, f70_cgm_metal, f70_galaxy_metal, f70_vir_metal, f70_igm = read_metal_masses(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    115,
    1570,
    step=2,
)
f35_times, f35_cgm_metal, f35_galaxy_metal, f35_vir_metal, f35_igm = read_metal_masses(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    200,
    1606,
    step=4,
)

cc_t_myr, cc_sn_metal = read_sne(
    "/home/fabg/container_tiramisu/sim_log_files/CC-Fiducial/logSN", interuped=True
)
f7_t_myr, f7_sn_metal = read_sne(
    "/home/fabg/container_tiramisu/sim_log_files/fs07_refine/logSN"
)
f3_t_myr, f3_sn_metal = read_sne(
    "/home/fabg/container_tiramisu/sim_log_files/fs035_ms10/logSN"
)


import colormaps as cmaps


cmap = matplotlib.colormaps["Dark2"]
cmap = cmap(np.linspace(0, 1, 8))
vsfe_clr = cmap[0]
high_clr = cmap[1]
low_clr = cmap[2]

fig, ax = plt.subplots(
    nrows=4,
    ncols=1,
    sharex=True,
    sharey=False,
    figsize=(5, 9),
    dpi=300,
    gridspec_kw={"height_ratios": [4, 4, 4, 3]},
)
plt.subplots_adjust(hspace=0, wspace=0)
redshft_ax = ax[0].twiny()

ax[0].plot(cc_t_myr, cc_sn_metal, color="k", lw=3, label=r"PopII SNe")
ax[1].plot(f7_t_myr, f7_sn_metal, color="k", lw=3)
ax[2].plot(f3_t_myr, f3_sn_metal, color="k", lw=3)


ax[0].plot(cc_times, cc_galaxy_metal, label="SF region", c=cmap[5], lw=3, alpha=0.9)
ax[1].plot(f70_times, f70_galaxy_metal, label="SF region", c=cmap[5], lw=3, alpha=0.9)
ax[2].plot(f35_times, f35_galaxy_metal, label="SF region", c=cmap[5], lw=3, alpha=0.9)

ax[0].plot(cc_times, cc_cgm_metal, label="CGM", color="grey", lw=3, alpha=0.9)
ax[1].plot(f70_times, f70_cgm_metal, label="CGM", color="grey", lw=3, alpha=0.9)
ax[2].plot(f35_times, f35_cgm_metal, label="CGM", color="grey", lw=3, alpha=0.9)

ax[0].plot(
    cc_times, cc_vir_metal, c=cmap[4], lw=3, label=r"$r < r_{\rm vir}$", alpha=0.9
)
ax[1].plot(f70_times, f70_vir_metal, c=cmap[4], lw=3, alpha=0.9)
ax[2].plot(f35_times, f35_vir_metal, c=cmap[4], lw=3, alpha=0.9)

ax[0].plot(cc_times[::5], cc_igm[::5], c=cmap[6], lw=3, alpha=0.9, label=r"IGM")
ax[1].plot(f70_times[::5], f70_igm[::5], c=cmap[6], lw=3, alpha=0.9)
ax[2].plot(f35_times[::5], f35_igm[::5], c=cmap[6], lw=3, alpha=0.9)


# mass in the sf region / mass produced by SNe
cc_t, cc_sf_ratio = metal_ratio(cc_t_myr, cc_sn_metal, cc_times, cc_galaxy_metal)
f70_t, f70_sf_ratio = metal_ratio(f7_t_myr, f7_sn_metal, f70_times, f70_galaxy_metal)
f35_t, f35_sf_ratio = metal_ratio(f3_t_myr, f3_sn_metal, f35_times, f35_galaxy_metal)

# mass in the CGM / mass produced by SNe
cc_t, cc_cgm_ratio = metal_ratio(cc_t_myr, cc_sn_metal, cc_times, cc_cgm_metal)
f70_t, f70_cgm_ratio = metal_ratio(f7_t_myr, f7_sn_metal, f70_times, f70_cgm_metal)
f35_t, f35_cgm_ratio = metal_ratio(f3_t_myr, f3_sn_metal, f35_times, f35_cgm_metal)


ax[3].plot(cc_t, cc_sf_ratio, lw=3, color=vsfe_clr, label="VSFE")
ax[3].plot(f70_t, f70_sf_ratio, lw=3, color=high_clr, label="high SFE")
ax[3].plot(f35_t, f35_sf_ratio, lw=3, color=low_clr, label="low SFE")

ax[3].plot(
    cc_t,
    cc_cgm_ratio,
    lw=1,
    color=vsfe_clr,
    ls=":",
)
ax[3].plot(
    f70_t,
    f70_cgm_ratio,
    lw=1,
    color=high_clr,
    ls=":",
)
ax[3].plot(
    f35_t,
    f35_cgm_ratio,
    lw=1,
    color=low_clr,
    ls=":",
)

ax[0].text(
    0.05, 0.90, "VSFE", ha="left", va="top", fontsize=9, transform=ax[0].transAxes
)
ax[1].text(
    0.05, 0.90, "high SFE", ha="left", va="top", fontsize=9, transform=ax[1].transAxes
)
ax[2].text(
    0.05, 0.90, "low SFE", ha="left", va="top", fontsize=9, transform=ax[2].transAxes
)

ax[0].legend(frameon=False, loc="lower right", ncols=2, fontsize=9)
ax[0].minorticks_on()
ax[3].minorticks_on()
ax[3].legend()
handles, labels = ax[3].get_legend_handles_labels()

colors = ["lightgrey", "lightgrey"]
lw = [3, 1]
ls = ["-", "--"]
labels = ["SF region", "CGM"]
lines = [
    Line2D([0], [0], color=c, linewidth=lw[i], linestyle=ls[i], label=labels[i])
    for i, c in enumerate(colors)
]
handles.extend(lines)
ax[3].legend(handles=handles, ncols=2, frameon=False, loc="upper right", fontsize=9)
# ax[2].legend(lines, labels, loc="upper right", bbox_to_anchor=(0.5, 0.5), frameon=False)


ax[0].set(ylim=(1000, 7e5), yscale="log")
ax[1].set(ylim=(1000, 7e5), yscale="log")
ax[2].set(ylim=(1000, 7e5), xlim=(340, 718), yscale="log")
ax[3].set(
    ylim=(0.01, 1.2), ylabel=r"$M_{\rm metals, region} / M_{\rm metals, PopII \: SNe}$"
)
ax[3].locator_params(axis="y", nbins=5)


ax[1].set(ylabel=r"$ M_{\rm metals} [\mathrm{M_\odot}$]", yscale="log")
ax[3].set(xlabel="time [Myr]")

redshft_ax.locator_params(axis="x")
redshft_ax.set(xlim=(340, 718), xlabel="$z$")
redshft_ax.set_xticklabels(
    list(np.round(z_from_t_myr(redshft_ax.get_xticks()), 1).astype("str"))
)

plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/metal_masses.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()
