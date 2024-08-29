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
