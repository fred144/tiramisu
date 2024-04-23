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


def read_mean_metallicities(path, start, stop, step=1):
    fpaths, snums = filter_snapshots(
        # "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
        # 306,
        # 466,
        path,
        start,
        stop,
        sampling=1,
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
        galaxy_metal.append(f["Galaxy/MeanMetallicity"][()])
        cgm_metal.append(f["CGM/MeanMetallicity"][()])
        igm_metal.append(f["IGM/MeanMetallicity"][()])
        mean_metal.append(f["Halo/MeanMetallicity"][()])
        times.append(f["Header/time"][()])
        f.close()

    return times, mean_metal, cgm_metal


def read_cloud_properties(logsfc_path):
    # logsfc_path = "/home/fabg/container_tiramisu/sim_log_files/CC-Fiducial/logSFC"
    log_sfc = np.loadtxt(logsfc_path)
    redshift = log_sfc[:, 2]
    t_myr = t_myr_from_z(redshift)
    cloud_metal = log_sfc[:, 9]
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

cc_times, cc_halo_metal, cc_cgm_metal = read_mean_metallicities(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
    306,
    466,
)

t_myr, cloud_metal = read_cloud_properties(
    "/home/fabg/container_tiramisu/sim_log_files/CC-Fiducial/logSFC"
)


f70_times, f70_halo_metal, f70_cgm_metal = read_mean_metallicities(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    115,
    1570,
)

f70_t_myr, f70_cloud_metal = read_cloud_properties(
    "/home/fabg/container_tiramisu/sim_log_files/fs07_refine/logSFC"
)

cmap = matplotlib.colormaps["Dark2"]
cmap = cmap(np.linspace(0, 1, 8))
vsfe_clr = cmap[0]

fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
ax.plot(cc_times, cc_cgm_metal, label="CGM", color="k", lw=2)
ax.plot(cc_times, cc_halo_metal, label="Halo", color="k", ls="--", lw=2)

ax.plot(f70_times, f70_cgm_metal, label="CGM", color=cmap[1], lw=2)
ax.plot(f70_times, f70_halo_metal, label="Halo", color=cmap[1], ls="--", lw=2)

# ax.plot(times, cgm_metal, label="cgm")
# ax.scatter(times, igm_metal, label=r"igm (virrad $<$ r $<$ 10kpc)")
# ax.plot(np.array(times), np.array(mean_metal) / 0.02, label="virrad $<$ r")


ax.scatter(t_myr, cloud_metal, alpha=0.1, s=10, c=vsfe_clr, marker="o")
ax.scatter(f70_t_myr, f70_cloud_metal, alpha=0.1, s=10, c=cmap[1], marker="o")
ax.set(
    ylabel=r"Mean Metallicity ($\mathrm{Z_\odot}$)", yscale="log", xlabel="time (Myr)"
)
# ax.axvline(x=588)
ax.legend()
ax.set(ylim=(1e-4, 2e-2))
ax.set(xlim=(425, 670), ylim=(5e-4, 3e-2))
plt.show()

# # %%
# fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
# ax.plot(times, wnm_mass, label="wnm")
# ax.plot(times, hot_mass, label="hot")
# ax.plot(times, cnm_mass, label="cnm")
# ax.set(ylabel="mean temperature (K)", xlabel="time (Myr)")
# ax.legend()
# plt.show()
# %%

times = np.array(times)

fig, ax = plt.subplots(
    1,
    1,
    figsize=(4, 3),
    dpi=400,
    sharey=True,
    sharex=True,
)
plt.subplots_adjust(wspace=0.0)
cbar_ax = ax.inset_axes([0, 1.1, 1, 0.05])

cmap = plt.cm.inferno
norm = colors.Normalize(vmin=np.min(times), vmax=np.max(times))

for i, t in enumerate(times):
    ax.plot(metal_radii[i], metal_profile[i], color=cmap(norm(t)))

ax.set(
    yscale="log",
    ylabel="Metallicity (Zsun)",
    xlabel="radial distance (pc)",
    xscale="log",
    ylim=(1e-4, 5),
)
cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation="horizontal")
plt.show()
# %%


fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
cbar_ax = ax.inset_axes([0, 1.1, 1, 0.05])

cmap = plt.cm.inferno
norm = colors.Normalize(vmin=np.min(times), vmax=np.max(times))

for i, t in enumerate(times):
    ax.plot(temp_radii[i], temp_profile[i], color=cmap(norm(t)), alpha=0.7)

ax.set(
    yscale="log",
    ylabel="Temperature (K)",
    xlabel="radial distance (pc)",
    xscale="log",
)

cb = mpl.colorbar.ColorbarBase(
    cbar_ax, cmap=cmap, norm=norm, orientation="horizontal", alpha=0.7
)
plt.show()


# %%


fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
cbar_ax = ax.inset_axes([0, 1.1, 1, 0.05])

cmap = plt.cm.spring
norm = colors.Normalize(vmin=np.min(times), vmax=np.max(times))

for i, t in enumerate(times):
    ax.plot(velocity_radii[i], velocity_profile[i], color=cmap(norm(t)), alpha=1)

ax.set(xscale="log", ylim=(-2e2, 2e2))
cb = mpl.colorbar.ColorbarBase(
    cbar_ax, cmap=cmap, norm=norm, orientation="horizontal", alpha=0.7
)
plt.show()
