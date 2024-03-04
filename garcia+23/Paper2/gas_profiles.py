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


fpaths, snums = filter_snapshots(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
    304,
    400,
    sampling=10,
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


# %%

times = np.array(times)

fig, ax = plt.subplots(
    1,
    3,
    figsize=(9.1, 3),
    dpi=400,
    sharey=True,
    sharex=True,
)
plt.subplots_adjust(wspace=0.0)
cbar_ax = ax[0].inset_axes([0, 1.1, 1, 0.05])

cmap = plt.cm.Dark2
norm = colors.Normalize(vmin=np.min(times), vmax=np.max(times))

for i, t in enumerate(times):
    ax[0].plot(metal_radii[i], metal_profile[i], color=cmap(norm(t)))

ax[0].set(yscale="log", xscale="log", ylim=(1e-4, 5))
cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation="horizontal")
plt.show()
# %%


fig, ax = plt.subplots(1, 1, figsize=(5, 4), dpi=300)
cbar_ax = ax.inset_axes([0, 1.1, 1, 0.05])

cmap = cmr.guppy
norm = colors.Normalize(vmin=np.min(times), vmax=np.max(times))

for i, t in enumerate(times):
    ax.plot(temp_radii[i], temp_profile[i], color=cmap(norm(t)), alpha=0.7)

ax.set(yscale="log", xscale="log")
cb = mpl.colorbar.ColorbarBase(
    cbar_ax, cmap=cmap, norm=norm, orientation="horizontal", alpha=0.7
)
plt.show()
