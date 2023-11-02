import sys

sys.path.append("../")
import yt
import numpy as np
import matplotlib.pyplot as plt
import os
from tools import plotstyle
from tools.fscanner import filter_snapshots
import glob
from matplotlib import cm
import matplotlib as mpl

bsc_container = "CC-Fiducial"
sim_path = os.path.join(
    "..", "..", "container_tiramisu", "post_processed", "bsc_catalogues", bsc_container
)
fpaths, snaps = filter_snapshots(
    sim_path, 320, 430, sampling=50, str_snaps=True, snapshot_type="bsc_processed"
)


fig, ax = plt.subplots(nrows=1, ncols=1, dpi=400, figsize=(5, 4))
clrmap = "jet_r"
age_map = cm.get_cmap(clrmap)
time_range = (450, 600)
cmap = age_map(np.linspace(0, 1, time_range[1] - time_range[0]))
evenly_spaced_times = np.arange(time_range[0], time_range[1] + 5)

for i, folder in enumerate(fpaths):
    fname = glob.glob(os.path.join(folder, "profiled_catalogue-{}-*".format(snaps[i])))[
        0
    ]
    time = float(fname.split("-")[2].replace("_", "."))
    print(fname)
    bsc_catalogue = np.loadtxt(fname)
    r_halfmass = bsc_catalogue[:, 21]
    vx_dispersion = bsc_catalogue[:, 10]
    vy_dispersion = bsc_catalogue[:, 11]
    vz_dispersion = bsc_catalogue[:, 12]
    mass = bsc_catalogue[:, 8]
    mask = mass < 200
    idx_of_nearest_c = np.argmin(np.abs(evenly_spaced_times - time))
    color = cmap[idx_of_nearest_c]
    color = color.reshape(1, -1)

    vdisp = np.sqrt(vx_dispersion**2 + vy_dispersion**2 + vz_dispersion**2)
    ax.scatter(r_halfmass[~mask], vdisp[~mask], c=color, s=1)

ax.set(
    xscale="log",
    yscale="log",
    xlabel=r"$r_{\rm half}$ (pc)",
    # xlabel=r"Mass ($\rm M_\odot$)",
    ylabel=r"$\sigma_{\rm 3D}\:({\rm km \:s^{-1}})$ ",
)

cbax = fig.add_axes([0.1, 0.95, 0.8, 0.02])
cb = mpl.colorbar.ColorbarBase(
    cbax,
    norm=mpl.colors.Normalize(time_range[0], time_range[1]),
    # ticks = [340,405,470],
    orientation="horizontal",
    cmap=clrmap,
    # label='Birth Epoch (Myr)'
)
cbax.set(title="time (Myr)")
plt.show()
