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


def read_outflow_rates(path, start, stop, step=1):
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

    mass_outflow = []
    metalmass_outflow = []
    times = []
    for i, file in enumerate(fpaths):
        f = h5.File(file, "r")
        mass_outflow.append(f["Winds/MassOutFlowRate"][()])
        metalmass_outflow.append(f["Winds/MetalMassOutFlowRate"][()])
        times.append(f["Header/time"][()])
        f.close()

    return np.array(times), np.array(mass_outflow), np.array(metalmass_outflow)


cc_times, cc_mass, cc_metalmass = read_outflow_rates(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
    153,
    466,
)

f7_times, f7_mass, f7_metalmass = read_outflow_rates(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    115,
    1570,
)

f3_times, f3_mass, f3_metalmass = read_outflow_rates(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    200,
    1606,
)

cmap = matplotlib.colormaps["Dark2"]
cmap = cmap(np.linspace(0, 1, 8))
vsfe_clr = cmap[0]
high_clr = cmap[1]
low_clr = cmap[2]

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(4.5, 5), dpi=300, sharex=True)
plt.subplots_adjust(hspace=0)
ax[0].plot(cc_times, cc_mass, color=vsfe_clr, label="VSFE")
ax[0].plot(f7_times, f7_mass, color=high_clr, label="high SFE")
ax[0].plot(f3_times, f3_mass, color=low_clr, label="low SFE")

ax[1].plot(cc_times, cc_metalmass, color=vsfe_clr)
ax[1].plot(f7_times, f7_metalmass, color=high_clr)
ax[1].plot(f3_times, f3_metalmass, color=low_clr)

ax[2].plot(cc_times, cc_metalmass / cc_mass, color=vsfe_clr)
ax[2].plot(f7_times, f7_metalmass / f7_mass, color=high_clr)
ax[2].plot(f3_times, f3_metalmass / f3_mass, color=low_clr)

ax[0].set(
    ylabel=r"$\dot{M}_{\rm gas, out} [{\rm M_\odot \: yr^{-1}}]$",
    yscale="log",
)
ax[0].legend(frameon=False)
ax[0].minorticks_on()
ax[1].set(
    ylabel=r"$\dot{M}_{\rm metal, out} [{\rm M_\odot \: yr^{-1} }]$",
    yscale="log",
    xlabel="t [myr]",
)

ax[2].set(
    ylabel=r"$\dot{Z}_{\rm metal} [{\rm Z_\odot}]$",
    yscale="log",
    xlabel="t [myr]",
)
plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/outflow_rate.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()
