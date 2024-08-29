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


def read_masses(path, start, stop, step=1):
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
        galaxy_gas_mass.append(f["Galaxy/GasMass"][()])
        galaxy_metal_mass.append(f["Galaxy/MetalMass"][()])
        f.close()

    return np.array(times), np.array(galaxy_gas_mass), np.array(galaxy_metal_mass)


cc_times, cc_m_out, cc_mz_out, cc_m_in, cc_mz_in = read_outflow_rates(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
    193,
    466,
)

f7_times, f7_m_out, f7_mz_out, f7_m_in, f7_mz_in = read_outflow_rates(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    115,
    1570,
)

f3_times, f3_m_out, f3_mz_out, f3_m_in, f3_mz_in = read_outflow_rates(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    200,
    1606,
)

cc_times, cc_mgas, cc_mzgas = read_masses(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/CC-Fiducial/",
    193,
    466,
)

f7_times, f7_mgas, f7_mzgas = read_masses(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs07_refine/",
    115,
    1570,
)

f3_times, f3_mgas, f3_mzgas = read_masses(
    "/home/fabg/container_tiramisu/post_processed/gas_properties/fs035_ms10/",
    200,
    1606,
)

cmap = matplotlib.colormaps["Dark2"]
cmap = cmap(np.linspace(0, 1, 8))
vsfe_clr = cmap[0]
high_clr = cmap[1]
low_clr = cmap[2]
# %%
fig, ax = plt.subplots(
    nrows=4, ncols=3, figsize=(12, 10), dpi=300, sharex=True, sharey="row"
)
ax = ax.ravel()
plt.subplots_adjust(hspace=0.1, wspace=0)

ax[0].plot(cc_times, cc_mgas, label="gas")
ax[1].plot(f7_times, f7_mgas)
ax[2].plot(f3_times, f3_mgas)

# ax[0].plot(cc_times, cc_mzgas, label="metals")
# ax[1].plot(f7_times, f7_mzgas)
# ax[2].plot(f3_times, f3_mzgas)


ax[0].set(
    ylabel=r"$M_{\rm gas} \: \left[{\rm M_\odot} \right]$",
    yscale="log",
)


# ax[3].plot(cc_times, cc_m_out, label="outflowing")
# ax[4].plot(f7_times, f7_m_out)
# ax[5].plot(f3_times, f3_m_out)

# ax[3].plot(cc_times, -cc_m_in, label="inflowing")
# ax[4].plot(f7_times, -f7_m_in)
# ax[5].plot(f3_times, -f3_m_in)

# ax[3].set(
#     ylabel=r"$|\dot{M}_{\rm gas}|\: \left[{\rm M_\odot \: yr^{-1}}\right]$",
#     yscale="log",
# )

# ax[6].plot(cc_times, cc_mz_out, label="outflowing")
# ax[7].plot(f7_times, f7_mz_out)
# ax[8].plot(f3_times, f3_mz_out)

# ax[6].plot(cc_times, -cc_mz_in, label="inflowing")
# ax[7].plot(f7_times, -f7_mz_in)
# ax[8].plot(f3_times, -f3_mz_in)

# ax[6].set(
#     ylabel=r"$|\dot{M}_{\rm metal}|\: \left[{\rm M_\odot \: yr^{-1}}\right]$",
#     yscale="log",
# )


# ax[1].plot(cc_times, cc_mz_out, color=vsfe_clr)
# ax[1].plot(f7_times, f7_mz_out, color=high_clr)
# ax[1].plot(f3_times, f3_mz_out, color=low_clr)

# ax[2].plot(cc_times, cc_mz_out / cc_m_out, color=vsfe_clr)
# ax[2].plot(f7_times, f7_mz_out / f7_m_out, color=high_clr)
# ax[2].plot(f3_times, f3_mz_out / f3_m_out, color=low_clr)


ax[3].legend(frameon=False, ncols=2)
ax[3].legend(frameon=False, ncols=2)
ax[0].minorticks_on()
# ax[1].set(
#     ylabel=r"$\dot{M}_{\rm metal, out} [{\rm M_\odot \: yr^{-1} }]$",
#     yscale="log",
#     xlabel="t [myr]",
# )

# ax[2].set(
#     ylabel=r"$\dot{Z}_{\rm metal} [{\rm Z_\odot}]$",
#     yscale="log",
#     xlabel="t [myr]",
# )

ax[0].text(
    0.05, 0.90, "VSFE", ha="left", va="top", fontsize=9, transform=ax[0].transAxes
)
ax[1].text(
    0.05, 0.90, "high SFE", ha="left", va="top", fontsize=9, transform=ax[1].transAxes
)
ax[2].text(
    0.05, 0.90, "low SFE", ha="left", va="top", fontsize=9, transform=ax[2].transAxes
)

plt.savefig(
    "../../../gdrive_columbia/research/massimo/paper2/outflow_rate.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.05,
)
plt.show()
