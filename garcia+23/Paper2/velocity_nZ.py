"""
how does SFE affect outflow velocities/ metal mixing?
Feedback times in the SNe dominated regime

35 SFE 435Myr - 465 Myr
70 SFE  654Myr - 684
VSFE 592Myr - 622Myr 
"""

import sys

sys.path.append("../../")

import yt
import numpy as np
import os
from tools.fscanner import filter_snapshots
from tools.fscanner import find_matching_time
from tools.ram_fields import ram_fields
import h5py as h5
from yt.funcs import mylog
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog
import warnings
import matplotlib.pyplot as plt

import matplotlib as mpl
from scipy.spatial.transform import Rotation as R
from yt.visualization.volume_rendering.api import Scene
from scipy.ndimage import gaussian_filter
from tools.check_path import check_path
from tools import plotstyle


import cmasher as cmr

mylog.setLevel(40)
warnings.simplefilter(action="ignore", category=RuntimeWarning)

cell_fields, epf = ram_fields()

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
    }
)

if __name__ == "__main__":
    m_h = 1.6735e-24  # grams
    r_sf = 500  # radii for sf in pc
    zsun = 0.02
    n_crit = 5e4
    # cold_phase_t = (0, 100)
    # warm_phase_t = (100, 5e4)
    # hot_phase = (5.001e4, 1e9)
    # temp_cuts = [cold_phase_t, warm_phase_t, hot_phase]
    # tlabels = [
    #     r"CNM ($T < 100$ K)",
    #     r"WNM  ($ 100 < T \leq 5 \times 10^4$ K)",
    #     r"Hot ($T > 5 \times 10^4$ K)",
    # ]

    lims = {
        ("gas", "radial_velocity"): ((-3.5e3, "km/s"), (3.5e3, "km/s")),
        # ("gas", "temperature"): ((50, "K"), (5e8, "K")),
        ("ramses", "Metallicity"): (2e-4 * zsun, 5 * zsun),
        ("gas", "mass"): ((1e-4, "msun"), (2e6, "msun")),
    }

    fpaths, snums = filter_snapshots(
        "/scratch/zt1/project/ricotti-prj/user/ricotti/GC-Fred/CC-Fiducial",
        374,
        438,
        sampling=2,
        str_snaps=True,
        snapshot_type="ramses_snapshot",
    )
    # for the other rows
    fpaths1, snums1 = filter_snapshots(
        "/afs/shell.umd.edu/project/ricotti-prj/user/fgarcia4/dwarf/data/cluster_evolution/fs07_refine",
        1300,
        1570,
        sampling=5,
        str_snaps=True,
        snapshot_type="ramses_snapshot",
    )

    fpaths2, snums2 = filter_snapshots(
        "/afs/shell.umd.edu/project/ricotti-prj/user/fgarcia4/dwarf/data/cluster_evolution/fs035_ms10",
        355,
        655,
        sampling=4,
        str_snaps=True,
        snapshot_type="ramses_snapshot",
    )

    # =============================================================================

    # datadir = os.path.expanduser("~/test_data/CC-Fiducial/")
    # logsfc_path = os.path.expanduser(
    #     "~/container_tiramisu/sim_log_files/CC-Fiducial/logSFC"
    # )
    # fpaths, snums = filter_snapshots(
    #     datadir,
    #     304,
    #     304,
    #     sampling=1,
    #     str_snaps=True,
    #     snapshot_type="ramses_snapshot",
    # )
    # # for the other rows
    # fpaths1, snums1 = filter_snapshots(
    #     datadir,
    #     375,
    #     388,
    #     sampling=1,
    #     str_snaps=True,
    #     snapshot_type="ramses_snapshot",
    # )

    # fpaths2, snums2 = filter_snapshots(
    #     datadir,
    #     390,
    #     390,
    #     sampling=1,
    #     str_snaps=True,
    #     snapshot_type="ramses_snapshot",
    # )

    # =============================================================================
    render_nickname = ""
    # run save
    sim_run = "run_comparison"
    render_container = os.path.join(
        "..",
        "..",
        "garcia+23",
        "Paper2",
    )
    check_path(render_container)

    snapshot_list = [
        [fpaths, snums],
        [fpaths1, snums1],
        [fpaths2, snums2],
    ]

    sfevariable = []
    sfe70 = []
    sfe35 = []

    sfevariable_myrs = []
    sfe70_myrs = []
    sfe35_myrs = []

    for sg, sn_group in enumerate(snapshot_list):
        # within each row  or grouping, read and update
        for r, (fpaths, snums) in enumerate(zip(sn_group[0], sn_group[1])):
            len_ofgroup = len(sn_group[0])
            print("# _________________________________________________________________")
            infofile = os.path.abspath(os.path.join(fpaths, f"info_{snums}.txt"))
            print("# reading in", infofile)

            ds = yt.load(infofile, fields=cell_fields, extra_particle_fields=epf)
            ad = ds.all_data()

            t_myr = float(ds.current_time.in_units("Myr"))
            redshift = ds.current_redshift

            x_pos = np.array(ad["star", "particle_position_x"])
            y_pos = np.array(ad["star", "particle_position_y"])
            z_pos = np.array(ad["star", "particle_position_z"])
            x_center = np.mean(x_pos)
            y_center = np.mean(y_pos)
            z_center = np.mean(z_pos)

            ctr_at_code = ds.arr([x_center, y_center, z_center], "code_length")

            mstar = np.sum(np.array(ad["star", "particle_mass"].to("Msun")))

            # for each grouping, go throught the CNM, WNM, and HOT phases

            sf_region = ds.sphere(ctr_at_code, (r_sf, "pc"))
            bulk_vel = sf_region.quantities.bulk_velocity()

            # Get the second sphere
            sf_region_corrected = ds.sphere(ctr_at_code, (r_sf, "pc"))

            # Set the bulk velocity field parameter
            sf_region_corrected.set_field_parameter("bulk_velocity", bulk_vel)

            profile2d = sf_region_corrected.profile(
                # the x bin field, the y bin field
                # metallicity is
                [("gas", "radial_velocity"), ("ramses", "Metallicity")],
                [("gas", "mass")],  # the profile field
                weight_field=None,  # sums each quantity in each bin
                n_bins=(150, 150),
                extrema=lims,
                logs={("gas", "radial_velocity"): False},
            )

            gas_mass = np.array(profile2d["gas", "mass"].to("msun")).T

            if sg == 0:
                sfevariable.append(gas_mass)
                sfevariable_myrs.append(t_myr)
            elif sg == 1:
                sfe70.append(gas_mass)
                sfe70_myrs.append(t_myr)
            else:
                sfe35.append(gas_mass)
                sfe35_myrs.append(t_myr)

    time_avg_vals = [
        np.mean(sfevariable, axis=0),
        np.mean(sfe70, axis=0),
        np.mean(sfe35, axis=0),
    ]

    names = [
        "VSFE",
        "high SFE",
        "low SFE",
    ]
    times = [np.array(sfevariable_myrs), np.array(sfe70_myrs), np.array(sfe35_myrs)]

    fig, ax = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
        figsize=(5, 9),
        dpi=300,
    )

    plt.subplots_adjust(hspace=0, wspace=0)

    ax = ax.ravel()
    for a, phase_plot in enumerate(time_avg_vals):
        nz_image = ax[a].imshow(
            np.log10(phase_plot),
            origin="lower",
            extent=[
                lims[("gas", "radial_velocity")][0][0],
                lims[("gas", "radial_velocity")][1][0],
                np.log10(lims[("ramses", "Metallicity")][0] / zsun),
                np.log10(lims[("ramses", "Metallicity")][1] / zsun),
            ],
            cmap=cmr.savanna_r,
            vmin=np.log10(lims[("gas", "mass")][0][0]),
            vmax=np.log10(lims[("gas", "mass")][1][0]),
            aspect=1200,
        )
        row_time = (r"$t = {:.0f}  - {:.0f}$ Myr").format(
            times[a].min(), times[a].max()
        )

        ax[a].text(
            0.05,
            0.95,
            names[a],
            ha="left",
            va="top",
            color="black",
            transform=ax[a].transAxes,
        )

        ax[a].text(
            0.05,
            0.05,
            row_time,
            ha="left",
            va="bottom",
            color="black",
            transform=ax[a].transAxes,
            # fontsize=10,
        )

    cbar_ax = ax[0].inset_axes([0, 1.02, 1, 0.05])
    bar = fig.colorbar(nz_image, cax=cbar_ax, pad=0, orientation="horizontal")
    # bar .ax.xaxis.set_tick_params(pad=2)
    bar.set_label(r"$\mathrm{log\:Gas\:Mass\:\left[M_{\odot}\right]}$")
    bar.ax.xaxis.set_label_position("top")
    bar.ax.xaxis.set_ticks_position("top")

    ax[1].set(ylabel=r"log Metallicity [Z$_\odot$]")
    ax[2].set(xlabel=r"radial velocity $[ \mathrm{km \: s^{-1} }  ]$")

    output_path = os.path.join(render_container, "radialvelocity_nZ.png")
    print("Saved", output_path)
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )

    plt.show()
